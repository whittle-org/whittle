# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from __future__ import annotations

import dataclasses
import math
import os
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any, Literal

import lightning as L
import torch
import yaml  # type: ignore
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import DDPStrategy
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import Alpaca, DataModule
from litgpt.generate.base import generate
from litgpt.lora import (
    lora_filter,
    mark_only_lora_as_trainable,
)
from litgpt.prompts import save_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    CycleIterator,
    auto_download_checkpoint,
    check_nvlink_connectivity,
    check_valid_checkpoint_dir,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    get_default_supported_precision,
    init_out_dir,
    instantiate_bnb_optimizer,
    instantiate_torch_optimizer,
    load_checkpoint,
    num_parameters,
    parse_devices,
    save_hyperparameters,
)
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import RunningMean

from search.search_spaces import search_spaces
from whittle.args import SamplerArgs
from whittle.data.llamamini import LLaMaMini
from whittle.eval.utils import compute_accuracy
from whittle.lora_model.config import LoRAConfig as Config
from whittle.lora_model.lora_gpt import GPT
from whittle.lora_model.merge import merge_lora
from whittle.sampling.samplers import get_sampler
from whittle.training_strategies.base_strategy import BaseTrainingStrategy
from whittle.training_strategies.sandwich import SandwichStrategy
from whittle.training_strategies.standard import StandardStrategy

# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""This script merges the LoRA weights with the base model"""


def compute_loss(model, val_dataloader, eval):
    losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(
            logits[..., :-1, :], targets[..., 1:], chunk_size=0
        )

    val_loss = losses.mean()
    return val_loss


def plot_validation_metrics(model, val_dataloader, eval, sampler):
    # compute loss for superent
    model.eval()
    model.reset_super_network()
    val_loss_largest = compute_loss(model, val_dataloader, eval)
    middle_config = sampler.get_medium_sub_network()
    model.set_sub_network(**middle_config)
    val_loss_medium = compute_loss(model, val_dataloader, eval)
    model.reset_super_network()
    smallest_config = sampler.get_smallest_sub_network()
    model.set_sub_network(**smallest_config)
    val_loss_smallest = compute_loss(model, val_dataloader, eval)
    model.reset_super_network()
    return val_loss_largest, val_loss_medium, val_loss_smallest


def plot_accuracies(model, sampler, dataset, checkpoint_dir):
    model.eval()
    model.reset_super_network()
    val_loss_largest = compute_accuracy(model, dataset, checkpoint_dir)
    middle_config = sampler.get_medium_sub_network()
    model.set_sub_network(**middle_config)
    val_loss_medium = compute_accuracy(model, dataset, checkpoint_dir)
    model.reset_super_network()
    smallest_config = sampler.get_smallest_sub_network()
    model.set_sub_network(**smallest_config)
    val_loss_smallest = compute_accuracy(model, dataset, checkpoint_dir)
    model.reset_super_network()
    return val_loss_largest, val_loss_medium, val_loss_smallest


def load_lora_metadata(
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], Path, str | None]:
    hparams_file = checkpoint_dir / "hyperparameters.yaml"
    if not hparams_file.is_file():
        raise FileNotFoundError(
            f"The path {str(hparams_file)!r} is not a valid checkpoint directory. It is missing a"
            f" `hyperparameters.yaml` file. Please point to the checkpoint directory that was produced by"
            f" the `litgpt/finetune/lora.py` script."
        )

    with open(hparams_file, encoding="utf-8") as file:
        hparams = yaml.safe_load(file)

    lora_params = {k: v for k, v in hparams.items() if k.startswith("lora_")}
    pretrained_checkpoint_dir = Path(hparams["checkpoint_dir"])
    precision = hparams.get("precision")
    return lora_params, pretrained_checkpoint_dir, precision


def find_resume_path(resume: bool | Path, out_dir: Path) -> bool | Path:
    resume_path = out_dir / "final" / "lit_model.pth.lora"
    if not resume_path.exists():
        return False
    return resume_path


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path("out/finetune/lora"),
    precision: str | None = None,
    devices: int = 1,
    num_nodes: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_query: bool = True,
    lora_key: bool = True,
    lora_value: bool = True,
    lora_projection: bool = True,
    lora_mlp: bool = True,
    lora_head: bool = True,
    lora_emb: bool = True,
    data: DataModule | None = None,
    train: TrainArgs = TrainArgs(
        save_interval=100,
        log_interval=1,
        global_batch_size=64,
        micro_batch_size=1,
        lr_warmup_steps=100,
        epochs=1,
        max_seq_length=None,
    ),
    train_strategy: str = "sandwich",
    search_space_type: str = "hw_gpt_bench",
    eval: EvalArgs = EvalArgs(interval=10, max_new_tokens=100, max_iters=100),
    optimizer: str | dict = "AdamW",
    logger_name: Literal["wandb", "tensorboard", "csv"] = "wandb",
    seed: int = 1337,
    access_token: str | None = None,
    downstream_test_iters: int = 500,
    downstream_dataset: str = "arc_easy",
    resume: bool | Path = True,
    dataset: str = "alpaca",
    sampler: SamplerArgs = SamplerArgs(),
) -> None:
    """Finetune a model using the LoRA method using super-network training strategies.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        devices: How many devices/GPUs to use.
        num_nodes: How many nodes the code is being run on.
        lora_r: The LoRA rank.
        lora_alpha: The LoRA alpha.
        lora_dropout: The LoRA dropout value.
        lora_emb: Whether to apply LoRA to the embedding weights in the model.
        lora_query: Whether to apply LoRA to the query weights in attention.
        lora_key: Whether to apply LoRA to the key weights in attention.
        lora_value: Whether to apply LoRA to the value weights in attention.
        lora_projection: Whether to apply LoRA to the output projection in the attention block.
        lora_mlp: Whether to apply LoRA to the weights of the MLP in the attention block.
        lora_head: Whether to apply LoRA to output head in GPT.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.Alpaca``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        train_strategy: The training strategy to use. Possible choices: "sandwich", "standard".
        search_space_type: The search space to use. Possible choices: "small", "medium", "hw_gpt_bench", "llama_joint".
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
        downstream_test_iters: The number of iterations after which to test the model on a downstream dataset.
        downstream_dataset: The downstream dataset to test on.
        resume: Whether to resume training from the last checkpoint.
        dataset: The dataset to use for finetuning. Possible choices: "alpaca", "llamamini".
        sampler: Sampler-related arguments. See ``whittle.args.SamplerArgs`` for details.
    """

    checkpoint_dir = auto_download_checkpoint(
        model_name=checkpoint_dir, access_token=access_token
    )
    pprint(locals())
    data_str = dataset
    if data_str == "alpaca":
        data = Alpaca()
    elif data_str == "llamamini":
        data = LLaMaMini()
    else:
        data = None
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(
        checkpoint_dir / "model_config.yaml",
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_query=lora_query,
        lora_key=lora_key,
        lora_value=lora_value,
        lora_projection=lora_projection,
        lora_mlp=lora_mlp,
        lora_head=lora_head,
        lora_emb=lora_emb,
    )
    config.fix_head_size = True
    config.tie_embeddings = False
    config.model_type = "gpt"
    now = datetime.now()
    sampling_strategy = sampler.sampling_strategy
    seed_sampler = sampler.seed_sampler
    num_configs = sampler.num_configs
    n_trials = sampler.n_trials
    # Create a timestamp with nanosecond precision
    time_string = now.strftime("%Y%m%d_%H%M%S")
    search_space = search_spaces[search_space_type](config)
    out_dir = Path(
        f"{config.name}-{train_strategy}-{search_space_type}-{sampling_strategy}-{data_str}/finetune/lora/"
    )
    id = f"{train_strategy}-{search_space_type}-{sampling_strategy}-{time_string}-{data_str}-lora"
    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"finetune-{config.name}",
        log_interval=train.log_interval,
        id=id,
        resume=bool(resume),
        config=dict(
            train_strategy=train_strategy,
            search_space_type=search_space_type,
            sampling_strategy=sampling_strategy,
            data=data_str,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_emb=lora_emb,
            lora_mlp=lora_mlp,
            lora_head=lora_head,
            lora_projection=lora_projection,
            lr_warmup_steps=train.lr_warmup_steps,
        ),
    )

    plugins = None

    if devices * num_nodes > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        loggers=logger,
        plugins=plugins,
    )

    sampler = get_sampler(
        sampling_strategy,
        search_space=search_space,
        seed=seed_sampler,
        num_configs=num_configs,
        n_trials=n_trials,
    )
    if train_strategy == "sandwich":
        strategy = SandwichStrategy(
            loss_function=chunked_cross_entropy,
            lora=True,
            sampler=sampler,
        )

    elif train_strategy == "standard":
        strategy = StandardStrategy(
            loss_function=chunked_cross_entropy,
            lora=True,
            sampler=sampler,
        )
    else:
        raise ValueError(f"Invalid training strategy: {train_strategy}")

    strategy.fabric = fabric
    strategy.gradient_accumulation_step = train.gradient_accumulation_iters(devices)
    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch(
        main,
        devices,
        seed,
        config,
        data,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        optimizer,
        sampling_strategy,
        strategy,
        downstream_dataset,
        downstream_test_iters,
        resume,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    config: Config,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: str | dict,
    sampling_strategy: str,
    strategy: BaseTrainingStrategy,
    downstream_dataset: str,
    downstream_test_iters: int,
    resume: bool,
) -> None:
    validate_args(train, eval)

    tokenizer = Tokenizer(checkpoint_dir)
    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train)
    steps_per_epoch = len(train_dataloader) // train.gradient_accumulation_iters(devices)
    lr_max_steps = min(train.epochs * steps_per_epoch, (train.max_steps or float("inf")))

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = checkpoint_dir / "lit_model.pth"  # type: ignore
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
        if "grid_params" in sampling_strategy:
            print("Initializing params grid....")
            strategy.sampler.initialize_grid(model)
            print("Grid Size", len(strategy.sampler.grid))
    model.name_or_path = checkpoint_dir
    mark_only_lora_as_trainable(model)

    fabric.print(
        f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}"
    )
    fabric.print(
        f"Number of non-trainable parameters: {num_parameters(model, requires_grad=False):,}"
    )

    model = fabric.setup_module(model)

    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        optimizer = instantiate_bnb_optimizer(optimizer, model.parameters())
    else:
        optimizer = instantiate_torch_optimizer(optimizer, model.parameters())

    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(
        optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps
    )
    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "iter_num": 0,
        "step_count": 0,
    }
    # strict=False because missing keys due to LoRA weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=False)
    if resume:
        resume = find_resume_path(resume, out_dir)  # type: ignore
        if resume:
            fabric.load(resume, state)
    train_time = time.perf_counter()
    fit(
        fabric,
        state,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        devices,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        data,
        strategy,
        downstream_dataset,
        downstream_test_iters,
        resume,
    )
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Final evaluation
    if eval.final_validation:
        val_loss_largest, val_loss_medium, val_loss_smallest = validate(
            fabric,
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=len(val_dataloader)),
            True,
            strategy.sampler,
        )
        metrics = {"val_loss": val_loss_largest, "val_ppl": math.exp(val_loss_largest)}
        fabric.log_dict(metrics)
        fabric.print(
            f"Final evaluation | val loss: {val_loss_largest.item():.3f} | val ppl: {math.exp(val_loss_largest):.3f}"
        )

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "final" / "lit_model.pth.lora"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_lora_checkpoint(fabric, model, save_path)
    if fabric.global_rank == 0:
        # Copy checkpoint files from original checkpoint dir
        copy_config_files(checkpoint_dir, save_path.parent)
        save_hyperparameters(setup, save_path.parent)
        save_prompt_style(data.prompt_style, save_path.parent)
        merge_lora(
            checkpoint_dir=save_path.parent,
            sampling_strategy=sampling_strategy,
        )


def fit(
    fabric: L.Fabric,
    state: dict,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    devices: int,
    checkpoint_dir: Path | str,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    data: DataModule,
    train_strategy: BaseTrainingStrategy,
    downstream_dataset: str,
    downstream_test_iters: int,
    resume: bool,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(
        ConcatDataset([train_dataloader.dataset, val_dataloader.dataset])
    )  # type: ignore
    model.max_seq_length = min(longest_seq_length, train.max_seq_length or float("inf"))  # type: ignore
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    if eval.initial_validation:
        val_loss_largest, val_loss_medium, val_loss_smallest = validate(
            fabric,
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=len(val_dataloader)),
            verbose=True,
            sampler=train_strategy.sampler,
        )
        val_loss_largest = f"{val_loss_largest:.3f}"
        val_loss_medium = f"{val_loss_medium:.3f}"
        val_loss_smallest = f"{val_loss_smallest:.3f}"
    else:
        fabric.print("Verifying settings ...")
        validate(
            fabric,
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=2),
            verbose=False,
            sampler=train_strategy.sampler,
        )  # sanity check
        val_loss_largest = "n/a"
        val_loss_medium = "n/a"
        val_loss_smallest = "n/a"
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)
    if resume:
        resume_t0 = time.perf_counter()
        for resume_iter in range(initial_iter):
            next(train_iterator)
            if resume_iter % 1000 == 0:
                fabric.print(f"Resuming dataset: {resume_iter} / {initial_iter}")
        fabric.barrier()
        fabric.print(
            f"Resuming data loader finished. Took {time.perf_counter() - resume_t0:.1f} seconds to reach iteration"
            f" {initial_iter}."
        )
    # throughput = ThroughputMonitor(fabric, window_size=50)
    running_loss = RunningMean(
        window=train.gradient_accumulation_iters(devices), sync_on_compute=False
    ).to(fabric.device)
    max_steps = train.max_steps or float("inf")
    total_lengths = 0
    time.perf_counter()

    while state["step_count"] < max_steps and train_iterator.epoch < train.epochs:
        state["iter_num"] += 1
        iter_t0 = time.perf_counter()
        batch = next(train_iterator)
        input_ids, targets = batch["input_ids"], batch["labels"]

        is_accumulating = (
            state["iter_num"] % train.gradient_accumulation_iters(devices) != 0
        )
        if hasattr(train_strategy, "random_samples"):
            # if we update multiple sub-networks in each step, we need to further scale the gradient
            scale_loss = 1 / (
                train.gradient_accumulation_iters(devices) * train_strategy.random_samples
            )
        else:
            scale_loss = 1 / train.gradient_accumulation_iters(devices)
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            loss = train_strategy(
                model,
                input_ids,
                targets,
                scale_loss=scale_loss,
            )

        running_loss.update(loss)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            state["step_count"] += 1

        total_lengths += input_ids.numel()
        if state["iter_num"] % train.log_interval == 0:
            loss = (
                running_loss.compute().item()
            )  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            # throughput.update(
            #    time=t1 - total_t0,
            #    batches=state["iter_num"],
            #    samples=state["iter_num"] * train.micro_batch_size,
            #    lengths=total_lengths,
            # )
            # throughput.compute_and_log(step=state["iter_num"])
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "tokens": state["iter_num"]
                * train.micro_batch_size
                * model.config.block_size,
                "total_tokens": (
                    state["iter_num"]
                    * train.micro_batch_size
                    * model.config.block_size
                    * fabric.world_size
                ),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            if isinstance(val_loss_largest, torch.Tensor):
                val_loss_largest = f"{val_loss_largest:.3f}"
                val_loss_medium = f"{val_loss_medium:.3f}"
                val_loss_smallest = f"{val_loss_smallest:.3f}"
            fabric.print(
                f"Epoch {metrics['epoch'] + 1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val_loss_largest: {val_loss_largest} |"
                f" val_loss_medium: {val_loss_medium} |"
                f" val_loss_smallest: {val_loss_smallest} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
            )
            fabric.log_dict(metrics, step=state["iter_num"])

        if not is_accumulating and state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss_largest, val_loss_medium, val_loss_smallest = validate(
                fabric, model, val_dataloader, eval, True, train_strategy.sampler
            )
            generate_example(fabric, model, tokenizer, eval, data)
            t1 = time.perf_counter() - t0
            fabric.print(
                f"iter {state['iter_num']}: val loss largest{val_loss_largest.item():.4f},val loss medium{val_loss_medium.item():.4f}, val loss  smallest{val_loss_smallest.item():.4f}, val time: {t1 * 1000:.2f} ms"
            )
            metrics = {
                "val_loss_largest": val_loss_largest,
                "val_ppl_largest": math.exp(val_loss_largest),
                "val_loss_medium": val_loss_medium,
                "val_ppl_medium": math.exp(val_loss_medium),
                "val_loss_smallest": val_loss_smallest,
                "val_ppl_smallest": math.exp(val_loss_smallest),
            }
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.barrier()
        if not is_accumulating and state["step_count"] % downstream_test_iters == 0:
            t0 = time.perf_counter()
            acc_largest, acc_medium, acc_smallest = test_downstream(
                fabric, model, downstream_dataset, train_strategy.sampler, out_dir
            )
            t1 = time.perf_counter() - t0
            fabric.print(
                f"iter {state['iter_num']}: acc largest{acc_largest:.4f},acc medium{acc_medium:.4f}, acc smallest{acc_smallest:.4f}, val time: {t1 * 1000:.2f} ms"
            )
            metrics = {
                "acc_largest": acc_largest,
                "acc_medium": acc_medium,
                "acc_smallest": acc_smallest,
            }
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.barrier()

        if (
            train.save_interval is not None
            and not is_accumulating
            and state["step_count"] % train.save_interval == 0
        ):
            checkpoint_file = out_dir / "final" / "lit_model.pth.lora"
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            fabric.save(checkpoint_file, state)
            if fabric.global_rank == 0:
                copy_config_files(checkpoint_dir, checkpoint_file.parent)
                save_hyperparameters(setup, checkpoint_file.parent)
                save_prompt_style(data.prompt_style, checkpoint_file.parent)


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: GPT,
    val_dataloader: DataLoader,
    eval: EvalArgs,
    verbose: bool = True,
    sampler=None,
) -> torch.Tensor:
    if verbose:
        fabric.print("Validating ...")
    model.eval()
    val_loss_largest, val_loss_middle, val_loss_smallest = plot_validation_metrics(
        model, val_dataloader, eval, sampler
    )
    model.train()
    return val_loss_largest, val_loss_middle, val_loss_smallest


@torch.no_grad()
def test_downstream(
    fabric: L.Fabric,
    model: GPT,
    dataset: str = "arc_easy",
    sampler=None,
    checkpoint_dir=None,
) -> torch.Tensor:
    model.eval()
    acc_largest, acc_middle, acc_smallest = plot_accuracies(
        model, sampler, dataset, checkpoint_dir
    )
    model.train()
    return acc_largest, acc_middle, acc_smallest


@torch.no_grad()
def generate_example(
    fabric: L.Fabric, model: GPT, tokenizer: Tokenizer, eval: EvalArgs, data: DataModule
):
    instruction = (
        "Recommend a movie for me to watch during the weekend and explain the reason."
    )
    fabric.print(instruction)
    prompt = data.prompt_style.apply(instruction)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    model.eval()
    model.reset_super_network()
    max_returned_tokens = len(encoded) + eval.max_new_tokens

    if max_returned_tokens < model.max_seq_length:
        with fabric.init_tensor():
            # do not set `max_seq_length=max_returned_token` because memory is not a concern here
            model.set_kv_cache(batch_size=1)
        output = generate(
            model,
            encoded,
            max_returned_tokens=max_returned_tokens,
            temperature=0.8,
            eos_id=tokenizer.eos_id,
        )
        model.clear_kv_cache()
        model.train()
        output = tokenizer.decode(output)
        fabric.print(output)
    else:
        print(
            f"Length of encoded instruction ({len(encoded)}) and eval.max_new_tokens ({eval.max_new_tokens}) "
            f"exceeds model.max_seq_length ({model.max_seq_length}) used for training. Skipping example generation for efficiency. "
            f"The model's supported context size (post-training) is {model.config.block_size}."
        )


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / warmup_steps
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(max_steps - warmup_steps)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [scheduler1, scheduler2], milestones=[warmup_steps]
    )


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs
) -> tuple[DataLoader, DataLoader]:
    data.connect(
        tokenizer=tokenizer,
        batch_size=train.micro_batch_size,
        max_seq_length=train.max_seq_length,
    )
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )
    return train_dataloader, val_dataloader


def get_longest_seq_length(data: list[dict]) -> tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def save_lora_checkpoint(
    fabric: L.Fabric, model: torch.nn.Module, file_path: Path
) -> None:
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})


def validate_args(train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [
        (train, ["max_tokens", "max_norm", "tie_embeddings", "lr_warmup_fraction"])
    ]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(
                    f"{__file__} doesn't support the {name!r} argument. This is set in {args}"
                )
    required = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(
                    f"{__file__} requires the {name!r} argument. This is set in {args}"
                )
    if not train.epochs and not train.max_steps:
        issues.append(
            f"{__file__} requires either epochs or max_steps to be set. This is set in {train}"
        )
    if issues:
        raise ValueError("\n".join(issues))


if __name__ == "__main__":
    from jsonargparse import CLI

    # setup("EleutherAI/pythia-1b", search_space_gpt)
    CLI(setup)
