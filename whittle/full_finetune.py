from __future__ import annotations

import dataclasses
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from pprint import pprint
from typing import Literal

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import DataModule
from litgpt.generate.base import generate
from litgpt.model import Config
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
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    load_checkpoint,
    num_parameters,
    parse_devices,
    save_hyperparameters,
)
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import RunningMean

from whittle.data.llamamini import LLaMaMini
from whittle.models.gpt import GPT
from whittle.models.gpt.blocks import Block
from whittle.pretrain_super_network import get_search_space, training_strategies_cls
from whittle.sampling.random_sampler import RandomSampler
from whittle.training_strategies.base_strategy import BaseTrainingStrategy


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path("out/finetune/full"),
    precision: str | None = None,
    devices: int | str = 1,
    num_nodes: int = 1,
    resume: bool | Literal["auto"] | Path = False,
    data: DataModule | None = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=16,
        micro_batch_size=1,
        lr_warmup_steps=100,
        epochs=5,
        max_seq_length=None,
        max_steps=10000,
        #        max_tokens=int(3e12),  # 3 trillion
        #        max_norm=1.0,
        min_lr=4e-5,
    ),
    eval: EvalArgs = EvalArgs(interval=600, max_new_tokens=100),
    optimizer: str | dict = "AdamW",
    training_strategy: str = "sandwich",
    logger_name: Literal["wandb", "tensorboard", "csv"] = "csv",
    seed: int = 1337,
    access_token: str | None = None,
    accelerator: str | None = None,
) -> None:
    """Finetune a model using super-network training strategies.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        devices: How many devices/GPUs to use
        num_nodes: How many nodes the code is being run on.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``. An error will be raised if no checkpoint is found. Passing
            ``'auto'`` will resume from the latest checkpoint but not error if no checkpoint exists.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.Alpaca``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        training_strategy: Training strategy for super-network training. Possible choices: sandwich, standard
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
        accelerator: The accelerator to use for training. Possible choices: "cuda", "cpu".
    """
    checkpoint_dir = auto_download_checkpoint(
        model_name=checkpoint_dir, access_token=access_token
    )
    pprint(locals())

    def init_LLaMaMini() -> LLaMaMini:
        return (
            LLaMaMini() if access_token is None else LLaMaMini(access_token=access_token)
        )

    data = init_LLaMaMini() if data is None else data
    num_devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    config.fix_head_size = True

    if accelerator is None:
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"finetune-{config.name}",
        resume=bool(resume),
        log_interval=train.log_interval,
    )

    assert training_strategy in training_strategies_cls, print(
        f"Training strategy is {training_strategy}. Should be in {list(training_strategies_cls)}"
    )

    # Use simple strategy for CPU or single-device setups
    if accelerator == "cpu" or num_devices * num_nodes <= 1:
        strategy = "auto"
    else:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )

    fabric = L.Fabric(
        devices=num_devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        loggers=logger,
        accelerator=accelerator,
    )

    # Only check NVLink for CUDA and multi-device setups
    if accelerator == "cuda" and num_devices > 1 and torch.cuda.is_available():
        check_nvlink_connectivity(fabric)

    fabric.launch(
        main,
        num_devices,
        resume,
        seed,
        config,
        data,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        optimizer,
        training_strategy,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    resume: bool | Literal["auto"] | Path,
    seed: int,
    config: Config,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: str | dict,
    training_strategy: str,
) -> None:
    validate_args(train, eval)

    tokenizer = Tokenizer(checkpoint_dir)
    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train)
    steps_per_epoch = len(train_dataloader) // train.gradient_accumulation_iters(devices)
    lr_max_steps = min(train.epochs * steps_per_epoch, (train.max_steps or float("inf")))

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)

    fabric.print(
        f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}"
    )

    model = fabric.setup(model)

    optimizer = instantiate_torch_optimizer(optimizer, model.parameters())
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(
        optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps
    )

    sampler = RandomSampler(search_space=get_search_space(config), seed=seed)
    training_strategy_kwargs = {
        "loss_function": chunked_cross_entropy,
        "sampler": sampler,
        "fabric": fabric,
    }
    strategy = training_strategies_cls[training_strategy](**training_strategy_kwargs)

    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "iter_num": 0,
        "step_count": 0,
    }

    resume = find_resume_path(resume, out_dir)
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)
    else:
        load_checkpoint(fabric, state["model"], checkpoint_path)

    train_time = time.perf_counter()
    fit(
        fabric,
        state,
        train_dataloader,
        val_dataloader,
        devices,
        resume,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        data,
        strategy,
    )
    train_duration = time.perf_counter() - train_time

    fabric.print(f"Training time: {train_duration:.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Final evaluation
    if eval.final_validation:
        val_loss = validate(
            fabric,
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=len(val_dataloader)),
        )
        metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(
            f"Final evaluation | val loss: {val_loss.item():.3f} | val ppl: {math.exp(val_loss):.3f}"
        )

    # Save the final checkpoint at the end of training
    save_path = out_dir / "final" / "lit_model.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fabric.save(save_path, {"model": state["model"]})
    if fabric.global_rank == 0:
        # Copy checkpoint files from original checkpoint dir
        copy_config_files(checkpoint_dir, save_path.parent)
        save_hyperparameters(setup, save_path.parent)
        save_prompt_style(data.prompt_style, save_path.parent)


def fit(
    fabric: L.Fabric,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    devices: int,
    resume: bool | Literal["auto"] | Path,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    data: DataModule,
    training_strategy: BaseTrainingStrategy,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(
        ConcatDataset([train_dataloader.dataset, val_dataloader.dataset])
    )
    model.max_seq_length = min(longest_seq_length, train.max_seq_length or float("inf"))
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    if eval.initial_validation:
        model.reset_super_network()
        val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
        val_loss = f"{val_loss:.3f}"
    else:
        model.reset_super_network()
        fabric.print("Verifying settings ...")
        validate(
            fabric,
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=2),
            verbose=False,
        )  # sanity check
        val_loss = "n/a"

    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("cpu"):
        meta_model = GPT(model.config)
        x = torch.randint(0, 1, (train.micro_batch_size, meta_model.max_seq_length))

        def model_fwd():
            return meta_model(x)  # noqa: F821

        def model_loss(y):
            return chunked_cross_entropy(y, x, chunk_size=0)  # noqa: F821

        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    log_iter_interval = train.log_interval * train.gradient_accumulation_iters(devices)
    initial_iter = state["iter_num"]
    max_steps = train.max_steps or float("inf")
    train_iterator = CycleIterator(train_dataloader)

    # resume data loader state by fast-forwarding through all seen batches
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

    running_loss = RunningMean(
        window=train.gradient_accumulation_iters(devices), sync_on_compute=False
    ).to(fabric.device)
    fabric.barrier()
    total_t0 = time.perf_counter()

    while state["step_count"] < max_steps:
        state["iter_num"] += 1
        iter_t0 = time.perf_counter()
        batch = next(train_iterator)
        if train_iterator.epoch >= train.epochs:
            break
        input_ids = batch["input_ids"].to(fabric.device)
        targets = batch["labels"].to(fabric.device)

        is_accumulating = (
            state["iter_num"] % train.gradient_accumulation_iters(devices) != 0
        )

        if hasattr(training_strategy, "random_samples"):
            # if we update multiple sub-networks in each step, we need to further scale the gradient
            scale_loss = 1 / (
                train.gradient_accumulation_iters(devices)
                * training_strategy.random_samples
            )
        else:
            scale_loss = 1 / train.gradient_accumulation_iters(devices)

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            loss = training_strategy(model, input_ids, targets, scale_loss=scale_loss)

        running_loss.update(loss)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            state["step_count"] += 1

        if state["iter_num"] % train.log_interval == 0:
            loss = (
                running_loss.compute().item()
            )  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * train.micro_batch_size),
                lengths=(
                    state["iter_num"] * train.micro_batch_size * model.max_seq_length
                ),
            )

            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0)
                    / (state["iter_num"] - initial_iter)
                    * (max_steps - state["iter_num"])
                ),
                "tokens": state["iter_num"]
                * train.micro_batch_size
                * model.max_seq_length,
                "total_tokens": (
                    state["iter_num"]
                    * train.micro_batch_size
                    * model.max_seq_length
                    * fabric.world_size
                ),
                "learning_rate": scheduler.get_last_lr(),
            }
            print(metrics)
            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.3f}"
            fabric.print(
                f"Epoch {metrics['epoch'] + 1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
                f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"] - 1)

        if not is_accumulating and state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()
            model.reset_super_network()
            val_loss = validate(fabric, model, val_dataloader, eval=eval)
            val_loss = val_loss.item()
            td = time.perf_counter() - t0

            fabric.print(
                f"iter {state['iter_num']}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms"
            )
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            fabric.log_dict(metrics, step=state["iter_num"] - 1)
            fabric.barrier()
        if (
            train.save_interval is not None
            and not is_accumulating
            and state["step_count"] % train.save_interval == 0
        ):
            checkpoint_file = (
                out_dir / f"step-{state['step_count']:06d}" / "lit_model.pth"
            )
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            fabric.print(f"Saving checkpoint to {str(checkpoint_file.parent)!r}")
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
) -> torch.Tensor:
    if verbose:
        fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(min(len(val_dataloader), eval.max_iters), device=fabric.device)
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        input_ids = batch["input_ids"].to(fabric.device)
        targets = batch["labels"].to(fabric.device)
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(
            logits[..., :-1, :], targets[..., 1:], chunk_size=0
        )

    val_loss = losses.mean()
    model.train()
    return val_loss


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

    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)

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
        fabric.print(f"{output}\n")
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

    CLI(setup)
