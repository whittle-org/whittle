# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import dataclasses
import math
import os
import time
import json
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Literal, Optional, Tuple, Union
from syne_tune.config_space import randint, choice
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics import RunningMean
from whittle.loss.loss_factory import LossFactory
from litgpt.args import EvalArgs
from litgpt.data import Alpaca, DataModule
from litgpt.generate.base import generate
from litgpt.model import Config
from litgpt.prompts import save_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    auto_download_checkpoint,
    CycleIterator,
    check_nvlink_connectivity,
    check_valid_checkpoint_dir,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    get_default_supported_precision,
    load_checkpoint,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
    save_hyperparameters,
)

from whittle.models.gpt import GPT
from whittle.models.gpt.blocks.transformer_block import Block
from whittle.training_strategies.base_strategy import BaseTrainingStrategy
from src.finetuning.sandwich_kd import SandwichStrategy as SandwichStrategyKD
from utils import plot_validation_metrics, plot_accuracies
from distillation_loss import KDLoss
from sandwich import SandwichStrategy
from standard import StandardStrategy
from train_args import FineTuningArgs
from search_spaces import search_spaces
from sampler import (
    RandomSampler,
    FixGridSampler,
    FixParamGridSampler,
    CalibFixGridSampler,
    ImportanceSampler,
    ImportanceParamGridSampler,
    ImportanceCalibFixGridSampler,
)


def find_resume_path(
    resume: Union[bool, Literal["auto"], Path], out_dir: Path
) -> Optional[Path]:
    resume_path = out_dir / "final" / "lit_model.pth"
    if not resume_path.exists():
        return False
    return resume_path


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path("out/finetune/full"),
    precision: Optional[str] = None,
    devices: Union[int, str] = 1,
    num_nodes: int = 1,
    resume: Union[bool, Literal["auto"], Path] = False,
    data: Optional[DataModule] = None,
    train: FineTuningArgs = FineTuningArgs(
        save_interval=100,
        log_interval=1,
        global_batch_size=16,
        micro_batch_size=1,
        lr_warmup_steps=100,
        epochs=5,
        max_seq_length=None,
        learning_rate=2e-5,
    ),
    train_strategy: str = "sandwich",
    search_space_type: str = "hw_gpt_bench",
    sampling_strategy: str = "random",
    eval: EvalArgs = EvalArgs(interval=100, max_new_tokens=100, max_iters=100),
    optimizer: Union[str, Dict] = "AdamW",
    logger_name: Literal["wandb", "tensorboard", "csv"] = "wandb",
    seed: int = 1337,
    access_token: Optional[str] = None,
    n_trials: int = 10000,
    downstream_test_iters: int = 500,
    downstream_dataset: str = "arc_easy",
    importance_objective: str = "norm",
    objective: str = "ppl",
    kd_loss: str = "forward_kl",
    kd_temperature: float = 0.9,
    kd_alpha: float = 0.5,
) -> None:
    """Finetune a model.

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
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
    """
    checkpoint_dir = auto_download_checkpoint(
        model_name=checkpoint_dir, access_token=access_token
    )
    pprint(locals())
    data = Alpaca() if data is None else data
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    config.fix_head_size = True
    config.model_type = "gpt"

    search_space = search_spaces[search_space_type](config)
    out_dir = Path(
        f"{config.name}-{train_strategy}-{search_space_type}-{sampling_strategy}-{kd_loss}/finetune/last_ft/"
    )
    # config_14m.tie_embeddings = False
    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"finetune-{config.name}",
        resume=bool(resume),
        log_interval=train.log_interval,
        id=f"{train_strategy}-{search_space_type}-{sampling_strategy}-{kd_loss}",
    )

    if devices * num_nodes > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        loggers=logger,
    )

    if sampling_strategy == "random":
        sampler = RandomSampler(search_space=search_space, seed=seed)
    elif sampling_strategy == "grid":
        sampler = FixGridSampler(search_space=search_space, seed=seed)
    elif sampling_strategy == "grid-params":
        sampler = FixParamGridSampler(
            search_space=search_space, seed=42, n_trials=n_trials
        )
    elif sampling_strategy == "calibrate":
        sampler = CalibFixGridSampler(
            objective=objective,
            checkpoint_dir=checkpoint_dir,
            search_space_type=search_space_type,
            search_space=search_space,
            seed=seed,
        )
    elif sampling_strategy == "importance-random":
        sampler = ImportanceSampler(
            os.path.join(checkpoint_dir, "sorted_ids.pkl"), search_space, seed=seed
        )
    elif sampling_strategy == "importance-grid-params":
        sampler = ImportanceParamGridSampler(
            sorted_ids_path=os.path.join(checkpoint_dir, "sorted_ids.pkl"),
            search_space=search_space,
            seed=42,
            n_trials=n_trials,
        )
    elif sampling_strategy == "importance-calibrate":
        sampler = ImportanceCalibFixGridSampler(
            objective=objective,
            importance_objective=importance_objective,
            sorted_ids_path=os.path.join(checkpoint_dir, "sorted_ids.pkl"),
            checkpoint_dir=checkpoint_dir,
            search_space_type=search_space_type,
            search_space=search_space,
            seed=seed,
        )
    loss_factory = LossFactory(alpha=kd_alpha, temperature=kd_temperature)
    if train_strategy == "sandwich-kd":
        strategy = SandwichStrategyKD(
            loss_function=loss_factory,
            sampler=sampler,
        )
    elif train_strategy == "sandwich":
        strategy = SandwichStrategy(
            loss_function=chunked_cross_entropy,
            sampler=sampler,
        )
    elif train_strategy == "standard":
        strategy = StandardStrategy(
            loss_function=chunked_cross_entropy,
            sampler=sampler,
        )

    strategy.fabric = fabric
    strategy.gradient_accumulation_step = train.gradient_accumulation_iters(devices)
    # strategy.sampler.config_space = config_space
    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch(
        main,
        devices,
        resume,
        seed,
        sampling_strategy,
        downstream_test_iters,
        downstream_dataset,
        config,
        data,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        optimizer,
        strategy,
        importance_objective,
        kd_loss,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    resume: Union[bool, Literal["auto"], Path],
    seed: int,
    sampling_strategy: str,
    downstream_test_iters: int,
    downstream_dataset: str,
    config: Config,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    train: FineTuningArgs,
    eval: EvalArgs,
    optimizer: Union[str, Dict],
    train_strategy: BaseTrainingStrategy,
    importance_objective: str,
    kd_loss: str,
) -> None:
    validate_args(train, eval)

    tokenizer = Tokenizer(checkpoint_dir)
    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train)
    steps_per_epoch = len(train_dataloader) // train.gradient_accumulation_iters(
        devices
    )
    lr_max_steps = min(
        train.epochs * steps_per_epoch, (train.max_steps or float("inf"))
    )

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if "importance" in sampling_strategy:
        checkpoint_path = (
            checkpoint_dir / f"lit_model_permuted_{importance_objective}.pth"
        )
    else:
        checkpoint_path = checkpoint_dir / f"lit_model.pth"
    with open(str(checkpoint_dir / "config.json")) as f:
        hf_config = json.load(f)
    config.tie_embeddings = hf_config["tie_word_embeddings"]
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
        if "grid-params" in sampling_strategy:
            print("Initializing params grid....")
            train_strategy.sampler.initialize_grid(model)
            print("Requested configs:", len(train_strategy.sampler.values))
            print("Grid Size", len(train_strategy.sampler.grid))

    model.name_or_path = checkpoint_dir
    fabric.print(
        f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}"
    )

    model = fabric.setup(model)
    for n, p in model.named_parameters():
        if "head" not in n:
            p.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=train.learning_rate)
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
    if resume:
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
        downstream_test_iters,
        downstream_dataset,
        data,
        train_strategy,
        kd_loss,
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
            train_strategy.sampler,
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
        fabric.print(
            f"Final evaluation | val loss: {val_loss_largest.item():.3f} | val ppl: {math.exp(val_loss_largest):.3f}"
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
    state: Dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    devices: int,
    resume: Union[bool, Literal["auto"], Path],
    checkpoint_dir: Path,
    out_dir: Path,
    train: FineTuningArgs,
    eval: EvalArgs,
    downstream_test_iters: int,
    downstream_dataset: str,
    data: DataModule,
    train_strategy: BaseTrainingStrategy,
    kd_loss: str,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(
        ConcatDataset([train_dataloader.dataset, val_dataloader.dataset])
    )
    model.max_seq_length = min(longest_seq_length, train.max_seq_length or float("inf"))
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

    while state["step_count"] < max_steps and train_iterator.epoch < train.epochs:
        state["iter_num"] += 1
        iter_t0 = time.perf_counter()
        batch = next(train_iterator)
        input_ids, targets = batch["input_ids"], batch["labels"]

        is_accumulating = (
            state["iter_num"] % train.gradient_accumulation_iters(devices) != 0
        )
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            loss = train_strategy(
                model,
                input_ids,
                targets,
                train.gradient_accumulation_iters(devices),
                kd_loss,
            )
            # logits = model(input_ids)
            # shift the targets such that output n predicts token n+1
            # loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
            # fabric.backward(loss / train.gradient_accumulation_iters(devices))

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
            checkpoint_file = out_dir / f"final" / "lit_model.pth"
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
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: FineTuningArgs
) -> Tuple[DataLoader, DataLoader]:
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


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def validate_args(train: FineTuningArgs, eval: EvalArgs) -> None:
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
