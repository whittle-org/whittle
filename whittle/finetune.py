from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal, Union

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from litgpt.args import EvalArgs
from litgpt.data import Alpaca, DataModule
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
    load_checkpoint,
    num_parameters,
    parse_devices,
    save_hyperparameters,
)

# from search_spaces import search_spaces
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import RunningMean

from whittle.args import FineTuningArgs
from whittle.models.gpt import GPT
from whittle.sampling import RandomSampler
from whittle.training_strategies import (
    RandomStrategy,
    SandwichStrategy,
    StandardStrategy,
)
from whittle.training_strategies.base_strategy import BaseTrainingStrategy


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path("out/finetune/full"),
    precision: str | None = None,
    devices: int | str = "auto",
    num_nodes: int = 1,
    num_dataloader_workers: int = 16,
    resume: bool | Literal["auto"] | Path = False,
    data: DataModule | None = None,
    train: FineTuningArgs = FineTuningArgs(
        save_interval=1000,
        log_interval=2048,
        global_batch_size=2048,
        micro_batch_size=16,
        lr_warmup_steps=100,
        epochs=5,
        max_seq_length=None,
        learning_rate=2e-5,
        temperature=10,
        distillation_weight=0.5,
    ),
    train_strategy: str = "sandwich",
    sampling_strategy: str = "random",
    eval: EvalArgs = EvalArgs(interval=1000, max_new_tokens=100, max_iters=100),
    optimizer: str | dict = "AdamW",
    logger_name: Literal["wandb", "tensorboard", "csv"] = "tensorboard",
    seed: int = 1337,
    access_token: str | None = None,
    downstream_test_iters: int = 500,
    downstream_dataset: str = "arc_easy",
    importance_objective: str = "norm",
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
    # pprint(locals())
    data = Alpaca(num_workers=num_dataloader_workers) if data is None else data
    # data = OpenWebText(num_workers=num_dataloader_workers) if data is None else data
    devices: int = parse_devices(devices)
    out_dir = init_out_dir(out_dir)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    config.fix_head_size = True
    config.model_type = "gpt"

    from syne_tune.config_space import lograndint, randint

    search_space = {
        "embed_dim": lograndint(1, config.n_embd),
        "num_heads": randint(1, config.n_head),
        "mlp_ratio": randint(1, 4),
        "depth": randint(1, config.n_layer),
    }

    # search_space = search_spaces[search_space_type](config)

    # config_14m.tie_embeddings = False
    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"finetune-{config.name}",
        resume=bool(resume),
        log_interval=train.log_interval,
    )

    if devices * num_nodes > 1:
        # For hybrid sharding, we want to keep parameter sharding within each node
        # strategy = FSDPStrategy(
        #     sharding_strategy="HYBRID_SHARD", device_mesh=(devices, num_nodes)
        # )
        strategy = FSDPStrategy()
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        loggers=logger,
    )

    sampler = RandomSampler(config_space=search_space, seed=seed)

    if train_strategy == "sandwich":
        strategy = SandwichStrategy(
            loss_function=chunked_cross_entropy,
            sampler=sampler,
        )
    elif train_strategy == "standard":
        strategy = StandardStrategy(
            loss_function=chunked_cross_entropy,
            sampler=sampler,
        )
    elif train_strategy == "random":
        strategy = RandomStrategy(
            loss_function=chunked_cross_entropy,
            sampler=sampler,
        )

    strategy.fabric = fabric
    strategy.gradient_accumulation_step = train.gradient_accumulation_iters(devices)
    # strategy.sampler.config_space = config_space
    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.print("launching")
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
    optimizer: Union[str, dict],
    train_strategy: BaseTrainingStrategy,
    importance_objective: str,
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
        checkpoint_path = checkpoint_dir / "lit_model.pth"
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

    resume = find_resume_path(resume, out_dir)
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)
    else:
        load_checkpoint(fabric, state["model"], checkpoint_path)

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
        data,
        train_strategy,
    )
    if fabric.device.type == "cuda" and fabric.is_global_zero:
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Final evaluation
    if eval.final_validation and fabric.is_global_zero:
        metrics = {
            "training_time": state["train_time"],
        }
        fabric.log_dict(metrics, step=state["iter_num"])


def fit(
    fabric: L.Fabric,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    devices: int,
    resume: Union[bool, Literal["auto"], Path],
    checkpoint_dir: Path,
    out_dir: Path,
    train: FineTuningArgs,
    data: DataModule,
    train_strategy: BaseTrainingStrategy,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    # tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(
        ConcatDataset([train_dataloader.dataset, val_dataloader.dataset])
    )
    model.max_seq_length = min(longest_seq_length, train.max_seq_length or float("inf"))
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

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

    if fabric.is_global_zero:
        train_start_time = time.perf_counter()

    forward_times = []
    while state["step_count"] < max_steps and train_iterator.epoch < train.epochs:
        if state["iter_num"] == 0:
            print(
                "Starting training at ", time.strftime("%H:%M:%S %Z", time.localtime())
            )
        state["iter_num"] += 1
        iter_t0 = time.perf_counter()
        batch = next(train_iterator)
        time.perf_counter()
        input_ids, targets = batch["input_ids"], batch["labels"]

        is_accumulating = (
            state["iter_num"] % train.gradient_accumulation_iters(devices) != 0
        )

        if fabric.is_global_zero:
            pre_forward_time = time.perf_counter()

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            loss = train_strategy(
                model, input_ids, targets, train.gradient_accumulation_iters(devices)
            )
            if fabric.is_global_zero:
                forward_times.append(time.perf_counter() - pre_forward_time)

        running_loss.update(loss)

        pre_step_time = time.perf_counter()
        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            state["step_count"] += 1
            print("Step ", state["step_count"])
            if state["step_count"] % 512 == 0:
                print(f"Step {state['step_count']} | Loss {loss:.3f}")
            post_step_time = time.perf_counter()
            print("train loss: ", loss)
            print("Step time: ", post_step_time - pre_step_time)

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
            fabric.print(
                f"Epoch {metrics['epoch'] + 1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" tokens: {metrics['tokens']} |"
                f" total_tokens: {metrics['total_tokens']} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
            )
            fabric.log_dict(metrics, step=state["iter_num"])

        if fabric.is_global_zero:
            state["train_time"] = time.perf_counter() - train_start_time

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
            # fabric.save(checkpoint_file, state)
            if fabric.is_global_zero:
                copy_config_files(checkpoint_dir, checkpoint_file.parent)
                save_hyperparameters(setup, checkpoint_file.parent)
                save_prompt_style(data.prompt_style, checkpoint_file.parent)

    if fabric.is_global_zero:
        print("Training time: ", state["train_time"])
        print("Forward times: ", torch.mean(torch.tensor(forward_times)))


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

    CLI(setup)
