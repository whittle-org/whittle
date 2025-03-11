from __future__ import annotations

import math
import os
import pprint
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import lightning as L
import torch
import torch._dynamo
from jsonargparse import CLI
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from litgpt import Config, Tokenizer
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import DataModule, TinyStories
from litgpt.pretrain import (
    get_dataloaders,
    get_lr,
    validate,
)
from litgpt.utils import (
    CycleIterator,
    capture_hparams,
    check_nvlink_connectivity,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    load_checkpoint,
    parse_devices,
    save_config,
    save_hyperparameters,
)
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

from whittle.args import DistillArgs
from whittle.loss.kd_loss import DistillLoss
from whittle.metrics.parameters import compute_parameters
from whittle.models.gpt.blocks import Block
from whittle.models.gpt.extract import extract_current_sub_network
from whittle.models.gpt.model import GPT
from whittle.pretrain_super_network import get_search_space
from whittle.sampling.random_sampler import RandomSampler

torch._dynamo.config.suppress_errors = True


def setup(
    out_dir: Path = Path("out/distill"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Path | None = None,
    student_dir: Path | None = None,
    data: DataModule | None = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(5e8),
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
    ),
    distill: DistillArgs = DistillArgs(
        method="logits",
        temperature=5,
        alpha=0.6,
        beta=0.4,
        loss="forward_kld",
        weight_scheme="other",
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100, initial_validation=True),
    optimizer: str | dict = "AdamW",
    devices: int | str = "auto",
    num_nodes: int = 1,
    tokenizer_dir: Path | None = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "csv",
    seed: int = 42,
    min_ratio: float = 0.3,
    max_ratio: float = 0.7,
):
    """Train a (random) subnet of the teacher model using knowledge distillation.

    Arguments:
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Determines a compatible precision setting by default.
        initial_checkpoint_dir: Path to a checkpoint directory to initialize the teacher model from.
        student_dir: Optional path to a directory to initialize the student model from.
            Checks for student model config and checkpoint.
            If not provided, the student model will be initialized as a random subnetwork of the teacher model.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyStories``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        distill: Distillation-related arguments. See ``whittle.args.DistillArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        devices: How many devices/GPUs to use. Uses all GPUs by default.
        num_nodes: How many nodes the code is being run on.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
            modules require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        min_ratio: Minimum allowed ratio (student_params / teacher_params).
        max_ratio: Maximum allowed ratio (student_params / teacher_params).
    """
    if initial_checkpoint_dir is not None:
        print(f"Loading teacher model config from {initial_checkpoint_dir}")
        config = Config.from_file(initial_checkpoint_dir / "model_config.yaml")
        config.fix_head_size = True
    else:
        raise ValueError("initial_checkpoint_dir is required")

    if student_dir:
        print(f"Loading student model config from {student_dir}")
        student_config = Config.from_file(student_dir / "model_config.yaml")
    else:
        student_config = None
        assert min_ratio < max_ratio, "min_ratio must be less than max_ratio"
        assert 0 < min_ratio < 1, "min_ratio must be between 0 and 1"
        assert 0 < max_ratio < 1, "max_ratio must be between 0 and 1"

    hparams = capture_hparams()
    data = TinyStories() if data is None else data

    precision = precision or get_default_supported_precision(training=True)
    num_devices = int(parse_devices(devices))
    out_dir = init_out_dir(out_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"distill-{config.name}",
        log_interval=train.log_interval,
    )

    if num_devices * num_nodes > 1:
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
        devices=num_devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        loggers=[logger],
    )

    if torch.cuda.is_available() and num_devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch()

    fabric.print(pprint.pformat(hparams))
    if logger_name in ("tensorboard", "wandb"):
        fabric.logger.log_hyperparams(hparams)

    main(
        fabric,
        initial_checkpoint_dir,
        num_devices,
        config,
        student_config,
        data,
        out_dir,
        tokenizer_dir,
        tokenizer,
        seed,
        train,
        distill,
        eval,
        optimizer,
        student_dir,
        min_ratio,
        max_ratio,
        logger_name,
    )


def main(
    fabric: L.Fabric,
    initial_checkpoint_dir: Path,
    devices: int,
    teacher_config: Config,
    student_config: Config | None = None,
    dataset: DataModule | None = None,
    out_dir: Path | None = None,
    tokenizer_dir: Path | None = None,
    tokenizer: Tokenizer | None = None,
    seed: int = 42,
    train: TrainArgs = TrainArgs(),
    distill: DistillArgs = DistillArgs(),
    eval: EvalArgs = EvalArgs(),
    optimizer: str | dict = "AdamW",
    student_dir: Path | None = None,
    min_ratio: float = 0.3,
    max_ratio: float = 0.7,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "csv",
):
    if fabric.global_rank == 0 and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    fabric.seed_everything(seed)

    # Check if torch.compile() should be used for student model only
    use_compile_student = hasattr(torch, "_dynamo") and not bool(
        os.environ.get("DISABLE_TORCH_COMPILE")
    )

    train_dataloader, val_dataloader = get_dataloaders(
        fabric, dataset, tokenizer, train, teacher_config.block_size
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        teacher = GPT(teacher_config)

    checkpoint = os.path.join(initial_checkpoint_dir, "lit_model.pth")
    teacher = fabric.setup(teacher)
    load_checkpoint(fabric, teacher, checkpoint, strict=False)

    teacher.eval()
    teacher_val_loss = validate(fabric, teacher, val_dataloader, max_iters=eval.max_iters)
    teacher_val_loss = teacher_val_loss.item()
    teacher_val_ppl = math.exp(teacher_val_loss)

    fabric.print(f"Teacher model loaded from {initial_checkpoint_dir} (not compiled)")
    fabric.print(f"Teacher model has {compute_parameters(teacher):,} parameters")
    fabric.print(
        f"Teacher model validation loss: {teacher_val_loss:.3f}, validation PPL: {teacher_val_ppl:.3f}"
    )
    fabric.log_dict(
        {"teacher_val_loss": teacher_val_loss, "teacher_val_ppl": teacher_val_ppl}
    )

    if fabric.global_rank == 0 and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    student_checkpoint = (
        os.path.join(student_dir, "lit_model.pth") if student_dir else None
    )

    if student_config is None:
        fabric.print(
            "Student model set to a random subnetwork of the teacher model, within the given ratio bounds"
        )
        search_space = get_search_space(teacher_config)
        sampler = RandomSampler(search_space, seed=seed)
        valid = False
        while not valid:
            random_config = sampler.sample()
            fabric.print(f"Random subnetwork config: {random_config}")
            subnetwork = {
                "embed_dim": random_config["embed_dim"],
                "mlp_ratio": random_config["mlp_ratio"],
                "num_heads": teacher_config.n_head,  # keep the same number of heads as teacher
                "depth": random_config["depth"],
            }
            teacher.select_sub_network(subnetwork)
            student = extract_current_sub_network(teacher)

            param_count = compute_parameters(student)
            teacher.reset_super_network()
            ratio = param_count / compute_parameters(teacher)
            if min_ratio <= ratio <= max_ratio:
                valid = True

    student = fabric.setup_module(student, move_to_device=True)

    if student_checkpoint:
        load_checkpoint(fabric, student, student_checkpoint)
        fabric.print(f"Student model loaded from {student_dir} (not compiled)")

    if use_compile_student:
        try:
            fabric.print("Compiling student model...")
            student = torch.compile(student)
            fabric.print("Student model compiled successfully")
        except Exception as e:
            fabric.print(f"Error compiling student model: {e}")
            fabric.print("Continuing with uncompiled student model")

    # Use fused optimizer only if not compiling student to avoid dtype/device mismatches.
    if use_compile_student:
        extra_kwargs = {}
    else:
        extra_kwargs = {"fused": fabric.device.type == "cuda"}

    optimizer = instantiate_torch_optimizer(
        optimizer, student.parameters(), **extra_kwargs
    )
    optimizer = fabric.setup_optimizers(optimizer)

    param_count = compute_parameters(student)
    fabric.print(f"Student model has {param_count:,} parameters")
    fabric.print(
        f"Model parameter reduction: {param_count / compute_parameters(teacher):.2%}"
    )

    exp_config = {
        "seed": seed,
        "embed_dim": student.config.n_embd,
        "mlp_ratio": student.config.intermediate_size / student.config.n_embd,
        "depth": student.config.n_layer,
        "parameter_count": param_count,
        "reduction_ratio": param_count / compute_parameters(teacher),
    }

    if logger_name in ("tensorboard", "wandb"):
        fabric.logger.log_hyperparams(exp_config)

    state = {
        "teacher": teacher,
        "model": student,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "iter_num": 0,
        "step_count": 0,
    }

    student_state = {
        "model": student,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "iter_num": state["iter_num"],
        "step_count": state["step_count"],
    }

    train_time = time.perf_counter()
    fit(
        fabric,
        devices,
        state,
        train_dataloader,
        val_dataloader,
        out_dir if out_dir is not None else Path(""),
        tokenizer_dir,
        train,
        eval,
        distill,
    )

    save_checkpoint(
        fabric,
        student_state,
        tokenizer_dir,
        out_dir / "final" / "lit_model.pth" if out_dir else None,
    )

    # Track experiment results
    total_tokens = (
        state["iter_num"]
        * train.micro_batch_size
        * student.max_seq_length
        * fabric.world_size
    )

    # Print formatted output
    separator = "-" * 40
    fabric.print(separator)
    fabric.print("| Performance")
    fabric.print(f"| - Total tokens  : {total_tokens:,}")
    fabric.print(f"| - Training Time : {(time.perf_counter() - train_time):.2f} s")
    fabric.print(f"| - Tok/sec       : {total_tokens / train_time:.2f} tok/s")
    fabric.print("| " + "-" * 40)

    if fabric.device.type == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        fabric.print("| Memory Usage")
        fabric.print(f"| - Memory Used   : {memory_used:.2f} GB")
    fabric.print(separator)


def fit(
    fabric: L.Fabric,
    devices: int,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    out_dir: Path,
    tokenizer_dir: Path | None,
    train: TrainArgs,
    eval: EvalArgs,
    distill: DistillArgs,
) -> dict[str, Any]:
    teacher = state["teacher"]
    student = state["model"]
    optimizer = state["optimizer"]

    teacher.eval()

    distill_loss = DistillLoss(
        alpha=distill.alpha,
        beta=distill.beta,
        temperature=distill.temperature,
        loss=distill.loss,
        weight_scheme=distill.weight_scheme,
    )

    if eval.initial_validation:
        val_loss = validate(fabric, student, val_dataloader, max_iters=eval.max_iters)
        val_loss = f"{val_loss:.3f}"
    else:
        fabric.print("Running quick sanity check...")
        validate(fabric, student, val_dataloader, max_iters=2, verbose=False)
        val_loss = "n/a"

    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = GPT(student.config)
        x = torch.randint(0, 1, (train.micro_batch_size, meta_model.max_seq_length))

        def model_fwd():
            return meta_model(x)  # noqa: F821

        def model_loss(y):
            return chunked_cross_entropy(y, x, chunk_size=0)  # noqa: F821

        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = train.max_tokens // fabric.world_size
    tokens_per_iter = train.micro_batch_size * student.max_seq_length
    max_iters = max_tokens_per_device // tokens_per_iter
    log_iter_interval = train.log_interval * train.gradient_accumulation_iters(devices)
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    running_loss = RunningMean(
        window=train.gradient_accumulation_iters(devices), sync_on_compute=False
    ).to(fabric.device)

    fabric.barrier()
    total_t0 = time.perf_counter()

    warmup_iters = train.warmup_iters(devices, max_iters, train_dataloader)

    for train_data in train_iterator:
        if state["iter_num"] >= max_iters:
            break

        lr = get_lr(
            optimizer.defaults["lr"],
            state["iter_num"],
            warmup_iters,
            max_iters,
            train.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : student.max_seq_length].contiguous().long()
        targets = train_data[:, 1 : (student.max_seq_length + 1)].contiguous().long()

        is_accumulating = (
            state["iter_num"] % train.gradient_accumulation_iters(devices) != 0
        )
        with fabric.no_backward_sync(student, enabled=is_accumulating):
            teacher.eval()
            with torch.inference_mode():  # no grads for teacher
                teacher_logits = teacher(input_ids)

            teacher_logits = teacher_logits.clone()
            logits = student(input_ids)
            logits_reshaped = logits.view(-1, logits.size(-1))
            targets_reshaped = targets.view(-1)
            teacher_logits_reshaped = teacher_logits.view(-1, teacher_logits.size(-1))

            loss = distill_loss(
                logits_reshaped, targets_reshaped, teacher_logits_reshaped
            )
            fabric.backward(loss / train.gradient_accumulation_iters(devices))

        running_loss.update(loss.detach())

        if not is_accumulating:
            fabric.clip_gradients(student, optimizer, max_norm=train.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        if state["iter_num"] % log_iter_interval == 0:
            loss = running_loss.compute().item()
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * train.micro_batch_size),
                lengths=(
                    state["iter_num"] * train.micro_batch_size * student.max_seq_length
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
                    * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"]
                * train.micro_batch_size
                * student.max_seq_length,
                "total_tokens": (
                    state["iter_num"]
                    * train.micro_batch_size
                    * student.max_seq_length
                    * fabric.world_size
                ),
                "learning_rate": lr,
            }
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

        if (
            val_dataloader is not None
            and not is_accumulating
            and state["step_count"] % eval.interval == 0
        ):
            t0 = time.perf_counter()
            val_loss = validate(fabric, student, val_dataloader, max_iters=eval.max_iters)
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
            student_state = {
                "model": student,
                "optimizer": optimizer,
                "train_dataloader": train_dataloader,
                "iter_num": state["iter_num"],
                "step_count": state["step_count"],
            }
            save_checkpoint(
                fabric,
                student_state,
                tokenizer_dir,
                out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth",
            )
            fabric.barrier()

    if eval.final_validation:
        val_loss = validate(fabric, student, val_dataloader, max_iters=eval.max_iters)
        val_loss_value = val_loss.item()
        ppl = math.exp(val_loss_value)
        metrics = {
            "val_loss": val_loss_value,
            "val_ppl": ppl,
            "params": compute_parameters(student),
        }

        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(
            f"Final evaluation | val loss: {val_loss_value:.3f} | val ppl: {ppl:.3f}"
        )
    return metrics


def save_checkpoint(fabric, state, tokenizer_dir, checkpoint_file):
    model = state["model"]
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
    fabric.save(checkpoint_file, state)
    if fabric.global_rank == 0:
        save_hyperparameters(setup, checkpoint_file.parent)
        if tokenizer_dir is not None:
            copy_config_files(tokenizer_dir, checkpoint_file.parent)
        save_config(model.config, checkpoint_file.parent)


if __name__ == "__main__":
    CLI(setup)
