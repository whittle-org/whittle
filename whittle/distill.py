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
from lightning.fabric.utilities.throughput import ThroughputMonitor
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
from whittle.metrics.flops import compute_flops
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
    teacher_checkpoint_dir: Path | None = None,
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
        temperature=10,
        alpha=0.3,
        beta=0.7,
        loss="forward_kld",
        weight_scheme="other",
    ),
    eval: EvalArgs = EvalArgs(interval=50, max_iters=100, initial_validation=True),
    optimizer: str | dict = "AdamW",
    devices: int | str = "auto",
    num_nodes: int = 1,
    tokenizer_dir: Path | None = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "csv",
    seed: int = 42,
    min_ratio: float = 0.6,
    max_ratio: float = 0.61,
    teacher_logits_dir: Path | None = None,
    use_saved_logits: bool = False,
    random_init_student: bool = False,
):
    """Train a (random) subnet of the teacher model using knowledge distillation.

    Arguments:
        out_dir: Directory in which to save checkpoints and logs.
        precision: The precision to use for finetuning.
        teacher_checkpoint_dir: Path to a checkpoint directory to initialize the teacher model from.
        student_dir: Optional path to a directory to initialize the student model from.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyStories``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        distill: Distillation-related arguments. See ``whittle.args.DistillArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        devices: How many devices/GPUs to use. Uses all GPUs by default.
        num_nodes: How many nodes the code is being run on.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        min_ratio: Minimum allowed ratio (student_params / teacher_params).
        max_ratio: Maximum allowed ratio (student_params / teacher_params).
        teacher_logits_dir: Directory containing pre-computed teacher logits. Required if use_saved_logits=True.
        use_saved_logits: Whether to use pre-computed teacher logits from files or compute them online.
        random_init_student: If True, the student sub-network will be randomly initialized instead of inheriting weights from the teacher.
    """
    if teacher_checkpoint_dir is not None:
        print(f"Loading teacher model config from {teacher_checkpoint_dir}")
        teacher_config = Config.from_file(teacher_checkpoint_dir / "model_config.yaml")
        teacher_config.fix_head_size = True
    else:
        raise ValueError("teacher_checkpoint_dir is required")

    if use_saved_logits and teacher_logits_dir is None:
        raise ValueError("teacher_logits_dir is required when use_saved_logits=True")

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
        name=f"distill-{teacher_config.name}",
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
        teacher_checkpoint_dir,
        num_devices,
        teacher_config,
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
        teacher_logits_dir,
        use_saved_logits,
        random_init_student,
    )


def main(
    fabric: L.Fabric,
    teacher_checkpoint_dir: Path,
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
    teacher_logits_dir: Path | None = None,
    use_saved_logits: bool = False,
    random_init_student: bool = False,
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

    # Load teacher model only if not using saved logits for training
    teacher = None
    if not use_saved_logits:
        with fabric.init_module(empty_init=(fabric.world_size > 1)):
            teacher = GPT(teacher_config)

        checkpoint = os.path.join(teacher_checkpoint_dir, "lit_model.pth")
        teacher = fabric.setup(teacher)
        load_checkpoint(fabric, teacher, checkpoint, strict=False)

        assert teacher is not None
        teacher.eval()
        teacher_val_loss = validate(
            fabric, teacher, val_dataloader, max_iters=eval.max_iters
        )
        teacher_val_loss = teacher_val_loss.item()
        teacher_val_ppl = math.exp(teacher_val_loss)

        fabric.print(f"Teacher model loaded from {teacher_checkpoint_dir} (not compiled)")
        fabric.print(f"Teacher model has {compute_parameters(teacher):,} parameters")
        fabric.print(
            f"Teacher model validation loss: {teacher_val_loss:.3f}, validation PPL: {teacher_val_ppl:.3f}"
        )
        fabric.log_dict(
            {"teacher_val_loss": teacher_val_loss, "teacher_val_ppl": teacher_val_ppl}
        )
    else:
        fabric.print(f"Using pre-computed teacher logits from {teacher_logits_dir}")

    # Initialize saved logits loader if needed
    logits_loader = None
    if use_saved_logits:
        assert teacher_logits_dir is not None
        logits_loader = SavedLogitsLoader(teacher_logits_dir, fabric.device)
        fabric.print(f"Loaded saved logits with {logits_loader.num_batches} batches")

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
                "embed_dim": random_config["sub_network_n_embd"],
                "mlp_ratio": random_config["sub_network_intermediate_size"]
                / random_config["sub_network_n_embd"],
                "num_heads": random_config["sub_network_num_heads"],
                "depth": random_config["sub_network_n_layers"],
            }

            # When no student checkpoint is provided, we need to create a teacher model to extract the subnetwork
            try:
                if teacher is None:
                    # When using saved logits, create teacher on CPU to avoid GPU OOM
                    if use_saved_logits:
                        with torch.device("meta"):
                            temp_teacher = GPT(teacher_config)
                        checkpoint_data = torch.load(
                            os.path.join(teacher_checkpoint_dir, "lit_model.pth"),
                            map_location="cpu",
                            weights_only=True,
                        )
                        # Create a CPU version for subnet extraction
                        with torch.device("cpu"):
                            temp_teacher_cpu = GPT(teacher_config)
                        temp_teacher_cpu.load_state_dict(checkpoint_data, strict=False)
                        temp_teacher = temp_teacher_cpu
                    else:
                        with fabric.init_module(empty_init=(fabric.world_size > 1)):
                            temp_teacher = GPT(teacher_config)
                        checkpoint = os.path.join(teacher_checkpoint_dir, "lit_model.pth")
                        temp_teacher = fabric.setup(temp_teacher)
                        load_checkpoint(fabric, temp_teacher, checkpoint, strict=False)
                else:
                    temp_teacher = teacher

                temp_teacher.select_sub_network(subnetwork)
                student = extract_current_sub_network(temp_teacher)

                if random_init_student:
                    fabric.print("Randomly initializing student weights.")
                    student_config = student.config
                    with fabric.init_module(empty_init=(fabric.world_size > 1)):
                        student = GPT(student_config)

                param_count = compute_parameters(student)
                temp_teacher.reset_super_network()
                ratio = param_count / compute_parameters(temp_teacher)
                if min_ratio <= ratio <= max_ratio:
                    valid = True
                    # Clean up CPU teacher if we created one
                    if use_saved_logits and teacher is None:
                        del temp_teacher
                        torch.cuda.empty_cache()

            except Exception as e:
                fabric.print(f"Error during subnet extraction: {e}")
                fabric.print("Retrying with different random config...")
                continue
    else:
        with fabric.init_module(empty_init=(fabric.world_size > 1)):
            student = GPT(student_config)

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

    # Fused optimizer causes dtype mismatch issues, so we disable it
    extra_kwargs = {"fused": False}

    optimizer = instantiate_torch_optimizer(
        optimizer, student.parameters(), **extra_kwargs
    )
    optimizer = fabric.setup_optimizers(optimizer)

    param_count = compute_parameters(student)
    fabric.print(f"Student model has {param_count:,} parameters")

    if teacher is not None:
        fabric.print(
            f"Model parameter reduction: {param_count / compute_parameters(teacher):.2%}"
        )
        reduction_ratio = param_count / compute_parameters(teacher)
    else:
        # Estimate teacher parameters from config for logging
        with torch.device("meta"):
            temp_teacher = GPT(teacher_config)
            teacher_params = compute_parameters(temp_teacher)
        reduction_ratio = param_count / teacher_params
        fabric.print(f"Model parameter reduction: {reduction_ratio:.2%}")

    exp_config = {
        "seed": seed,
        "embed_dim": student.config.n_embd,
        "mlp_ratio": student.config.intermediate_size / student.config.n_embd,
        "depth": student.config.n_layer,
        "parameter_count": param_count,
        "reduction_ratio": reduction_ratio,
        "use_saved_logits": use_saved_logits,
    }

    if logger_name in ("tensorboard", "wandb"):
        fabric.logger.log_hyperparams(exp_config)

    train_dataloader, val_dataloader = get_dataloaders(
        fabric, dataset, tokenizer, train, teacher_config.block_size
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

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
        logits_loader,
        use_saved_logits,
    )

    save_checkpoint(
        fabric,
        student_state,
        tokenizer_dir,
        out_dir / "distill" / "lit_model.pth" if out_dir else None,
    )

    total_tokens = (
        state["iter_num"]
        * train.micro_batch_size
        * student.max_seq_length
        * fabric.world_size
    )

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


class SavedLogitsLoader:
    """Utility class to load pre-computed teacher logits from saved files."""

    def __init__(self, logits_dir: Path, device: torch.device):
        self.logits_dir = Path(logits_dir)
        self.device = device

        # Find all logits files
        self.logits_files = sorted((self.logits_dir / "logits").glob("logits_batch_*.pt"))
        self.indices_files = sorted(
            (self.logits_dir / "indices").glob("indices_batch_*.pt")
        )

        self.num_batches = len(self.logits_files)
        self.current_batch = 0
        self.current_sample = 0
        self.epoch = 0  # Track how many times we've cycled through all batches

        if self.num_batches > 0:
            first_batch = torch.load(self.logits_files[0], map_location="cpu")
            self.top_k = first_batch.get("top_k", None)
            self.vocab_size = (
                first_batch["logits"].shape[-1] if self.top_k is None else None
            )
            self.batch_size = first_batch["logits"].shape[0]
            self.seq_len = first_batch["logits"].shape[1]

        self._current_logits = None
        self._current_indices = None
        self._current_input_ids = None
        self._load_batch(0)

    def _load_batch(self, batch_idx: int):
        """Load a specific batch of logits to CPU first."""
        if batch_idx >= self.num_batches:
            return

        # Load to CPU first to avoid OOM
        logits_data = torch.load(self.logits_files[batch_idx], map_location="cpu")
        self._current_logits = logits_data["logits"]
        self._current_input_ids = logits_data["input_ids"]

        if self.top_k and batch_idx < len(self.indices_files):
            indices_data = torch.load(self.indices_files[batch_idx], map_location="cpu")
            self._current_indices = indices_data["indices"]
        else:
            self._current_indices = None

        self.current_batch = batch_idx
        self.current_sample = 0

    def reset_to_beginning(self):
        """Reset to the beginning of the saved logits for another epoch."""
        self.epoch += 1
        self._load_batch(0)

    def get_logits_for_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get teacher logits for given input_ids. Cycles through data if needed."""
        batch_size, seq_len = input_ids.shape

        # Check if we need to load next batch
        if (
            self._current_logits is None
            or self.current_sample + batch_size > self._current_logits.shape[0]
        ):
            if self.current_batch + 1 < self.num_batches:
                self._load_batch(self.current_batch + 1)
            else:
                # We've reached the end, cycle back to the beginning
                self.reset_to_beginning()

        # Extract logits for current batch
        start_idx = self.current_sample
        end_idx = start_idx + batch_size

        if self.top_k:
            if self._current_indices is None:
                raise ValueError("Top-k indices are missing for the current batch.")

            # Reconstruct full logits from top-k
            top_k_logits = self._current_logits[start_idx:end_idx].to(self.device)
            top_k_indices = self._current_indices[start_idx:end_idx].to(self.device)

            # Create full logits tensor with very negative values for non-top-k
            full_logits = torch.full(
                (batch_size, seq_len, 50304),  # GPT-2 vocab size
                fill_value=-1e9,
                device=self.device,
                dtype=top_k_logits.dtype,
            )

            # Fill in top-k values
            batch_indices = (
                torch.arange(batch_size, device=self.device).unsqueeze(1).unsqueeze(2)
            )
            seq_indices = (
                torch.arange(seq_len, device=self.device).unsqueeze(0).unsqueeze(2)
            )

            full_logits[batch_indices, seq_indices, top_k_indices] = top_k_logits
            teacher_logits = full_logits
        else:
            if self._current_logits is None:
                raise ValueError("Logits are missing for the current batch.")

            teacher_logits = self._current_logits[start_idx:end_idx].to(self.device)

        self.current_sample = end_idx
        return teacher_logits


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
    logits_loader: SavedLogitsLoader | None = None,
    use_saved_logits: bool = False,
) -> dict[str, Any]:
    teacher = state["teacher"]
    student = state["model"]
    optimizer = state["optimizer"]

    if teacher is not None:
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

    measured_flops = compute_flops(
        student,
        batch_size=train.micro_batch_size,
        sequence_length=student.max_seq_length,
        device=fabric.device.type,
        verbose=True,
    )
    fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")

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
            # Get teacher logits
            if use_saved_logits and logits_loader is not None:
                teacher_logits = logits_loader.get_logits_for_input(input_ids)
                # Log epoch information when cycling through saved logits
                if state["iter_num"] % log_iter_interval == 0 and logits_loader.epoch > 0:
                    fabric.print(
                        f"Using saved logits epoch {logits_loader.epoch + 1}, batch {logits_loader.current_batch + 1}/{logits_loader.num_batches}"
                    )
            else:
                teacher.eval()
                with torch.inference_mode():
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
        val_loss = val_loss.item()
        ppl = math.exp(val_loss)

        metrics = {
            "val_loss": val_loss,
            "val_ppl": ppl,
            "params": compute_parameters(student),
        }

        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(f"Final evaluation | val loss: {val_loss:.3f} | val ppl: {ppl:.3f}")

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
