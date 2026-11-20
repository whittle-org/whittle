# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from __future__ import annotations

import math
import os
import pprint
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Literal

import lightning as L
import torch
from lightning.fabric.strategies import DDPStrategy, FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from litgpt import Tokenizer
from litgpt.args import EvalArgs, LogArgs, TrainArgs
from litgpt.config import name_to_config
from litgpt.data import DataModule, TinyLlama
from litgpt.model import GPT, Block, Config
from litgpt.pretrain import (
    copy_config_files,
    get_dataloaders,
    get_lr,
    initialize_weights,
    save_config,
    validate,
    validate_args,
)
from litgpt.utils import (
    CycleIterator,
    capture_hparams,
    check_nvlink_connectivity,
    choose_logger,
    chunked_cross_entropy,
    extend_checkpoint_dir,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
)
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

from whittle.args import DistillArgs
from whittle.hyperparameters import dump_hyperparameters, save_hyperparameters
from whittle.loss.kd_loss import DistillLoss
from whittle.metrics.parameters import compute_parameters
from whittle.models.gpt import GPT as WhittleGPT
from whittle.models.gpt.blocks import Block as WhittleBlock
from whittle.models.gpt.extract import extract_current_sub_network
from whittle.pretrain_super_network import get_search_space
from whittle.sampling.random_sampler import RandomSampler

MAX_STUDENT_SAMPLES = 1000


def setup(
    model_name: str | None = None,
    model_config: Config | None = None,
    out_dir: Path = Path("out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    teacher_checkpoint_dir: Path | None = None,
    initial_checkpoint_dir: Path | None = None,
    resume: bool | Literal["auto"] | Path = False,
    data: DataModule | None = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e12),  # 3 trillion
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
    ),
    distill: DistillArgs = DistillArgs(
        method="logits",
        temperature=0.9,
        alpha=0.3,
        beta=0.7,
        loss="forward_kld",
        weight_scheme="other",
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    log: LogArgs = LogArgs(),
    optimizer: str | dict = "AdamW",
    devices: int | str = "auto",
    num_nodes: int = 1,
    tokenizer_dir: Path | None = None,
    logger_name: Literal["wandb", "tensorboard", "csv", "mlflow"] = "tensorboard",
    seed: int = 42,
    init_from: str = "scratch",
    config_path: str | None = None,
    subnetwork_student: bool = False,
    min_ratio: float = 0.6,
    max_ratio: float = 0.61,
    random_init_student: bool = False,
    strategy: Literal["ddp", "fsdp"] = "ddp",
):
    """Distil a teacher litgpt model into a student litgpt model.

    Arguments:
        model_name: The name of the student model. Choose from names in
            ``litgpt.config``. Use "list" to list the supported models.
        model_config: A ``litgpt.Config`` object for the student architecture.
            Overrides the `model_name` if specified. Mutually exclusive with
            ``config_path``.
        out_dir: Directory in which to save checkpoints and logs. If running in a
            Lightning Studio Job, look for it in /teamspace/jobs/<job-name>/share.
        precision: The precision to use for training. Determines a compatible
            precision setting by default.
        teacher_checkpoint_dir: The litgpt checkpoint directory of the teacher. It must
            contain ``model_config.yaml`` and ``lit_model.pth``. Required.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize
            the student from. Mutually exclusive with ``resume``.
        resume: Path to a checkpoint directory to resume from in case training was
            interrupted, or ``True`` to resume from the latest checkpoint in
            ``out_dir``. An error will be raised if no checkpoint is found. Passing
            ``'auto'`` will resume from the latest checkpoint but not error if no
            checkpoint exists.
        data: Data-related arguments. If not provided, the default is
            ``litgpt.data.TinyLlama``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        distill: Distillation-related arguments (loss, temperature, and the weights
            ``alpha`` and ``beta``). See ``whittle.args.DistillArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        log: Logger-related arguments. See ``litgpt.args.LogArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        devices: How many devices/GPUs to use. Uses all GPUs by default.
        num_nodes: How many nodes the code is being run on.
        tokenizer_dir: Optional path to the tokenizer dir that was used for
            preprocessing the dataset. Only some data module require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        init_from: ``"scratch"`` to initialize the student weights at random, or the
            path to a ``.pth`` file with a raw state dict to load.
        config_path: Optional path to the ``model_config.yaml`` file of the student.
            Overrides the `model_name` if specified. Mutually exclusive with
            ``model_config``.
        subnetwork_student: If True, the student is a random sub-network of the teacher
            whose parameter count is within ``min_ratio`` and ``max_ratio`` of the
            teacher's. It inherits the teacher's weights. ``model_name``,
            ``model_config``, ``config_path`` and ``init_from`` are not used.
        min_ratio: Minimum allowed ratio (student_params / teacher_params). Only used
            with ``subnetwork_student``.
        max_ratio: Maximum allowed ratio (student_params / teacher_params). Only used
            with ``subnetwork_student``.
        random_init_student: If True, the sub-network student is randomly initialized
            instead of inheriting the teacher's weights. Only used with
            ``subnetwork_student``.
        strategy: ``"ddp"`` or ``"fsdp"``.
    """
    if model_name == "list":
        available_models = "\n".join(sorted(name_to_config))
        print(f"Available values:\n{available_models}")
        quit()
    # saved with each checkpoint; `locals()` holds only the arguments at this point
    hyperparameters = dump_hyperparameters(setup, locals())
    if (
        teacher_checkpoint_dir is not None
    ):  # We currently only use litgpt models - no further pretraining/finetuning
        print(f"Loading teacher model config from {teacher_checkpoint_dir}")
        teacher_config = Config.from_file(teacher_checkpoint_dir / "model_config.yaml")
    else:
        raise ValueError(
            "A teacher model checkpoint directory must be provided for distillation."
        )
    if initial_checkpoint_dir is not None:
        initial_checkpoint_dir = extend_checkpoint_dir(initial_checkpoint_dir)

    if tokenizer_dir is not None:
        tokenizer_dir = extend_checkpoint_dir(tokenizer_dir)

    student_config = model_config
    if config_path is not None:
        if model_config is not None:
            raise ValueError("Pass either `model_config` or `config_path`, not both.")
        student_config = Config.from_file(config_path)

    if subnetwork_student:
        if student_config is not None:
            raise ValueError(
                "`subnetwork_student` samples the student architecture; do not pass "
                "`model_config` or `config_path`."
            )
        assert min_ratio < max_ratio, "min_ratio must be less than max_ratio"
        assert 0 < min_ratio < 1, "min_ratio must be between 0 and 1"
        assert 0 < max_ratio < 1, "max_ratio must be between 0 and 1"
    elif student_config is None:
        if model_name is None:
            raise ValueError(
                "Pass `model_name`, `model_config` or `config_path` for the student, "
                "or set `subnetwork_student`."
            )
        student_config = Config.from_name(model_name)

    hparams = capture_hparams()
    data = TinyLlama() if data is None else data

    precision = precision or get_default_supported_precision(training=True)
    num_devices = int(parse_devices(devices))
    out_dir = init_out_dir(out_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"distill-{(student_config or teacher_config).name}",
        resume=bool(resume),
        log_interval=train.log_interval,
        log_args=asdict(log),
    )

    if num_devices * num_nodes == 1:
        fabric_strategy = "auto"
    elif strategy == "fsdp":
        fabric_strategy = FSDPStrategy(
            auto_wrap_policy={Block, WhittleBlock},
            activation_checkpointing_policy={Block, WhittleBlock},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        fabric_strategy = DDPStrategy()

    fabric = L.Fabric(
        devices=num_devices,
        num_nodes=num_nodes,
        strategy=fabric_strategy,
        precision=precision,
        loggers=[logger],
    )

    if torch.cuda.is_available() and num_devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch()

    fabric.print(pprint.pformat(hparams))
    if logger_name in ("tensorboard", "wandb", "mlflow"):
        fabric.logger.log_hyperparams(hparams)

    main(
        fabric=fabric,
        teacher_checkpoint_dir=teacher_checkpoint_dir,
        devices=num_devices,
        num_nodes=num_nodes,
        seed=seed,
        initial_checkpoint_dir=initial_checkpoint_dir,
        resume=resume,
        student_config=student_config,
        teacher_config=teacher_config,
        data=data,
        distill=distill,
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        tokenizer=tokenizer,
        train=train,
        eval=eval,
        optimizer=optimizer,
        init_from=init_from,
        subnetwork_student=subnetwork_student,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        random_init_student=random_init_student,
        hyperparameters=hyperparameters,
    )


def main(
    fabric: L.Fabric,
    teacher_checkpoint_dir: Path,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Path | None,
    resume: bool | Literal["auto"] | Path,
    data: DataModule,
    teacher_config: Config,
    out_dir: Path,
    tokenizer_dir: Path | None,
    tokenizer: Tokenizer | None,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: str | dict,
    num_nodes: int = 1,
    distill: DistillArgs = DistillArgs(),
    student_config: Config | None = None,
    init_from: str = "scratch",
    subnetwork_student: bool = False,
    min_ratio: float = 0.6,
    max_ratio: float = 0.61,
    random_init_student: bool = False,
    hyperparameters: str | None = None,
) -> None:
    validate_args(train, eval, initial_checkpoint_dir, resume)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    teacher_checkpoint = os.path.join(teacher_checkpoint_dir, "lit_model.pth")
    t0 = time.perf_counter()
    if subnetwork_student:
        student_model = sample_subnetwork_student(
            fabric, teacher_config, teacher_checkpoint, seed, min_ratio, max_ratio
        )
        if random_init_student:
            with fabric.init_module():
                student_model = WhittleGPT(student_model.config)
            fabric.print("Randomly initialized the student sub-network")
    else:
        assert student_config is not None
        with fabric.init_module(empty_init=True):
            student_model = GPT(student_config)
        if init_from == "scratch":
            initialize_weights(
                fabric,
                student_model,
                n_layer=student_config.n_layer,
                n_embd=student_config.n_embd,
            )
            fabric.print("Initialized student model from scratch")
        else:  # load from init from as path
            if not init_from:
                raise ValueError(
                    "init_from must be 'scratch' or a path to a checkpoint .pth file; got an empty value."
                )
            state_dict = torch.load(init_from, map_location="cpu")
            student_model.load_state_dict(state_dict)
            fabric.print(f"Initialized student model from {init_from}")
    if train.tie_embeddings:
        student_model.transformer.wte.weight = student_model.lm_head.weight
    if train.max_seq_length:
        student_model.max_seq_length = train.max_seq_length

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(student_model):,}")

    if not os.environ.get("DISABLE_TORCH_COMPILE"):
        if subnetwork_student:
            torch._dynamo.config.suppress_errors = True
        student_model = torch.compile(student_model)
    student_model = fabric.setup(student_model)

    extra_kwargs = {"fused": fabric.device.type == "cuda"}
    optimizer = instantiate_torch_optimizer(
        optimizer, student_model.parameters(), **extra_kwargs
    )
    optimizer = fabric.setup_optimizers(optimizer)

    train_dataloader, val_dataloader = get_dataloaders(
        fabric, data, tokenizer, train, student_model.max_seq_length
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    if initial_checkpoint_dir:
        load_weights(fabric, student_model, initial_checkpoint_dir / "lit_model.pth")

    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        teacher = GPT(teacher_config)

    teacher = fabric.setup(teacher)
    load_weights(fabric, teacher, teacher_checkpoint)
    teacher.eval()
    teacher_val_loss = validate(fabric, teacher, val_dataloader, max_iters=eval.max_iters)
    teacher_val_loss = teacher_val_loss.item()
    teacher_val_ppl = math.exp(teacher_val_loss)

    fabric.print(f"Teacher model loaded from {teacher_checkpoint_dir} (not compiled)")
    fabric.print(
        f"Teacher model validation loss: {teacher_val_loss:.3f}, validation PPL: {teacher_val_ppl:.3f}"
    )
    fabric.log_dict(
        {"teacher_val_loss": teacher_val_loss, "teacher_val_ppl": teacher_val_ppl}
    )
    state = {
        "model": student_model,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "iter_num": 0,
        "step_count": 0,
    }

    resume = find_resume_path(resume, out_dir)
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()

    # work around PyTorch issue https://github.com/pytorch/pytorch/issues/152162
    # which does not like the lazy initialization to be called in dynamo.
    # Happens with PyTorch 2.7.
    if (
        torch.__version__.startswith("2.7.")
        and (student_model._forward_module.__class__.__name__ == "OptimizedModule")
        and (
            student_model._forward_module._orig_mod.__class__.__name__
            == "FullyShardedDataParallel"
        )
    ):
        from torch.distributed.fsdp._runtime_utils import _root_pre_forward

        _root_pre_forward(
            student_model._forward_module._orig_mod,
            student_model._forward_module._orig_mod,
            [],
            {},
        )
    fit(
        fabric=fabric,
        devices=devices,
        num_nodes=num_nodes,
        state=state,
        teacher=teacher,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        train=train,
        eval=eval,
        distill=distill,
        hyperparameters=hyperparameters,
    )

    # Save final checkpoint
    save_checkpoint(
        fabric, state, tokenizer_dir, out_dir / "final" / "lit_model.pth", hyperparameters
    )

    total_tokens = (
        state["iter_num"]
        * train.micro_batch_size
        * student_model.max_seq_length
        * fabric.world_size
    )

    # Print formatted output
    train_duration = time.perf_counter() - train_time
    separator = "-" * 40
    fabric.print(separator)
    fabric.print("| Performance")
    fabric.print(f"| - Total tokens  : {total_tokens:,}")
    fabric.print(f"| - Training Time : {train_duration:.2f} s")
    fabric.print(f"| - Tok/sec       : {total_tokens / train_duration:.2f} tok/s")
    fabric.print("| " + "-" * 40)

    if fabric.device.type == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        fabric.print("| Memory Usage")
        fabric.print(f"| - Memory Used   : {memory_used:.2f} GB")
    fabric.print(separator)


def load_weights(
    fabric: L.Fabric, model: torch.nn.Module, checkpoint_path: str | Path
) -> None:
    """Load weights saved as a raw state dict, or under "model" as in whittle sub-network
    and Fabric checkpoints."""
    checkpoint = torch.load(checkpoint_path, mmap=True, weights_only=False)
    if "model" in checkpoint:
        fabric.load(checkpoint_path, {"model": model}, weights_only=False)
    else:
        fabric.load_raw(checkpoint_path, model, weights_only=False)


def sample_subnetwork_student(
    fabric: L.Fabric,
    teacher_config: Config,
    teacher_checkpoint: str,
    seed: int,
    min_ratio: float,
    max_ratio: float,
) -> WhittleGPT:
    """Extract a random sub-network of the teacher with a parameter ratio in
    ``[min_ratio, max_ratio]``.

    It is taken from a memory-mapped CPU copy of the teacher, since the training
    teacher is sharded under FSDP.
    """
    state_dict = torch.load(teacher_checkpoint, mmap=True, weights_only=False)
    state_dict = state_dict.get("model", state_dict)
    with torch.device("meta"):
        teacher = WhittleGPT(teacher_config)
    teacher.load_state_dict(state_dict, strict=False, assign=True)
    missing = [name for name, p in teacher.named_parameters() if p.is_meta]
    if missing:
        raise ValueError(f"{teacher_checkpoint} has no weights for {missing}")

    sampler = RandomSampler(get_search_space(teacher_config), seed=seed)
    teacher_params = compute_parameters(teacher)
    last_error: Exception | None = None
    for _ in range(MAX_STUDENT_SAMPLES):
        config = sampler.sample()
        fabric.print(f"Random subnetwork config: {config}")
        try:
            teacher.select_sub_network(
                {
                    "embed_dim": config["sub_network_n_embd"],
                    "mlp_ratio": config["sub_network_intermediate_size"]
                    / config["sub_network_n_embd"],
                    "num_heads": config["sub_network_num_heads"],
                    "depth": config["sub_network_n_layers"],
                }
            )
            student = extract_current_sub_network(teacher)
        except Exception as e:
            last_error = e
            fabric.print(f"Error during subnet extraction: {e}")
            fabric.print("Retrying with different random config...")
            continue
        finally:
            teacher.reset_super_network()
        ratio = compute_parameters(student) / teacher_params
        if min_ratio <= ratio <= max_ratio:
            fabric.print(f"Model parameter reduction: {ratio:.2%}")
            return student
    raise RuntimeError(
        f"No student sub-network with a parameter ratio in "
        f"[{min_ratio}, {max_ratio}] after {MAX_STUDENT_SAMPLES} samples."
    ) from last_error


def fit(
    fabric: L.Fabric,
    devices: int,
    state: dict,
    teacher: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    out_dir: Path,
    tokenizer_dir: Path | None,
    train: TrainArgs,
    distill: DistillArgs,
    eval: EvalArgs,
    num_nodes: int = 1,
    hyperparameters: str | None = None,
) -> None:
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
    vocab_size_student = student.config.vocab_size
    vocab_size_teacher = teacher.config.vocab_size
    print(
        f"Student vocab size: {vocab_size_student}, Teacher vocab size: {vocab_size_teacher}"
    )
    if vocab_size_student > vocab_size_teacher:
        vocab_size = vocab_size_teacher
    else:
        vocab_size = vocab_size_student  # noqa: F841  # unused until vocab slicing is fixed
    if eval.initial_validation:
        val_loss = validate(fabric, student, val_dataloader, max_iters=eval.max_iters)
        val_loss = f"{val_loss:.3f}"
    else:
        fabric.print("Verifying settings ...")
        validate(
            fabric, student, val_dataloader, max_iters=2, verbose=False
        )  # sanity check
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
    log_iter_interval = train.log_interval * train.gradient_accumulation_iters(
        devices, num_nodes
    )
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    running_loss = RunningMean(
        window=train.gradient_accumulation_iters(devices, num_nodes),
        sync_on_compute=False,
    ).to(fabric.device)
    fabric.barrier()
    total_t0 = time.perf_counter()

    warmup_iters = train.warmup_iters(devices, num_nodes, max_iters, train_dataloader)

    for train_data in train_iterator:
        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
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
            state["iter_num"] % train.gradient_accumulation_iters(devices, num_nodes) != 0
        )
        with fabric.no_backward_sync(student, enabled=is_accumulating):
            logits = student(input_ids)
            teacher.eval()
            with torch.inference_mode():
                teacher_logits = teacher(input_ids)
            logits_reshaped = logits.view(-1, logits.size(-1))
            targets_reshaped = targets.view(-1)
            teacher_logits_reshaped = teacher_logits.view(-1, teacher_logits.size(-1))
            loss = distill_loss(
                logits_reshaped, targets_reshaped, teacher_logits_reshaped
            )
            fabric.backward(loss / train.gradient_accumulation_iters(devices, num_nodes))
        running_loss.update(loss.detach())

        if not is_accumulating:
            fabric.clip_gradients(student, optimizer, max_norm=train.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        if state["iter_num"] % log_iter_interval == 0:
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
            save_checkpoint(
                fabric,
                state,
                tokenizer_dir,
                out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth",
                hyperparameters,
            )

    # Final validation
    if eval.final_validation:
        val_loss = validate(fabric, student, val_dataloader, max_iters=eval.max_iters)
        metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(
            f"Final evaluation | val loss: {val_loss.item():.3f} | val ppl: {math.exp(val_loss):.3f}"
        )


def save_checkpoint(fabric, state, tokenizer_dir, checkpoint_file, hyperparameters=None):
    model = state["model"]
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
    fabric.save(checkpoint_file, state)
    if fabric.global_rank == 0:
        if hyperparameters is not None:
            save_hyperparameters(hyperparameters, checkpoint_file.parent)
        if tokenizer_dir is not None:
            copy_config_files(tokenizer_dir, checkpoint_file.parent)
        save_config(model.config, checkpoint_file.parent)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
