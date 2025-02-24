from __future__ import annotations

import os
import torch
import math
import pprint
import time
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional

from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from litgpt.data import DataModule, TinyStories
from litgpt.args import EvalArgs, TrainArgs
from litgpt import Tokenizer
from litgpt.config import name_to_config
from litgpt import Config

from litgpt.pretrain import (
    get_dataloaders,
    get_lr,
    initialize_weights,
    save_checkpoint,
    validate,
)

from litgpt.utils import (
    CycleIterator,
    capture_hparams,
    check_nvlink_connectivity,
    choose_logger,
    chunked_cross_entropy,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    parse_devices,
)

from whittle.sampling.random_sampler import RandomSampler
from whittle.pretrain_super_network import get_search_space
from whittle.models.gpt.model import GPT
from whittle.models.gpt.blocks import Block
from whittle.args import DistillArgs
from whittle.loss.kd_loss import DistillLoss
from whittle.metrics.parameters import compute_parameters

from jsonargparse import CLI
from pathlib import Path

def setup(
    model_name: str | None = None,
    model_config: Config | None = None,
    load_from_checkpoint: bool = True,
    out_dir: Path = Path("examples/gpt/out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Path = Path('checkpoints/standard-step-00190000'),
    resume: bool | Literal["auto"] | Path = False,
    data: DataModule | None = None,
    train: TrainArgs = TrainArgs(
        micro_batch_size=16,
        log_interval=100,
        max_norm=1.0,
        max_tokens=int(3e09)
    ),
    distill: DistillArgs = DistillArgs(
        method='logits',
        kd_epochs=1,
        temperature=5,
        alpha=0.7, # Higher weight since dataset is small
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    optimizer: str | dict = "AdamW",
    devices: int | str = "auto",
    num_nodes: int = 1,
    tokenizer_dir: Path | None = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "wandb",
    seed: int = 42,
):
    """Pretrain a model.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Use "list" to list the supported models.
        model_config: A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
            ``model_config``. Overrides the `model_name` if specified.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Determines a compatible precision setting by default.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize the model from.
            Useful for continued pretraining. Mutually exclusive with ``resume``.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``. An error will be raised if no checkpoint is found. Passing
            ``'auto'`` will resume from the latest checkpoint but not error if no checkpoint exists.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyStories``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        devices: How many devices/GPUs to use. Uses all GPUs by default.
        num_nodes: How many nodes the code is being run on.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
            module require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
    """
    if load_from_checkpoint:
        print(f"Loading teacher model from {initial_checkpoint_dir}")
        config = Config.from_file(initial_checkpoint_dir / "model_config.yaml")
        config.fix_head_size = True

    elif model_name == "list":
        available_models = "\n".join(sorted(name_to_config))
        print(f"Available values:\n{available_models}")
        quit()

    else:
        if model_config is None:
            # Support both model_name options: meta-llama/Meta-Llama-3-8B & Meta-Llama-3-8B
            try:
                model_config = Config.from_name(model_name)
            except ValueError:
                print(f"Model name {model_name} is not supported.\n")
                available_models = "\n".join(sorted(name_to_config))
                print(f"Available values:\n{available_models}")
                quit()

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
        resume=bool(resume),
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
        num_devices,
        seed,
        initial_checkpoint_dir,
        config,
        data,
        out_dir,
        tokenizer_dir,
        tokenizer,
        train,
        distill,
        eval,
        optimizer
    )

def main(
    fabric: L.Fabric,
    device: int,
    seed: int = 42,
    initial_checkpoint_dir: Path = Path('checkpoints/standard-step-00190000'),
    teacher_config: Config | None = None,
    dataset: DataModule | None = None,
    out_dir: Path = Path("examples/gpt/out/distillation"),
    tokenizer_dir: Path | None = None,
    tokenizer: Tokenizer | None = None,
    train: TrainArgs = TrainArgs(),
    distill: DistillArgs = DistillArgs(),
    eval: EvalArgs = EvalArgs(),
    optimizer: str | dict = "AdamW",
    resume: Path | bool = False,
    student_ckpt_path: Path | None = None,
    student_config_path: Path | None = None,
):  
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(initial_checkpoint_dir)

    fabric.seed_everything(seed)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        teacher = GPT(teacher_config)
    
    fabric.print(f"Loaded teacher model from {initial_checkpoint_dir}") 
    fabric.print(f"Teacher model has {compute_parameters(teacher)} parameters")

    teacher = fabric.setup(teacher)
    teacher = torch.compile(teacher)

    extra_kwargs = {"fused": fabric.device.type == "cuda"}
    optimizer = instantiate_torch_optimizer(optimizer, teacher.parameters(), **extra_kwargs)
    optimizer = fabric.setup_optimizers(optimizer)

    train_dataloader, val_dataloader = get_dataloaders(
        fabric, dataset, tokenizer, train, teacher.max_seq_length
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )
    
    if initial_checkpoint_dir:
        checkpoint = os.path.join(initial_checkpoint_dir, "lit_model.pth")
        ckpt = torch.load(checkpoint, map_location=fabric.device, weights_only=True)
        teacher.load_state_dict(ckpt, strict=False)
        # fabric.load(checkpoint, teacher)
        teacher.reset_super_network()

    fabric.print("Teacher model loaded.")

    # If no student model is provided, use a smallest subnet of the teacher model
    fabric.print("No student model provided. Initializing student model with smallest subnet of teacher model.")

    search_space = get_search_space(teacher_config)
    sampler = RandomSampler(search_space, seed=seed)
    # random_config = sampler.get_smallest_sub_network()
    random_config = sampler.sample()

    student = GPT(teacher_config)

    subnetwork = {
            "sub_network_n_embd": random_config["embed_dim"],
            "sub_network_intermediate_size": int(random_config["mlp_ratio"] * random_config["embed_dim"]),
            "sub_network_num_heads": random_config["num_heads"],
            "sub_network_n_layers": random_config["depth"]
            }   

    student.set_sub_network(**subnetwork)
    initialize_weights(fabric, student, n_layer=random_config["depth"], n_embd=random_config["embed_dim"])

    student = fabric.setup(student)
    student = torch.compile(student)

    fabric.print(f"Student model has {compute_parameters(student)} parameters") 

    # Log the subnet configuration as part of hyperparameters for this subnet.
    fabric.logger.log_hyperparams(subnetwork)

    state = {
        "teacher": teacher,
        "model": student,
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
    fit(fabric, device, state, train_dataloader, val_dataloader, out_dir, tokenizer_dir, train, eval, distill)

    # Requires only one model to save checkpoint, rewriting state
    state2 = {
        "model": student,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "iter_num": 0,
        "step_count": 0,
    }

    save_checkpoint(fabric, state2, tokenizer_dir, out_dir / "final" / "lit_model.pth")

    total_tokens = state["iter_num"] * train.micro_batch_size * student.max_seq_length * fabric.world_size

    # Print formatted output
    separator = "-" * 40
    fabric.print(separator)
    fabric.print("| Performance")
    fabric.print(f"| - Total tokens  : {total_tokens:,}")
    fabric.print(f"| - Training Time : {(time.perf_counter()-train_time):.2f} s")
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
    tokenizer_dir: Optional[Path],
    train: TrainArgs,
    eval: EvalArgs,
    distill: DistillArgs,
) -> None:
    teacher = state["teacher"]
    student = state["model"]
    optimizer = state["optimizer"]

    if eval.initial_validation:
        val_loss = validate(fabric, student, val_dataloader, max_iters=eval.max_iters)
        val_loss = f"{val_loss:.3f}"
    else:
        fabric.print("Verifying settings ...")
        validate(fabric, student, val_dataloader, max_iters=2, verbose=False)   # sanity check
        val_loss = "n/a"

    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = GPT(student.config)
        x = torch.randint(0, 1, (train.micro_batch_size, meta_model.max_seq_length))
        model_fwd = lambda: meta_model(x)
        model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = train.max_tokens // fabric.world_size
    tokens_per_iter = train.micro_batch_size * student.max_seq_length
    max_iters = max_tokens_per_device // tokens_per_iter
    log_iter_interval = train.log_interval * train.gradient_accumulation_iters(devices)
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices), sync_on_compute=False).to(
        fabric.device
    )
    fabric.barrier()
    total_t0 = time.perf_counter()

    warmup_iters = train.warmup_iters(devices, max_iters, train_dataloader)

    for train_data in train_iterator:
        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(optimizer.defaults["lr"], state["iter_num"], warmup_iters, max_iters, train.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : student.max_seq_length].contiguous().long()
        targets = train_data[:, 1 : (student.max_seq_length + 1)].contiguous().long()

        distill_loss = DistillLoss(
                    temperature=distill.temperature,
                    distillation_weight=distill.alpha
                )

        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices) != 0
        with fabric.no_backward_sync(student, enabled=is_accumulating):
            teacher_logits = teacher(input_ids)
            logits = student(input_ids)
            loss = distill_loss(logits, targets, teacher_logits)
            fabric.backward(loss / train.gradient_accumulation_iters(devices))

        running_loss.update(loss.detach())

        if not is_accumulating:
            fabric.clip_gradients(student, optimizer, max_norm=train.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        if state["iter_num"] % log_iter_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * train.micro_batch_size),
                lengths=(state["iter_num"] * train.micro_batch_size * student.max_seq_length),
            )
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"] * train.micro_batch_size * student.max_seq_length,
                "total_tokens": (state["iter_num"] * train.micro_batch_size * student.max_seq_length * fabric.world_size),
                "learning_rate": lr,
            }
            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.3f}"
            fabric.print(
                f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
                f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"] - 1)

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, student, val_dataloader, max_iters=eval.max_iters)
            val_loss = val_loss.item()
            td = time.perf_counter() - t0

            fabric.print(f"iter {state['iter_num']}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms")
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            fabric.log_dict(metrics, step=state["iter_num"] - 1)
            fabric.barrier()

        if train.save_interval is not None and not is_accumulating and state["step_count"] % train.save_interval == 0:
            save_checkpoint(fabric, state, tokenizer_dir, out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth")

    # Final validation
    if eval.final_validation:
        val_loss = validate(fabric, student, val_dataloader, max_iters=eval.max_iters)
        metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(f"Final evaluation | val loss: {val_loss.item():.3f} | val ppl: {math.exp(val_loss):.3f}")

if __name__ == "__main__":
    CLI(setup)