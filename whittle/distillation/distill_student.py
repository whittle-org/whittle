from __future__ import annotations

import os
import random
import torch
import math
import pprint
import time
import json
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional, Dict, Any

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
    save_hyperparameters,
    copy_config_files,
    save_config,
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
    load_from_checkpoint: bool = False,
    out_dir: Path = Path("examples/gpt/out/distill_experiments"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Path | None = None,
    resume: bool | Literal["auto"] | Path = False,
    data: DataModule | None = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e7),
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
    ),
    distill: DistillArgs = DistillArgs(
        method='logits',
        temperature=5,
        alpha=0.7,
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    optimizer: str | dict = "AdamW",
    devices: int | str = "auto",
    num_nodes: int = 1,
    tokenizer_dir: Path | None = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "wandb",
    seed: int = 42,
):
    """Run multiple distillation experiments with different subnetwork configurations.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Use "list" to list the supported models.
        model_config: A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
            ``model_config``. Overrides the `model_name` if specified.
        load_from_checkpoint: Whether to load the teacher model from a checkpoint. 
            If True, the model will be loaded from the initial checkpoint directory.
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
            modules require this.
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
        name=f"manual-distill-experiments-{config.name}",
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
        optimizer,
        resume,
        load_from_checkpoint,
    )

def main(
    fabric: L.Fabric,
    devices: int,
    seed: int = 42,
    initial_checkpoint_dir: Path = Path('checkpoints/standard-step-00190000'),
    teacher_config: Config | None = None,
    dataset: DataModule | None = None,
    out_dir: Path = Path("examples/gpt/out/distillation_experiments"),
    tokenizer_dir: Path | None = None,
    tokenizer: Tokenizer | None = None,
    train: TrainArgs = TrainArgs(),
    distill: DistillArgs = DistillArgs(),
    eval: EvalArgs = EvalArgs(),
    optimizer: str | dict = "AdamW",
    resume: bool | Literal["auto"] | Path = False,
    load_from_checkpoint: bool = False,
):  
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(initial_checkpoint_dir)
    random.seed(random.randint(0, 2**32 - 1))
    exp_seed = random.randint(0, 2**32 - 1)
    fabric.print(f"Experiment seed: {exp_seed}")

    fabric.seed_everything(seed)

    # Check if torch.compile() should be used for student model only
    use_compile_student = hasattr(torch, "_dynamo") and not bool(os.environ.get("DISABLE_TORCH_COMPILE"))
    
    train_dataloader, val_dataloader = get_dataloaders(
        fabric, dataset, tokenizer, train, teacher_config.block_size
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    # Load the teacher model once
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        teacher = GPT(teacher_config)
    
    if initial_checkpoint_dir:
        checkpoint = os.path.join(initial_checkpoint_dir, "lit_model.pth")
        ckpt = torch.load(checkpoint, map_location=fabric.device, weights_only=True)
        teacher.load_state_dict(ckpt, strict=False)
        teacher.reset_super_network()
    
    teacher = fabric.setup_module(teacher, move_to_device=True)
    teacher.eval()
    
    fabric.print(f"Teacher model loaded from {initial_checkpoint_dir} (not compiled)")
    fabric.print(f"Teacher model has {compute_parameters(teacher):,} parameters")

    # Get search space for creating different subnetworks
    search_space = get_search_space(teacher_config)

    sampler = RandomSampler(search_space, seed=exp_seed)
    random_config = sampler.sample()
            
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        student = GPT(teacher_config)

    subnetwork = {
        "sub_network_n_embd": random_config["embed_dim"],
        "sub_network_intermediate_size": int(random_config["mlp_ratio"] * random_config["embed_dim"]),
        "sub_network_num_heads": random_config["num_heads"],
        "sub_network_n_layers": random_config["depth"]
    }   

    student.set_sub_network(**subnetwork)
    initialize_weights(fabric, student, n_layer=random_config["depth"], n_embd=random_config["embed_dim"])
    
    student = fabric.setup_module(student, move_to_device=True)
    
    if use_compile_student:
        try:
            fabric.print("Compiling student model...")
            student = torch.compile(student)
            fabric.print("Student model compiled successfully")
        except Exception as e:
            fabric.print(f"Error compiling student model: {e}")
            fabric.print("Continuing with uncompiled student model")

    extra_kwargs = {"fused": fabric.device.type == "cuda"}
    optimizer = instantiate_torch_optimizer(optimizer, student.parameters(), **extra_kwargs)
    optimizer = fabric.setup_optimizers(optimizer)

    param_count = compute_parameters(student)
    fabric.print(f"Student model has {param_count:,} parameters")
    fabric.print(f"Model parameter reduction: {param_count/compute_parameters(teacher):.2%}")

    # Log the subnetwork configuration to the logger
    exp_config = {
        "seed": exp_seed,
        "embed_dim": random_config["embed_dim"],
        "mlp_ratio": random_config["mlp_ratio"],
        "num_heads": random_config["num_heads"],
        "depth": random_config["depth"],
        "parameter_count": param_count,
        "reduction_ratio": param_count/compute_parameters(teacher)
    }
    fabric.logger.log_hyperparams(exp_config)

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
    train_results = fit(
        fabric, devices, state, train_dataloader, val_dataloader, 
        out_dir, tokenizer_dir, train, eval, distill
    )

    # Save student model
    student_state = {
        "model": student,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "iter_num": state["iter_num"],
        "step_count": state["step_count"],
    }

    save_checkpoint(
        fabric, student_state, tokenizer_dir, 
        out_dir / "final" / "lit_model.pth"
    )

    # Track experiment results
    total_tokens = state["iter_num"] * train.micro_batch_size * student.max_seq_length * fabric.world_size
    training_time = time.perf_counter() - train_time
    
    # Create a results summary
    experiment_summary = {
        "config": {
            "embed_dim": random_config["embed_dim"],
            "mlp_ratio": random_config["mlp_ratio"],
            "num_heads": random_config["num_heads"],
            "depth": random_config["depth"],
        },
        "subnetwork": subnetwork,
        "parameters": param_count,
        "parameter_reduction": float(param_count/compute_parameters(teacher)),
        "training_time_seconds": training_time,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / training_time,
        "final_train_loss": train_results.get("final_train_loss", None),
        "final_val_loss": train_results.get("final_val_loss", None),
        "final_val_ppl": train_results.get("final_val_ppl", None),
    }
    
    experiment_results = []
    experiment_results["students"].append(experiment_summary)
    
    for result in experiment_results["students"]:
        fabric.print(f"\n Embed dim: {result['config']['embed_dim']}, Depth: {result['config']['depth']}")
        fabric.print(f"  - Parameters: {result['parameters']:,} ({result['parameter_reduction']:.2%} of teacher)")
        if result.get("final_val_loss") is not None:
            fabric.print(f"  - Val loss: {result['final_val_loss']:.4f}, Val ppl: {result['final_val_ppl']:.4f}")
        fabric.print(f"  - Training time: {result['training_time_seconds']:.2f}s")
    
    fabric.print("\n" + "=" * 80)

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
) -> Dict[str, Any]:
    teacher = state["teacher"]
    student = state["model"]
    optimizer = state["optimizer"]

    teacher.eval()

    distill_loss = DistillLoss(
        temperature=distill.temperature,
        distillation_weight=distill.alpha
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
        meta_model.set_sub_network(**{
            "sub_network_n_embd": student.config.n_embd,
            "sub_network_intermediate_size": student.config.intermediate_size,
            "sub_network_num_heads": student.config.n_head,
            "sub_network_n_layers": student.config.n_layer,
        })
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

        lr = get_lr(optimizer.defaults["lr"], state["iter_num"], warmup_iters, max_iters, train.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : student.max_seq_length].contiguous().long()
        targets = train_data[:, 1 : (student.max_seq_length + 1)].contiguous().long()

        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices) != 0
        with fabric.no_backward_sync(student, enabled=is_accumulating):
            teacher.eval()
            with torch.inference_mode(): # no grads for teacher
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
            loss = running_loss.compute().item()
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
            student_state = {
                "model": student,
                "optimizer": optimizer,
                "train_dataloader": train_dataloader,
                "iter_num": state["iter_num"],
                "step_count": state["step_count"],
            }
            save_checkpoint(fabric, student_state, tokenizer_dir, out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth")
            fabric.log_dict(metrics, step=state["iter_num"] - 1)
            fabric.barrier()

    if eval.final_validation:
        val_loss = validate(fabric, student, val_dataloader, max_iters=eval.max_iters)
        val_loss_value = val_loss.item()
        ppl = math.exp(val_loss_value)
        metrics = {"val_loss": val_loss_value, "val_ppl": ppl}
        
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(f"Final evaluation | val loss: {val_loss_value:.3f} | val ppl: {ppl:.3f}")
    

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