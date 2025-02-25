from __future__ import annotations

import math
import pprint
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import lightning as L
import torch

# TODO: wrap this in a try-import in case deepspeed is not installed
from deepspeed.utils.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
)

# Monkeypatch Lightning Fabric's Strategy.get_module_state_dict method
from lightning.fabric.strategies.strategy import Strategy
from torch.nn import Module

# Store the original method for later use
original_get_module_state_dict = Strategy.get_module_state_dict


# Define our custom implementation
def custom_get_module_state_dict(self, module: Module) -> dict[str, Any | torch.Tensor]:
    """Custom implementation that handles DeepSpeed module state dict differently."""
    from lightning.fabric.strategies import DeepSpeedStrategy

    if isinstance(self, DeepSpeedStrategy):
        # For DeepSpeed, we need to handle the module differently
        # First get the state dict using the original method
        state_dict = original_get_module_state_dict(self, module)

        # Add "module." prefix to all keys
        prefixed_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith("module."):
                prefixed_state_dict[f"module.{key}"] = value
            else:
                prefixed_state_dict[key] = value

        return prefixed_state_dict
    else:
        # For other strategies, use the original implementation
        return original_get_module_state_dict(self, module)


# Apply the monkeypatch
Strategy.get_module_state_dict = custom_get_module_state_dict

from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from lightning.fabric.wrappers import _FabricDataLoader
from litgpt import Tokenizer
from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import name_to_config
from litgpt.data import DataModule, TinyLlama
from litgpt.model import Config
from litgpt.pretrain import (
    get_dataloaders,
    get_lr,
    initialize_weights,
    validate,
    validate_args,
)
from litgpt.utils import (
    CycleIterator,
    capture_hparams,
    check_nvlink_connectivity,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    extend_checkpoint_dir,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
    save_config,
    save_hyperparameters,
)
from syne_tune.config_space import lograndint, randint
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

from whittle.models.gpt import GPT
from whittle.models.gpt.blocks import Block
from whittle.sampling.random_sampler import RandomSampler
from whittle.training_strategies import (
    RandomStrategy,
    SandwichStrategy,
    StandardStrategy,
)
from whittle.training_strategies.base_strategy import BaseTrainingStrategy

training_strategies_cls = {
    "sandwich": SandwichStrategy,
    "random": RandomStrategy,
    "standard": StandardStrategy,
}


def get_search_space(config):
    return {
        "embed_dim": lograndint(1, config.n_embd),
        "num_heads": randint(1, config.n_head),
        "mlp_ratio": randint(1, 4),
        "depth": randint(1, config.n_layer),
    }


def setup(
    model_name: str,
    model_config: Config | None = None,
    out_dir: Path = Path("../examples/gpt/out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
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
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    optimizer: str | dict = "AdamW",
    devices: int | str = "auto",
    num_nodes: int = 1,
    training_strategy: str = "sandwich",
    distributed_strategy: Literal["auto", "fsdp", "deepspeed"] = "auto",
    tokenizer_dir: Path | None = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "tensorboard",
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
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyLlama``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        training_strategy:
        devices: How many devices/GPUs to use. Uses all GPUs by default.
        num_nodes: How many nodes the code is being run on.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
            module require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
    """
    if model_name == "list":
        available_models = "\n".join(sorted(name_to_config))
        print(f"Available values:\n{available_models}")
        quit()

    if initial_checkpoint_dir is not None:
        initial_checkpoint_dir = extend_checkpoint_dir(initial_checkpoint_dir)

    if tokenizer_dir is not None:
        tokenizer_dir = extend_checkpoint_dir(tokenizer_dir)

    if model_config is None:
        # Support both model_name options: meta-llama/Meta-Llama-3-8B & Meta-Llama-3-8B
        try:
            model_config = Config.from_name(model_name)
        except ValueError:
            print(f"Model name {model_name} is not supported.\n")
            available_models = "\n".join(sorted(name_to_config))
            print(f"Available values:\n{available_models}")
            quit()

    assert training_strategy in training_strategies_cls, print(
        f"Training strategy is {training_strategy}. Should be in {list(training_strategies_cls)}"
    )

    hparams = capture_hparams()
    data = TinyLlama() if data is None else data

    config = Config.from_name(model_name) if model_config is None else model_config
    config.fix_head_size = True

    precision = precision or get_default_supported_precision(training=True)
    num_devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"pretrain-{config.name}",
        resume=bool(resume),
        log_interval=train.log_interval,
    )

    if num_devices * num_nodes > 1:
        if distributed_strategy == "fsdp":
            distributed_strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                state_dict_type="full",
                sharding_strategy="HYBRID_SHARD",
            )
        elif distributed_strategy == "deepspeed":
            ds_config = {
                "train_micro_batch_size_per_gpu": train.micro_batch_size,
                "gradient_accumulation_steps": train.gradient_accumulation_iters(
                    num_devices
                ),
                "zero_optimization": {"stage": 2},
                "gradient_clipping": train.max_norm,
            }
            distributed_strategy = DeepSpeedStrategy(config=ds_config)
    else:
        distributed_strategy = "auto"

    fabric = L.Fabric(
        devices=num_devices,
        num_nodes=num_nodes,
        strategy=distributed_strategy,
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
        resume,
        config,
        data,
        out_dir,
        tokenizer_dir,
        tokenizer,
        train,
        eval,
        optimizer,
        training_strategy,
    )


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
    training_strategy: BaseTrainingStrategy,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]

    if eval.initial_validation:
        model.reset_super_network()
        val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
        val_loss = f"{val_loss:.3f}"
    else:
        model.reset_super_network()
        fabric.print("Verifying settings ...")
        validate(
            fabric, model, val_dataloader, max_iters=2, verbose=False
        )  # sanity check
        val_loss = "n/a"

    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = GPT(model.config)
        x = torch.randint(0, 1, (train.micro_batch_size, meta_model.max_seq_length))

        def model_fwd():
            return meta_model(x)  # noqa: F821

        def model_loss(y):
            return chunked_cross_entropy(y, x, chunk_size=0)  # noqa: F821

        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = train.max_tokens // fabric.world_size
    tokens_per_iter = train.micro_batch_size * model.max_seq_length
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

        # determine and set the learning rate for this iteration
        lr = get_lr(
            optimizer.defaults["lr"]
            if isinstance(optimizer, torch.optim.Optimizer)
            else optimizer.optimizer.optimizer.defaults["lr"],
            state["iter_num"],
            warmup_iters,
            max_iters,
            train.min_lr,
        )
        if isinstance(optimizer, torch.optim.Optimizer):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            for param_group in optimizer.optimizer.optimizer.param_groups:
                param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.max_seq_length].contiguous().long()
        targets = train_data[:, 1 : (model.max_seq_length + 1)].contiguous().long()

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
        #            fabric.backward(loss / train.gradient_accumulation_iters(devices))

        running_loss.update(loss)

        if not is_accumulating:
            if isinstance(optimizer, torch.optim.Optimizer):
                fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)
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
                    * (max_iters - state["iter_num"])
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
            model.reset_super_network()
            val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
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
            ckpt_dir = out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth"
            save_checkpoint(
                fabric,
                state,
                tokenizer_dir,
                ckpt_dir,
            )

    # Final validation
    if eval.final_validation:
        model.reset_super_network()
        val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
        metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(
            f"Final evaluation | val loss: {val_loss.item():.3f} | val ppl: {math.exp(val_loss):.3f}"
        )


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Path | None,
    resume: bool | Literal["auto"] | Path,
    config: Config,
    data: DataModule,
    out_dir: Path,
    tokenizer_dir: Path | None,
    tokenizer: Tokenizer | None,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: str | dict,
    training_strategy: str,
) -> None:
    validate_args(train, eval, initial_checkpoint_dir, resume)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)

    initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)

    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight
    if train.max_seq_length:
        model.max_seq_length = train.max_seq_length

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    # model = torch.compile(model, mode="max-autotune")

    extra_kwargs = {"fused": fabric.device.type == "cuda"}
    if isinstance(fabric.strategy, DeepSpeedStrategy):
        # model = fabric.setup(model, move_to_device=False)
        optimizer = instantiate_torch_optimizer(
            optimizer, model.parameters(), **extra_kwargs
        )
        model, optimizer = fabric.setup(model, optimizer)
    else:
        model = fabric.setup(model)

        optimizer = instantiate_torch_optimizer(
            optimizer, model.parameters(), **extra_kwargs
        )
        optimizer = fabric.setup_optimizers(optimizer)

    sampler = RandomSampler(config_space=get_search_space(config), seed=seed)
    training_strategy_kwargs = {
        "loss_function": chunked_cross_entropy,
        "sampler": sampler,
        "fabric": fabric,
    }
    strategy = training_strategies_cls[training_strategy](**training_strategy_kwargs)

    train_dataloader, val_dataloader = get_dataloaders(
        fabric, data, tokenizer, train, model.max_seq_length
    )

    if initial_checkpoint_dir:
        fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

    state = {
        "model": model,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "iter_num": 0,
        "step_count": 0,
    }

    resume = find_resume_path(resume, out_dir)
    if resume:
        fabric.print(f"Resuming training from {resume}")
        if isinstance(fabric.strategy, DeepSpeedStrategy):
            state.pop("train_dataloader")
            fabric.load(resume, state, strict=False)

            train_dataloader.load_state_dict(
                torch.load(resume / "lit_model.pth", weights_only=False)[
                    "train_dataloader"
                ]
            )
            state["train_dataloader"] = train_dataloader
        else:
            fabric.load(resume, state)

    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        state["train_dataloader"], val_dataloader
    )

    state["train_dataloader"] = train_dataloader

    train_time = time.perf_counter()
    fit(
        fabric,
        devices,
        state,
        train_dataloader,
        val_dataloader,
        out_dir,
        tokenizer_dir,
        train,
        eval,
        strategy,
    )

    # Save final checkpoint
    save_checkpoint(fabric, state, tokenizer_dir, out_dir / "final" / "lit_model.pth")

    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def save_checkpoint(
    fabric: L.Fabric,
    state: dict,
    tokenizer_dir: Path | str | None,
    checkpoint_file: Path,
):
    model: GPT = state["model"]
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
    if not isinstance(fabric.strategy, DeepSpeedStrategy):
        fabric.save(checkpoint_file, state)
    else:
        # Create a new state dict with the "module." prefix for each key

        fabric.strategy.save_checkpoint(checkpoint_file.parent, state)

        # Get the FP32 state dict from the saved checkpoint
        model_state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_file.parent)
        dataloader: _FabricDataLoader = state["train_dataloader"]._dataloader

        with open(checkpoint_file, "wb") as f:
            new_model_state = {}
            for key, value in model_state.items():
                new_model_state[f"module.{key}"] = value

            torch.save(
                dict(
                    model=new_model_state,
                    train_dataloader=dataloader.state_dict(),
                    iter_num=state["iter_num"],
                    step_count=state["step_count"],
                ),
                f,
            )
    if fabric.global_rank == 0:
        save_hyperparameters(setup, checkpoint_file.parent)
        if tokenizer_dir is not None:
            copy_config_files(tokenizer_dir, checkpoint_file.parent)
        save_config(model.config, checkpoint_file.parent)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
