from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

import lightning as L
import torch
import yaml  # type: ignore[import-untyped]
from lightning.fabric.strategies import FSDPStrategy
from litgpt import Tokenizer
from litgpt.args import TrainArgs
from litgpt.data import DataModule
from litgpt.data.alpaca import Alpaca
from litgpt.model import Config
from litgpt.pretrain import get_dataloaders
from litgpt.utils import (
    auto_download_checkpoint,
    check_nvlink_connectivity,
    check_valid_checkpoint_dir,
    choose_logger,
    get_default_supported_precision,
    init_out_dir,
    load_checkpoint,
    parse_devices,
)

from whittle.args import PruningArgs
from whittle.models.gpt import GPT
from whittle.models.gpt.blocks import Block
from whittle.pruning import MagnitudePruner, SparseGPTPruner, WandaPruner

pruner_classes = {
    "mag": MagnitudePruner,
    "wanda": WandaPruner,
    "sparse_gpt": SparseGPTPruner,
}


def setup(
    checkpoint_dir: Path,
    out_dir: Path | None = None,
    precision: str | None = None,
    data: DataModule | None = None,
    devices: int | str | None = 1,
    num_nodes: int = 1,
    prune: PruningArgs = PruningArgs(
        pruning_strategy="mag",
        prune_n_weights_per_group=2,
        weights_per_group=4,
    ),
    max_seq_length: int | None = 512,
    seed: int | None = 1337,
    access_token: str | None = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "tensorboard",
    accelerator: str | None = None,
) -> None:
    """
    Performs structural pruning on a specified model checkpoint and saves a new checkpoint with the pruned weights set to zero.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for pruning.
        out_dir: Directory in which to save checkpoints and logs. If None, final checkpoint is saved in checkpoint_dir/pruning/<pruning_strategy>
        precision: The precision to use for loading the model. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        data: Data module for pruning calibration. If None, uses Alpaca dataset.
        devices: How many devices/GPUs to use.
        num_nodes: How many nodes the code is being run on.
        prune: Pruning-related arguments. See ``whittle.args.PruneArgs`` for details.
        max_seq_length: Maximum sequence length for the dataloader.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
        logger_name: Logger to use for tracking metrics.
        accelerator: Device type to use ("cpu", "cuda", etc.). If None, defaults to CUDA if available, else CPU.
    """

    checkpoint_dir = auto_download_checkpoint(
        model_name=checkpoint_dir, access_token=access_token
    )

    num_devices = int(parse_devices(devices))

    if out_dir is None:
        out_dir = checkpoint_dir / "pruning" / prune.pruning_strategy
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
        name=f"prune-{config.name}",
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
        accelerator=accelerator,
    )

    if accelerator == "cuda" and num_devices > 1 and torch.cuda.is_available():
        check_nvlink_connectivity(fabric)

    fabric.launch(
        main,
        seed,
        max_seq_length,
        config,
        data,
        checkpoint_dir,
        out_dir,
        prune,
        accelerator,
    )


def main(
    fabric: L.Fabric,
    seed: int,
    max_seq_length: int,
    config: Config,
    data: DataModule | None,
    checkpoint_dir: Path,
    out_dir: Path,
    prune: PruningArgs,
    accelerator: str,
) -> None:
    fabric.seed_everything(seed)

    tokenizer = Tokenizer(checkpoint_dir)

    data = Alpaca() if data is None else data

    train_dataloader, val_dataloader = get_dataloaders(
        fabric,
        data,
        tokenizer,
        TrainArgs(max_seq_length=max_seq_length),
        max_seq_length,
    )

    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
        # If the GPT model needs max_seq_length explicitly set after init
        # model.max_seq_length = effective_max_seq_length

    model = fabric.setup(model)

    load_checkpoint(fabric, model, checkpoint_path)

    start_time = time.perf_counter()

    fabric.print("Start structural pruning")
    pruner = pruner_classes[prune.pruning_strategy]()
    sparsity_ratio = pruner(
        model,
        prune_n=prune.prune_n_weights_per_group,
        prune_m=prune.weights_per_group,
        dataloader=val_dataloader,
        nsamples=prune.n_samples,
        device=accelerator,
    )

    pruning_time = time.perf_counter() - start_time

    fabric.print(f"Total time for pruning: {pruning_time:.02f} seconds.")
    fabric.print(f"Sparsity ratio: {sparsity_ratio:.02f}.")
    fabric.print(f"Save checkpoints to {out_dir}.")

    fabric.log_dict({"sparsity_ratio": sparsity_ratio, "pruning_time": pruning_time})

    # Save model weights (existing code)
    save_path = out_dir / "lit_model.pth"
    fabric.save(save_path, {"model": model})
    fabric.print(f"Saved model weights to {save_path}")

    # Save the config file only on the main process
    if fabric.global_rank == 0:
        config_save_path = out_dir / "model_config.yaml"
        try:
            # Convert the config object to a dictionary for saving
            if hasattr(config, "as_dict"):
                config_dict = config.as_dict()
            elif hasattr(config, "__dict__"):  # For Namespace-like objects
                config_dict = vars(config)
            else:  # Fallback: try converting specific known attrs? Or raise error?
                # For now, assume it's convertible or log warning
                config_dict = vars(config)
                fabric.print(
                    "Warning: Config object type might not be fully serializable. Trying vars()."
                )

            with open(config_save_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, indent=4)  # Added indent for readability
            fabric.print(f"Saved model config to {config_save_path}")
        except Exception as e:
            fabric.print(f"ERROR: Failed to save model_config.yaml: {e}")
            # Optionally raise the error or exit depending on severity
            # raise e


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
