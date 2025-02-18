from __future__ import annotations

import os
import time
from pathlib import Path

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from litgpt import Tokenizer
from litgpt.args import TrainArgs
from litgpt.data import DataModule
from litgpt.model import Config
from litgpt.pretrain import get_dataloaders
from litgpt.utils import (
    auto_download_checkpoint,
    check_nvlink_connectivity,
    check_valid_checkpoint_dir,
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
    out_dir: Path | None = Path("out/finetune/full"),
    precision: str | None = None,
    data: DataModule | None = None,
    devices: int | str | None = 1,
    num_nodes: int = 1,
    prune: PruningArgs = PruningArgs(
        pruning_strategy="mag",
        n=2,
        m=4,
    ),
    max_seq_length: int | None = 512,
    seed: int | None = 1337,
    access_token: str | None = None,
) -> None:
    """
    Performs structural pruning on a specified model checkpoint and saves a new checkpoint with the pruned weights set to zero.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for pruning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        devices: How many devices/GPUs to use
        num_nodes: How many nodes the code is being run on.
        prune: Pruning-related arguments. See ``whittle.args.PruneArgs`` for details.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
    """

    checkpoint_dir = auto_download_checkpoint(
        model_name=checkpoint_dir, access_token=access_token
    )

    num_devices = int(parse_devices(devices))
    out_dir = init_out_dir(out_dir)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    config.fix_head_size = True

    precision = precision or get_default_supported_precision(training=True)

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
    )

    if torch.cuda.is_available() and num_devices > 1:
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
    )


def main(
    fabric: L.Fabric,
    seed: int,
    max_seq_length: int,
    config: Config,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    prune: PruningArgs,
) -> None:
    fabric.seed_everything(seed)

    tokenizer = Tokenizer(checkpoint_dir)

    if data is None:
        train_dataloader, val_dataloader = None, None
    else:
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

    model = fabric.setup(model)

    load_checkpoint(fabric, model, checkpoint_path)

    start_time = time.perf_counter()

    fabric.print("Start structural pruning")
    pruner = pruner_classes[prune.pruning_strategy]()
    sparsity_ratio = pruner(
        model,
        prune_n=prune.n,
        prune_m=prune.m,
        dataloader=val_dataloader,
        nsamples=prune.nsamples,
    )

    pruning_time = time.perf_counter() - start_time

    fabric.print(f"Total time for pruning: {pruning_time:.02f} seconds.")
    fabric.print(f"Sparsity ratio: {sparsity_ratio:.02f}.")
    fabric.print(f"Save checkpoints to {out_dir}.")

    fabric.save(out_dir / "lit_model.pth", {"model": model})


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
