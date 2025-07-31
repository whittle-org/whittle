from __future__ import annotations

import os
import pprint
import time
from pathlib import Path
from typing import Literal

import lightning as L
import torch
import torch._dynamo
from jsonargparse import CLI
from lightning.fabric.strategies import FSDPStrategy
from litgpt import Config, Tokenizer
from litgpt.args import TrainArgs
from litgpt.data import DataModule, TinyStories
from litgpt.pretrain import get_dataloaders
from litgpt.utils import (
    capture_hparams,
    check_nvlink_connectivity,
    choose_logger,
    get_default_supported_precision,
    init_out_dir,
    load_checkpoint,
    parse_devices,
)
from torch.utils.data import DataLoader

from whittle.models.gpt.blocks import Block
from whittle.models.gpt.model import GPT

torch._dynamo.config.suppress_errors = True


def setup(
    teacher_checkpoint_dir: Path | None = None,
    out_dir: Path = Path("out/teacher_logits"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    data: DataModule | None = None,
    train: TrainArgs = TrainArgs(
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(5e8),
    ),
    devices: int | str = "auto",
    num_nodes: int = 1,
    tokenizer_dir: Path | None = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "csv",
    seed: int = 42,
    top_k: int | None = None,
    save_interval: int = 10,
):
    """Save teacher model logits for knowledge distillation.

    Arguments:
        teacher_checkpoint_dir: Path to teacher model checkpoint directory.
        out_dir: Directory to save teacher logits.
        precision: The precision to use for inference.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyStories``.
        train: Training-related arguments for dataloader setup.
        devices: How many devices/GPUs to use.
        num_nodes: How many nodes the code is being run on.
        tokenizer_dir: Path to the tokenizer directory.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        top_k: If specified, save only top-k logits and their indices. Otherwise save full logits.
        save_interval: Save logits every N batches.
    """
    if teacher_checkpoint_dir is None:
        raise ValueError("teacher_checkpoint_dir is required")

    print(f"Loading teacher model config from {teacher_checkpoint_dir}")
    config = Config.from_file(teacher_checkpoint_dir / "model_config.yaml")
    config.fix_head_size = True

    hparams = capture_hparams()
    data = TinyStories() if data is None else data

    precision = precision or get_default_supported_precision(training=False)
    num_devices = int(parse_devices(devices))
    out_dir = init_out_dir(out_dir)

    # Set up tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"store-logits-{config.name}",
        log_interval=1,
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
        config,
        data,
        out_dir,
        tokenizer_dir,
        tokenizer,
        seed,
        train,
        top_k,
        save_interval,
    )


def main(
    fabric: L.Fabric,
    teacher_checkpoint_dir: Path,
    teacher_config: Config,
    dataset: DataModule,
    out_dir: Path,
    tokenizer_dir: Path | None = None,
    tokenizer: Tokenizer | None = None,
    seed: int = 42,
    train: TrainArgs = TrainArgs(),
    top_k: int | None = None,
    save_interval: int = 1,
):
    # Create save directory structure
    save_dir = (
        out_dir / f"{teacher_config.name}_top_{top_k}"
        if top_k
        else out_dir / teacher_config.name
    )

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)

        logits_dir = save_dir / "logits"
        logits_dir.mkdir(exist_ok=True)

        if top_k:
            indices_dir = save_dir / "indices"
            indices_dir.mkdir(exist_ok=True)

    fabric.seed_everything(seed)

    # Set up data loaders
    train_dataloader, _ = get_dataloaders(
        fabric, dataset, tokenizer, train, teacher_config.block_size
    )
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    # Load teacher model
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        teacher = GPT(teacher_config)

    checkpoint = os.path.join(teacher_checkpoint_dir, "lit_model.pth")
    teacher = fabric.setup(teacher)
    load_checkpoint(fabric, teacher, checkpoint, strict=False)
    teacher.eval()

    fabric.print(f"Teacher model loaded from {teacher_checkpoint_dir}")
    fabric.print(
        f"Teacher model has {sum(p.numel() for p in teacher.parameters()):,} parameters"
    )

    if top_k:
        fabric.print(f"Saving top-{top_k} logits and indices")
    else:
        fabric.print("Saving full logits")

    # Save teacher logits
    save_teacher_logits(fabric, teacher, train_dataloader, save_dir, top_k, save_interval)

    fabric.print("Teacher logits saved successfully!")


def save_teacher_logits(
    fabric: L.Fabric,
    teacher: GPT,
    dataloader: DataLoader,
    save_dir: Path,
    top_k: int | None = None,
    save_interval: int = 1,
):
    """Save teacher logits for the entire dataset."""
    teacher.eval()

    batch_count = 0
    total_batches = 0

    fabric.print("Starting to save teacher logits...")
    start_time = time.time()

    with torch.inference_mode():
        for batch_idx, batch_data in enumerate(dataloader):
            input_ids = batch_data[:, 0 : teacher.max_seq_length].contiguous().long()
            teacher_logits = teacher(input_ids)

            save_data = {
                "input_ids": input_ids.cpu(),
                "batch_idx": batch_idx,
            }

            if top_k:
                top_k_logits, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
                save_data["logits"] = top_k_logits.cpu()
                save_data["top_k"] = top_k

                indices_data = {
                    "indices": top_k_indices.cpu(),
                    "batch_idx": batch_idx,
                }
            else:
                save_data["logits"] = teacher_logits.cpu()

            if (batch_idx + 1) % save_interval == 0:
                if fabric.global_rank == 0:
                    logits_file = (
                        save_dir / "logits" / f"logits_batch_{batch_count:06d}.pt"
                    )
                    torch.save(save_data, logits_file)

                    if top_k:
                        indices_file = (
                            save_dir / "indices" / f"indices_batch_{batch_count:06d}.pt"
                        )
                        torch.save(indices_data, indices_file)

                    batch_count += 1

                    elapsed = time.time() - start_time
                    fabric.print(
                        f"Saved batch {batch_idx + 1} "
                        f"(file {batch_count}) - "
                        f"Elapsed: {elapsed:.1f}s - "
                        f"Rate: {(batch_idx + 1) / elapsed:.2f} batches/s"
                    )

            total_batches = batch_idx + 1

    # Save any remaining data that wasn't saved due to save_interval
    if total_batches % save_interval != 0 and fabric.global_rank == 0:
        logits_file = save_dir / "logits" / f"logits_batch_{batch_count:06d}.pt"
        torch.save(save_data, logits_file)

        if top_k:
            indices_file = save_dir / "indices" / f"indices_batch_{batch_count:06d}.pt"
            torch.save(indices_data, indices_file)

        batch_count += 1

    if fabric.global_rank == 0:
        # Save metadata
        metadata = {
            "total_batches": total_batches,
            "total_files": batch_count,
            "top_k": top_k,
            "vocab_size": teacher.config.vocab_size,
            "max_seq_length": teacher.max_seq_length,
            "save_interval": save_interval,
        }
        metadata_file = save_dir / "metadata.pt"
        torch.save(metadata, metadata_file)

        fabric.print(f"Processed {total_batches} batches from dataset")
        fabric.print(f"Saved {batch_count} logits files")
        fabric.print(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    CLI(setup)
