"""
Adapted from the original LitGPT code.
"""

from __future__ import annotations

import os
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import Config
from litgpt.data.alpaca import Alpaca
from litgpt.utils import auto_download_checkpoint, check_valid_checkpoint_dir
from torch.utils.data import DataLoader, Dataset

from whittle import full_finetune

MODEL_NAME = "EleutherAI/pythia-14m"


class MockDataset(Dataset):
    """Custom dataset to return dictionary format expected by full_finetune.py."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Input: tensor([x1, x2, x3]), Labels: tensor([x2, x3, x4])
        input_ids = self.data[idx]
        # Shift labels by one and append a token (use 0 as a placeholder)
        labels = torch.cat([input_ids[1:], torch.tensor([0], dtype=input_ids.dtype)])
        token_counts = {
            "raw": 2,
            "raw_plus_prompt_template": 3,
        }
        return {"input_ids": input_ids, "labels": labels, "token_counts": token_counts}


@pytest.fixture(scope="module")
def ensure_checkpoint():
    """Fixture to ensure the model checkpoint is downloaded."""
    try:
        checkpoint_dir = auto_download_checkpoint(MODEL_NAME)
        check_valid_checkpoint_dir(checkpoint_dir)
    except Exception as e:
        pytest.skip(f"Failed to download or verify checkpoint for {MODEL_NAME}: {str(e)}")
    return checkpoint_dir


@mock.patch("whittle.full_finetune.save_hyperparameters")
@pytest.mark.parametrize("strategy", ["standard", "random", "sandwich"])
def test_training_strategies(
    save_hyper_mock, strategy, tmp_path, accelerator_device, ensure_checkpoint
):
    Config(block_size=2, n_layer=2, n_embd=4, n_head=2, padded_vocab_size=8)

    # Use tokens within vocab size (0 to 7)
    dataset = MockDataset(torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6]]))
    dataloader = DataLoader(dataset)
    full_finetune.get_dataloaders = Mock(return_value=(dataloader, dataloader))

    fixed_config = {
        "sub_network_n_embd": 4,
        "sub_network_intermediate_size": 93,
        "sub_network_num_heads": 4,
        "sub_network_n_layers": 2,
    }

    smallest_config = {
        "sub_network_n_embd": 4,
        "sub_network_intermediate_size": 5,
        "sub_network_num_heads": 4,
        "sub_network_n_layers": 1,
    }

    with (
        mock.patch(
            "whittle.sampling.random_sampler.RandomSampler.sample",
            return_value=fixed_config,
        ),
        mock.patch(
            "whittle.sampling.random_sampler.RandomSampler.get_smallest_sub_network",
            return_value=smallest_config,
        ),
    ):
        full_finetune.setup(
            MODEL_NAME,
            devices=1,
            optimizer="RMSprop",
            training_strategy=strategy,
            data=Alpaca(),
            out_dir=tmp_path,
            train=TrainArgs(
                global_batch_size=2,
                epochs=5,  # Required by validate_args
                save_interval=1,
                micro_batch_size=1,
                max_steps=4,  # Set to ensure termination
            ),
            eval=EvalArgs(
                interval=1,
                max_new_tokens=10,  # Required by validate_args
                max_iters=1,
                final_validation=False,
            ),
            precision="32-true",  # Full precision for CPU compatibility
            accelerator=accelerator_device,
        )


# Set CUDA_VISIBLE_DEVICES for FSDP hybrid-shard, if fewer GPUs are used than are available
@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
# If we were to use `save_hyperparameters()`, we would have to patch `sys.argv` or otherwise
# the CLI would capture pytest args, but unfortunately patching would mess with subprocess
# launching, so we need to mock `save_hyperparameters()`
@mock.patch("whittle.full_finetune.save_hyperparameters")
def test_full_finetune(save_hyper_mock, tmp_path, accelerator_device, ensure_checkpoint):
    Config(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)

    # Use tokens within vocab size (0 to 7)
    dataset = MockDataset(torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6]]))
    dataloader = DataLoader(dataset)
    full_finetune.get_dataloaders = Mock(return_value=(dataloader, dataloader))

    out_dir = tmp_path / "out"
    stdout = StringIO()
    with redirect_stdout(stdout):
        fixed_config = {
            "sub_network_n_embd": 4,
            "sub_network_intermediate_size": 93,
            "sub_network_num_heads": 4,
            "sub_network_n_layers": 2,
        }

        min_config = {
            "sub_network_n_embd": 4,
            "sub_network_intermediate_size": 4,
            "sub_network_num_heads": 4,
            "sub_network_n_layers": 1,
        }

        _rs_str = "whittle.sampling.random_sampler.RandomSampler"
        with (
            mock.patch(f"{_rs_str}.sample", return_value=fixed_config),
            mock.patch(f"{_rs_str}.get_smallest_sub_network", return_value=min_config),
        ):
            full_finetune.setup(
                MODEL_NAME,
                devices=1,
                out_dir=out_dir,
                data=Alpaca(),
                train=TrainArgs(
                    global_batch_size=2,
                    epochs=5,  # Required by validate_args
                    save_interval=1,
                    micro_batch_size=1,
                    max_steps=4,  # Set to ensure termination
                ),
                eval=EvalArgs(
                    interval=1,
                    max_new_tokens=10,  # Required by validate_args
                    max_iters=1,
                    final_validation=False,
                ),
                precision="32-true",  # Full precision for CPU compatibility
                accelerator=accelerator_device,
            )

    # tmp_path is not the same across all ranks, run assert only on rank 0
    out_dir_contents = set(os.listdir(out_dir))
    checkpoint_dirs = {
        "step-000001",
        "step-000002",
        "step-000003",
        "step-000004",
        "final",
    }
    assert checkpoint_dirs.issubset(out_dir_contents)
    assert all((out_dir / p).is_dir() for p in checkpoint_dirs)
    for checkpoint_dir in checkpoint_dirs:
        required_files = {"lit_model.pth", "model_config.yaml"}
        actual_files = set(os.listdir(out_dir / checkpoint_dir))
        assert required_files.issubset(actual_files)

    # logs only appear on rank 0
    logs = stdout.getvalue()
    assert logs.count("(step)") == 4
    assert logs.count("val loss") == 4

    assert "Number of trainable parameters: 14,067,712" in logs
