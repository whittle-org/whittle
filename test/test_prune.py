from __future__ import annotations

import os
import pathlib
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from litgpt.scripts.download import download_from_hub
from torch.utils.data import DataLoader, TensorDataset

from whittle import prune
from whittle.args import PruningArgs

methods = ["mag", "wanda", "sparse_gpt"]


@pytest.fixture(scope="session")
def checkpoint_dir(tmp_path_factory):
    checkpoint_dir = tmp_path_factory.getbasetemp()
    download_from_hub(repo_id="EleutherAI/pythia-14m", checkpoint_dir=checkpoint_dir)
    return pathlib.Path(checkpoint_dir) / "EleutherAI" / "pythia-14m"


# Set CUDA_VISIBLE_DEVICES for FSDP hybrid-shard, if fewer GPUs are used than are available
@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
@pytest.mark.parametrize("pruning_strategy", methods)
def test_checkpoints(tmp_path, checkpoint_dir, pruning_strategy, accelerator_device):
    out_dir = tmp_path / "out"
    # Dynamically adjust the number of sequences based on the device to prevent OOM issue on CPU
    num_sequences = 16 if accelerator_device == "cpu" else 128
    max_seq_length = 512
    batch_size = 8
    nsamples = int(num_sequences / batch_size)

    dataset = TensorDataset(
        torch.randint(0, 1000, size=(num_sequences, max_seq_length)),
        torch.randint(0, 1000, size=(num_sequences, 1)),
    )

    dataloader = DataLoader(dataset, batch_size=batch_size)
    prune.get_dataloaders = Mock(return_value=(dataloader, dataloader))

    stdout = StringIO()
    with redirect_stdout(stdout):
        prune.setup(
            checkpoint_dir=checkpoint_dir,
            devices=1,
            out_dir=out_dir,
            data="test",
            prune=PruningArgs(
                pruning_strategy=pruning_strategy,
                n_samples=nsamples,
                prune_n_weights_per_group=2,
                weights_per_group=4,
            ),
            precision="32-true",  # Full precision for CPU compatibility
            accelerator=accelerator_device,
        )

    out_dir_content = {
        "lit_model.pth",
        "model_config.yaml",
    }
    assert out_dir_content.issubset(set(os.listdir(out_dir)))

    logs = stdout.getvalue()
    assert "Total time for pruning" in logs
    assert "Sparsity ratio" in logs
