from __future__ import annotations

import os
import pathlib
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock

import pytest
import torch
from lightning.fabric import Fabric
from litgpt.args import EvalArgs
from litgpt.config import Config
from litgpt.scripts.download import download_from_hub
from torch.utils.data import DataLoader

from whittle import search_sub_networks
from whittle.args import SearchArgs
from whittle.models.gpt import GPT


def test_objective():
    model_config = Config(
        block_size=2, n_layer=2, n_embd=4, n_head=2, padded_vocab_size=8
    )
    model_config.fix_head_size = True

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    model = GPT(model_config)

    sub_network_config = {"embed_dim": 2, "mlp_ratio": 1, "num_heads": 1, "depth": 1}
    x, y = search_sub_networks._objective(
        config=sub_network_config,
        fabric=Fabric(),
        model=model,
        val_dataloader=dataloader,
        eval=EvalArgs(interval=1, max_iters=1, final_validation=False),
    )

    assert type(x) is float
    assert type(y) is int
    assert y > 0


@pytest.fixture(scope="session")
def checkpoint_dir(tmp_path_factory):
    checkpoint_dir = tmp_path_factory.getbasetemp()
    download_from_hub(repo_id="EleutherAI/pythia-14m", checkpoint_dir=checkpoint_dir)
    return pathlib.Path(checkpoint_dir) / "EleutherAI" / "pythia-14m"


def test_checkpoints(tmp_path, checkpoint_dir):
    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset, batch_size=3)
    search_sub_networks.get_dataloaders = Mock(return_value=(dataloader, dataloader))
    search_sub_networks._objective = Mock(return_value=(1, 1))
    out_dir = tmp_path / "out"
    stdout = StringIO()
    with redirect_stdout(stdout):
        search_sub_networks.setup(
            checkpoint_dir,
            devices=1,
            out_dir=out_dir,
            search=SearchArgs(iterations=3),
            eval=EvalArgs(interval=1, max_iters=1, final_validation=False),
        )

    out_dir_contents = set(os.listdir(out_dir))

    checkpoint_dirs = {
        "sub_network_0",
        "sub_network_1",
        "sub_network_2",
    }
    assert checkpoint_dirs.issubset(out_dir_contents)
    assert all((out_dir / p).is_dir() for p in checkpoint_dirs)
    for checkpoint_dir in checkpoint_dirs:
        assert set(os.listdir(out_dir / checkpoint_dir)) == {
            "lit_model.pth",
            "model_config.yaml",
        }
