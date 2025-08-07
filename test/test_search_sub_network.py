from __future__ import annotations

import os
import pathlib
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from lightning.fabric import Fabric
from litgpt.args import EvalArgs
from litgpt.config import Config
from litgpt.scripts.download import download_from_hub
from litgpt.utils import lazy_load
from syne_tune.backend.trial_status import Trial
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

    sub_network_config = {"embed_dim": 2, "mlp_ratio": 1, "num_heads": 2, "depth": 1}
    x, y = search_sub_networks._objective(
        config=sub_network_config,
        fabric=Fabric(devices=1),
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


def get_checkpoint_contents(copy_config_files, save_checkpoints):
    # model_config.yaml if in the parent super-net directory
    if not save_checkpoints:
        return {"lit_model.pth"}
    # we copied config files, model_config.yaml and tokenizer configs are the most important
    if copy_config_files:
        return {
            "lit_model.pth",
            "model_config.yaml",
            "tokenizer.json",
            "tokenizer_config.json",
        }
    # other config files are in the parent super-net directory
    return {"lit_model.pth", "model_config.yaml"}


@pytest.mark.parametrize("copy_config_files", [True, False])
@pytest.mark.parametrize("save_checkpoints", [True, False])
def test_checkpoints(tmp_path, checkpoint_dir, copy_config_files, save_checkpoints):
    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset, batch_size=3)
    search_sub_networks.get_dataloaders = Mock(return_value=(dataloader, dataloader))
    search_sub_networks._objective = Mock(return_value=(1, 1))
    out_dir = tmp_path / "out"
    stdout = StringIO()
    with redirect_stdout(stdout):
        fixed_config = {
            "sub_network_n_embd": 4,
            "sub_network_intermediate_size": 93,
            "sub_network_num_heads": 2,
            "sub_network_n_layers": 2,
        }

        fixed_config_syne_tune = {
            "embed_dim": 4,
            "num_heads": 4,
            "mlp_ratio": 4,
            "depth": 6,
        }
        fixed_trial = Trial(
            trial_id=1,
            config=fixed_config_syne_tune,
            creation_time=datetime.now(),
        )

        with (
            mock.patch(
                "whittle.sampling.random_sampler.RandomSampler.sample",
                return_value=fixed_config,
            ),
            mock.patch(
                "whittle.search.ask_tell_scheduler.AskTellScheduler.ask",
                return_value=fixed_trial,
            ),
        ):
            search_sub_networks.setup(
                checkpoint_dir,
                devices=1,
                out_dir=out_dir,
                search=SearchArgs(iterations=3),
                eval=EvalArgs(interval=1, max_iters=1, final_validation=False),
                save_checkpoints=save_checkpoints,
                copy_config_files=copy_config_files,
                verbose=False,
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
        # Check that the checkpoint directory contains the expected files
        contents = get_checkpoint_contents(copy_config_files, save_checkpoints)
        assert contents.issubset(set(os.listdir(out_dir / checkpoint_dir)))

        # Check the contents of lit_model.pth
        checkpoint = lazy_load(out_dir / checkpoint_dir / "lit_model.pth")
        if save_checkpoints:
            assert "model" in checkpoint
            if not copy_config_files:
                assert "parent_dir" in checkpoint
        else:
            assert "sub_network_config" in checkpoint
            assert "parent_dir" in checkpoint

    assert os.path.exists(out_dir / "pareto_optimal_paths.json")
