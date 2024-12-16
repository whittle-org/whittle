"""
Adapted from the original LitGPT code.
"""

import os
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import ANY, Mock

import torch
from torch.utils.data import DataLoader

from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import Config

from whittle import pretrain_super_network

from test.conftest import RunIf


@RunIf(min_cuda_gpus=1, standalone=True)
@mock.patch("litgpt.pretrain.save_hyperparameters")
def test_training_strategies(_, tmp_path):
    model_config = Config(
        block_size=2, n_layer=2, n_embd=4, n_head=2, padded_vocab_size=8
    )

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    pretrain_super_network.get_dataloaders = Mock(return_value=(dataloader, dataloader))

    for strategy in ("standard", "random", "sandwich"):
        pretrain_super_network.setup(
            "pythia-14m",
            devices=1,
            optimizer="RMSprop",
            training_strategy=strategy,
            model_config=model_config,
            out_dir=tmp_path,
            train=TrainArgs(
                global_batch_size=2,
                max_tokens=16,
                save_interval=1,
                micro_batch_size=1,
                max_norm=1.0,
            ),
            eval=EvalArgs(interval=1, max_iters=1, final_validation=False),
        )


@RunIf(min_cuda_gpus=1, standalone=True)
# Set CUDA_VISIBLE_DEVICES for FSDP hybrid-shard, if fewer GPUs are used than are available
@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
# If we were to use `save_hyperparameters()`, we would have to patch `sys.argv` or otherwise
# the CLI would capture pytest args, but unfortunately patching would mess with subprocess
# launching, so we need to mock `save_hyperparameters()`
@mock.patch("litgpt.pretrain.save_hyperparameters")
def test_pretrain(_, tmp_path):
    model_config = Config(
        block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8
    )

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    pretrain_super_network.get_dataloaders = Mock(return_value=(dataloader, dataloader))

    out_dir = tmp_path / "out"
    stdout = StringIO()
    with redirect_stdout(stdout):
        pretrain_super_network.setup(
            "pythia-14m",
            devices=1,
            model_config=model_config,
            out_dir=out_dir,
            train=TrainArgs(
                global_batch_size=2,
                max_tokens=16,
                save_interval=1,
                micro_batch_size=1,
                max_norm=1.0,
            ),
            eval=EvalArgs(interval=1, max_iters=1, final_validation=False),
        )

        # tmp_path is not the same across all ranks, run assert only on rank 0
    out_dir_contents = set(os.listdir(out_dir))
    checkpoint_dirs = {
        "step-00000001",
        "step-00000002",
        "step-00000003",
        "step-00000004",
        "final",
    }
    assert checkpoint_dirs.issubset(out_dir_contents)
    assert all((out_dir / p).is_dir() for p in checkpoint_dirs)
    for checkpoint_dir in checkpoint_dirs:
        # the `tokenizer_dir` is None by default, so only 'lit_model.pth' shows here
        assert set(os.listdir(out_dir / checkpoint_dir)) == {
            "lit_model.pth",
            "model_config.yaml",
        }

    assert (out_dir / "logs" / "tensorboard" / "version_0").is_dir()

    # logs only appear on rank 0
    logs = stdout.getvalue()
    assert logs.count("(step)") == 4
    assert logs.count("val loss") == 4
    assert "Total parameters: 1,888" in logs


@RunIf(min_cuda_gpus=1, standalone=True)
# Set CUDA_VISIBLE_DEVICES for FSDP hybrid-shard, if fewer GPUs are used than are available
@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("litgpt.pretrain.L.Fabric.load_raw")
# See comment in `test_pretrain` why we need to mock `save_hyperparameters()`
@mock.patch("litgpt.pretrain.save_hyperparameters")
def test_initial_checkpoint_dir(_, load_mock, tmp_path):
    model_config = Config(
        block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8
    )

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    pretrain_super_network.get_dataloaders = Mock(return_value=(dataloader, dataloader))
    pretrain_super_network.fit = Mock()

    pretrain_super_network.setup(
        "pythia-14m",
        initial_checkpoint_dir=tmp_path,
        devices=1,
        model_config=model_config,
        out_dir=tmp_path,
    )

    load_mock.assert_called_once_with(tmp_path / "lit_model.pth", ANY)
