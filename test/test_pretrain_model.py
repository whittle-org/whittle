from __future__ import annotations

from unittest import mock

import pytest
import torch
from litgpt import Config
from litgpt.args import EvalArgs, TrainArgs
from litgpt.utils import save_config
from torch.utils.data import DataLoader

from whittle import pretrain_model


def test_config_path_and_model_config_are_mutually_exclusive(tmp_path):
    config = Config(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    save_config(config, tmp_path)

    with pytest.raises(ValueError, match="not both"):
        pretrain_model.setup(
            "pythia-14m",
            model_config=config,
            config_path=str(tmp_path / "model_config.yaml"),
            out_dir=tmp_path / "out",
        )


def test_lr_schedule_arguments_reach_the_schedule(tmp_path):
    config = Config(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)

    with (
        mock.patch(
            "whittle.pretrain_model.get_dataloaders",
            return_value=(dataloader, dataloader),
        ),
        mock.patch(
            "whittle.pretrain_model.get_wsd_lr", wraps=pretrain_model.get_wsd_lr
        ) as get_wsd_lr_mock,
    ):
        pretrain_model.setup(
            "pythia-14m",
            model_config=config,
            out_dir=tmp_path / "out",
            precision="32-true",
            devices=1,
            train=TrainArgs(
                global_batch_size=2,
                micro_batch_size=1,
                max_tokens=8,
                max_norm=1.0,
                lr_warmup_steps=1,
                save_interval=None,
            ),
            eval=EvalArgs(interval=10, max_iters=1, final_validation=False),
            logger_name="csv",
            lr_stable_ratio=0.5,
            lr_decay_type="cosine",
        )

    assert get_wsd_lr_mock.called
    for call in get_wsd_lr_mock.call_args_list:
        assert call.kwargs == {"stable_ratio": 0.5, "decay_type": "cosine"}
