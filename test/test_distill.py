from __future__ import annotations

import os
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import Config
from litgpt.utils import save_config
from torch.utils.data import DataLoader

from whittle import distill
from whittle.args import DistillArgs
from whittle.models.gpt import GPT

STUDENT_CONFIG = {
    "sub_network_n_embd": 4,
    "sub_network_intermediate_size": 8,
    "sub_network_num_heads": 2,
    "sub_network_n_layers": 1,
}


def run_distill(tmp_path, sample_mock_kwargs, min_ratio=0.01, max_ratio=0.99):
    teacher_config = Config(
        block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8
    )
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    save_config(teacher_config, teacher_dir)
    torch.save(GPT(teacher_config).state_dict(), teacher_dir / "lit_model.pth")

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)

    out_dir = tmp_path / "out"
    with (
        mock.patch(
            "whittle.distill.get_dataloaders", Mock(return_value=(dataloader, dataloader))
        ),
        mock.patch(
            "whittle.sampling.random_sampler.RandomSampler.sample", **sample_mock_kwargs
        ),
    ):
        distill.setup(
            out_dir=out_dir,
            precision="32-true",
            teacher_checkpoint_dir=teacher_dir,
            train=TrainArgs(
                global_batch_size=2,
                micro_batch_size=1,
                max_tokens=8,
                save_interval=1,
                max_norm=1.0,
                lr_warmup_steps=1,
            ),
            distill=DistillArgs(),
            eval=EvalArgs(interval=1, max_iters=1, initial_validation=False),
            optimizer="RMSprop",
            devices=1,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
        )
    return out_dir


@mock.patch.dict(os.environ, {"DISABLE_TORCH_COMPILE": "1"})
# See `test_pretrain` in test_pretrain_super_network.py for why we mock this
@mock.patch("whittle.distill.save_hyperparameters")
def test_distill(save_hyperparameters_mock, tmp_path):
    # a single-element side_effect stops the test if the student sampling retries
    out_dir = run_distill(tmp_path, {"side_effect": [STUDENT_CONFIG]})

    for checkpoint_dir in ["step-00000001", "step-00000002", "distill"]:
        assert (out_dir / checkpoint_dir / "lit_model.pth").is_file()
        student = Config.from_file(out_dir / checkpoint_dir / "model_config.yaml")
        assert (student.n_embd, student.n_layer, student.n_head) == (4, 1, 2)

    save_hyperparameters_mock.assert_called()


@mock.patch("whittle.distill.MAX_STUDENT_SAMPLES", 3)
def test_distill_stops_when_ratio_is_unreachable(tmp_path):
    # the student has about 13% of the teacher parameters
    with pytest.raises(RuntimeError, match=r"\[0.9, 0.95\] after 3 samples"):
        run_distill(
            tmp_path,
            {"return_value": STUDENT_CONFIG},
            min_ratio=0.9,
            max_ratio=0.95,
        )


@mock.patch("whittle.distill.MAX_STUDENT_SAMPLES", 3)
@mock.patch(
    "whittle.distill.extract_current_sub_network",
    side_effect=ValueError("extraction failed"),
)
def test_distill_stops_when_extraction_fails(extract_mock, tmp_path):
    with pytest.raises(RuntimeError, match="after 3 samples") as exc_info:
        run_distill(tmp_path, {"return_value": STUDENT_CONFIG})

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert extract_mock.call_count == 3
