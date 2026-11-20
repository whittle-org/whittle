from __future__ import annotations

import os
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
import yaml  # type: ignore[import-untyped]
from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import Config
from litgpt.utils import save_config
from torch.utils.data import DataLoader

from whittle import distill_super_network
from whittle.args import DistillArgs
from whittle.models.gpt import GPT

STUDENT_CONFIG = {
    "sub_network_n_embd": 4,
    "sub_network_intermediate_size": 8,
    "sub_network_num_heads": 2,
    "sub_network_n_layers": 1,
}


def save_model(config, checkpoint_dir, whittle_format):
    checkpoint_dir.mkdir()
    save_config(config, checkpoint_dir)
    state_dict = GPT(config).state_dict()
    if whittle_format:
        state_dict = {"model": state_dict, "parent_dir": checkpoint_dir}
    torch.save(state_dict, checkpoint_dir / "lit_model.pth")


def run_distill(
    tmp_path,
    sample_mock_kwargs,
    min_ratio=0.01,
    max_ratio=0.99,
    whittle_format=False,
    student_config=None,
):
    teacher_config = Config(
        block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8
    )
    teacher_dir = tmp_path / "teacher"
    save_model(teacher_config, teacher_dir, whittle_format)
    student_dir = None
    if student_config is not None:
        student_dir = tmp_path / "student"
        save_model(student_config, student_dir, whittle_format)

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)

    out_dir = tmp_path / "out"
    with (
        mock.patch(
            "whittle.distill_model.get_dataloaders",
            Mock(return_value=(dataloader, dataloader)),
        ),
        mock.patch(
            "whittle.sampling.random_sampler.RandomSampler.sample", **sample_mock_kwargs
        ),
    ):
        distill_super_network.setup(
            out_dir=out_dir,
            precision="32-true",
            teacher_checkpoint_dir=teacher_dir,
            student_dir=student_dir,
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


@pytest.mark.parametrize("whittle_format", [False, True])
@mock.patch.dict(os.environ, {"DISABLE_TORCH_COMPILE": "1"})
def test_distill(tmp_path, whittle_format):
    # a single-element side_effect stops the test if the student sampling retries
    out_dir = run_distill(
        tmp_path, {"side_effect": [STUDENT_CONFIG]}, whittle_format=whittle_format
    )

    for checkpoint_dir in ["step-00000001", "step-00000002", "final"]:
        assert (out_dir / checkpoint_dir / "lit_model.pth").is_file()
        checkpoint = torch.load(
            out_dir / checkpoint_dir / "lit_model.pth", weights_only=False
        )
        assert "teacher" not in checkpoint
        student = Config.from_file(out_dir / checkpoint_dir / "model_config.yaml")
        assert (student.n_embd, student.n_layer, student.n_head) == (4, 1, 2)
        # the arguments of the call are saved, not the arguments of pytest
        hyperparameters = yaml.safe_load(
            (out_dir / checkpoint_dir / "hyperparameters.yaml").read_text()
        )
        assert hyperparameters["min_ratio"] == 0.01
        assert hyperparameters["train"]["max_tokens"] == 8


@mock.patch.dict(os.environ, {"DISABLE_TORCH_COMPILE": "1"})
def test_distill_from_student_dir(tmp_path):
    student_config = Config(
        block_size=2, n_layer=1, n_embd=4, n_head=2, padded_vocab_size=8
    )
    out_dir = run_distill(
        tmp_path, {"side_effect": []}, whittle_format=True, student_config=student_config
    )

    student = Config.from_file(out_dir / "final" / "model_config.yaml")
    assert (student.n_embd, student.n_layer, student.n_head) == (4, 1, 2)


@mock.patch("whittle.distill_model.MAX_STUDENT_SAMPLES", 3)
def test_distill_stops_when_ratio_is_unreachable(tmp_path):
    # the student has about 13% of the teacher parameters
    with pytest.raises(RuntimeError, match=r"\[0.9, 0.95\] after 3 samples"):
        run_distill(
            tmp_path,
            {"return_value": STUDENT_CONFIG},
            min_ratio=0.9,
            max_ratio=0.95,
        )


@mock.patch("whittle.distill_model.MAX_STUDENT_SAMPLES", 3)
@mock.patch(
    "whittle.distill_model.extract_current_sub_network",
    side_effect=ValueError("extraction failed"),
)
def test_distill_stops_when_extraction_fails(extract_mock, tmp_path):
    with pytest.raises(RuntimeError, match="after 3 samples") as exc_info:
        run_distill(tmp_path, {"return_value": STUDENT_CONFIG})

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert extract_mock.call_count == 3
