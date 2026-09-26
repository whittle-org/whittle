from __future__ import annotations

from unittest import mock

import pytest
from litgpt import Config
from litgpt.utils import save_config

from whittle import distill_model


def test_config_path_and_model_config_are_mutually_exclusive(tmp_path):
    config = Config(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    save_config(config, tmp_path)

    with pytest.raises(ValueError, match="not both"):
        distill_model.setup(
            "pythia-14m",
            model_config=config,
            config_path=str(tmp_path / "model_config.yaml"),
            teacher_checkpoint_dir=tmp_path,
            out_dir=tmp_path / "out",
        )


@pytest.mark.parametrize("source", ["model_config", "config_path", "model_name"])
def test_student_config_source(tmp_path, source):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    save_config(Config.from_name("pythia-14m"), teacher_dir)
    student = Config(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    save_config(student, tmp_path)

    kwargs = {}
    if source == "model_config":
        kwargs["model_config"] = student
    elif source == "config_path":
        kwargs["config_path"] = str(tmp_path / "model_config.yaml")

    with mock.patch("whittle.distill_model.main") as main_mock:
        distill_model.setup(
            "pythia-14m",
            teacher_checkpoint_dir=teacher_dir,
            out_dir=tmp_path / "out",
            precision="32-true",
            devices=1,
            logger_name="csv",
            **kwargs,
        )

    student_config = main_mock.call_args.kwargs["student_config"]
    if source == "model_name":
        assert student_config.name == "pythia-14m"
        assert student_config.n_layer == Config.from_name("pythia-14m").n_layer
    else:
        assert (student_config.n_layer, student_config.n_embd) == (2, 8)
