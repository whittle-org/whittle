from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import yaml  # type: ignore[import-untyped]
from jsonargparse import ArgumentParser
from litgpt import Config
from litgpt.args import TrainArgs
from litgpt.data import JSON, TextFiles

from whittle import full_finetune, pretrain_super_network
from whittle.hyperparameters import dump_hyperparameters, save_hyperparameters


def read_back(function, text: str, tmp_path: Path):
    """Reads the YAML text with the parser of `function`, as `--config` does."""
    path = tmp_path / "hyperparameters.yaml"
    path.write_text(text)
    parser = ArgumentParser(exit_on_error=False)
    parser.add_function_arguments(function)
    return parser.instantiate_classes(parser.parse_path(path))


@mock.patch("sys.argv", ["some_script.py", "--unrelated", "argument"])
def test_dump_hyperparameters_round_trip(tmp_path):
    arguments = {
        "model_name": "pythia-14m",
        "model_config": Config.from_name("pythia-14m", n_layer=2),
        "data": TextFiles(train_data_path=Path("data")),
        "train": TrainArgs(max_tokens=100),
        "out_dir": Path("out"),
    }
    text = dump_hyperparameters(pretrain_super_network.setup, arguments)
    config = read_back(pretrain_super_network.setup, text, tmp_path)

    assert config.model_name == "pythia-14m"
    assert config.model_config.n_layer == 2
    assert isinstance(config.data, TextFiles)
    assert config.data.train_data_path == Path("data")
    assert config.train.max_tokens == 100
    assert config.out_dir == Path("out")
    # arguments that were not given keep their default
    assert config.training_strategy == "sandwich"


def test_dump_hyperparameters_instruction_data(tmp_path):
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps([{"instruction": "a", "input": "", "output": "b"}]))
    arguments = {
        "checkpoint_dir": Path("checkpoint"),
        "data": JSON(json_path=json_path, val_split_fraction=0.5),
    }
    text = dump_hyperparameters(full_finetune.setup, arguments)
    config = read_back(full_finetune.setup, text, tmp_path)

    assert isinstance(config.data, JSON)
    assert config.data.json_path == json_path
    assert config.data.val_split_fraction == 0.5


def test_dump_hyperparameters_invalid_argument():
    with pytest.raises(Exception, match="devices"):
        dump_hyperparameters(
            pretrain_super_network.setup, {"model_name": "pythia-14m", "devices": [1]}
        )


def test_save_hyperparameters(tmp_path):
    text = dump_hyperparameters(
        pretrain_super_network.setup, {"model_name": "pythia-14m"}
    )
    save_hyperparameters(text, tmp_path)

    saved = yaml.safe_load((tmp_path / "hyperparameters.yaml").read_text())
    assert saved["model_name"] == "pythia-14m"
