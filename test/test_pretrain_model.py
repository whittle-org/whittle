from __future__ import annotations

import pytest
from litgpt import Config
from litgpt.utils import save_config

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
