from __future__ import annotations
import pytest


from whittle.models.gpt import GPT, Config
from whittle.prunning.pruner import NMPruner


@pytest.mark.parametrize(
    "model_config",
    [
        {
            "name": "pythia-14m",
            "config_params": {
                "block_size": 6,
                "sliding_window_size": 3,
                "n_layer": 2,
                "n_embd": 32,
                "intermediate_size": 86,
            },
        },
        {
            "name": "gemma-2-9b",
            "config_params": {
                "block_size": 6,
                "sliding_window_size": 3,
                "n_layer": 2,
                "n_embd": 32,
                "intermediate_size": 86,
            },
        },
        {
            "name": "Llama-3-8B",
            "config_params": {
                "n_layer": 2,
                "n_embd": 32,
                "intermediate_size": 86,
                "padded_vocab_size": 10000,
            },
        },
        {
            "name": "Llama-3.2-1B",
            "config_params": {
                "n_layer": 2,
                "n_embd": 32,
                "intermediate_size": 86,
                "padded_vocab_size": 10000,
            },
        },
    ],
)
def test_mag_model_pruning(model_config):
    config = Config.from_name(model_config["name"], **model_config["config_params"])

    config.fix_head_size = True

    model = GPT(config)

    pruner = NMPruner()
    sparsity = pruner(model, prune_n=2, prune_m=4, prune_methode="magnitude")

    assert abs(sparsity - 0.5) <= 0.1
