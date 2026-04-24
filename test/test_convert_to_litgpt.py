from __future__ import annotations

import json
import os
import pathlib

import pytest
import torch
from litgpt import Config
from litgpt.scripts.download import download_from_hub

from whittle.convert import convert_subnet_to_litgpt
from whittle.models.gpt import GPT


def _halved_subnet_config(config):
    return {
        "sub_network_n_embd": max(1, config.n_embd // 2),
        "sub_network_intermediate_size": max(1, config.intermediate_size // 2),
        "sub_network_num_heads": max(1, config.n_head // 2),
        "sub_network_n_layers": max(1, config.n_layer // 2),
        "sub_network_query_groups": max(1, config.n_query_groups // 2),
        "sub_network_head_size": config.head_size,
    }


@pytest.fixture(scope="session")
def supernet():
    model_id = "EleutherAI/pythia-14m"
    config_path = os.path.join("checkpoints", model_id, "model_config.yaml")

    # Download model if not present
    if not os.path.exists(config_path):
        download_from_hub(
            model_id, checkpoint_dir=pathlib.Path(os.path.join("checkpoints"))
        )

    config = Config.from_file(config_path)
    supernet = GPT(config)
    return supernet


CONFIG_IDS = list(range(100))
SEARCH_SPACE_TYPES = ["coarse", "finegrained"]


@pytest.mark.parametrize("config_id", CONFIG_IDS)
@pytest.mark.parametrize("search_space_type", SEARCH_SPACE_TYPES)
def test_equivalence(supernet, search_space_type, config_id):
    with open(f"test/config_{search_space_type}.json") as f:
        subnet_config = json.load(f)[config_id]

    lit_model = convert_subnet_to_litgpt(supernet, subnet_config)

    x = torch.tensor([[9856, 23, 491, 1536, 304, 1234]], dtype=torch.int32)
    supernet.set_sub_network(**subnet_config)
    out_whittle = supernet(x)
    lit_out = lit_model(x)

    assert torch.allclose(out_whittle, lit_out)


@pytest.mark.parametrize(
    "model_name",
    [
        "Llama-3-8B",
        "gemma-2-9b",
        "gemma-3-1b-it",
        "OLMo-2-1124-7B",
        "Qwen3-0.6B",
    ],
)
def test_halved_subnet_matches_converted_litgpt(model_name):
    config = Config.from_name(
        model_name,
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
        padded_vocab_size=10000,
    )

    whittle_model = GPT(config)
    whittle_model.eval()

    subnet_config = _halved_subnet_config(config)
    lit_model = convert_subnet_to_litgpt(whittle_model, subnet_config)
    lit_model.eval()

    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    whittle_model.set_sub_network(**subnet_config)
    out_whittle = whittle_model(x)
    out_lit = lit_model(x)

    assert torch.allclose(out_whittle, out_lit, atol=1e-6)
