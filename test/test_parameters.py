from __future__ import annotations

import pytest
from litgpt.config import Config

from test.test_training_strategies import MLP
from whittle.metrics.parameters import (
    compute_all_parameters,
    compute_parameters,
)
from whittle.models.gpt import GPT

mlp_types = ["GptNeoxMLP", "LLaMAMLP", "GemmaMLP"]
norm_types = ["LayerNorm", "RMSNorm"]


def test_compute_parameters():
    model = MLP(8)
    assert compute_all_parameters(model) == 641


@pytest.mark.parametrize("mlp_type", mlp_types)
@pytest.mark.parametrize("norm_type", norm_types)
def test_compute_parameters_sub_network(mlp_type, norm_type):
    config = Config()
    config.padded_vocab_size = 512
    config.n_embd = 64
    config.intermediate_size = 64 * 4
    config.n_head = 8
    config.bias = False
    config.n_query_groups = 8
    config.head_size = 8
    config.n_layer = 2
    config.block_size = 512
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)
    config.norm_eps = 1e-5
    config.lm_head_bias = True
    config.fix_head_size = True
    config.mlp_class_name = mlp_type
    config.norm_class_name = norm_type

    gpt = GPT(config)

    params_super_network = compute_parameters(gpt)

    params_sub_network = compute_parameters(gpt)
    assert params_sub_network == params_super_network

    # reduce super-network by one single head
    gpt.set_sub_network(
        sub_network_n_embd=config.n_embd,
        sub_network_intermediate_size=config.intermediate_size,
        sub_network_num_heads=config.n_head - 1,
        sub_network_n_layers=config.n_layer,
        sub_network_head_size=config.head_size,
        sub_network_query_groups=config.n_query_groups - 1,
    )
    params_sub_network = compute_parameters(gpt)
    params_single_head = (
        config.n_embd * config.head_size + config.head_size * config.n_embd * 3
    ) * (config.n_layer)
    assert params_sub_network == params_super_network - params_single_head

    # remove larger part of the network
    gpt.set_sub_network(
        sub_network_n_embd=config.n_embd // 2,
        sub_network_intermediate_size=config.intermediate_size,
        sub_network_num_heads=config.n_head - 5,
        sub_network_n_layers=config.n_layer - 1,
    )
    params_sub_network = compute_parameters(gpt)
    assert params_sub_network < params_super_network
