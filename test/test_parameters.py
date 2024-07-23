from litgpt.config import Config

from whittle.metrics.parameters import (
    compute_parameters,
    compute_parameters_sub_network_gpt,
)
from whittle.models.gpt import GPT

from test.test_training_strategies import MLP


def test_compute_parameters():
    model = MLP(8)
    assert compute_parameters(model) == 641


def test_compute_parameters_sub_network():
    config = Config()
    config.padded_vocab_size = 512
    config.n_embd = 64
    config.intermediate_size = 64 * 4
    config.n_head = 8
    config.n_query_groups = 8
    config.head_size = 8
    config.n_layer = 2
    config.block_size = 512
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)
    config.norm_eps = 1e-5
    config.lm_head_bias = True
    config.fix_head_size = True
    gpt = GPT(config)

    params_super_network = compute_parameters(gpt)

    params_sub_network = compute_parameters_sub_network_gpt(gpt)
    assert params_sub_network == params_super_network

    gpt.set_sub_network(
        sub_network_n_embd=config.n_embd,
        sub_network_intermediate_size=[
            config.intermediate_size for _ in range(config.n_layer)
        ],
        sub_network_num_heads=[config.n_head - 1]
        + [config.n_head for _ in range(1, config.n_layer)],
        sub_network_n_layers=config.n_layer,
    )
    params_sub_network = compute_parameters_sub_network_gpt(gpt)
    params_single_head = (config.n_embd * config.head_size + config.head_size) * 3
    assert params_sub_network == params_super_network - params_single_head
