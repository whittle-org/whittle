import pytest

from litgpt.config import Config

from whittle.models.gpt import GPT
from whittle.metrics import compute_flops


mlp_types = ["GptNeoxMLP", "LLaMAMLP", "GemmaMLP"]
norm_types = ["LayerNorm", "RMSNorm"]


@pytest.mark.parametrize("mlp_type", mlp_types)
@pytest.mark.parametrize("norm_type", norm_types)
def test_compute_flops(mlp_type, norm_type):
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
    config.mlp_class_name = mlp_type
    config.norm_class_name = norm_type

    gpt = GPT(config)
    flops = compute_flops(gpt)
    assert flops > 0
