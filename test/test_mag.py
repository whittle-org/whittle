import pytest
import torch

from litgpt.config import Config

from whittle.models.gpt import GPT
from whittle.metrics.mag import compute_weight_magnitude


mlp_types = ["GptNeoxMLP", "LLaMAMLP", "GemmaMLP"]
norm_types = ["LayerNorm", "RMSNorm"]


@pytest.mark.parametrize("mlp_type", mlp_types)
@pytest.mark.parametrize("norm_type", norm_types)
def test_compute_weight_magnitude(mlp_type, norm_type):
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
    gpt.transformer.wte.weight.data = torch.ones_like(gpt.transformer.wte.weight.data)
    gpt.lm_head.weight.data = torch.ones_like(gpt.lm_head.weight.data)
    gpt.lm_head.bias.data = torch.ones_like(gpt.lm_head.bias.data)
    gpt.transformer.ln_f.weight.data = torch.ones_like(gpt.transformer.ln_f.weight.data)
    if norm_type == "LayerNorm":
        gpt.transformer.ln_f.bias.data = torch.ones_like(gpt.transformer.ln_f.bias.data)

    for block in gpt.transformer.h:
        block.attn.attn.weight.data = torch.ones_like(block.attn.attn.weight.data)
        block.attn.attn.bias.data = torch.ones_like(block.attn.attn.bias.data)
        block.attn.proj.bias.data = torch.ones_like(block.attn.proj.bias.data)
        block.attn.proj.weight.data = torch.ones_like(block.attn.proj.weight.data)

        if mlp_type == "GptNeoxMLP":
            block.mlp.fc.weight.data = torch.ones_like(block.mlp.fc.weight.data)
            block.mlp.fc.bias.data = torch.ones_like(block.mlp.fc.bias.data)
            block.mlp.proj.weight.data = torch.ones_like(block.mlp.proj.weight.data)
            block.mlp.proj.bias.data = torch.ones_like(block.mlp.proj.bias.data)
        else:
            block.mlp.fc_1.weight.data = torch.ones_like(block.mlp.fc_1.weight.data)
            block.mlp.fc_1.bias.data = torch.ones_like(block.mlp.fc_1.bias.data)
            block.mlp.fc_2.weight.data = torch.ones_like(block.mlp.fc_2.weight.data)
            block.mlp.fc_2.bias.data = torch.ones_like(block.mlp.fc_2.bias.data)
            block.mlp.proj.weight.data = torch.ones_like(block.mlp.proj.weight.data)
            block.mlp.proj.bias.data = torch.ones_like(block.mlp.proj.bias.data)

        block.norm_1.weight.data = torch.ones_like(block.norm_1.weight.data)
        block.norm_2.weight.data = torch.ones_like(block.norm_2.weight.data)
        if norm_type == "LayerNorm":
            block.norm_1.bias.data = torch.ones_like(block.norm_1.bias.data)
            block.norm_2.bias.data = torch.ones_like(block.norm_2.bias.data)
    mag_super = compute_weight_magnitude(gpt)

    num_params_super = sum(p.numel() for p in gpt.parameters() if p.requires_grad)

    assert mag_super == num_params_super

    sub_network_config = {
        "sub_network_n_embd": 256,
        "sub_network_intermediate_size": [1024] * 1,
        "sub_network_num_heads": [4] * 1,
        "sub_network_n_layers": 1,
    }
    gpt.set_sub_network(**sub_network_config)
    mag_sub_network = compute_weight_magnitude(gpt)

    assert mag_sub_network < mag_super
