import torch

from litgpt.config import Config

from whittle.models.gpt import GPT
from whittle.metrics.mag import weight_magnitude


def test_weight_magnitude():
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
    gpt.transformer.wte.weight.data = torch.ones_like(gpt.transformer.wte.weight.data)
    gpt.lm_head.weight.data = torch.ones_like(gpt.lm_head.weight.data)
    gpt.lm_head.bias.data = torch.ones_like(gpt.lm_head.bias.data)
    gpt.transformer.ln_f.weight.data = torch.ones_like(gpt.transformer.ln_f.weight.data)

    for block in gpt.transformer.h:
        block.attn.attn.weight.data = torch.ones_like(block.attn.attn.weight.data)
        block.attn.attn.bias.data = torch.ones_like(block.attn.attn.bias.data)
        block.attn.proj.bias.data = torch.ones_like(block.attn.proj.bias.data)
        block.attn.proj.weight.data = torch.ones_like(block.attn.proj.weight.data)
        block.mlp.fc.weight.data = torch.ones_like(block.mlp.fc.weight.data)
        block.mlp.fc.bias.data = torch.ones_like(block.mlp.fc.bias.data)
        block.mlp.proj.weight.data = torch.ones_like(block.mlp.proj.weight.data)
        block.mlp.proj.bias.data = torch.ones_like(block.mlp.proj.bias.data)
        block.norm_1.weight.data = torch.ones_like(block.norm_1.weight.data)
        block.norm_2.weight.data = torch.ones_like(block.norm_2.weight.data)

    mag_super = weight_magnitude(gpt)

    num_params_super = sum(p.numel() for p in gpt.parameters() if p.requires_grad)

    assert mag_super == num_params_super

    sub_network_config = {
        "sub_network_n_embd": 256,
        "sub_network_intermediate_size": [1024] * 3,
        "sub_network_num_heads": [4] * 3,
        "sub_network_n_layers": 3,
    }
    gpt.set_sub_network(**sub_network_config)
    mag_sub_network = weight_magnitude(gpt)

    assert mag_sub_network > mag_super
