from __future__ import annotations

import torch
from whittle.lora.config import LoRAConfig as Config

from litgpt import Config as LitConfig
from whittle.lora.lora_block import LoRABlock as Block
from litgpt.model import (
    Block as LitBlock,
    build_mask_cache,
    build_rope_cache,
)


def test_block():
    config = Config()
    config.n_embd = 64
    config.n_head = 8
    config.n_query_groups = 4
    config.head_size = 8
    config.intermediate_size = 64 * 4
    config.fix_head_size = False
    config.mlp_class_name = "LLaMAMLP"
    config.max_seq_len = 512
    config.rotary_percentage = 0.25
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)
    cos, sin = build_rope_cache(config.max_seq_len, n_elem=config.rope_n_elem)
    litconfig = LitConfig()
    litconfig.n_embd = 64
    litconfig.n_head = 8
    litconfig.n_query_groups = 4
    litconfig.head_size = 8
    litconfig.intermediate_size = 64 * 4
    litconfig.fix_head_size = False
    litconfig.mlp_class_name = "LLaMAMLP"
    litconfig.max_seq_len = 512
    litconfig.rotary_percentage = 0.25
    litconfig.rope_n_elem = int(litconfig.rotary_percentage * litconfig.head_size)
    block = Block(config, 0)
    input = torch.rand(8, 512, 64)
    mask = build_mask_cache(512)
    block.attn.attn.linear.linear.weight.data = torch.ones_like(
        block.attn.attn.linear.linear.weight.data
    )
    block.attn.attn.linear.linear.bias.data = torch.ones_like(
        block.attn.attn.linear.linear.bias.data
    )
    block.attn.proj.linear.bias.data = torch.ones_like(block.attn.proj.linear.bias.data)
    block.attn.proj.linear.weight.data = torch.ones_like(
        block.attn.proj.linear.weight.data
    )
    block.mlp.fc_1.linear.weight.data = torch.ones_like(
        block.mlp.fc_1.linear.weight.data
    )
    block.mlp.fc_1.linear.bias.data = torch.ones_like(block.mlp.fc_1.linear.bias.data)
    block.mlp.fc_2.linear.weight.data = torch.ones_like(
        block.mlp.fc_2.linear.weight.data
    )
    block.mlp.fc_2.linear.bias.data = torch.ones_like(block.mlp.fc_2.linear.bias.data)
    block.mlp.proj.linear.weight.data = torch.ones_like(
        block.mlp.proj.linear.weight.data
    )
    block.mlp.proj.linear.bias.data = torch.ones_like(block.mlp.proj.linear.bias.data)
    block.reset_super_network()
    out_large = block(input, cos, sin, mask)
    assert out_large.shape == (8, 512, 64)
    block.set_sub_network(
        sub_network_n_embd=32,
        sub_network_intermediate_size=32 * 4,
        sub_network_num_heads=8,
        sub_network_query_groups=config.n_query_groups // 2,
        sub_network_head_size=32 // 4,
    )
    out_small = block(input[:, :, :32], cos, sin, mask)
    assert out_small.shape == (8, 512, 32)

    lit_block = LitBlock(litconfig, 0)
    print(lit_block)
    lit_block.attn.attn.weight.data = torch.ones_like(lit_block.attn.attn.weight.data)
    lit_block.attn.attn.bias.data = torch.ones_like(lit_block.attn.attn.bias.data)
    lit_block.attn.proj.bias.data = torch.ones_like(lit_block.attn.proj.bias.data)
    lit_block.attn.proj.weight.data = torch.ones_like(lit_block.attn.proj.weight.data)
    lit_block.mlp.fc_1.weight.data = torch.ones_like(lit_block.mlp.fc_1.weight.data)
    lit_block.mlp.fc_1.bias.data = torch.ones_like(lit_block.mlp.fc_1.bias.data)
    lit_block.mlp.fc_2.weight.data = torch.ones_like(lit_block.mlp.fc_2.weight.data)
    lit_block.mlp.fc_2.bias.data = torch.ones_like(lit_block.mlp.fc_2.bias.data)
    lit_block.mlp.proj.weight.data = torch.ones_like(lit_block.mlp.proj.weight.data)
    lit_block.mlp.proj.bias.data = torch.ones_like(lit_block.mlp.proj.bias.data)
    out_lit_large = lit_block(input, cos, sin, mask)
    assert torch.all(out_lit_large == out_large)

    litconfig.n_embd = 32
    litconfig.n_head = 4
    litconfig.n_query_groups = 2
    litconfig.intermediate_size = 32 * 4
    lit_block_small = LitBlock(litconfig, 0)
    lit_block_small.attn.attn.weight.data = torch.ones_like(
        lit_block_small.attn.attn.weight.data
    )
    lit_block_small.attn.attn.bias.data = torch.ones_like(
        lit_block_small.attn.attn.bias.data
    )
    lit_block_small.attn.proj.bias.data = torch.ones_like(
        lit_block_small.attn.proj.bias.data
    )
    lit_block_small.attn.proj.weight.data = torch.ones_like(
        lit_block_small.attn.proj.weight.data
    )
    lit_block_small.mlp.fc_1.weight.data = torch.ones_like(
        lit_block_small.mlp.fc_1.weight.data
    )
    lit_block_small.mlp.fc_1.bias.data = torch.ones_like(
        lit_block_small.mlp.fc_1.bias.data
    )
    lit_block_small.mlp.fc_2.weight.data = torch.ones_like(
        lit_block_small.mlp.fc_2.weight.data
    )
    lit_block_small.mlp.fc_2.bias.data = torch.ones_like(
        lit_block_small.mlp.fc_2.bias.data
    )
    lit_block_small.mlp.proj.weight.data = torch.ones_like(
        lit_block_small.mlp.proj.weight.data
    )
    lit_block_small.mlp.proj.bias.data = torch.ones_like(
        lit_block_small.mlp.proj.bias.data
    )
    out_lit_small = lit_block_small(input[:, :, :32], cos, sin, mask)
    assert torch.all(out_lit_small == out_small)
