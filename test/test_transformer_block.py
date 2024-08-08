from __future__ import annotations

import torch
from litgpt import Config
from litgpt.model import (
    Block as LitBlock,
    build_mask_cache,
    build_rope_cache,
)

from whittle.models.gpt.blocks import Block


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

    block = Block(config, 0)
    input = torch.rand(8, 512, 64)
    mask = build_mask_cache(512)
    block.attn.attn.weight.data = torch.ones_like(block.attn.attn.weight.data)
    block.attn.attn.bias.data = torch.ones_like(block.attn.attn.bias.data)
    block.attn.proj.bias.data = torch.ones_like(block.attn.proj.bias.data)
    block.attn.proj.weight.data = torch.ones_like(block.attn.proj.weight.data)
    block.mlp.fc_1.weight.data = torch.ones_like(block.mlp.fc_1.weight.data)
    block.mlp.fc_1.bias.data = torch.ones_like(block.mlp.fc_1.bias.data)
    block.mlp.fc_2.weight.data = torch.ones_like(block.mlp.fc_2.weight.data)
    block.mlp.fc_2.bias.data = torch.ones_like(block.mlp.fc_2.bias.data)
    block.mlp.proj.weight.data = torch.ones_like(block.mlp.proj.weight.data)
    block.mlp.proj.bias.data = torch.ones_like(block.mlp.proj.bias.data)
    block.reset_super_network()
    out_large = block(input, cos, sin, mask)
    assert out_large.shape == (8, 512, 64)
    block.set_sub_network(
        sub_network_n_embd=32,
        sub_network_intermediate_size=32 * 4,
        sub_network_num_heads=4,
    )
    out_small = block(input[:, :, :32], cos, sin, mask)
    assert out_small.shape == (8, 512, 32)

    lit_block = LitBlock(config, 0)
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

    config.n_embd = 32
    config.n_head = 4
    config.n_query_groups = 2
    config.intermediate_size = 32 * 4
    lit_block_small = LitBlock(config, 0)
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
