from __future__ import annotations

import pytest
import torch
from litgpt import Config
from litgpt.model import (
    CausalSelfAttention as LitCausalSelfAttention,
    build_mask_cache,
    build_rope_cache,
)

from whittle.models.gpt.blocks import CausalSelfAttention

attention_configs = {
    "mha_fix_head_size_sliding": {
        "config": Config(
            n_embd=64,
            n_head=16,
            n_query_groups=16,
            head_size=64,
            sliding_window_size=256,
            sliding_window_layer_placing="interleaved",
        ),
        "fix_head_size": True,
    },
    "mha_fix_head_size": {
        "config": Config(n_embd=64, n_head=16, n_query_groups=16, head_size=64),
        "fix_head_size": True,
    },
    "gqa_fix_head_size": {
        "config": Config(n_embd=64, n_head=16, n_query_groups=2, head_size=64),
        "fix_head_size": True,
    },
    "mqa_fix_head_size": {
        "config": Config(n_embd=64, n_head=16, n_query_groups=1, head_size=64),
        "fix_head_size": True,
    },
    "mha_flexible_head_size": {
        "config": Config(n_embd=64, n_head=16, n_query_groups=16),
        "fix_head_size": False,
    },
    "gqa_flexible_head_size": {
        "config": Config(n_embd=64, n_head=16, n_query_groups=2),
        "fix_head_size": False,
    },
    "mqa_flexible_head_size": {
        "config": Config(n_embd=64, n_head=16, n_query_groups=1),
        "fix_head_size": False,
    },
}


def init_attention(config):
    attention = CausalSelfAttention(config, 2)
    torch.manual_seed(0)
    attention.attn.weight.data = torch.randn_like(attention.attn.weight.data)
    attention.attn.bias.data = torch.randn_like(attention.attn.bias.data)
    attention.proj.bias.data = torch.randn_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.randn_like(attention.proj.weight.data)
    return attention


def init_lit_attention(config):
    attention = LitCausalSelfAttention(config, 2)
    torch.manual_seed(0)
    attention.attn.weight.data = torch.randn_like(attention.attn.weight.data)
    attention.attn.bias.data = torch.randn_like(attention.attn.bias.data)
    attention.proj.bias.data = torch.randn_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.randn_like(attention.proj.weight.data)
    return attention


def init_lit_small_attention(config, base_attention):
    attention = LitCausalSelfAttention(config, 2)
    torch.manual_seed(0)
    slices = tuple(slice(0, s) for s in attention.attn.weight.data.size())
    attention.attn.weight.data = base_attention.attn.weight.data[slices]
    slices = tuple(slice(0, s) for s in attention.attn.bias.data.size())
    attention.attn.bias.data = base_attention.attn.bias.data[slices]
    slices = tuple(slice(0, s) for s in attention.proj.bias.data.size())
    attention.proj.bias.data = base_attention.proj.bias.data[slices]
    slices = tuple(slice(0, s) for s in attention.proj.weight.data.size())
    attention.proj.weight.data = base_attention.proj.weight.data[slices]
    return attention


@pytest.mark.parametrize("attention_config", attention_configs.keys())
def test_attention(attention_config):
    config = attention_configs[attention_config]["config"]
    config.fix_head_size = attention_configs[attention_config]["fix_head_size"]
    config.max_seq_len = 512
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)

    seq_len = config.max_seq_len
    cos, sin = build_rope_cache(seq_len, n_elem=config.rope_n_elem)
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    input = torch.rand(8, seq_len, config.n_embd)
    mask = build_mask_cache(seq_len)

    attention = init_attention(config)
    out_large = attention(input, mask=mask, cos=cos, sin=sin)

    # check shape of super network attention
    assert out_large.shape == (8, seq_len, config.n_embd)
    lit_attention = init_lit_attention(config)
    out_lit_large = lit_attention(input, mask=mask, cos=cos, sin=sin)

    attention.set_sub_network(
        sub_network_n_embd=config.n_embd // 2, sub_network_n_head=config.n_head // 4
    )
    cos, sin = build_rope_cache(
        seq_len, n_elem=int(config.rotary_percentage * attention.sub_network_head_size)
    )
    out_small = attention(
        input[:, :, : config.n_embd // 2], mask=mask, cos=cos, sin=sin
    )

    # check shape of sub-network attention
    assert out_small.shape == (8, seq_len, config.n_embd // 2)

    # check that our custom model produces the same output as LitGPT
    assert torch.all(out_lit_large == out_large)

    config.n_embd = attention.sub_network_n_embd
    config.n_head = attention.sub_network_n_head
    config.n_query_groups = attention.sub_network_query_groups
    config.head_size = attention.sub_network_head_size
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)

    lit_attention_small = init_lit_small_attention(config, lit_attention)

    out_lit_small = lit_attention_small(
        input[:, :, : config.n_embd], mask=mask, cos=cos, sin=sin
    )

    # check that our sub-networks the same output as equally sized LitGPT attention layer
    assert torch.all(out_lit_small == out_small)
