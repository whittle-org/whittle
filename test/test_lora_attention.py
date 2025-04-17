from __future__ import annotations

import pytest
import torch
from litgpt.model import (
    CausalSelfAttention as LitCausalSelfAttention,
    build_mask_cache,
    build_rope_cache,
)

from whittle.lora_model.config import LoRAConfig as Config
from whittle.lora_model.lora_attention import CausalSelfAttention

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
    attention.qkv.linear.weight.data = torch.randn_like(attention.qkv.linear.weight.data)
    attention.qkv.linear.bias.data = torch.randn_like(attention.qkv.linear.bias.data)
    attention.proj.linear.bias.data = torch.randn_like(attention.proj.linear.bias.data)
    attention.proj.linear.weight.data = torch.randn_like(
        attention.proj.linear.weight.data
    )
    return attention


def init_lit_attention(config):
    attention = LitCausalSelfAttention(config, 2)
    torch.manual_seed(0)
    attention.qkv.weight.data = torch.randn_like(attention.qkv.weight.data)
    attention.qkv.bias.data = torch.randn_like(attention.qkv.bias.data)
    attention.proj.bias.data = torch.randn_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.randn_like(attention.proj.weight.data)
    return attention


def init_lit_small_attention(config, base_attention, attention_super):
    attention = LitCausalSelfAttention(config, 2)
    torch.manual_seed(0)
    slices = tuple(slice(0, s) for s in attention.qkv.weight.data.size())[1]
    qkv_indices = (
        attention_super.qkv_indices
        if attention_super.qkv_indices is not None
        else slice(0, attention.qkv.weight.data.size()[0])
    )
    attention.qkv.weight.data = base_attention.qkv.weight.data[qkv_indices, :][
        :, 0 : attention.qkv.weight.data.size()[1]
    ]
    attention.qkv.bias.data = base_attention.qkv.bias.data[qkv_indices]
    proj_indices = (
        attention_super.proj_indices
        if attention_super.proj_indices is not None
        else slice(0, attention.proj.weight.data.size()[-1])
    )
    slices = tuple(slice(0, s) for s in attention.proj.bias.data.size())
    attention.proj.bias.data = base_attention.proj.bias.data[slices]

    attention.proj.weight.data = base_attention.proj.weight.data[
        0 : attention.proj.weight.data.size()[0], :
    ][:, proj_indices]
    return attention


@pytest.mark.parametrize("attention_config", attention_configs.keys())
def test_attention(attention_config):
    config = attention_configs[attention_config]["config"]
    if config.sliding_window_size is not None:
        config.sliding_window_layer_stride = (
            1
            if (
                config.sliding_window_layer_placing is None
                or config.sliding_window_layer_placing == "all"
            )
            else 2
        )

    config.fix_head_size = attention_configs[attention_config]["fix_head_size"]
    if not config.fix_head_size:
        config.head_size = 32
    config.max_seq_len = 512
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)

    seq_len = config.max_seq_len
    cos, sin = build_rope_cache(seq_len, n_elem=config.rope_n_elem)
    cos = cos[:seq_len].unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0)
    input = torch.rand(8, seq_len, config.n_embd)
    mask = build_mask_cache(seq_len)

    attention = init_attention(config)
    out_large = attention(input, mask=mask, cos=cos, sin=sin)

    # check shape of super network attention
    assert out_large.shape == (8, seq_len, config.n_embd)
    lit_attention = init_lit_attention(config)
    out_lit_large = lit_attention(input, mask=mask, cos=cos, sin=sin)
    if not config.fix_head_size:
        sub_network_head_size = config.head_size // 2
    else:
        sub_network_head_size = config.head_size
    if config.n_query_groups == 1:
        sub_network_query_groups = 1
        sub_network_n_head = config.n_head // 4
    elif config.n_query_groups == config.n_head:
        sub_network_n_head = config.n_head // 4
        sub_network_query_groups = sub_network_n_head
    else:
        sub_network_query_groups = config.n_query_groups // 2
        sub_network_n_head = config.n_head // 2
    attention.set_sub_network(
        sub_network_n_embd=config.n_embd // 2,
        sub_network_n_head=sub_network_n_head,
        sub_network_query_groups=sub_network_query_groups,
        sub_network_head_size=sub_network_head_size,
    )
    cos, sin = build_rope_cache(
        seq_len, n_elem=int(config.rotary_percentage * sub_network_head_size)
    )
    cos = cos[:seq_len].unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0)
    out_small = attention(input[:, :, : config.n_embd // 2], mask=mask, cos=cos, sin=sin)

    # check shape of sub-network attention
    assert out_small.shape == (8, seq_len, config.n_embd // 2)

    # check that our custom model produces the same output as LitGPT
    assert torch.all(out_lit_large == out_large)
    config.n_embd = attention.sub_network_n_embd
    if config.n_query_groups == config.n_head:
        config.n_head = attention.sub_network_n_head
    else:
        config.n_head = (
            attention.sub_network_n_head // config.n_query_groups
        ) * attention.sub_network_query_groups
    config.n_query_groups = attention.sub_network_query_groups
    config.head_size = attention.sub_network_head_size
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)
    lit_attention_small = init_lit_small_attention(config, lit_attention, attention)

    out_lit_small = lit_attention_small(
        input[:, :, : config.n_embd], mask=mask, cos=cos, sin=sin
    )
    # check that our sub-networks the same output as equally sized LitGPT attention layer
    assert torch.all(out_lit_small == out_small)
