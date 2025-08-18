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
            # only layer with idx 0, 2, 4, ... have sliding window attention
            sliding_window_indices=[1 if i % 2 == 0 else 0 for i in range(1)],
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
    attention = CausalSelfAttention(config, 0)
    torch.manual_seed(0)
    attention.qkv.weight.data = torch.randn_like(attention.qkv.weight.data)
    attention.qkv.bias.data = torch.randn_like(attention.qkv.bias.data)
    attention.proj.bias.data = torch.randn_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.randn_like(attention.proj.weight.data)
    return attention


def init_lit_attention(config):
    attention = LitCausalSelfAttention(config, 0)
    torch.manual_seed(0)
    attention.qkv.weight.data = torch.randn_like(attention.qkv.weight.data)
    attention.qkv.bias.data = torch.randn_like(attention.qkv.bias.data)
    attention.proj.bias.data = torch.randn_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.randn_like(attention.proj.weight.data)
    return attention


def init_lit_small_attention(config, base_attention, attention_super):
    attention = LitCausalSelfAttention(config, 0)
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
    config.n_head = attention.sub_network_n_head
    config.n_query_groups = attention.sub_network_query_groups
    config.head_size = attention.sub_network_head_size
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)
    lit_attention_small = init_lit_small_attention(config, lit_attention, attention)
    if attention.qkv_indices is not None:
        print(lit_attention_small.qkv.weight.data.size())
        print(
            attention.qkv.weight.data[attention.qkv_indices, :][
                :, 0 : config.n_embd
            ].size()
        )
        assert torch.all(
            lit_attention_small.qkv.weight.data
            == attention.qkv.weight.data[attention.qkv_indices, :][:, 0 : config.n_embd]
        )

    out_lit_small = lit_attention_small(
        input[:, :, : config.n_embd], mask=mask, cos=cos, sin=sin
    )
    # check that our sub-networks the same output as equally sized LitGPT attention layer
    assert torch.all(out_lit_small == out_small)


QKV_INDICES_TEST_CONFIGS = {
    "gqa_to_gqa": [
        (64, 12, 3, 2),  # supernet  n_embd, n_head, query_groups, head_size
        (64, 8, 2, 2),  # subnet    n_embd, n_head, query_groups, head_size
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # queries
            + [24, 25, 26, 27]  # keys
            + [
                30,
                31,
                32,
                33,
            ]  # values
        ),
    ],
    "gqa_to_gqa_2": [
        (64, 16, 4, 2),  # supernet  n_embd, n_head, query_groups, head_size
        (64, 6, 3, 1),  # subnet    n_embd, n_head, query_groups, head_size
        (
            [0, 2, 8, 10, 16, 18]  # queries
            + [32, 34, 36]  # keys
            + [40, 42, 44]  # values
        ),
    ],
    "gqa_to_gqa_slice_head": [
        (64, 12, 3, 2),  # supernet  n_embd, n_head, query_groups, head_size
        (64, 8, 2, 1),  # subnet    n_embd, n_head, query_groups, head_size
        (
            [0, 2, 4, 6, 8, 10, 12, 14]  # queries
            + [24, 26]  # keys
            + [30, 32]  # values
        ),
    ],
    "gqa_to_mqa": [
        (64, 12, 3, 2),  # supernet  n_embd, n_head, query_groups, head_size
        (64, 4, 1, 2),  # subnet    n_embd, n_head, query_groups, head_size
        (
            [0, 1, 2, 3, 4, 5, 6, 7]  # queries
            + [24, 25]  # keys
            + [30, 31]  # values
        ),
    ],
    "gqa_to_mqa_fewer_than_max_heads": [
        (64, 12, 3, 2),  # supernet  n_embd, n_head, query_groups, head_size
        (64, 3, 1, 2),  # subnet    n_embd, n_head, query_groups, head_size
        (
            [0, 1, 2, 3, 4, 5]  # queries
            + [24, 25]  # keys
            + [30, 31]  # values
        ),
    ],
    "gqa_to_mqa_slice_head_fewer_than_max_heads": [
        (64, 12, 3, 2),  # supernet  n_embd, n_head, query_groups, head_size
        (64, 3, 1, 1),  # subnet    n_embd, n_head, query_groups, head_size
        (
            [0, 2, 4]  # queries
            + [24]  # keys
            + [30]  # values
        ),
    ],
    "gqa_to_mha": [
        (64, 12, 3, 2),  # supernet  n_embd, n_head, query_groups, head_size
        (64, 2, 2, 2),  # subnet    n_embd, n_head, query_groups, head_size
        (
            [0, 1, 8, 9]  # queries
            + [24, 25, 26, 27]  # keys
            + [
                30,
                31,
                32,
                33,
            ]  # values
        ),
    ],
    "gqa_to_mha_slice_head": [
        (64, 12, 3, 2),  # supernet  n_embd, n_head, query_groups, head_size
        (64, 2, 2, 1),  # subnet    n_embd, n_head, query_groups, head_size
        (
            [0, 8]  # queries
            + [24, 26]  # keys
            + [30, 32]  # values
        ),
    ],
    "mha_to_mha": [
        (64, 4, 4, 2),  # supernet  n_embd, n_head, query_groups, head_size
        (64, 3, 3, 2),  # subnet    n_embd, n_head, query_groups, head_size
        (
            [0, 1, 2, 3, 4, 5]  # queries
            + [8, 9, 10, 11, 12, 13]  # keys
            + [16, 17, 18, 19, 20, 21]  # values
        ),
    ],
    "mha_to_mha_slice_head": [
        (64, 4, 4, 2),  # supernet  n_embd, n_head, query_groups, head_size
        (64, 3, 3, 1),  # subnet    n_embd, n_head, query_groups, head_size
        (
            [0, 2, 4]  # queries
            + [8, 10, 12]  # keys
            + [16, 18, 20]  # values
        ),
    ],
    "mqa_to_mqa": [
        (64, 4, 1, 2),  # supernet  n_embd, n_head, query_groups, head_size
        (64, 3, 1, 2),  # subnet    n_embd, n_head, query_groups, head_size
        (
            [0, 1, 2, 3, 4, 5]  # queries
            + [8, 9]  # keys
            + [10, 11]  # values
        ),
    ],
    "mqa_to_mqa_slice_head": [
        (64, 4, 1, 2),  # supernet  n_embd, n_head, query_groups, head_size
        (64, 3, 1, 1),  # subnet    n_embd, n_head, query_groups, head_size
        (
            [
                0,
                2,
                4,
            ]  # queries
            + [8]  # keys
            + [10]  # values
        ),
    ],
}


@pytest.mark.parametrize("config_key", QKV_INDICES_TEST_CONFIGS.keys())
def test_qkv_indices(config_key):
    supernet_config, subnet_config, correct_indices = QKV_INDICES_TEST_CONFIGS[config_key]
    n_embd, n_head, n_query_groups, head_size = supernet_config
    config = Config(
        n_embd=n_embd, n_head=n_head, n_query_groups=n_query_groups, head_size=head_size
    )
    attn = CausalSelfAttention(config, 2)

    attn.set_sub_network(*subnet_config)

    qkv_indices = attn.qkv_indices
    assert qkv_indices is not None

    _, sub_n_head, sub_n_query_groups, sub_head_size = subnet_config
    assert qkv_indices.shape == (
        (sub_n_head + 2 * sub_n_query_groups) * sub_head_size,
    )  # (n heads + 2*num_groups)*head_size

    correct_indices = torch.tensor(correct_indices)
    assert torch.all(qkv_indices == correct_indices)
