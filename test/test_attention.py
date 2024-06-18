import torch
from lobotomy.models.gpt.blocks import CausalSelfAttention
from litgpt.model import CausalSelfAttention as LitCausalSelfAttention

# from lobotomy.models.gpt.blocks.causal_self_attention import CausalSelfAttentionLit as LitCausalSelfAttention
from litgpt import Config
from litgpt.model import build_mask_cache, build_rope_cache
import numpy as np


def test_attention_mha_fix_head_size():
    config = Config(n_embd=64, n_head=8, n_query_groups=8, head_size=64)
    config.fix_head_size = True
    config.max_seq_len = 512
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)

    seq_len = config.max_seq_len
    cos, sin = build_rope_cache(seq_len, n_elem=config.rope_n_elem)
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    input = torch.rand(8, seq_len, 64)
    mask = build_mask_cache(seq_len)

    attention = CausalSelfAttention(config)
    attention.attn.weight.data = torch.ones_like(attention.attn.weight.data)
    attention.attn.bias.data = torch.ones_like(attention.attn.bias.data)
    attention.proj.bias.data = torch.ones_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.ones_like(attention.proj.weight.data)
    attention.reset_super_network()
    out_large = attention(input, mask=mask, cos=cos, sin=sin)

    # check shape of super network attention
    assert out_large.shape == (8, seq_len, 64)

    attention.set_sub_network(sub_network_n_embd=32, sub_network_n_head=4)
    out_small = attention(input[:, :, :32], mask=mask, cos=cos, sin=sin)

    # check shape of sub-network attention
    assert out_small.shape == (8, seq_len, 32)

    lit_attention = LitCausalSelfAttention(config)
    lit_attention.attn.weight.data = torch.ones_like(lit_attention.attn.weight.data)
    lit_attention.attn.bias.data = torch.ones_like(lit_attention.attn.bias.data)
    lit_attention.proj.bias.data = torch.ones_like(lit_attention.proj.bias.data)
    lit_attention.proj.weight.data = torch.ones_like(lit_attention.proj.weight.data)
    out_lit_large = lit_attention(input, mask=mask, cos=cos, sin=sin)

    # check that our custom model produces the same output as LitGPT
    assert torch.all(out_lit_large == out_large)

    config.n_embd = 32
    config.n_head = 4
    config.n_query_groups = config.n_head

    lit_attention_small = LitCausalSelfAttention(config)
    lit_attention_small.attn.weight.data = torch.ones_like(
        lit_attention_small.attn.weight.data
    )
    lit_attention_small.attn.bias.data = torch.ones_like(
        lit_attention_small.attn.bias.data
    )
    lit_attention_small.proj.bias.data = torch.ones_like(
        lit_attention_small.proj.bias.data
    )
    lit_attention_small.proj.weight.data = torch.ones_like(
        lit_attention_small.proj.weight.data
    )
    out_lit_small = lit_attention_small(input[:, :, :32], mask=mask, cos=cos, sin=sin)

    # check that our sub-networks the same output as equally sized LitGPT attention layer
    assert torch.all(out_lit_small == out_small)


def test_gqa_attention_mha_fix_head_size():
    config = Config(n_embd=64, n_head=8, n_query_groups=2, head_size=64)
    config.fix_head_size = True
    config.max_seq_len = 512
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)

    seq_len = config.max_seq_len
    cos, sin = build_rope_cache(seq_len, n_elem=config.rope_n_elem)
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    input = torch.rand(8, seq_len, 64)
    mask = build_mask_cache(seq_len)

    attention = CausalSelfAttention(config)
    attention.attn.weight.data = torch.ones_like(attention.attn.weight.data)
    attention.attn.bias.data = torch.ones_like(attention.attn.bias.data)
    attention.proj.bias.data = torch.ones_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.ones_like(attention.proj.weight.data)
    attention.reset_super_network()
    out_large = attention(input, mask=mask, cos=cos, sin=sin)

    # check shape of super network attention
    assert out_large.shape == (8, seq_len, 64)

    attention.set_sub_network(sub_network_n_embd=32, sub_network_n_head=4)
    out_small = attention(input[:, :, :32], mask=mask, cos=cos, sin=sin)

    # check shape of sub-network attention
    assert out_small.shape == (8, seq_len, 32)

    lit_attention = LitCausalSelfAttention(config)
    lit_attention.attn.weight.data = torch.ones_like(lit_attention.attn.weight.data)
    lit_attention.attn.bias.data = torch.ones_like(lit_attention.attn.bias.data)
    lit_attention.proj.bias.data = torch.ones_like(lit_attention.proj.bias.data)
    lit_attention.proj.weight.data = torch.ones_like(lit_attention.proj.weight.data)
    out_lit_large = lit_attention(input, mask=mask, cos=cos, sin=sin)

    # check that our custom model produces the same output as LitGPT
    assert torch.all(out_lit_large == out_large)

    config.n_embd = 32
    config.n_head = 4
    config.n_query_groups = 2

    lit_attention_small = LitCausalSelfAttention(config)
    lit_attention_small.attn.weight.data = torch.ones_like(
        lit_attention_small.attn.weight.data
    )
    lit_attention_small.attn.bias.data = torch.ones_like(
        lit_attention_small.attn.bias.data
    )
    lit_attention_small.proj.bias.data = torch.ones_like(
        lit_attention_small.proj.bias.data
    )
    lit_attention_small.proj.weight.data = torch.ones_like(
        lit_attention_small.proj.weight.data
    )
    out_lit_small = lit_attention_small(input[:, :, :32], mask=mask, cos=cos, sin=sin)

    # check that our sub-networks the same output as equally sized LitGPT attention layer

    assert torch.all(out_lit_small == out_small)


def test_attention_mqa_fix_head_size():
    config = Config(n_embd=64, n_head=8, n_query_groups=1, head_size=64)
    config.fix_head_size = True
    config.max_seq_len = 512
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)

    seq_len = config.max_seq_len
    cos, sin = build_rope_cache(seq_len, n_elem=config.rope_n_elem)
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    input = torch.rand(8, seq_len, 64)
    mask = build_mask_cache(seq_len)

    attention = CausalSelfAttention(config)
    attention.attn.weight.data = torch.ones_like(attention.attn.weight.data)
    attention.attn.bias.data = torch.ones_like(attention.attn.bias.data)
    attention.proj.bias.data = torch.ones_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.ones_like(attention.proj.weight.data)
    attention.reset_super_network()
    out_large = attention(input, mask=mask, cos=cos, sin=sin)

    # check shape of super network attention
    assert out_large.shape == (8, seq_len, 64)

    attention.set_sub_network(sub_network_n_embd=32, sub_network_n_head=4)
    out_small = attention(input[:, :, :32], mask=mask, cos=cos, sin=sin)

    # check shape of sub-network attention
    assert out_small.shape == (8, seq_len, 32)

    lit_attention = LitCausalSelfAttention(config)
    lit_attention.attn.weight.data = torch.ones_like(lit_attention.attn.weight.data)
    lit_attention.attn.bias.data = torch.ones_like(lit_attention.attn.bias.data)
    lit_attention.proj.bias.data = torch.ones_like(lit_attention.proj.bias.data)
    lit_attention.proj.weight.data = torch.ones_like(lit_attention.proj.weight.data)
    out_lit_large = lit_attention(input, mask=mask, cos=cos, sin=sin)

    # check that our custom model produces the same output as LitGPT
    assert torch.all(out_lit_large == out_large)

    config.n_embd = 32
    config.n_head = 4
    config.n_query_groups = 1

    lit_attention_small = LitCausalSelfAttention(config)
    lit_attention_small.attn.weight.data = torch.ones_like(
        lit_attention_small.attn.weight.data
    )
    lit_attention_small.attn.bias.data = torch.ones_like(
        lit_attention_small.attn.bias.data
    )
    lit_attention_small.proj.bias.data = torch.ones_like(
        lit_attention_small.proj.bias.data
    )
    lit_attention_small.proj.weight.data = torch.ones_like(
        lit_attention_small.proj.weight.data
    )
    out_lit_small = lit_attention_small(input[:, :, :32], mask=mask, cos=cos, sin=sin)

    # check that our sub-networks the same output as equally sized LitGPT attention layer
    assert torch.all(out_lit_small == out_small)


def test_attention_mha_flexible_head_size():

    config = Config(n_embd=16, n_head=8, n_query_groups=8)
    config.fix_head_size = False
    config.max_seq_len = 2
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)

    seq_len = config.max_seq_len
    cos, sin = build_rope_cache(seq_len, n_elem=config.rope_n_elem)
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    input = torch.rand(1, seq_len, 16)
    mask = build_mask_cache(seq_len)
    attention = CausalSelfAttention(config)
    attention.attn.weight.data = torch.ones_like(attention.attn.weight.data)
    attention.attn.bias.data = torch.ones_like(attention.attn.bias.data)
    attention.proj.bias.data = torch.ones_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.ones_like(attention.proj.weight.data)

    attention.set_sub_network(sub_network_n_embd=4, sub_network_n_head=2)
    cos_small, sin_small = build_rope_cache(
        seq_len, n_elem=int(config.rotary_percentage * 2)
    )
    cos_small = cos_small[:seq_len]
    sin_small = sin_small[:seq_len]
    out_small = attention(input[:, :, :4], mask=mask, cos=cos_small, sin=sin_small)
    attention.set_sub_network(sub_network_n_embd=16, sub_network_n_head=8)
    out_large = attention(input, mask=mask, cos=cos, sin=sin)

    # check a reduced head size
    assert out_small.shape == (1, seq_len, 4)
    # assert out_large.shape == (1, seq_len, 16)

    lit_attention_large = LitCausalSelfAttention(config)
    lit_attention_large.attn.weight.data = torch.ones_like(
        lit_attention_large.attn.weight.data
    )
    lit_attention_large.attn.bias.data = torch.ones_like(
        lit_attention_large.attn.bias.data
    )
    lit_attention_large.proj.bias.data = torch.ones_like(
        lit_attention_large.proj.bias.data
    )
    lit_attention_large.proj.weight.data = torch.ones_like(
        lit_attention_large.proj.weight.data
    )
    out_lit_large = lit_attention_large(input, mask=mask, cos=cos, sin=sin)

    # check that our custom model produces the same output as LitGPT
    assert torch.all(out_lit_large == out_large)

    config.n_embd = 4
    config.n_head = 2
    config.head_size = 2
    config.n_query_groups = 2
    config.rope_n_elem = int(config.rotary_percentage * 2)
    cos, sin = cos_small, sin_small

    lit_attention_small = LitCausalSelfAttention(config)
    lit_attention_small.attn.weight.data = torch.ones_like(
        lit_attention_small.attn.weight.data
    )
    lit_attention_small.attn.bias.data = torch.ones_like(
        lit_attention_small.attn.bias.data
    )
    lit_attention_small.proj.bias.data = torch.ones_like(
        lit_attention_small.proj.bias.data
    )
    lit_attention_small.proj.weight.data = torch.ones_like(
        lit_attention_small.proj.weight.data
    )
    out_lit_small = lit_attention_small(input[:, :, :4], mask=mask, cos=cos, sin=sin)

    # check that our sub-networks the same output as equally sized LitGPT attention layer
    assert torch.all(out_lit_small == out_small)


def test_attention_gqa_flexible_head_size():

    config = Config(n_embd=64, n_head=8, n_query_groups=4)
    config.fix_head_size = False
    config.max_seq_len = 512
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)

    seq_len = config.max_seq_len
    cos, sin = build_rope_cache(seq_len, n_elem=config.rope_n_elem)
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    input = torch.rand(8, seq_len, 64)
    mask = build_mask_cache(seq_len)
    attention = CausalSelfAttention(config)
    attention.attn.weight.data = torch.ones_like(attention.attn.weight.data)
    attention.attn.bias.data = torch.ones_like(attention.attn.bias.data)
    attention.proj.bias.data = torch.ones_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.ones_like(attention.proj.weight.data)
    attention.set_sub_network(sub_network_n_embd=32, sub_network_n_head=2)
    cos_small, sin_small = build_rope_cache(
        seq_len, n_elem=int(config.rotary_percentage * 16)
    )
    cos_small = cos_small[:seq_len]
    sin_small = sin_small[:seq_len]
    out_small = attention(input[:, :, :32], mask=mask, cos=cos_small, sin=sin_small)
    attention.set_sub_network(sub_network_n_embd=64, sub_network_n_head=8)
    out_large = attention(input, mask=mask, cos=cos, sin=sin)

    # check a reduced head size
    assert out_small.shape == (8, seq_len, 32)
    assert out_large.shape == (8, seq_len, 64)

    lit_attention_large = LitCausalSelfAttention(config)
    lit_attention_large.attn.weight.data = torch.ones_like(
        lit_attention_large.attn.weight.data
    )
    lit_attention_large.attn.bias.data = torch.ones_like(
        lit_attention_large.attn.bias.data
    )
    lit_attention_large.proj.bias.data = torch.ones_like(
        lit_attention_large.proj.bias.data
    )
    lit_attention_large.proj.weight.data = torch.ones_like(
        lit_attention_large.proj.weight.data
    )
    out_lit_large = lit_attention_large(input, mask=mask, cos=cos, sin=sin)

    # check that our custom model produces the same output as LitGPT
    assert torch.all(out_lit_large == out_large)

    config.n_embd = 32
    config.n_head = 2
    config.n_query_groups = 1
    config.head_size = config.n_embd // config.n_head
    config.rope_n_elem = int(
        config.rotary_percentage * (config.n_embd // config.n_head)
    )
    cos, sin = build_rope_cache(seq_len, n_elem=config.rope_n_elem)
    cos = cos[:seq_len]
    sin = sin[:seq_len]

    lit_attention_small = LitCausalSelfAttention(config)
    lit_attention_small.attn.weight.data = torch.ones_like(
        lit_attention_small.attn.weight.data
    )
    lit_attention_small.attn.bias.data = torch.ones_like(
        lit_attention_small.attn.bias.data
    )
    lit_attention_small.proj.bias.data = torch.ones_like(
        lit_attention_small.proj.bias.data
    )
    lit_attention_small.proj.weight.data = torch.ones_like(
        lit_attention_small.proj.weight.data
    )
    out_lit_small = lit_attention_small(input[:, :, :32], mask=mask, cos=cos, sin=sin)
    print(torch.sum(out_lit_small - out_small))
    # check that our sub-networks the same output as equally sized LitGPT attention layer
    assert torch.all(out_lit_small == out_small)


def test_attention_mqa_flexible_head_size():
    config = Config(n_embd=64, n_head=8, n_query_groups=1)
    config.fix_head_size = False
    config.max_seq_len = 512
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)

    seq_len = config.max_seq_len
    cos, sin = build_rope_cache(seq_len, n_elem=config.rope_n_elem)
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    input = torch.rand(8, seq_len, 64)
    mask = build_mask_cache(seq_len)
    attention = CausalSelfAttention(config)
    attention.attn.weight.data = torch.ones_like(attention.attn.weight.data)
    attention.attn.bias.data = torch.ones_like(attention.attn.bias.data)
    attention.proj.bias.data = torch.ones_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.ones_like(attention.proj.weight.data)

    attention.set_sub_network(sub_network_n_embd=32, sub_network_n_head=2)
    cos_small, sin_small = build_rope_cache(
        seq_len, n_elem=int(config.rotary_percentage * 16)
    )
    cos_small = cos_small[:seq_len]
    sin_small = sin_small[:seq_len]
    out_small = attention(input[:, :, :32], mask=mask, cos=cos_small, sin=sin_small)
    attention.set_sub_network(sub_network_n_embd=64, sub_network_n_head=8)
    out_large = attention(input, mask=mask, cos=cos, sin=sin)

    # check a reduced head size
    assert out_small.shape == (8, seq_len, 32)
    assert out_large.shape == (8, seq_len, 64)

    lit_attention_large = LitCausalSelfAttention(config)
    lit_attention_large.attn.weight.data = torch.ones_like(
        lit_attention_large.attn.weight.data
    )
    lit_attention_large.attn.bias.data = torch.ones_like(
        lit_attention_large.attn.bias.data
    )
    lit_attention_large.proj.bias.data = torch.ones_like(
        lit_attention_large.proj.bias.data
    )
    lit_attention_large.proj.weight.data = torch.ones_like(
        lit_attention_large.proj.weight.data
    )
    out_lit_large = lit_attention_large(input, mask=mask, cos=cos, sin=sin)

    # check that our custom model produces the same output as LitGPT
    assert torch.all(out_lit_large == out_large)

    config.n_embd = 32
    config.n_head = 2
    config.head_size = 16
    config.n_query_groups = 1
    config.rope_n_elem = int(
        config.rotary_percentage * (config.n_embd // config.n_head)
    )
    cos, sin = build_rope_cache(seq_len, n_elem=config.rope_n_elem)
    cos = cos[:seq_len]
    sin = sin[:seq_len]

    lit_attention_small = LitCausalSelfAttention(config)
    lit_attention_small.attn.weight.data = torch.ones_like(
        lit_attention_small.attn.weight.data
    )
    lit_attention_small.attn.bias.data = torch.ones_like(
        lit_attention_small.attn.bias.data
    )
    lit_attention_small.proj.bias.data = torch.ones_like(
        lit_attention_small.proj.bias.data
    )
    lit_attention_small.proj.weight.data = torch.ones_like(
        lit_attention_small.proj.weight.data
    )
    out_lit_small = lit_attention_small(input[:, :, :32], mask=mask, cos=cos, sin=sin)

    # check that our sub-networks the same output as equally sized LitGPT attention layer
    assert torch.all(out_lit_small == out_small)
