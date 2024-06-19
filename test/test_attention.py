import torch
from lobotomy.models.gpt.blocks import CausalSelfAttention
from litgpt.model import CausalSelfAttention as LitCausalSelfAttention

# from lobotomy.models.gpt.blocks.causal_self_attention import CausalSelfAttentionLit as LitCausalSelfAttention
from litgpt import Config
from litgpt.model import build_mask_cache, build_rope_cache
import pytest

attention_configs = {
    "mha_fix_head_size": {"config":Config(n_embd=64, n_head=16, n_query_groups=16, head_size=64), "fix_head_size": True},
    "gqa_fix_head_size": {"config":Config(n_embd=64, n_head=16, n_query_groups=2, head_size=64), "fix_head_size": True},
    "mqa_fix_head_size": {"config":Config(n_embd=64, n_head=16, n_query_groups=1, head_size=64), "fix_head_size": True},
    "mha_flexible_head_size": {"config":Config(n_embd=64, n_head=16, n_query_groups=16), "fix_head_size": False},
    "gqa_flexible_head_size": {"config":Config(n_embd=64, n_head=16, n_query_groups=2), "fix_head_size": False},
    "mqa_flexible_head_size": {"config":Config(n_embd=64, n_head=16, n_query_groups=1), "fix_head_size": False}
}
def init_attention(config):
    attention = CausalSelfAttention(config)
    attention.attn.weight.data = torch.ones_like(attention.attn.weight.data)
    attention.attn.bias.data = torch.ones_like(attention.attn.bias.data)
    attention.proj.bias.data = torch.ones_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.ones_like(attention.proj.weight.data)
    return attention

def init_lit_attention(config):
    attention = LitCausalSelfAttention(config)
    attention.attn.weight.data = torch.ones_like(attention.attn.weight.data)
    attention.attn.bias.data = torch.ones_like(attention.attn.bias.data)
    attention.proj.bias.data = torch.ones_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.ones_like(attention.proj.weight.data)
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
    input = torch.rand(8, seq_len, 64)
    mask = build_mask_cache(seq_len)

    attention = init_attention(config)
    attention.reset_super_network()
    out_large = attention(input, mask=mask, cos=cos, sin=sin)

    # check shape of super network attention
    assert out_large.shape == (8, seq_len, 64)

    attention.set_sub_network(sub_network_n_embd=config.n_embd // 2, sub_network_n_head=config.n_head // 4)
    cos, sin = build_rope_cache(seq_len, n_elem=int(config.rotary_percentage * attention.sub_network_head_size))
    out_small = attention(input[:, :, :32], mask=mask, cos=cos, sin=sin)

    # check shape of sub-network attention
    assert out_small.shape == (8, seq_len, 32)

    lit_attention = init_lit_attention(config)
    out_lit_large = lit_attention(input, mask=mask, cos=cos, sin=sin)

    # check that our custom model produces the same output as LitGPT
    assert torch.all(out_lit_large == out_large)

    config.n_embd = attention.sub_network_n_embd
    config.n_head = attention.sub_network_n_head
    config.n_query_groups = attention.sub_network_query_groups
    config.head_size = attention.sub_network_head_size
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)

    lit_attention_small = init_lit_attention(config)
    
    out_lit_small = lit_attention_small(input[:, :, :32], mask=mask, cos=cos, sin=sin)

    # check that our sub-networks the same output as equally sized LitGPT attention layer
    assert torch.all(out_lit_small == out_small)