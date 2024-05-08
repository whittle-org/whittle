import torch
from lobotomy.modules.rotary_embedding import RotaryEmbedding
from lobotomy.models.gpt.blocks import CausalSelfAttention
from litgpt.model import CausalSelfAttention as LitCausalSelfAttention
from litgpt import Config
from litgpt.model import build_mask_cache
def test_attention_mha():
    config = Config()
    config.n_embd = 64
    config.n_head = 8
    config.n_query_groups = 4
    config.head_size = 8
    config.fix_head_size = True
    config.max_seq_len = 512
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)
    input = torch.rand(8, 512, 64)
    rotary_emb = RotaryEmbedding(config, 512)
    mask = build_mask_cache(512)
    attention = CausalSelfAttention(config, rotary_emb)
    attention.attn.weight.data = torch.ones_like(attention.attn.weight.data)
    attention.attn.bias.data = torch.ones_like(attention.attn.bias.data)
    attention.proj.bias.data = torch.ones_like(attention.proj.bias.data)
    attention.proj.weight.data = torch.ones_like(attention.proj.weight.data)
    attention.reset_super_network()
    out_large = attention(input, mask)
    sin, cos = attention.reset_parameters()
    assert out_large.shape == (8, 512, 64)
    attention.set_sub_network(sub_network_n_embd=32, sub_network_n_head=4)
    out_small = attention(input[:,:,:32], mask)
    assert out_small.shape == (8, 512, 32)
    print(sin.shape)
    print(cos.shape)
    lit_attention = LitCausalSelfAttention(config)
    lit_attention.attn.weight.data = torch.ones_like(lit_attention.attn.weight.data)
    lit_attention.attn.bias.data = torch.ones_like(lit_attention.attn.bias.data)
    lit_attention.proj.bias.data = torch.ones_like(lit_attention.proj.bias.data)
    lit_attention.proj.weight.data = torch.ones_like(lit_attention.proj.weight.data)
    out_lit_large = lit_attention(input, mask=mask,cos=cos,sin=sin)
    assert torch.all(out_lit_large == out_large)

    config.n_embd = 32
    config.n_head = 4
    config.n_query_groups = 2

    lit_attention_small = LitCausalSelfAttention(config)
    lit_attention_small.attn.weight.data = torch.ones_like(lit_attention_small.attn.weight.data)
    lit_attention_small.attn.bias.data = torch.ones_like(lit_attention_small.attn.bias.data)
    lit_attention_small.proj.bias.data = torch.ones_like(lit_attention_small.proj.bias.data)
    lit_attention_small.proj.weight.data = torch.ones_like(lit_attention_small.proj.weight.data)
    out_lit_small = lit_attention_small(input[:,:,:32], mask=mask,cos=cos,sin=sin)
    assert torch.all(out_lit_small == out_small)



    