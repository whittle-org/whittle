import torch
from lobotomy.models.litgpt.super_layers.linear_super import SuperLinear

super_embed_dim = 1024
sample_embed_dim_in = [1024, 512]
sample_embed_dim_out = [512, 256]
bias = True
linear_super = SuperLinear(super_embed_dim, super_embed_dim, bias=bias)
linear_super.weight.data = torch.ones(super_embed_dim, super_embed_dim)
linear_super.bias.data = torch.ones(super_embed_dim)

for emb_in in sample_embed_dim_in:
  for emb_out in sample_embed_dim_out:
        linear_super.set_sample_config(emb_in, emb_out)
        x = torch.ones(1, emb_in)
        y = linear_super(x)
        print(torch.sum(y).item())
        if bias:
            assert torch.sum(y) == emb_in*emb_out + emb_out
        else:
            assert torch.sum(y) == emb_in*emb_out

