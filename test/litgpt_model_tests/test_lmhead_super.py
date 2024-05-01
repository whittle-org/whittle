import torch
from lobotomy.models.litgpt.super_layers.lmhead_super import LMHeadSuper

output_dim = 1024
super_input_dim = 1024
sample_embed_dim_in = [512, 256]
bias = True
head_super = LMHeadSuper(super_input_dim, output_dim, bias=bias)
head_super.weight.data = torch.ones(output_dim, super_input_dim)
head_super.bias.data = torch.ones(output_dim)

for emb_in in sample_embed_dim_in:
        head_super.set_sample_config(emb_in)
        x = torch.ones(1, emb_in)
        y = head_super(x)
        print(torch.sum(y).item())
        if bias:
            assert torch.sum(y) == emb_in*output_dim + output_dim
        else:
            assert torch.sum(y) == emb_in*output_dim