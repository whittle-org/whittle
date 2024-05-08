import torch
from lobotomy.models.litgpt.super_layers.linear_super import SuperLinear

def test_linear():

    input_features = torch.rand(8, 64)
    l = SuperLinear(super_dim_in=64, super_dim_out=32)

    out = l(input_features)
    assert out.shape == (8, 32)
    l.set_sample_config(sample_dim_in=64, sample_dim_out=16)
    out = l(input_features)
    assert out.shape == (8, 16)

    l.set_sample_config(sample_dim_in=64, sample_dim_out=32)
    out = l(input_features)
    assert out.shape == (8, 32)

    input_small = torch.rand(8, 16)
    l.weight.data = torch.ones_like(l.weight.data)
    l.bias.data = torch.ones_like(l.bias.data)
    l.set_sample_config(sample_dim_in=64, sample_dim_out=16)
    out_small = l(input_features)
    l.set_sample_config(sample_dim_in=64, sample_dim_out=32)
    out_large = l(input_features)
    l.set_sample_config(sample_dim_in=16, sample_dim_out=32)
    out_small_large = l(input_small)

    

    small_layer = torch.nn.Linear(64, 16)

    small_layer.weight.data = torch.ones_like(small_layer.weight.data)
    small_layer.bias.data = torch.ones_like(small_layer.bias.data)
    out_small_layer = small_layer(input_features)

    large_layer = torch.nn.Linear(64, 32)
    large_layer.weight.data = torch.ones_like(large_layer.weight.data)
    large_layer.bias.data = torch.ones_like(large_layer.bias.data)
    out_large_layer = large_layer(input_features)

    small_large_layer = torch.nn.Linear(16, 32)
    small_large_layer.weight.data = torch.ones_like(small_large_layer.weight.data)
    small_large_layer.bias.data = torch.ones_like(small_large_layer.bias.data)
    out_small_large_layer = small_large_layer(input_small)



    assert torch.all(out_small == out_small_layer)
    assert torch.all(out_large == out_large_layer)
    assert torch.all(out_small_large == out_small_large_layer)


