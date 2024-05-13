import torch
from lobotomy.modules.linear import Linear

def test_linear():

    input_features = torch.rand(8, 64)
    l = Linear(64, 10, bias=True)
    l.reset_super_network()
    out = l(input_features)
    assert out.shape == (8, 10)
    input_features = torch.rand(8, 16)
    l.set_sub_network(sub_network_in_features=16, sub_network_out_features=10)
    out = l(input_features)
    assert out.shape == (8, 10)

    l.set_sub_network(sub_network_in_features=64, sub_network_out_features=10)
    input_features = torch.rand(8, 64)
    out = l(input_features)
    assert out.shape == (8, 10)


    l.weight.data = torch.ones_like(l.weight.data)
    l.bias.data = torch.ones_like(l.bias.data)
    l.set_sub_network(sub_network_in_features=16, sub_network_out_features=10)
    input_features_small = torch.rand(8, 16)
    out_small = l(input_features_small)
    input_features_large = torch.rand(8, 64)
    l.set_sub_network(sub_network_in_features=64, sub_network_out_features=10)
    out_large = l(input_features_large)
    

    small_layer = torch.nn.Linear(16, 10, bias=True)

    small_layer.weight.data = torch.ones_like(small_layer.weight.data)
    small_layer.bias.data = torch.ones_like(small_layer.bias.data)
    out_small_layer = small_layer(input_features_small)

    large_layer = torch.nn.Linear(64, 10)
    large_layer.weight.data = torch.ones_like(large_layer.weight.data)
    large_layer.bias.data = torch.ones_like(large_layer.bias.data)
    out_large_layer = large_layer(input_features_large)

    assert torch.all(out_small == out_small_layer)
    assert torch.all(out_large == out_large_layer)