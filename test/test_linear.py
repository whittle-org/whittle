import torch
from lobotomy.modules.linear import Linear


def test_linear():
    input_features = torch.rand(8, 64)
    linear = Linear(64, 32, bias=True)
    linear.reset_super_network()
    out = linear(input_features)
    assert out.shape == (8, 32)
    linear.set_sub_network(
        sub_network_in_features=64, sub_network_out_features=16
    )
    out = linear(input_features)
    assert out.shape == (8, 16)
    linear.set_sub_network(
        sub_network_in_features=64, sub_network_out_features=32
    )
    out = linear(input_features)
    assert out.shape == (8, 32)

    input_small = torch.rand(8, 16)
    linear.weight.data = torch.ones_like(linear.weight.data)
    linear.bias.data = torch.ones_like(linear.bias.data)
    linear.set_sub_network(
        sub_network_in_features=64, sub_network_out_features=16
    )
    out_small = linear(input_features)
    linear.set_sub_network(
        sub_network_in_features=64, sub_network_out_features=32
    )
    out_large = linear(input_features)
    linear.set_sub_network(
        sub_network_in_features=16, sub_network_out_features=32
    )
    out_small_large = linear(input_small)

    small_layer = torch.nn.Linear(64, 16, bias=True)

    small_layer.weight.data = torch.ones_like(small_layer.weight.data)
    small_layer.bias.data = torch.ones_like(small_layer.bias.data)
    out_small_layer = small_layer(input_features)

    large_layer = torch.nn.Linear(64, 32)
    large_layer.weight.data = torch.ones_like(large_layer.weight.data)
    large_layer.bias.data = torch.ones_like(large_layer.bias.data)
    out_large_layer = large_layer(input_features)

    small_large_layer = torch.nn.Linear(16, 32)
    small_large_layer.weight.data = torch.ones_like(
        small_large_layer.weight.data
    )
    small_large_layer.bias.data = torch.ones_like(
        small_large_layer.bias.data
    )
    out_small_large_layer = small_large_layer(input_small)

    assert torch.all(out_small == out_small_layer)
    assert torch.all(out_large == out_large_layer)
    assert torch.all(out_small_large == out_small_large_layer)
