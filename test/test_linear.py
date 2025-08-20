from __future__ import annotations

import torch

from whittle.modules.linear import Linear


def test_linear_shapes():
    input_features = torch.rand(8, 64)
    linear = Linear(64, 32, bias=True)
    linear.reset_super_network()
    out = linear(input_features)
    assert out.shape == (8, 32)
    linear.set_sub_network(sub_network_in_features=64, sub_network_out_features=16)
    out = linear(input_features)
    assert out.shape == (8, 16)
    linear.set_sub_network(sub_network_in_features=64, sub_network_out_features=32)
    out = linear(input_features)
    assert out.shape == (8, 32)

def test_linear_shapes_indices_sampling():
    input_features = torch.rand(8, 4)
    linear = Linear(64, 32, bias=True)
    linear.set_sub_network(
        sub_network_in_features=4,
        sub_network_out_features=7,
        sampled_in_indices=[0, 31, 32, 63],
    )
    out = linear(input_features)
    assert out.shape == (8, 7)

    linear.set_sub_network(
        sub_network_in_features=2, # indices take precedence over sub_network_in_features
        sub_network_out_features=2, # indices take precedence over sub_network_out_features
        sampled_in_indices=[0, 31, 32, 63],
        sampled_out_indices=[0, 1, 11, 12, 13, 30, 31],
    )
    out = linear(input_features)
    assert out.shape == (8, 7)

def test_linear_equivalence():
    input_features = torch.rand(8, 64)
    input_small = torch.rand(8, 16)
    linear = Linear(64, 32, bias=True)
    linear.weight.data = torch.randn_like(linear.weight.data)
    linear.bias.data = torch.randn_like(linear.bias.data)

    linear.set_sub_network(sub_network_in_features=64, sub_network_out_features=16)
    out_small = linear(input_features)
    linear.set_sub_network(sub_network_in_features=64, sub_network_out_features=32)
    out_large = linear(input_features)
    linear.set_sub_network(sub_network_in_features=16, sub_network_out_features=32)
    out_small_large = linear(input_small)

    small_layer = torch.nn.Linear(64, 16, bias=True)
    small_layer.weight.data = linear.weight.data[:16, :]
    small_layer.bias.data = linear.bias.data[:16]
    out_small_layer = small_layer(input_features)

    large_layer = torch.nn.Linear(64, 32)
    large_layer.weight.data = linear.weight.data
    large_layer.bias.data = linear.bias.data
    out_large_layer = large_layer(input_features)

    small_large_layer = torch.nn.Linear(16, 32)
    small_large_layer.weight.data = linear.weight.data[:32, :16]
    small_large_layer.bias.data = linear.bias.data[:32]
    out_small_large_layer = small_large_layer(input_small)

    assert torch.all(out_small == out_small_layer)
    assert torch.all(out_large == out_large_layer)
    assert torch.all(out_small_large == out_small_large_layer)

def test_linear_equivalence_sample_in_indices():
    supernet_in_features, supernet_out_features = 8, 5
    linear = Linear(supernet_in_features, supernet_out_features)
    linear.weight.data = torch.arange(supernet_out_features)[:, None] + torch.arange(supernet_in_features)*3.14
    linear.bias.data = torch.arange(supernet_out_features, dtype=torch.float32)

    new_n_in_features = 3
    input_features = torch.rand(16, new_n_in_features)
    sampled_in_indices = [0, 2, 4]

    linear.set_sub_network(
        sub_network_in_features=new_n_in_features,
        sub_network_out_features=supernet_out_features,
        sampled_in_indices=sampled_in_indices,
    )

    subnet_out = linear(input_features)

    new_linear = Linear(new_n_in_features, supernet_out_features)
    new_linear.weight.data = torch.arange(supernet_out_features)[:, None] + torch.tensor(sampled_in_indices)*3.14
    new_linear.bias.data = torch.arange(supernet_out_features, dtype=torch.float32)

    new_layer_out = new_linear(input_features)
    assert subnet_out.shape == new_layer_out.shape
    assert torch.all(subnet_out == new_layer_out)

def test_linear_equivalence_sample_out_indices():
    supernet_in_features, supernet_out_features = 8, 5
    linear = Linear(supernet_in_features, supernet_out_features)
    linear.weight.data = torch.arange(supernet_out_features)[:, None] + torch.arange(supernet_in_features)*3.14
    linear.bias.data = torch.arange(supernet_out_features, dtype=torch.float32)

    new_n_out_features = 3
    input_features = torch.rand(16, supernet_in_features)
    sampled_out_indices = [0, 2, 4]

    linear.set_sub_network(
        sub_network_in_features=supernet_in_features,
        sub_network_out_features=new_n_out_features,
        sampled_out_indices=sampled_out_indices,
    )

    subnet_out = linear(input_features)

    new_linear = Linear(supernet_in_features, new_n_out_features)
    new_linear.weight.data = torch.tensor(sampled_out_indices)[:, None] + torch.arange(supernet_in_features)*3.14
    new_linear.bias.data = torch.tensor(sampled_out_indices, dtype=torch.float32)

    new_layer_out = new_linear(input_features)
    assert subnet_out.shape == new_layer_out.shape
    assert torch.all(subnet_out == new_layer_out)
