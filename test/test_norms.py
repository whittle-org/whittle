from __future__ import annotations

import torch
from litgpt.model import RMSNorm

from whittle.modules.layernorm import LayerNorm as LayerNormSuper
from whittle.modules.rmsnorm import RMSNorm as RMSNormSuper


def test_rmsnorm():
    input_features_large = torch.rand(8, 64)
    input_features_small = torch.rand(8, 32)
    rmsnorm = RMSNormSuper(in_features=64, add_unit_offset=True)
    rmsnorm.reset_super_network()
    out = rmsnorm(input_features_large)
    assert out.shape == (8, 64)
    rmsnorm.set_sub_network(sub_network_in_features=32)
    out = rmsnorm(input_features_small)
    assert out.shape == (8, 32)
    rmsnorm.set_sub_network(sub_network_in_features=64)
    out = rmsnorm(input_features_large)
    assert out.shape == (8, 64)

    rmsnorm.weight.data = torch.randn_like(rmsnorm.weight.data)
    rmsnorm.set_sub_network(sub_network_in_features=32)
    out_small = rmsnorm(input_features_small)
    rmsnorm.set_sub_network(sub_network_in_features=64)
    out_large = rmsnorm(input_features_large)

    small_layer = RMSNorm(32, add_unit_offset=True)
    small_layer.weight.data = rmsnorm.weight.data[:32]
    out_small_layer = small_layer(input_features_small)

    large_layer = RMSNorm(64, add_unit_offset=True)
    large_layer.weight.data = rmsnorm.weight.data[:64]
    out_large_layer = large_layer(input_features_large)

    assert torch.all(out_small == out_small_layer)
    assert torch.all(out_large == out_large_layer)


def test_layernorm():
    input_features_large = torch.rand(8, 64)
    input_features_small = torch.rand(8, 32)
    layernorm = LayerNormSuper(in_features=64)
    layernorm.reset_super_network()
    out = layernorm(input_features_large)
    assert out.shape == (8, 64)
    layernorm.set_sub_network(sub_network_in_features=32)
    out = layernorm(input_features_small)
    assert out.shape == (8, 32)
    layernorm.set_sub_network(sub_network_in_features=64)
    out = layernorm(input_features_large)
    assert out.shape == (8, 64)

    layernorm.weight.data = torch.randn_like(layernorm.weight.data)
    layernorm.bias.data = torch.randn_like(layernorm.bias.data)
    layernorm.set_sub_network(sub_network_in_features=32)
    out_small = layernorm(input_features_small)
    layernorm.set_sub_network(sub_network_in_features=64)
    out_large = layernorm(input_features_large)

    small_layer = torch.nn.LayerNorm(32)
    small_layer.weight.data = layernorm.weight.data[:32]
    small_layer.bias.data = layernorm.bias.data[:32]
    out_small_layer = small_layer(input_features_small)

    large_layer = torch.nn.LayerNorm(64)
    large_layer.weight.data = layernorm.weight.data[:64]
    large_layer.bias.data = layernorm.bias.data[:64]
    out_large_layer = large_layer(input_features_large)

    assert torch.all(out_small == out_small_layer)
    assert torch.all(out_large == out_large_layer)
