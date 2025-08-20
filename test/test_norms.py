from __future__ import annotations

import torch
from litgpt.model import RMSNorm

from whittle.modules.layernorm import LayerNorm as LayerNormSuper
from whittle.modules.rmsnorm import RMSNorm as RMSNormSuper

class TestRMSNorm:
    def test_rmsnorm_shapes(self):
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
        rmsnorm.set_sub_network(sub_network_in_features=0, sampled_ln_indices=list(range(50)))
        out = rmsnorm(input_features_large[:, :50])
        assert out.shape == (8, 50)

    def test_rmsnorm_equivalence(self):
        input_features_large = torch.rand(8, 64)
        input_features_small = torch.rand(8, 32)
        rmsnorm = RMSNormSuper(in_features=64, add_unit_offset=True)
        rmsnorm.reset_super_network()
        rmsnorm.weight.data = torch.randn_like(rmsnorm.weight.data)
        rmsnorm.set_sub_network(sub_network_in_features=32)
        out_small = rmsnorm(input_features_small)
        rmsnorm.set_sub_network(sub_network_in_features=64)
        out_large = rmsnorm(input_features_large)

        small_layer = RMSNorm(32, add_unit_offset=True)
        small_layer.weight.data = rmsnorm.weight.data[:32]
        out_small_layer = small_layer(input_features_small)

        new_layer = RMSNorm(64, add_unit_offset=True)
        new_layer.weight.data = rmsnorm.weight.data[:64]
        out_new_layer = new_layer(input_features_large)

        assert torch.all(out_small == out_small_layer)
        assert torch.all(out_large == out_new_layer)

        rmsnorm = RMSNormSuper(in_features=64, add_unit_offset=True)
        rmsnorm.reset_super_network()
        rmsnorm.weight.data = torch.arange(64, dtype=torch.float32)
        rmsnorm.set_sub_network(sub_network_in_features=0, sampled_ln_indices=list(range(0, 64, 2)))
        out_super_net = rmsnorm(input_features_small[:, :32])

        new_layer = RMSNorm(32, add_unit_offset=True)
        new_layer.weight.data = torch.arange(0, 64, 2, dtype=torch.float32)
        out_new_layer = new_layer(input_features_small[:, :32])

        assert torch.all(out_super_net == out_new_layer)

class TestLayerNorm:
    def test_layernorm_shapes(self):
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
        layernorm.set_sub_network(sub_network_in_features=4, sampled_ln_indices=[0, 1, 62, 63])
        out = layernorm(input_features_large[:, :4])
        assert out.shape == (8, 4)


    def test_layernorm_equivalence(self):
        input_features_large = torch.rand(8, 64)
        input_features_small = torch.rand(8, 32)
        layernorm = LayerNormSuper(in_features=64)
        layernorm.reset_super_network()
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

        layernorm_super = LayerNormSuper(in_features=64)
        layernorm_super.weight.data = torch.arange(64, dtype=torch.float32)
        layernorm_super.reset_super_network()
        layernorm_super.set_sub_network(4, [0, 32, 33, 63])
        out_super = layernorm_super(input_features_large[:, :4])

        new_layer = torch.nn.LayerNorm(4)
        new_layer.weight.data = torch.tensor([0, 32, 33, 63.0])
        new_layer_out = new_layer(input_features_large[:, :4])

        assert torch.all(out_super == new_layer_out)