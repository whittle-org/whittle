from lobotomy.models.litgpt.super_layers.rmsnorm_super import RMSNormSuper
from lobotomy.models.litgpt.super_layers.layernorm_super import LayerNormSuper
from litgpt_utils.base_model import RMSNorm
import torch

def test_rmsnorm():
    input_features_large = torch.rand(8, 64)
    input_features_small = torch.rand(8, 32)
    rmsnorm = RMSNormSuper(super_embed_dim=64, add_unit_offset=True)
    rmsnorm.set_sample_config(sample_embed_dim=64)
    out = rmsnorm(input_features_large)
    assert out.shape == (8, 64)
    rmsnorm.set_sample_config(sample_embed_dim=32)
    out = rmsnorm(input_features_small)
    assert out.shape == (8, 32)
    rmsnorm.set_sample_config(sample_embed_dim=64)
    out = rmsnorm(input_features_large)
    assert out.shape == (8, 64)

    rmsnorm.weight.data = torch.ones_like(rmsnorm.weight.data)
    rmsnorm.set_sample_config(sample_embed_dim=32)
    out_small = rmsnorm(input_features_small)
    rmsnorm.set_sample_config(sample_embed_dim=64)
    out_large = rmsnorm(input_features_large)

    small_layer = RMSNorm(32, add_unit_offset=True)
    small_layer.weight.data = torch.ones_like(small_layer.weight.data)
    out_small_layer = small_layer(input_features_small)

    large_layer = RMSNorm(64, add_unit_offset=True)
    large_layer.weight.data = torch.ones_like(large_layer.weight.data)
    out_large_layer = large_layer(input_features_large)


    assert torch.all(out_small == out_small_layer)
    assert torch.all(out_large == out_large_layer)


def test_layernorm():
    input_features_large = torch.rand(8, 64)
    input_features_small = torch.rand(8, 32)
    layernorm = LayerNormSuper(super_embed_dim=64)
    layernorm.set_sample_config(sample_embed_dim=64)
    out = layernorm(input_features_large)
    assert out.shape == (8, 64)
    layernorm.set_sample_config(sample_embed_dim=32)
    out = layernorm(input_features_small)
    assert out.shape == (8, 32)
    layernorm.set_sample_config(sample_embed_dim=64)
    out = layernorm(input_features_large)
    assert out.shape == (8, 64)

    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
    layernorm.bias.data = torch.ones_like(layernorm.bias.data)
    layernorm.set_sample_config(sample_embed_dim=32)
    out_small = layernorm(input_features_small)
    layernorm.set_sample_config(sample_embed_dim=64)
    out_large = layernorm(input_features_large)

    small_layer = torch.nn.LayerNorm(32)
    small_layer.weight.data = torch.ones_like(small_layer.weight.data)
    small_layer.bias.data = torch.ones_like(small_layer.bias.data)
    out_small_layer = small_layer(input_features_small)


    large_layer = torch.nn.LayerNorm(64)
    large_layer.weight.data = torch.ones_like(large_layer.weight.data)
    large_layer.bias.data = torch.ones_like(large_layer.bias.data)
    out_large_layer = large_layer(input_features_large)


    assert torch.all(out_small == out_small_layer)
    assert torch.all(out_large == out_large_layer)