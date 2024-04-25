import torch

from lobotomy.modules import Linear


def test_linear():

    input_features = torch.rand(8, 64)
    l = Linear(in_features=64, out_features=32)


    out = l(input_features)
    assert out.shape == (8, 32)
    l.set_sub_network(64, 16)
    out = l(input_features)
    assert out.shape == (8, 16)

    l.reset_super_network()
    out = l(input_features)
    assert out.shape == (8, 32)

    l.weight.data = torch.ones_like(l.weight.data)
    l.bias.data = torch.ones_like(l.bias.data)
    l.set_sub_network(64, 16)
    out = l(input_features)

    small_layer = torch.nn.Linear(64, 16)

    small_layer.weight.data = torch.ones_like(small_layer.weight.data)
    small_layer.bias.data = torch.ones_like(small_layer.bias.data)

    out_small_layer = small_layer(input_features)

    assert torch.all(out == out_small_layer)

def test_comparison_masking():
    pass
    # import numpy as np
    # a = torch.randn([32, 1024, 128])  # B, T, C
    # model = GptMLPWeightSlicing(max(choices["embed_dim"]), max(choices["mlp_ratio"]))
    # model.fc.weight.data = torch.ones_like(model.fc.weight.data)
    # model.proj.weight.data = torch.ones_like(model.proj.weight.data)
    # sample_embed_dim = np.random.choice(choices["embed_dim"])
    # sample_mlp_ratio = np.random.choice(choices["mlp_ratio"])
    # model.set_sample_config(sample_embed_dim, sample_mlp_ratio)
    # print(model(a).shape)
    # out_slicing = model(a)