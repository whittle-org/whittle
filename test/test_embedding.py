from __future__ import annotations

import torch

from whittle.modules.embedding import Embedding


def test_embedding():
    input_features = torch.randint(low=1, high=64, size=(4, 8))
    emb = Embedding(64, 32)

    out = emb(input_features)
    assert out.shape == (4, 8, 32)
    emb.set_sub_network(16)
    out = emb(input_features)
    assert out.shape == (4, 8, 16)
    emb.set_sub_network(32)
    out = emb(input_features)
    assert out.shape == (4, 8, 32)

    emb.weight.data = torch.randn_like(emb.weight.data)
    emb.set_sub_network(16)
    out_small = emb(input_features)
    emb.set_sub_network(32)
    out_large = emb(input_features)

    small_layer = torch.nn.Embedding(64, 16)

    small_layer.weight.data = emb.weight.data[:, :16]

    out_small_layer = small_layer(input_features)

    large_layer = torch.nn.Embedding(64, 32)
    large_layer.weight.data = emb.weight.data[:, :32]
    out_large_layer = large_layer(input_features)

    assert torch.all(out_small == out_small_layer)
    assert torch.all(out_large == out_large_layer)
