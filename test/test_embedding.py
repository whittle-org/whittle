from __future__ import annotations

import torch

from whittle.modules.embedding import Embedding


def test_embedding_shapes():
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


def test_embedding_shapes_indices_sampling():
    input_features = torch.randint(low=1, high=64, size=(4, 8))
    emb = Embedding(64, 32)

    out = emb(input_features)
    assert out.shape == (4, 8, 32)
    emb.set_sub_network(
        0, [0, 2, 3]
    )  # indices take precedence over sub_network_embedding_dim
    out = emb(input_features)
    assert out.shape == (4, 8, 3)
    emb.set_sub_network(32)
    out = emb(input_features)
    assert out.shape == (4, 8, 32)


def test_embedding_equivalence():
    input_features = torch.randint(low=1, high=64, size=(4, 8))
    emb = Embedding(64, 32)
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


def test_embedding_equivalence_indices_sampling():
    input_features = torch.randint(low=1, high=6, size=(4, 8))
    emb = Embedding(6, 12)
    emb.weight.data = torch.arange(6)[:, None] + torch.arange(12) * 3.14

    emb.set_sub_network(2, [0, 11])
    out_small = emb(input_features)

    small_layer = torch.nn.Embedding(6, 2)
    small_layer.weight.data = torch.arange(6)[:, None] + torch.tensor([0, 11]) * 3.14
    out_small_layer = small_layer(input_features)

    assert torch.all(out_small == out_small_layer)
