import torch

from lobotomy.modules.embedding import Embedding


def test_embedding():

    input_features = torch.randint(low=1, high=64, size=(4, 8))
    emb = Embedding(num_embeddings=64, embedding_dim=32)

    out = emb(input_features)
    assert out.shape == (4, 8, 32)
    emb.set_sub_network(16)
    out = emb(input_features)
    assert out.shape == (4, 8, 16)

    emb.reset_super_network()
    out = emb(input_features)
    assert out.shape == (4, 8, 32)

    emb.weight.data = torch.ones_like(emb.weight.data)
    emb.set_sub_network(16)
    out = emb(input_features)

    small_layer = torch.nn.Embedding(64, 16)

    small_layer.weight.data = torch.ones_like(small_layer.weight.data)

    out_small_layer = small_layer(input_features)

    assert torch.all(out == out_small_layer)


