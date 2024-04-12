import torch

from transformers import AutoConfig

from nas_fine_tuning.sampling import SmallSearchSpace, FullSearchSpace, MediumSearchSpace, LayerSearchSpace

config = AutoConfig.from_pretrained('bert-base-cased')


def test_small():
    ss = SmallSearchSpace(config)
    hm, nm = ss(config)
    num_layers = hm[:, 0].sum()
    num_heads = hm[0, :].sum()
    num_units = nm[0, :].sum()

    assert hm.shape[0] == config.num_hidden_layers
    assert hm.shape[1] == config.num_attention_heads

    assert nm.shape[0] == config.num_hidden_layers
    assert nm.shape[1] == config.intermediate_size

    assert num_units <= config.intermediate_size
    assert num_heads <= config.num_attention_heads
    assert num_layers <= config.num_hidden_layers


def test_medium():
    ss = MediumSearchSpace(config)
    hm, nm = ss(config)
    num_layers = hm.shape[0]
    num_heads = hm.sum(dim=1)
    num_units = nm.sum(dim=1)

    assert hm.shape[0] == config.num_hidden_layers
    assert hm.shape[1] == config.num_attention_heads

    assert nm.shape[0] == config.num_hidden_layers
    assert nm.shape[1] == config.intermediate_size

    for i in range(num_layers):
        assert hm[i, int(num_heads[i]):].sum() == 0
        assert nm[i, int(num_units[i]):].sum() == 0


def test_large():
    ss = FullSearchSpace(config)
    hm, nm = ss(config)

    assert hm.shape[0] == config.num_hidden_layers
    assert hm.shape[1] == config.num_attention_heads

    assert nm.shape[0] == config.num_hidden_layers
    assert nm.shape[1] == config.intermediate_size
