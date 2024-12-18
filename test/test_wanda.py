from __future__ import annotations
import argparse

import pytest


from whittle.models.gpt import GPT, Config


from whittle.prunning.wanda_structured import find_layers
from whittle.prunning.pruner import NMPruner


@pytest.mark.parametrize(
    "model_info",
    [
        {
            "config_name": "pythia-14m",
        },
        {
            "config_name": "gemma-2-9b",
        },
        {
            "config_name": "Llama-3-8B",
        },
        {
            "config_name": "Llama-3.2-1B",
        },
    ],
)
def test_wanda_model_pruning(model_info, mock_tokenizer, compute_sparsity_ratio):
    args = argparse.Namespace(nsamples=32, seed=9001, batch_size=128)

    config = Config.from_name(
        model_info["config_name"],
        block_size=6,
        sliding_window_size=3,
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
    )
    config.fix_head_size = True
    config.model_type = "gpt"
    config.tie_embeddings = False
    config.use_cache = True

    model = GPT(config)

    pruner = NMPruner(args)
    pruner(
        model,
        prune_n=2,
        prune_m=4,
        dev="cpu",
        prune_methode="wanda",
        tokenizer=mock_tokenizer,
    )

    for layer in model.transformer.h:
        subset = find_layers(layer)
        for name in subset:
            sparsity = compute_sparsity_ratio(subset[name])
            assert abs(sparsity - 0.5) <= 0.1
