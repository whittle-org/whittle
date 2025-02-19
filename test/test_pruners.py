from __future__ import annotations

import pytest
import torch

from whittle.models.gpt import GPT, Config
from whittle.pruning.pruners.magnitude import MagnitudePruner
from whittle.pruning.pruners.sparsegpt import SparseGPTPruner
from whittle.pruning.pruners.wanda import WandaPruner
from whittle.pruning.utils.data import get_c4_dataloader


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
def test_model_pruning(model_info, mock_tokenizer):
    torch.manual_seed(0)
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
    pruner_wanda = WandaPruner()
    pruner_sparsegpt = SparseGPTPruner()
    pruner_magnitude = MagnitudePruner()

    dataloader, _ = get_c4_dataloader(
        n_samples=32,
        seed=9001,
        seqlen=model.max_seq_length,
        tokenizer=mock_tokenizer,
    )

    sparsity_ratio_magnitude = pruner_magnitude(
        model,
        prune_n=2,
        prune_m=4,
    )
    sparsity_ratio_wanda = pruner_wanda(
        model=model,
        dataloader=dataloader,
        prune_n=2,
        prune_m=4,
        dev="cpu",
        nsamples=32,
    )

    sparsity_ratio_sparsegpt = pruner_sparsegpt(
        model=model,
        dataloader=dataloader,
        prune_n=2,
        prune_m=4,
        dev="cpu",
        nsamples=32,
    )

    assert abs(sparsity_ratio_magnitude - 0.5) <= 0.1
    assert abs(sparsity_ratio_wanda - 0.5) <= 0.1
    assert abs(sparsity_ratio_sparsegpt - 0.5) <= 0.1
