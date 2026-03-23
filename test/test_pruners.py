from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from whittle.models.gpt import GPT, Config
from whittle.pruning.pruners.magnitude import MagnitudePruner
from whittle.pruning.pruners.sparsegpt import SparseGPTPruner
from whittle.pruning.pruners.wanda import WandaPruner


@pytest.fixture(
    params=[
        "pythia-14m",
        "gemma-2-9b",
        "Llama-3-8B",
        "Llama-3.2-1B",
    ]
)
def pruner_model(request, accelerator_device):
    torch.manual_seed(0)
    config = Config.from_name(
        request.param,
        block_size=64,
        sliding_window_size=3,
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
    )
    config.fix_head_size = True
    config.model_type = "gpt"
    config.tie_embeddings = False
    config.use_cache = True
    return GPT(config).to(accelerator_device)


@pytest.fixture
def pruner_dataloader(accelerator_device):
    dataset = TensorDataset(
        torch.randint(0, 1000, size=(256, 64)),
        torch.randint(0, 1000, size=(256, 1)),
    )
    return DataLoader(dataset, batch_size=8)


# @pytest.mark.skip("Fix later")
def test_magnitude_pruning(pruner_model):
    pruner = MagnitudePruner()
    sparsity_ratio = pruner(pruner_model, prune_n=2, prune_m=4)
    assert abs(sparsity_ratio - 0.5) <= 0.4


def test_wanda_pruning(pruner_model, pruner_dataloader, accelerator_device):
    pruner = WandaPruner()
    sparsity_ratio = pruner(
        model=pruner_model,
        dataloader=pruner_dataloader,
        prune_n=2,
        prune_m=4,
        device=accelerator_device,
        nsamples=32,
    )
    assert abs(sparsity_ratio - 0.5) <= 0.4


def test_sparsegpt_pruning(pruner_model, pruner_dataloader, accelerator_device):
    pruner = SparseGPTPruner()
    sparsity_ratio = pruner(
        model=pruner_model,
        dataloader=pruner_dataloader,
        prune_n=2,
        prune_m=4,
        device=accelerator_device,
        nsamples=32,
    )
    assert abs(sparsity_ratio - 0.5) <= 0.4
