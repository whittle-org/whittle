from __future__ import annotations
import argparse


from litgpt import Config
from litgpt.model import GPT
from transformers import AutoTokenizer

from whittle.baselines.wanda_structured import find_layers
from whittle.baselines.pruner import NMPruner
from whittle.metrics.parameters import compute_sparsity_ratio


# def compute_sparsity_ratio(layer):
#     W = layer.weight.data
#     total_elements = W.numel()
#     zero_elements = torch.sum(W == 0).item()
#     sparsity_ratio = zero_elements / total_elements
#     return sparsity_ratio


def test_pythia():
    args = argparse.Namespace()
    args.nsamples = 32
    args.seed = 9001
    args.dataset = "sciq"
    args.batch_size = 128
    config = Config(
        name="pythia-14m",
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

    pythia_model = GPT(config)
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-14m", trust_remote_code=True, use_fast=True
    )
    pruner = NMPruner(args)
    sparsity = pruner(
        pythia_model, prune_n=2, prune_m=4, prune_methode="wanda", tokenizer=tokenizer
    )

    for layer in pythia_model.transformer.h:
        subset = find_layers(layer)

        for name in subset:
            sparsity = compute_sparsity_ratio(subset[name])
            assert abs(sparsity - 0.5) <= 0.1


def test_gamma():
    args = argparse.Namespace()
    args.nsamples = 32
    args.seed = 9001
    args.dataset = "sciq"
    args.batch_size = 128

    config_gemma = Config.from_name(
        "gemma-2-9b",
        block_size=6,
        sliding_window_size=3,
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
    )

    config_gemma.fix_head_size = True
    config_gemma.model_type = "gpt"
    config_gemma.tie_embeddings = False
    config_gemma.use_cache = True

    gamma_model = GPT(config_gemma)
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-9b", trust_remote_code=True, use_fast=True
    )

    pruner = NMPruner(args)
    sparsity = pruner(
        gamma_model, prune_n=2, prune_m=4, prune_methode="wanda", tokenizer=tokenizer
    )

    for layer in gamma_model.transformer.h:
        subset = find_layers(layer)

        for name in subset:
            sparsity = compute_sparsity_ratio(subset[name])
            assert abs(sparsity - 0.5) <= 0.1


def test_Llama_3_8b():
    args = argparse.Namespace()
    args.nsamples = 32
    args.seed = 9001
    args.dataset = "sciq"
    args.batch_size = 128

    config_gemma = Config.from_name(
        "Llama-3-8B",
        block_size=6,
        sliding_window_size=3,
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
    )

    config_gemma.fix_head_size = True
    config_gemma.fix_head_size = True
    config_gemma.model_type = "gpt"
    config_gemma.tie_embeddings = False
    config_gemma.use_cache = True
    gamma_model = GPT(config_gemma)

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", trust_remote_code=True, use_fast=True
    )

    pruner = NMPruner(args)
    sparsity = pruner(
        gamma_model, prune_n=2, prune_m=4, prune_methode="wanda", tokenizer=tokenizer
    )
    for layer in gamma_model.transformer.h:
        subset = find_layers(layer)
        for name in subset:
            sparsity = compute_sparsity_ratio(subset[name])
            assert abs(sparsity - 0.5) <= 0.1


def test_Llama3_2_1b():
    args = argparse.Namespace()
    args.nsamples = 32
    args.seed = 9001
    args.dataset = "sciq"
    args.batch_size = 128

    config_llama = Config.from_name(
        "Llama-3.2-1B",
        block_size=6,
        sliding_window_size=3,
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
    )
    config_llama.fix_head_size = True
    config_llama.model_type = "gpt"
    config_llama.tie_embeddings = False
    config_llama.use_cache = True
    gamma_model = GPT(config_llama)

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B", trust_remote_code=True, use_fast=True
    )

    pruner = NMPruner(args)
    sparsity = pruner(
        gamma_model, prune_n=2, prune_m=4, prune_methode="wanda", tokenizer=tokenizer
    )
    for layer in gamma_model.transformer.h:
        subset = find_layers(layer)
        for name in subset:
            sparsity = compute_sparsity_ratio(subset[name])
            assert abs(sparsity - 0.5) <= 0.1
