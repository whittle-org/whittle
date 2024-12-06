from __future__ import annotations


from litgpt.model import GPT, Config
from whittle.baselines.pruner import NMPruner


def test_pythia():
    config = Config(
        name="pythia-14m",
        block_size=6,
        sliding_window_size=3,
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
    )
    config.fix_head_size = True
    pythia_model = GPT(config)
    pruner = NMPruner()
    sparsity = pruner(pythia_model, prune_n=2, prune_m=4, prune_methode="magnitude")
    assert abs(sparsity - 0.5) <= 0.1


def test_gamma():
    config_gemma = Config.from_name(
        "gemma-2-9b",
        block_size=6,
        sliding_window_size=3,
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
    )
    config_gemma.fix_head_size = True
    gamma_model = GPT(config_gemma)
    pruner = NMPruner()
    sparsity = pruner(gamma_model, prune_n=2, prune_m=4, prune_methode="magnitude")
    print(f"Sparsity: {sparsity:.6f}")
    assert abs(sparsity - 0.5) <= 0.1


def test_llama_3_8B():
    config_llama = Config.from_name(
        "Llama-3-8B",
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
        padded_vocab_size=10000,
    )
    config_llama.fix_head_size = True
    llama_model = GPT(config_llama)
    pruner = NMPruner()
    sparsity = pruner(llama_model, prune_n=2, prune_m=4, prune_methode="magnitude")
    print(f"Sparsity: {sparsity:.6f}")
    assert abs(sparsity - 0.5) <= 0.1


def test_llama_3_2_1B():
    config_llama = Config.from_name(
        "Llama-3.2-1B",
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
        padded_vocab_size=10000,
    )
    config_llama.fix_head_size = True
    llama_model = GPT(config_llama)
    pruner = NMPruner()
    sparsity = pruner(llama_model, prune_n=2, prune_m=4, prune_methode="magnitude")
    assert abs(sparsity - 0.5) <= 0.1
