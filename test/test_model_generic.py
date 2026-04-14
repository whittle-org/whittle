from __future__ import annotations

import pytest
import torch
from litgpt import Config
from litgpt.model import GPT as LitGPT

from whittle.models.gpt import GPT


def copy_weights(model_source, model_target):
    for (_, p1), (_, p2) in zip(
        model_source.named_parameters(), model_target.named_parameters()
    ):
        p1.data = p2.data


# One representative per model family, <=8B params, no MoEs.
# Extra kwargs override defaults that would otherwise break with n_embd=32
# (e.g. n_head must divide n_embd, sliding window needs block_size).
MODEL_CONFIGS = {
    # GptNeoxMLP + MHA + LayerNorm + parallel_residual
    "stablelm-base-alpha-3b": {},
    # LLaMAMLP + MHA + LayerNorm + parallel_residual
    "stablelm-zephyr-3b": {},
    # GptNeoxMLP + MHA
    "pythia-14m": {},
    # GptNeoxMLP + MQA + LayerNorm + parallel_residual (n_head=71 doesn't divide 32)
    "falcon-7b": {"n_head": 8, "n_query_groups": 1},
    # LLaMAMLP + GQA
    "Falcon3-1B-Base": {},
    # LLaMAMLP + MHA
    "open_llama_3b": {},
    # LLaMAMLP + MHA
    "Llama-2-7b-hf": {},
    # LLaMAMLP + GQA
    "Llama-3-8B": {},
    # LLaMAMLP + GQA (different head_size from Llama-3)
    "Llama-3.2-1B": {},
    # LLaMAMLP + MHA + LayerNorm (no parallel_residual)
    "OLMo-1B-hf": {},
    # LLaMAMLP + MHA + RMSNorm
    "OLMo-2-1124-7B": {},
    # GemmaMLP + MQA
    "Gemma-2b": {},
    # GemmaMLP + GQA + sliding window
    "Gemma-2-2b": {"block_size": 6, "sliding_window_size": 3},
    # GemmaMLP + MQA + sliding window
    "Gemma-3-1b-it": {"block_size": 6, "sliding_window_size": 3},
    # GptNeoxMLP + MHA + LayerNorm + bias + parallel_residual
    "phi-2": {},
    # LLaMAMLP + MHA
    "Phi-3-mini-4k-instruct": {},
    # LLaMAMLP + GQA (n_head=24 doesn't divide 32)
    "Phi-4-mini-instruct": {"n_head": 8, "n_query_groups": 4},
    # LLaMAMLP + GQA
    "Mistral-7B-v0.1": {},
    # LLaMAMLP + GQA
    "tiny-llama-1.1b": {},
    # LLaMAMLP + GQA
    "micro-llama-300M": {},
    # LLaMAMLP + GQA (n_head=14 doesn't divide 32)
    "Qwen2.5-0.5B": {"n_head": 8, "n_query_groups": 2},
    # LLaMAMLP + GQA
    "Qwen3-0.6B": {"n_head": 16},
    # LLaMAMLP + MHA
    "salamandra-2b": {},
    # LLaMAMLP + GQA (n_head=9 doesn't divide 32)
    "SmolLM2-135M": {"n_head": 16, "n_query_groups": 4},
}


@pytest.mark.parametrize("model_name", MODEL_CONFIGS.keys())
def test_model_output_matches_litgpt(model_name):
    config_kwargs = {
        "n_layer": 2,
        "n_embd": 32,
        "intermediate_size": 86,
        "padded_vocab_size": 10000,
        **MODEL_CONFIGS[model_name],
    }

    config = Config.from_name(model_name, **config_kwargs)
    config.fix_head_size = True

    lit_model = LitGPT(config)
    whittle_model = GPT(config)
    copy_weights(lit_model, whittle_model)

    seq_len = min(config.block_size, 6)
    tokens = [9856, 23, 491, 1536, 304, 1234][:seq_len]
    x = torch.tensor([tokens], dtype=torch.int32)

    whittle_out = whittle_model(x)
    lit_out = lit_model(x)
    assert torch.allclose(whittle_out, lit_out, atol=1e-3), (
        f"Output mismatch for {model_name}: "
        f"max diff = {(whittle_out - lit_out).abs().max().item()}"
    )
