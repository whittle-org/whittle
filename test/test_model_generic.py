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


# All litgpt models <10B params, no MoEs.
# Extra kwargs override defaults that would otherwise break with n_embd=32
# (e.g. n_head must divide n_embd, sliding_window_size for memory).
MODEL_CONFIGS = {
    # === Pythia family (GptNeoxMLP + MHA) ===
    "pythia-14m": {},
    "pythia-31m": {},
    "pythia-70m": {},
    "pythia-70m-deduped": {},
    "pythia-160m": {"n_head": 16},  # n_head=12 doesn't divide 32
    "pythia-160m-deduped": {"n_head": 16},
    "pythia-410m": {},
    "pythia-410m-deduped": {},
    "pythia-1b": {},
    "pythia-1b-deduped": {},
    "pythia-1.4b": {},
    "pythia-1.4b-deduped": {},
    "pythia-2.8b": {},
    "pythia-2.8b-deduped": {},
    "pythia-6.9b": {},
    "pythia-6.9b-deduped": {},
    # === StableLM family ===
    # GptNeoxMLP + MHA + LayerNorm + parallel_residual
    "stablelm-base-alpha-3b": {},
    "stablelm-tuned-alpha-3b": {},
    "stablelm-base-alpha-7b": {"n_head": 32, "n_query_groups": 32},  # n_head=48
    "stablelm-tuned-alpha-7b": {"n_head": 32, "n_query_groups": 32},  # n_head=48
    # LLaMAMLP + MHA + LayerNorm + parallel_residual
    "stablelm-zephyr-3b": {},
    "stablelm-3b-4e1t": {},
    "stable-code-3b": {},
    "stablecode-completion-alpha-3b": {},
    "stablecode-completion-alpha-3b-4k": {},
    "stablecode-instruct-alpha-3b": {},
    # === Falcon family ===
    # GptNeoxMLP + MQA + LayerNorm + parallel_residual
    "falcon-7b": {"n_head": 8, "n_query_groups": 1},  # n_head=71
    "falcon-7b-instruct": {"n_head": 8, "n_query_groups": 1},
    # LLaMAMLP + GQA
    "Falcon3-1B-Base": {},
    "Falcon3-1B-Instruct": {},
    "Falcon3-3B-Base": {"n_head": 8, "n_query_groups": 4},  # n_head=12
    "Falcon3-3B-Instruct": {"n_head": 8, "n_query_groups": 4},
    "Falcon3-7B-Base": {"n_head": 8, "n_query_groups": 4},  # n_head=12
    "Falcon3-7B-Instruct": {"n_head": 8, "n_query_groups": 4},
    # === OpenLLaMA family (LLaMAMLP + MHA) ===
    "open_llama_3b": {},
    "open_llama_7b": {},
    # === LLaMA 2 family (LLaMAMLP + MHA) ===
    "Llama-2-7b-hf": {},
    "Llama-2-7b-chat-hf": {},
    "Llama-2-7b-chat-hf-function-calling-v2": {},
    "LLaMA-2-7B-32K": {},
    "Platypus2-7B": {},
    # === CodeLlama family (LLaMAMLP + MHA) ===
    "CodeLlama-7b-hf": {},
    "CodeLlama-7b-Instruct-hf": {},
    "CodeLlama-7b-Python-hf": {},
    # === LLaMA 3 family (LLaMAMLP + GQA) ===
    "Llama-3-8B": {},
    "Llama-3-8B-Instruct": {},
    "Llama-3.1-8B": {},
    "Llama-3.1-8B-Instruct": {},
    "Llama-3.2-1B": {},
    "Llama-3.2-1B-Instruct": {},
    "Llama-3.2-3B": {"n_head": 16, "n_query_groups": 8},  # n_head=24
    "Llama-3.2-3B-Instruct": {"n_head": 16, "n_query_groups": 8},
    "R1-Distill-Llama-8B": {},
    # === TinyLlama / micro-llama ===
    "tiny-llama-1.1b": {},
    "tiny-llama-1.1b-chat": {},
    "micro-llama-300M": {},
    # === OLMo family ===
    # LLaMAMLP + MHA + LayerNorm
    "OLMo-1B-hf": {},
    "OLMo-7B-hf": {},
    "OLMo-7B-Instruct-hf": {},
    # LLaMAMLP + MHA + RMSNorm
    "OLMo-2-1124-7B": {},
    "OLMo-2-1124-7B-DPO": {},
    "OLMo-2-1124-7B-Instruct": {},
    "OLMo-2-1124-7B-SFT": {},
    # === Gemma family ===
    # GemmaMLP + MQA
    "Gemma-2b": {},
    "Gemma-2b-it": {},
    "Gemma-7b": {},
    "Gemma-7b-it": {},
    "CodeGemma-7b-it": {},
    # GemmaMLP + GQA + sliding window
    "Gemma-2-2b": {"sliding_window_size": 1024},
    "Gemma-2-2b-it": {"sliding_window_size": 1024},
    "Gemma-2-9b": {"sliding_window_size": 1024},
    "Gemma-2-9b-it": {"sliding_window_size": 1024},
    # GemmaMLP + MQA/GQA + sliding window
    "Gemma-3-1b-it": {"sliding_window_size": 512},
    "Gemma-3-4b-it": {"sliding_window_size": 1024},
    # === Phi family ===
    # GptNeoxMLP + MHA + LayerNorm + bias + parallel_residual
    "phi-1_5": {},
    "phi-2": {},
    # LLaMAMLP + MHA + sliding window
    "Phi-3-mini-4k-instruct": {"sliding_window_size": 1024},
    "Phi-3-mini-128k-instruct": {"sliding_window_size": 1024},
    "Phi-3.5-mini-instruct": {},
    # LLaMAMLP + GQA
    "Phi-4-mini-instruct": {
        "n_head": 8,
        "n_query_groups": 4,
        "sliding_window_size": 1024,
    },  # n_head=24
    "Phi-4-mini-reasoning": {
        "n_head": 8,
        "n_query_groups": 4,
        "sliding_window_size": 1024,
    },
    # === Mistral family (LLaMAMLP + GQA) ===
    "Mistral-7B-v0.1": {"sliding_window_size": 1024},
    "Mistral-7B-v0.2": {},
    "Mistral-7B-v0.3": {},
    "Mistral-7B-Instruct-v0.1": {"sliding_window_size": 1024},
    "Mistral-7B-Instruct-v0.2": {},
    "Mistral-7B-Instruct-v0.3": {},
    "Mathstral-7B-v0.1": {"sliding_window_size": 1024},
    # === Qwen 2.5 family (LLaMAMLP + GQA) ===
    "Qwen2.5-0.5B": {"n_head": 8, "n_query_groups": 2},  # n_head=14
    "Qwen2.5-0.5B-Instruct": {"n_head": 8, "n_query_groups": 2},
    "Qwen2.5-1.5B": {"n_head": 8, "n_query_groups": 2},  # n_head=12
    "Qwen2.5-1.5B-Instruct": {"n_head": 8, "n_query_groups": 2},
    "Qwen2.5-3B": {},
    "Qwen2.5-3B-Instruct": {},
    "Qwen2.5-7B": {"n_head": 16, "n_query_groups": 4},  # n_head=28
    "Qwen2.5-7B-Instruct": {"n_head": 16, "n_query_groups": 4},
    "Qwen2.5-7B-Instruct-1M": {"n_head": 16, "n_query_groups": 4},
    # Qwen 2.5 Coder
    "Qwen2.5-Coder-0.5B": {"n_head": 8, "n_query_groups": 2},  # n_head=14
    "Qwen2.5-Coder-0.5B-Instruct": {"n_head": 8, "n_query_groups": 2},
    "Qwen2.5-Coder-1.5B": {"n_head": 8, "n_query_groups": 2},  # n_head=12
    "Qwen2.5-Coder-1.5B-Instruct": {"n_head": 8, "n_query_groups": 2},
    "Qwen2.5-Coder-3B": {},
    "Qwen2.5-Coder-3B-Instruct": {},
    "Qwen2.5-Coder-7B": {"n_head": 16, "n_query_groups": 4},  # n_head=28
    "Qwen2.5-Coder-7B-Instruct": {"n_head": 16, "n_query_groups": 4},
    # Qwen 2.5 Math
    "Qwen2.5-Math-1.5B": {"n_head": 8, "n_query_groups": 2},  # n_head=12
    "Qwen2.5-Math-1.5B-Instruct": {"n_head": 8, "n_query_groups": 2},
    "Qwen2.5-Math-7B": {"n_head": 16, "n_query_groups": 4},  # n_head=28
    "Qwen2.5-Math-7B-Instruct": {"n_head": 16, "n_query_groups": 4},
    # === Qwen 3 family (LLaMAMLP + GQA) ===
    "Qwen3-0.6B": {},
    "Qwen3-0.6B-Base": {},
    "Qwen3-1.7B": {},
    "Qwen3-1.7B-Base": {},
    "Qwen3-4B": {},
    "Qwen3-4B-Base": {},
    "Qwen3-4B-Instruct-2507": {},
    "Qwen3-4B-Thinking-2507": {},
    "Qwen3-8B": {},
    "Qwen3-8B-Base": {},
    # === SmolLM2 family (LLaMAMLP + GQA) ===
    "SmolLM2-135M": {"n_head": 16, "n_query_groups": 4},  # n_head=9
    "SmolLM2-135M-Instruct": {"n_head": 16, "n_query_groups": 4},
    "SmolLM2-360M": {"n_head": 16, "n_query_groups": 4},  # n_head=15
    "SmolLM2-360M-Instruct": {"n_head": 16, "n_query_groups": 4},
    "SmolLM2-1.7B": {},
    "SmolLM2-1.7B-Instruct": {},
    # === Salamandra family (LLaMAMLP + MHA/GQA) ===
    "salamandra-2b": {},
    "salamandra-2b-instruct": {},
    "salamandra-7b": {},
    "salamandra-7b-instruct": {},
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

    torch.manual_seed(0)
    x = torch.randint(0, config.padded_vocab_size, (1, 2048))

    whittle_out = whittle_model(x)
    lit_out = lit_model(x)
    assert torch.allclose(whittle_out, lit_out, atol=1e-3), (
        f"Output mismatch for {model_name}: "
        f"max diff = {(whittle_out - lit_out).abs().max().item()}"
    )
