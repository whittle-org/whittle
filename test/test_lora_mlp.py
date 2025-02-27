from __future__ import annotations

import torch
from litgpt.model import (
    GemmaMLP as LitGemmaMLP,
    GptNeoxMLP as LitGptNeoxMLP,
    LLaMAMLP as LitLLaMAMLP,
)

from whittle.lora_model.config import LoRAConfig as Config
from whittle.lora_model.lora_mlps import (
    LoRAGemmaMLP as GemmaMLP,
    LoRAGptNeoxMLP as GptNeoxMLP,
    LoRALLaMAMLP as LLaMAMLP,
)


def test_GptNeoxMLP():
    config = Config()
    input = torch.rand(8, 64)
    # update config
    config.n_embd = 64
    config.intermediate_size = 64 * 4
    gpt_neox_mlp = GptNeoxMLP(config)
    # init weights and biases
    gpt_neox_mlp.fc.linear.weight.data = torch.randn_like(
        gpt_neox_mlp.fc.linear.weight.data
    )
    gpt_neox_mlp.fc.linear.bias.data = torch.randn_like(gpt_neox_mlp.fc.linear.bias.data)
    gpt_neox_mlp.proj.linear.weight.data = torch.randn_like(
        gpt_neox_mlp.proj.linear.weight.data
    )
    gpt_neox_mlp.proj.linear.bias.data = torch.randn_like(
        gpt_neox_mlp.proj.linear.bias.data
    )
    gpt_neox_mlp.reset_super_network()
    out_large = gpt_neox_mlp(input)
    assert out_large.shape == (8, 64)
    gpt_neox_mlp.set_sub_network(
        sub_network_n_embd=32, sub_network_intermediate_size=32 * 4
    )
    out_small = gpt_neox_mlp(input[:8, :32])
    assert out_small.shape == (8, 32)

    litgpt_neox_mlp_large = LitGptNeoxMLP(config)
    litgpt_neox_mlp_large.fc.weight.data = gpt_neox_mlp.fc.linear.weight.data
    litgpt_neox_mlp_large.fc.bias.data = gpt_neox_mlp.fc.linear.bias.data
    litgpt_neox_mlp_large.proj.weight.data = gpt_neox_mlp.proj.linear.weight.data
    litgpt_neox_mlp_large.proj.bias.data = gpt_neox_mlp.proj.linear.bias.data
    out_large_lit = litgpt_neox_mlp_large(input)
    config.n_embd = 32
    config.intermediate_size = 32 * 4
    litgpt_neox_mlp_small = LitGptNeoxMLP(config)
    litgpt_neox_mlp_small.fc.weight.data = gpt_neox_mlp.fc.linear.weight.data[
        : config.intermediate_size, : config.n_embd
    ]
    litgpt_neox_mlp_small.fc.bias.data = gpt_neox_mlp.fc.linear.bias.data[
        : config.intermediate_size
    ]
    litgpt_neox_mlp_small.proj.weight.data = gpt_neox_mlp.proj.linear.weight.data[
        : config.n_embd, : config.intermediate_size
    ]
    litgpt_neox_mlp_small.proj.bias.data = gpt_neox_mlp.proj.linear.bias.data[
        : config.n_embd
    ]
    out_small_lit = litgpt_neox_mlp_small(input[:8, :32])
    assert torch.all(out_small == out_small_lit)
    assert torch.all(out_large == out_large_lit)


def test_LLaMAMLP():
    config = Config()
    input = torch.rand(8, 64)
    # update config
    config.n_embd = 64
    config.intermediate_size = 64 * 4
    llama_mlp = LLaMAMLP(config)
    # init weights and biases
    llama_mlp.fc_1.linear.weight.data = torch.randn_like(
        llama_mlp.fc_1.linear.weight.data
    )
    llama_mlp.fc_1.linear.bias.data = torch.randn_like(llama_mlp.fc_1.linear.bias.data)
    llama_mlp.fc_2.linear.weight.data = torch.randn_like(
        llama_mlp.fc_2.linear.weight.data
    )
    llama_mlp.fc_2.linear.bias.data = torch.randn_like(llama_mlp.fc_2.linear.bias.data)
    llama_mlp.proj.linear.weight.data = torch.randn_like(
        llama_mlp.proj.linear.weight.data
    )
    llama_mlp.proj.linear.bias.data = torch.randn_like(llama_mlp.proj.linear.bias.data)
    llama_mlp.reset_super_network()
    out_large = llama_mlp(input)
    assert out_large.shape == (8, 64)
    llama_mlp.set_sub_network(sub_network_n_embd=32, sub_network_intermediate_size=32 * 4)
    out_small = llama_mlp(input[:8, :32])
    assert out_small.shape == (8, 32)

    litllama_mlp_large = LitLLaMAMLP(config)
    litllama_mlp_large.fc_1.weight.data = llama_mlp.fc_1.linear.weight.data
    litllama_mlp_large.fc_1.bias.data = llama_mlp.fc_1.linear.bias.data
    litllama_mlp_large.fc_2.weight.data = llama_mlp.fc_2.linear.weight.data
    litllama_mlp_large.fc_2.bias.data = llama_mlp.fc_2.linear.bias.data
    litllama_mlp_large.proj.weight.data = llama_mlp.proj.linear.weight.data
    litllama_mlp_large.proj.bias.data = llama_mlp.proj.linear.bias.data
    out_large_lit = litllama_mlp_large(input)
    config.n_embd = 32
    config.intermediate_size = 32 * 4
    litllama_mlp_small = LitLLaMAMLP(config)
    litllama_mlp_small.fc_1.weight.data = llama_mlp.fc_1.linear.weight.data[
        : config.intermediate_size, : config.n_embd
    ]
    litllama_mlp_small.fc_1.bias.data = llama_mlp.fc_1.linear.bias.data[
        : config.intermediate_size
    ]
    litllama_mlp_small.fc_2.weight.data = llama_mlp.fc_2.linear.weight.data[
        : config.intermediate_size, : config.n_embd
    ]
    litllama_mlp_small.fc_2.bias.data = llama_mlp.fc_2.linear.bias.data[
        : config.intermediate_size
    ]
    litllama_mlp_small.proj.weight.data = llama_mlp.proj.linear.weight.data[
        : config.n_embd, : config.intermediate_size
    ]
    litllama_mlp_small.proj.bias.data = llama_mlp.proj.linear.bias.data[: config.n_embd]
    out_small_lit = litllama_mlp_small(input[:8, :32])
    assert torch.all(out_small == out_small_lit)
    assert torch.all(out_large == out_large_lit)


def test_GemmaMLP():
    config = Config()
    input = torch.rand(8, 64)
    # update config
    config.n_embd = 64
    config.intermediate_size = 64 * 4
    gemma_mlp = GemmaMLP(config)
    # init weights and biases
    gemma_mlp.fc_1.linear.weight.data = torch.randn_like(
        gemma_mlp.fc_1.linear.weight.data
    )
    gemma_mlp.fc_1.linear.bias.data = torch.randn_like(gemma_mlp.fc_1.linear.bias.data)
    gemma_mlp.fc_2.linear.weight.data = torch.randn_like(
        gemma_mlp.fc_2.linear.weight.data
    )
    gemma_mlp.fc_2.linear.bias.data = torch.randn_like(gemma_mlp.fc_2.linear.bias.data)
    gemma_mlp.proj.linear.weight.data = torch.randn_like(
        gemma_mlp.proj.linear.weight.data
    )
    gemma_mlp.proj.linear.bias.data = torch.randn_like(gemma_mlp.proj.linear.bias.data)
    gemma_mlp.reset_super_network()
    out_large = gemma_mlp(input)
    assert out_large.shape == (8, 64)
    gemma_mlp.set_sub_network(sub_network_n_embd=32, sub_network_intermediate_size=32 * 4)
    out_small = gemma_mlp(input[:8, :32])
    assert out_small.shape == (8, 32)

    litgemma_mlp_large = LitGemmaMLP(config)
    litgemma_mlp_large.fc_1.weight.data = gemma_mlp.fc_1.linear.weight.data
    litgemma_mlp_large.fc_1.bias.data = gemma_mlp.fc_1.linear.bias.data
    litgemma_mlp_large.fc_2.weight.data = gemma_mlp.fc_2.linear.weight.data
    litgemma_mlp_large.fc_2.bias.data = gemma_mlp.fc_2.linear.bias.data
    litgemma_mlp_large.proj.weight.data = gemma_mlp.proj.linear.weight.data
    litgemma_mlp_large.proj.bias.data = gemma_mlp.proj.linear.bias.data
    out_large_lit = litgemma_mlp_large(input)
    config.n_embd = 32
    config.intermediate_size = 32 * 4
    litgemma_mlp_small = LitGemmaMLP(config)
    litgemma_mlp_small.fc_1.weight.data = gemma_mlp.fc_1.linear.weight.data[
        : config.intermediate_size, : config.n_embd
    ]
    litgemma_mlp_small.fc_1.bias.data = gemma_mlp.fc_1.linear.bias.data[
        : config.intermediate_size
    ]
    litgemma_mlp_small.fc_2.weight.data = gemma_mlp.fc_2.linear.weight.data[
        : config.intermediate_size, : config.n_embd
    ]
    litgemma_mlp_small.fc_2.bias.data = gemma_mlp.fc_2.linear.bias.data[
        : config.intermediate_size
    ]
    litgemma_mlp_small.proj.weight.data = gemma_mlp.proj.linear.weight.data[
        : config.n_embd, : config.intermediate_size
    ]
    litgemma_mlp_small.proj.bias.data = gemma_mlp.proj.linear.bias.data[: config.n_embd]
    out_small_lit = litgemma_mlp_small(input[:8, :32])
    assert torch.all(out_small == out_small_lit)
    assert torch.all(out_large == out_large_lit)
