from __future__ import annotations

import torch
from litgpt import Config as LitConfig
from whittle.lora.config import LoRAConfig as Config
from litgpt.model import GPT as LitGPT

from whittle.lora.lora_gpt import GPT


def test_gpt():
    torch.manual_seed(0)
    config = Config()
    config.padded_vocab_size = 128
    config.n_embd = 64
    config.intermediate_size = 64 * 4
    config.n_head = 8
    config.n_query_groups = 4
    config.head_size = 8
    config.n_layer = 2
    config.block_size = 128
    config.norm_class_name = "RMSNorm"
    config.mlp_class_name = "LLaMAMLP"
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)
    config.norm_eps = 1e-5
    config.lm_head_bias = True
    config.fix_head_size = False
    litconfig = LitConfig()
    litconfig.padded_vocab_size = 128
    litconfig.n_embd = 64
    litconfig.intermediate_size = 64 * 4
    litconfig.n_head = 8
    litconfig.n_query_groups = 4
    litconfig.head_size = 8
    litconfig.n_layer = 2
    litconfig.block_size = 128
    litconfig.norm_class_name = "RMSNorm"
    litconfig.mlp_class_name = "LLaMAMLP"
    litconfig.rope_n_elem = int(litconfig.rotary_percentage * litconfig.head_size)
    litconfig.norm_eps = 1e-5
    litconfig.lm_head_bias = True
    litconfig.fix_head_size = False
    gpt = GPT(config)
    gpt.transformer.wte.embedding.weight.data = torch.randn_like(
        gpt.transformer.wte.embedding.weight.data
    )
    gpt.lm_head.linear.weight.data = torch.randn_like(gpt.lm_head.linear.weight.data)
    gpt.lm_head.linear.bias.data = torch.randn_like(gpt.lm_head.linear.bias.data)
    gpt.transformer.ln_f.weight.data = torch.randn_like(
        gpt.transformer.ln_f.weight.data
    )

    for block in gpt.transformer.h:
        block.attn.attn.linear.linear.weight.data = torch.randn_like(
            block.attn.attn.linear.linear.weight.data
        )
        block.attn.attn.linear.linear.bias.data = torch.randn_like(
            block.attn.attn.linear.linear.bias.data
        )
        block.attn.proj.linear.bias.data = torch.randn_like(
            block.attn.proj.linear.bias.data
        )
        block.attn.proj.linear.weight.data = torch.randn_like(
            block.attn.proj.linear.weight.data
        )
        block.mlp.fc_1.linear.weight.data = torch.randn_like(
            block.mlp.fc_1.linear.weight.data
        )
        block.mlp.fc_1.linear.bias.data = torch.randn_like(
            block.mlp.fc_1.linear.bias.data
        )
        block.mlp.fc_2.linear.weight.data = torch.randn_like(
            block.mlp.fc_2.linear.weight.data
        )
        block.mlp.fc_2.linear.bias.data = torch.randn_like(
            block.mlp.fc_2.linear.bias.data
        )
        block.mlp.proj.linear.weight.data = torch.randn_like(
            block.mlp.proj.linear.weight.data
        )
        block.mlp.proj.linear.bias.data = torch.randn_like(
            block.mlp.proj.linear.bias.data
        )
        block.norm_1.weight.data = torch.randn_like(block.norm_1.weight.data)
        block.norm_2.weight.data = torch.randn_like(block.norm_2.weight.data)

    gpt.reset_super_network()
    input = torch.randint(0, 64, (1, 64))
    out_large = gpt(input)
    assert out_large.shape == (1, 64, 128)

    lit_gpt = LitGPT(litconfig)
    lit_gpt.lm_head.weight.data = gpt.lm_head.linear.weight.data
    lit_gpt.lm_head.bias.data = gpt.lm_head.linear.bias.data
    lit_gpt.transformer.wte.weight.data = gpt.transformer.wte.embedding.weight.data
    lit_gpt.transformer.ln_f.weight.data = gpt.transformer.ln_f.weight.data
    for i, block in enumerate(lit_gpt.transformer.h):
        block_orig = gpt.transformer.h[i]
        block.attn.attn.weight.data = block_orig.attn.attn.linear.linear.weight.data
        block.attn.attn.bias.data = block_orig.attn.attn.linear.linear.bias.data
        block.attn.proj.bias.data = block_orig.attn.proj.linear.bias.data
        block.attn.proj.weight.data = block_orig.attn.proj.linear.weight.data
        block.mlp.fc_1.weight.data = block_orig.mlp.fc_1.linear.weight.data
        block.mlp.fc_1.bias.data = block_orig.mlp.fc_1.linear.bias.data
        block.mlp.fc_2.weight.data = block_orig.mlp.fc_2.linear.weight.data
        block.mlp.fc_2.bias.data = block_orig.mlp.fc_2.linear.bias.data
        block.mlp.proj.weight.data = block_orig.mlp.proj.linear.weight.data
        block.mlp.proj.bias.data = block_orig.mlp.proj.linear.bias.data
        block.norm_1.weight.data = block_orig.norm_1.weight.data
        block.norm_2.weight.data = block_orig.norm_2.weight.data

    out_lit_large = lit_gpt(input)
    assert torch.allclose(out_lit_large, out_large, atol=1e-3)
    gpt.set_sub_network(
        sub_network_n_embd=32,
        sub_network_intermediate_size=32 * 4,
        sub_network_num_heads=4,
        sub_network_n_layers=1,
        sub_network_query_groups=2,
        sub_network_head_size=config.head_size,
    )
    out_small = gpt(input)
    assert out_small.shape == (1, 64, 128)
    litconfig.n_embd = 32
    litconfig.n_head = 2
    litconfig.n_query_groups = 2
    litconfig.intermediate_size = 32 * 4
    litconfig.n_layer = 1
    print(config)
    lit_gpt_small = LitGPT(litconfig)
    lit_gpt_small.lm_head.weight.data = gpt.lm_head.linear.weight.data[
        : gpt.lm_head.sub_network_out_features, : gpt.lm_head.sub_network_in_features
    ]
    lit_gpt_small.lm_head.bias.data = gpt.lm_head.linear.bias.data[:]
    lit_gpt_small.transformer.wte.weight.data = (
        gpt.transformer.wte.embedding.weight.data[
            :, : gpt.transformer.wte.sub_network_embedding_dim
        ]
    )
    lit_gpt_small.transformer.ln_f.weight.data = gpt.transformer.ln_f.weight.data[
        : gpt.transformer.ln_f.sub_network_in_features
    ]

    for i, block in enumerate(lit_gpt_small.transformer.h):
        block_orig = gpt.transformer.h[i]
        if block_orig.attn.qkv_indices is not None:
            block.attn.attn.weight.data = (
                block_orig.attn.attn.linear.linear.weight.data[
                    block_orig.attn.qkv_indices,
                    : block_orig.attn.attn.sub_network_in_features,
                ]
            )
            block.attn.attn.bias.data = block_orig.attn.attn.linear.linear.bias.data[
                block_orig.attn.qkv_indices
            ]
            print(torch.tensor(block_orig.attn.qkv_indices).shape)
        else:
            block.attn.attn.weight.data = (
                block_orig.attn.attn.linear.linear.weight.data[
                    : block_orig.attn.attn.sub_network_out_features,
                    : block_orig.attn.attn.sub_network_in_features,
                ]
            )
            block.attn.attn.bias.data = block_orig.attn.attn.linear.linear.bias.data[
                : block_orig.attn.attn.sub_network_out_features
            ]
        if block_orig.attn.proj_indices is not None:
            block.attn.proj.weight.data = block_orig.attn.proj.linear.weight.data[
                : block_orig.attn.proj.sub_network_out_features,
                block_orig.attn.proj_indices,
            ]
            block.attn.proj.bias.data = block_orig.attn.proj.linear.bias.data[
                : block_orig.attn.proj.sub_network_out_features
            ]
        else:
            block.attn.proj.bias.data = block_orig.attn.proj.linear.bias.data[
                : block_orig.attn.proj.sub_network_out_features
            ]
            block.attn.proj.weight.data = block_orig.attn.proj.linear.weight.data[
                : block_orig.attn.proj.sub_network_out_features,
                : block_orig.attn.proj.sub_network_in_features,
            ]
        block.mlp.fc_1.weight.data = block_orig.mlp.fc_1.linear.weight.data[
            : block_orig.mlp.fc_1.sub_network_out_features,
            : block_orig.mlp.fc_1.sub_network_in_features,
        ]
        block.mlp.fc_1.bias.data = block_orig.mlp.fc_1.linear.bias.data[
            : block_orig.mlp.fc_1.sub_network_out_features
        ]
        block.mlp.fc_2.weight.data = block_orig.mlp.fc_2.linear.weight.data[
            : block_orig.mlp.fc_2.sub_network_out_features,
            : block_orig.mlp.fc_2.sub_network_in_features,
        ]
        block.mlp.fc_2.bias.data = block_orig.mlp.fc_2.linear.bias.data[
            : block_orig.mlp.fc_2.sub_network_out_features
        ]
        block.mlp.proj.weight.data = block_orig.mlp.proj.linear.weight.data[
            : block_orig.mlp.proj.sub_network_out_features,
            : block_orig.mlp.proj.sub_network_in_features,
        ]
        block.mlp.proj.bias.data = block_orig.mlp.proj.linear.bias.data[
            : block_orig.mlp.proj.sub_network_out_features
        ]
        block.norm_1.weight.data = block_orig.norm_1.weight.data[
            : block_orig.norm_1.sub_network_in_features
        ]
        block.norm_2.weight.data = block_orig.norm_2.weight.data[
            : block_orig.norm_2.sub_network_in_features
        ]
    out_lit_small = lit_gpt_small(input)
    assert torch.allclose(out_lit_small, out_small, atol=1e-3)


def copy_weights(model_source, model_target):
    for name, param_source in model_source.named_parameters():
        if "linear." in name:
            target_name = name.replace("linear.", "")
        elif "embedding." in name:
            target_name = name.replace("embedding.", "")
        else:
            target_name = name
        print(name, target_name)
        if target_name in model_target.state_dict():
            param_target = model_target.state_dict()[target_name]
            if param_source.shape == param_target.shape:
                param_source.data.copy_(param_target.data)
                print(f"Copying {name} to {target_name}")


def test_llama_3_1():
    config_llama = Config.from_name(
        "Llama-3-8B",
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
        padded_vocab_size=10000,
    )
    config_llama.fix_head_size = True
    config_llama_lit = LitConfig.from_name(
        "Llama-3-8B",
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
        padded_vocab_size=10000,
    )
    lit_model = LitGPT(config_llama_lit)
    whittle_model = GPT(config_llama)
    copy_weights(whittle_model, lit_model)
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    whittle_out = whittle_model(x)
    lit_out = lit_model(x)
    assert torch.allclose(whittle_out, lit_out, atol=1e-3)


def test_llama_3_2():
    config_llama = Config.from_name(
        "Llama-3.2-1B",
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
        padded_vocab_size=10000,
    )
    config_llama.fix_head_size = True
    config_llama_lit = LitConfig.from_name(
        "Llama-3.2-1B",
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
        padded_vocab_size=10000,
    )
    lit_model = LitGPT(config_llama_lit)
    whittle_model = GPT(config_llama)
    copy_weights(whittle_model, lit_model)
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32)
    whittle_out = whittle_model(x)
    lit_out = lit_model(x)
    assert torch.allclose(whittle_out, lit_out, atol=1e-3)


def test_gemma_2():
    config_gemma = Config.from_name(
        "gemma-2-9b",
        block_size=6,
        sliding_window_size=3,
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
    )
    config_gemma.head_size = 256
    config_gemma.fix_head_size = True
    config_gemma_lit = LitConfig.from_name(
        "gemma-2-9b",
        block_size=6,
        sliding_window_size=3,
        n_layer=2,
        n_embd=32,
        intermediate_size=86,
    )
    print(config_gemma)
    lit_model = LitGPT(config_gemma_lit)
    whittle_model = GPT(config_gemma)
    copy_weights(whittle_model, lit_model)
    x = torch.tensor([[9856, 23, 491, 1536, 304, 1234]], dtype=torch.int32)
    whittle_out = whittle_model(x)
    lit_out = lit_model(x)
    assert torch.allclose(whittle_out, lit_out, atol=1e-3)
