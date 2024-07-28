from whittle.models.gpt import GPT
from litgpt import Config
import torch
import pytest
from litgpt.model import GPT as LitGPT


@pytest.mark.parametrize("sample_random_indices", [False, True])
def test_gpt(sample_random_indices):
    torch.manual_seed(0)
    config = Config()
    config.padded_vocab_size = 512
    config.n_embd = 64
    config.intermediate_size = 64 * 4
    config.n_head = 8
    config.n_query_groups = 4
    config.head_size = 8
    config.n_layer = 2
    config.block_size = 512
    config.norm_class_name = "RMSNorm"
    config.mlp_class_name = "LLaMAMLP"
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)
    config.norm_eps = 1e-5
    config.lm_head_bias = True
    config.fix_head_size = False
    gpt = GPT(config)
    gpt.transformer.wte.weight.data = torch.randn_like(gpt.transformer.wte.weight.data)
    gpt.lm_head.weight.data = torch.randn_like(gpt.lm_head.weight.data)
    gpt.lm_head.bias.data = torch.randn_like(gpt.lm_head.bias.data)
    gpt.transformer.ln_f.weight.data = torch.randn_like(
        gpt.transformer.ln_f.weight.data
    )

    for block in gpt.transformer.h:
        block.attn.attn.weight.data = torch.randn_like(block.attn.attn.weight.data)
        block.attn.attn.bias.data = torch.randn_like(block.attn.attn.bias.data)
        block.attn.proj.bias.data = torch.randn_like(block.attn.proj.bias.data)
        block.attn.proj.weight.data = torch.randn_like(block.attn.proj.weight.data)
        block.mlp.fc_1.weight.data = torch.randn_like(block.mlp.fc_1.weight.data)
        block.mlp.fc_1.bias.data = torch.randn_like(block.mlp.fc_1.bias.data)
        block.mlp.fc_2.weight.data = torch.randn_like(block.mlp.fc_2.weight.data)
        block.mlp.fc_2.bias.data = torch.randn_like(block.mlp.fc_2.bias.data)
        block.mlp.proj.weight.data = torch.randn_like(block.mlp.proj.weight.data)
        block.mlp.proj.bias.data = torch.randn_like(block.mlp.proj.bias.data)
        block.norm_1.weight.data = torch.randn_like(block.norm_1.weight.data)
        block.norm_2.weight.data = torch.randn_like(block.norm_2.weight.data)

    gpt.reset_super_network()
    input = torch.randint(0, 512, (1, 512))
    out_large = gpt(input)
    assert out_large.shape == (1, 512, 512)

    lit_gpt = LitGPT(config)
    lit_gpt.lm_head.weight.data = gpt.lm_head.weight.data
    lit_gpt.lm_head.bias.data = gpt.lm_head.bias.data
    lit_gpt.transformer.wte.weight.data = gpt.transformer.wte.weight.data
    lit_gpt.transformer.ln_f.weight.data = gpt.transformer.ln_f.weight.data
    for i, block in enumerate(lit_gpt.transformer.h):
        block_orig = gpt.transformer.h[i]
        block.attn.attn.weight.data = block_orig.attn.attn.weight.data
        block.attn.attn.bias.data = block_orig.attn.attn.bias.data
        block.attn.proj.bias.data = block_orig.attn.proj.bias.data
        block.attn.proj.weight.data = block_orig.attn.proj.weight.data
        block.mlp.fc_1.weight.data = block_orig.mlp.fc_1.weight.data
        block.mlp.fc_1.bias.data = block_orig.mlp.fc_1.bias.data
        block.mlp.fc_2.weight.data = block_orig.mlp.fc_2.weight.data
        block.mlp.fc_2.bias.data = block_orig.mlp.fc_2.bias.data
        block.mlp.proj.weight.data = block_orig.mlp.proj.weight.data
        block.mlp.proj.bias.data = block_orig.mlp.proj.bias.data
        block.norm_1.weight.data = block_orig.norm_1.weight.data
        block.norm_2.weight.data = block_orig.norm_2.weight.data

    out_lit_large = lit_gpt(input)
    assert torch.allclose(out_lit_large, out_large, atol=1e-3)
    gpt.set_sub_network(
        sub_network_n_embd=32,
        sub_network_intermediate_size=[32 * 4 for i in range(4)],
        sub_network_num_heads=[4 for i in range(4)],
        sub_network_n_layers=1,
        sub_network_query_groups=2,
        sample_random_indices=sample_random_indices,
    )
    out_small = gpt(input)
    if sample_random_indices:
        assert torch.all(
            gpt.transformer.wte.random_indices
            == torch.tensor(
                [
                    21,
                    63,
                    10,
                    43,
                    44,
                    14,
                    59,
                    3,
                    36,
                    55,
                    11,
                    40,
                    5,
                    27,
                    45,
                    0,
                    17,
                    41,
                    33,
                    25,
                    50,
                    60,
                    12,
                    20,
                    52,
                    24,
                    47,
                    58,
                    38,
                    32,
                    26,
                    61,
                ]
            )
        )
    assert out_small.shape == (1, 512, 512)
    config.n_embd = 32
    config.n_head = 4
    config.n_query_groups = 2
    config.intermediate_size = 32 * 4
    config.n_layer = 1
    lit_gpt_small = LitGPT(config)
    lit_gpt_small.lm_head.weight.data = gpt.lm_head.weight.data[
        gpt.lm_head.random_indices_out_features, :
    ][:, gpt.lm_head.random_indices_in_features]
    lit_gpt_small.lm_head.bias.data = gpt.lm_head.bias.data[:]
    lit_gpt_small.transformer.wte.weight.data = gpt.transformer.wte.weight.data[
        :, gpt.transformer.wte.random_indices
    ]
    lit_gpt_small.transformer.ln_f.weight.data = gpt.transformer.ln_f.weight.data[
        gpt.transformer.ln_f.random_indices
    ]

    for i, block in enumerate(lit_gpt_small.transformer.h):
        block_orig = gpt.transformer.h[gpt.random_layers[i]]
        block.attn.attn.weight.data = block_orig.attn.attn.weight.data[
            block_orig.attn.attn.random_indices_out_features, :
        ][:, block_orig.attn.attn.random_indices_in_features]
        block.attn.attn.bias.data = block_orig.attn.attn.bias.data[
            block_orig.attn.attn.random_indices_out_features
        ]
        block.attn.proj.bias.data = block_orig.attn.proj.bias.data[
            block_orig.attn.proj.random_indices_out_features
        ]
        block.attn.proj.weight.data = block_orig.attn.proj.weight.data[
            block_orig.attn.proj.random_indices_out_features, :
        ][:, block_orig.attn.proj.random_indices_in_features]
        block.mlp.fc_1.weight.data = block_orig.mlp.fc_1.weight.data[
            block_orig.mlp.fc_1.random_indices_out_features, :
        ][:, block_orig.mlp.fc_1.random_indices_in_features]
        block.mlp.fc_1.bias.data = block_orig.mlp.fc_1.bias.data[
            block_orig.mlp.fc_1.random_indices_out_features
        ]
        block.mlp.fc_2.weight.data = block_orig.mlp.fc_2.weight.data[
            block_orig.mlp.fc_2.random_indices_out_features, :
        ][:, block_orig.mlp.fc_2.random_indices_in_features]
        block.mlp.fc_2.bias.data = block_orig.mlp.fc_2.bias.data[
            block_orig.mlp.fc_2.random_indices_out_features
        ]
        block.mlp.proj.weight.data = block_orig.mlp.proj.weight.data[
            block_orig.mlp.proj.random_indices_out_features, :
        ][:, block_orig.mlp.proj.random_indices_in_features]
        block.mlp.proj.bias.data = block_orig.mlp.proj.bias.data[
            block_orig.mlp.proj.random_indices_out_features
        ]
        block.norm_1.weight.data = block_orig.norm_1.weight.data[
            block_orig.norm_1.random_indices
        ]
        block.norm_2.weight.data = block_orig.norm_2.weight.data[
            block_orig.norm_2.random_indices
        ]
    out_lit_small = lit_gpt_small(input)
    assert torch.allclose(out_lit_small, out_small, atol=1e-3)
