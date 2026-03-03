from __future__ import annotations

import copy

from litgpt.model import GPT as LitGPT


def create_litgpt_config_for_subnet(supernet):
    config = copy.deepcopy(supernet.config)
    config.fix_head_size = True
    config.n_embd = supernet.sub_network_n_embd
    config.intermediate_size = supernet.sub_network_intermediate_size
    config.n_head = supernet.sub_network_num_heads
    config.n_layer = supernet.sub_network_n_layers
    config.head_size = supernet.sub_network_head_size
    config.rope_n_elem = supernet.sub_network_rope_n_elem
    config.n_query_groups = supernet.sub_network_query_groups
    return config


def copy_weights_to_litgpt(whittle_model, lit_model):
    def _copy_weights_and_biases(whittle_module, lit_module):
        if hasattr(whittle_module, "extract_weights"):
            W, b = whittle_module.extract_weights()
        else:
            print(f"{whittle_module} has no method extract_weights")

        if hasattr(lit_module, "weight"):
            lit_module.weight.data = W.data

        if hasattr(lit_module, "bias"):
            if b is None:
                lit_module.bias = None
            else:
                lit_module.bias.data = b.data

    _copy_weights_and_biases(whittle_model.lm_head, lit_model.lm_head)
    _copy_weights_and_biases(whittle_model.transformer.wte, lit_model.transformer.wte)
    _copy_weights_and_biases(whittle_model.transformer.ln_f, lit_model.transformer.ln_f)

    chosen_layers = (
        range(whittle_model.sub_network_n_layers)
        if whittle_model.sampled_layer_indices is None
        else whittle_model.sampled_layer_indices
    )
    whittle_blocks = [whittle_model.transformer.h[idx] for idx in chosen_layers]
    litgpt_blocks = lit_model.transformer.h

    for whittle_block, litgpt_block in zip(whittle_blocks, litgpt_blocks):
        _copy_weights_and_biases(whittle_block.norm_1, litgpt_block.norm_1)
        _copy_weights_and_biases(whittle_block.attn.qkv, litgpt_block.attn.qkv)
        _copy_weights_and_biases(whittle_block.attn.proj, litgpt_block.attn.proj)
        _copy_weights_and_biases(
            whittle_block.post_attention_norm, litgpt_block.post_attention_norm
        )
        _copy_weights_and_biases(whittle_block.norm_2, litgpt_block.norm_2)

        if hasattr(whittle_block.mlp, "fc"):
            _copy_weights_and_biases(whittle_block.mlp.fc, litgpt_block.mlp.fc)
        if hasattr(whittle_block.mlp, "fc_1"):
            _copy_weights_and_biases(whittle_block.mlp.fc_1, litgpt_block.mlp.fc_1)
        if hasattr(whittle_block.mlp, "fc_2"):
            _copy_weights_and_biases(whittle_block.mlp.fc_2, litgpt_block.mlp.fc_2)

        _copy_weights_and_biases(whittle_block.mlp.proj, litgpt_block.mlp.proj)
        _copy_weights_and_biases(whittle_block.post_mlp_norm, litgpt_block.post_mlp_norm)


def convert_subnet_to_litgpt(whittle_model, subnet_config):
    whittle_model.set_sub_network(**subnet_config)
    litgpt_config = create_litgpt_config_for_subnet(whittle_model)
    lit_model = LitGPT(litgpt_config)
    copy_weights_to_litgpt(whittle_model, lit_model)
    whittle_model.reset_super_network()
    return lit_model
