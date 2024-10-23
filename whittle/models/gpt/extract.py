from __future__ import annotations

import torch.nn as nn

from collections import OrderedDict

from whittle.models.gpt import GPT
from whittle.models.gpt.blocks.mlp import GptNeoxMLP, LLaMAMLP
from whittle.modules.layernorm import LayerNorm
from whittle.modules.rmsnorm import RMSNorm


def extract_sub_network(model, sub_network_config):
    sub_network = GPT(sub_network_config)

    state_dict = extract_linear(model.lm_head)
    sub_network.lm_head.load_state_dict(state_dict)

    state_dict = extract_embedding(model.transformer.wte)
    sub_network.transformer.wte.load_state_dict(state_dict)

    extract_norm(model.transformer.ln_f, sub_network.transformer.ln_f)

    for i in range(sub_network_config.n_layer):
        block = model.transformer.h[i]
        sub_network_block = sub_network.transformer.h[i]

        # Attention
        state_dict = extract_linear(block.attn.attn)
        sub_network_block.attn.attn.load_state_dict(state_dict)
        state_dict = extract_linear(block.attn.proj)
        sub_network_block.attn.proj.load_state_dict(state_dict)

        # MLP
        extract_mlp(block.mlp, sub_network_block.mlp)

        # norm
        extract_norm(block.norm_1, sub_network_block.norm_1)
        extract_norm(block.post_attention_norm, sub_network_block.post_attention_norm)
        extract_norm(block.norm_2, sub_network_block.norm_2)
        extract_norm(block.post_mlp_norm, sub_network_block.post_mlp_norm)

    return sub_network


def extract_mlp(mlp, sub_mlp):
    if isinstance(mlp, GptNeoxMLP):
        state_dict = extract_linear(mlp.fc)
        sub_mlp.fc.load_state_dict(state_dict)

        state_dict = extract_linear(mlp.proj)
        sub_mlp.proj.load_state_dict(state_dict)
    elif isinstance(mlp, LLaMAMLP):
        state_dict = extract_linear(mlp.fc_1)
        sub_mlp.fc_1.load_state_dict(state_dict)

        state_dict = extract_linear(mlp.fc_2)
        sub_mlp.fc_2.load_state_dict(state_dict)

        state_dict = extract_linear(mlp.proj)
        sub_mlp.proj.load_state_dict(state_dict)
    else:
        raise ValueError(
            "Cannot extract MLP, supported MLP classes are GptNeoxMLP, LLaMAMLP and GemmaMLP."
        )


def extract_norm(norm, sub_norm):
    # nothing to extract
    if norm is None:
        assert sub_norm is None
        return

    if isinstance(norm, nn.Identity):
        assert isinstance(sub_norm, nn.Identity)
        return

    # extract depending on norm type
    assert isinstance(norm, RMSNorm) or isinstance(norm, LayerNorm)

    in_feat_sub = sub_norm.in_features
    super_state = norm.state_dict()

    new_state_dict = OrderedDict()
    new_state_dict["weight"] = super_state["weight"][:in_feat_sub]

    if isinstance(norm, RMSNorm):
        assert isinstance(sub_norm, RMSNorm)
    elif isinstance(norm, LayerNorm):
        assert isinstance(sub_norm, LayerNorm)
        new_state_dict["bias"] = super_state["bias"][:in_feat_sub]
    else:
        raise ValueError(
            "Cannot extract norm, supported norm classes are RMSNorm and LayerNorm."
        )

    sub_norm.load_state_dict(new_state_dict)


def extract_linear(super_network_linear):
    super_network_state = super_network_linear.state_dict()
    in_feat_sub = super_network_linear.sub_network_in_features
    out_feat_sub = super_network_linear.sub_network_out_features

    new_state_dict = OrderedDict()
    new_state_dict["weight"] = super_network_state["weight"][
        :out_feat_sub, :in_feat_sub
    ]

    if super_network_linear.use_bias:
        new_state_dict["bias"] = super_network_state["bias"][:out_feat_sub]

    return new_state_dict


def extract_embedding(super_network_embedding):
    super_network_state = super_network_embedding.state_dict()
    new_state_dict = OrderedDict()
    sub_network_embedding_dim = super_network_embedding.sub_network_embedding_dim

    new_state_dict["weight"] = super_network_state["weight"][
        :, :sub_network_embedding_dim
    ]

    return new_state_dict
