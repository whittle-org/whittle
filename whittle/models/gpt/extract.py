from __future__ import annotations

from collections import OrderedDict

from whittle.models.gpt import GPT
from whittle.models.gpt_flex import GPTFlex


def set_subnet_attention_sizes(model, subnet_config):
    n_query_groups = []
    head_size = []
    for i in range(subnet_config.n_layer):
        block = model.transformer.h[i]
        n_query_groups.append(block.attn.sub_network_query_groups)
        head_size.append(block.attn.sub_network_head_size)
    subnet_config.n_query_groups = n_query_groups
    subnet_config.head_size = head_size


def extract_sub_network(model, sub_network_config, use_flex=False):
    sub_network = GPT(sub_network_config) if use_flex else GPTFlex(sub_network_config)

    state_dict = extract_linear(model.lm_head)
    sub_network.lm_head.load_state_dict(state_dict)

    state_dict = extract_embedding(model.transformer.wte)
    sub_network.transformer.wte.load_state_dict(state_dict)

    for i in range(sub_network_config.n_layer):
        block = model.transformer.h[i]
        sub_network_block = sub_network.transformer.h[i]

        # Attention
        state_dict = extract_linear(block.attn.attn)
        sub_network_block.attn.attn.load_state_dict(state_dict)
        state_dict = extract_linear(block.attn.proj)
        sub_network_block.attn.proj.load_state_dict(state_dict)

        # MLP
        state_dict = extract_linear(block.mlp.fc)
        sub_network_block.mlp.fc.load_state_dict(state_dict)

        state_dict = extract_linear(block.mlp.proj)
        sub_network_block.mlp.proj.load_state_dict(state_dict)

    return sub_network


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
