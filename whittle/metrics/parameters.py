from __future__ import annotations

import logging

import torch.nn as nn

from whittle.models.gpt import GPT
from whittle.models.gpt.blocks import CausalSelfAttention, GemmaMLP, GptNeoxMLP, LLaMAMLP
from whittle.modules.embedding import Embedding
from whittle.modules.layernorm import LayerNorm
from whittle.modules.linear import Linear
from whittle.modules.rmsnorm import RMSNorm


def compute_all_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def params_linear_layer(layer: Linear):
    params = layer.sub_network_in_features * layer.sub_network_out_features
    if layer.use_bias:
        params += layer.sub_network_out_features
    return params


def params_embedding_layer(embedding: Embedding):
    return embedding.num_embeddings * embedding.sub_network_embedding_dim


def params_layer_normalization(normalization_layer: nn.Module):
    if normalization_layer is None:
        return 0
    if isinstance(normalization_layer, LayerNorm):
        return 2 * normalization_layer.sub_network_in_features
    elif isinstance(normalization_layer, RMSNorm):
        return normalization_layer.sub_network_in_features
    else:
        logging.error(
            f"Normalization layer type: {type(normalization_layer)} not supported!"
        )
        raise


def params_attention_layer(attention: CausalSelfAttention):
    dmodel = attention.sub_network_n_embd
    dhead = attention.sub_network_head_size
    if attention.config.n_query_groups != attention.config.n_head:
        q_per_kv = attention.sub_network_n_head // attention.config.n_query_groups
        num_query_groups = attention.sub_network_query_groups
    else:
        q_per_kv = 1
        num_query_groups = attention.sub_network_n_head
    qkv_dim = (q_per_kv + 2) * dhead * num_query_groups
    n_attention = dmodel * qkv_dim
    if attention.qkv.use_bias:
        n_attention += qkv_dim
    n_attention += dmodel * dhead * num_query_groups * q_per_kv
    if attention.proj.use_bias:
        n_attention += dmodel
    if attention.config.norm_qk:
        n_attention += dhead * num_query_groups
        n_attention += dhead * num_query_groups * q_per_kv

    return n_attention


def params_mlp(mlp: nn.Module):
    layers = []
    if isinstance(mlp, GptNeoxMLP):
        layers = [mlp.proj, mlp.fc]

    elif isinstance(mlp, LLaMAMLP) or isinstance(mlp, GemmaMLP):
        layers = [mlp.proj, mlp.fc_1, mlp.fc_2]

    num_params = 0
    for layer in layers:
        num_params += params_linear_layer(layer)
    return num_params


def compute_parameters(model: GPT) -> float:
    """
    Computes parameters of the current sub-network of a GPT mmodel. Make sure to set the sub-network before
    calling this function.

    Refs:
        https://towardsdatascience.com/how-to-estimate-the-number-of-parameters-in-transformer-models-ca0f57d8dff0

    Args:
        model: GPT model

    Returns:
        float: number of parameters of the activated sub-network
    """

    num_params = 0
    num_params += params_linear_layer(model.lm_head)
    num_params += params_embedding_layer(model.transformer.wte)
    for i in range(model.sub_network_n_layers):
        block = model.transformer.h[i]
        num_params += params_mlp(block.mlp)
        num_params += params_attention_layer(block.attn)

        num_params += params_layer_normalization(block.norm_1)
        num_params += params_layer_normalization(block.norm_2)
    num_params += params_layer_normalization(model.transformer.ln_f)
    return num_params
