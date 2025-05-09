from __future__ import annotations

import logging

import torch

from whittle.models.gpt import GPT
from whittle.models.gpt.blocks import GemmaMLP, GptNeoxMLP, LLaMAMLP
from whittle.modules.layernorm import LayerNorm
from whittle.modules.rmsnorm import RMSNorm


def compute_weight_magnitude(model: GPT) -> float:
    """
    Computes the sum of the weight magnitudes of the current sub-network of a GPT model. Make sure to set the
    sub-network before calling this function.

    Args:
        model: GPT model

    Returns:
        float: magnitude of the weights of the activated sub-network
    """
    magnitude = 0
    magnitude += compute_weight_magnitude_linear_layer(model.lm_head)
    magnitude += compute_weight_magnitude_embedding(model.transformer.wte)
    for i in range(model.sub_network_n_layers):
        block = model.transformer.h[i]
        magnitude += compute_weight_magnitude_layer_norm(block.norm_1)
        magnitude += compute_weight_magnitude_attention(block.attn)
        magnitude += compute_weight_magnitude_mlp(block.mlp)
        magnitude += compute_weight_magnitude_layer_norm(block.norm_2)
    magnitude += compute_weight_magnitude_layer_norm(model.transformer.ln_f)
    return magnitude


def compute_weight_magnitude_mlp(mlp):
    if isinstance(mlp, GptNeoxMLP):
        layers = [mlp.proj, mlp.fc]

    elif isinstance(mlp, LLaMAMLP) or isinstance(mlp, GemmaMLP):
        layers = [mlp.proj, mlp.fc_1, mlp.fc_2]

    else:
        logging.error(f"MLP type: {type(mlp)} not supported!")
        raise

    mag = 0
    for layer in layers:
        mag += compute_weight_magnitude_linear_layer(layer)
    return mag


def compute_weight_magnitude_layer_norm(layer):
    if layer is None:
        return 0
    if isinstance(layer, LayerNorm):
        n = layer.sub_network_in_features
        mag = torch.sum(torch.abs(layer.weight[:n]))
        mag += torch.sum(torch.abs(layer.bias[:n]))

    elif isinstance(layer, RMSNorm):
        n = layer.sub_network_in_features
        mag = torch.sum(torch.abs(layer.weight[:n]))
    else:
        logging.error(f"Normalization layer type: {type(layer)} not supported!")
        raise

    return float(mag)


def compute_weight_magnitude_linear_layer(layer):
    n = layer.sub_network_in_features
    m = layer.sub_network_out_features
    mag = torch.sum(torch.abs(layer.weight[:m, :n]))
    if layer.use_bias:
        mag += torch.sum(torch.abs(layer.bias[:m]))
    return float(mag)


def compute_weight_magnitude_embedding(layer):
    n = layer.sub_network_embedding_dim
    mag = torch.sum(torch.abs(layer.weight[:, :n]))
    return float(mag)


def compute_weight_magnitude_attention(layer):
    mag = 0
    mag = mag + compute_weight_magnitude_linear_layer(layer.qkv)
    mag += compute_weight_magnitude_linear_layer(layer.proj)
    if layer.config.norm_qk:
        mag += compute_weight_magnitude_layer_norm(layer.norm_q)
        mag += compute_weight_magnitude_layer_norm(layer.norm_k)
    return float(mag)
