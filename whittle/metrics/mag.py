import torch

from whittle.models.gpt import GPT


def weight_magnitude(model: GPT):
    magnitude = 0
    magnitude += weight_magnitude_linear_layer(model.lm_head)
    magnitude += weight_magnitude_embedding(model.transformer.wte)
    for block in model.transformer.h:
        magnitude += weight_magnitude_layer_norm(block.norm_1)
        magnitude += weight_magnitude_attention(block.attn)
        magnitude += weight_magnitude_linear_layer(block.mlp.fc)
        magnitude += weight_magnitude_linear_layer(block.mlp.proj)
        magnitude += weight_magnitude_layer_norm(block.norm_2)
    magnitude += weight_magnitude_layer_norm(model.transformer.ln_f)
    return magnitude


def weight_magnitude_layer_norm(layer):
    n = layer.sub_network_in_features
    mag = torch.sum(torch.abs(layer.weight[:n]))
    mag += torch.sum(torch.abs(layer.bias[:n]))
    return float(mag)


def weight_magnitude_linear_layer(layer):
    n = layer.sub_network_in_features
    m = layer.sub_network_out_features
    mag = torch.sum(torch.abs(layer.weight[:m, :n]))
    if layer.use_bias:
        mag += torch.sum(torch.abs(layer.bias[:m]))
    return float(mag)


def weight_magnitude_embedding(layer):
    n = layer.sub_network_embedding_dim
    mag = torch.sum(torch.abs(layer.weight[:, :n]))
    return float(mag)


def weight_magnitude_attention(layer):
    mag = weight_magnitude_linear_layer(layer.attn)
    mag += weight_magnitude_linear_layer(layer.proj)
    return float(mag)
