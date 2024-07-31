from whittle.models.gpt import GPT
from whittle.models.gpt.blocks import CausalSelfAttention
from whittle.modules.embedding import Embedding
from whittle.modules.linear import Linear
from whittle.modules.layernorm import LayerNorm
from whittle.modules.rmsnorm import RMSNorm


def compute_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def params_linear_layer(layer: Linear):
    params = layer.sub_network_in_features * layer.sub_network_out_features
    if layer.use_bias:
        params += layer.sub_network_out_features
    return params


def params_rmsnorm(rmsnorm: RMSNorm):
    return rmsnorm.sub_network_in_features


def params_embedding_layer(embedding: Embedding):
    return embedding.num_embeddings * embedding.sub_network_embedding_dim


def params_layer_norm(layer_norm: LayerNorm):
    return 2 * layer_norm.sub_network_in_features


def params_attention_layer(attention: CausalSelfAttention):
    # TODO: generalize to QHA
    dmodel = attention.sub_network_n_embd
    dhead = attention.sub_network_head_size
    num_heads = attention.sub_network_n_head
    n_attention = (dmodel * dhead + dhead) * num_heads * 3
    n_attention += dmodel * dmodel + dmodel  # output

    return n_attention


def compute_parameters_sub_network_gpt(model: GPT):
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
    for block in model.transformer.h:
        num_params += params_layer_norm(block.norm_1)

        num_params += params_attention_layer(block.attn)
        num_params += params_linear_layer(block.mlp.fc)
        num_params += params_linear_layer(block.mlp.proj)
        num_params += params_layer_norm(block.norm_2)
    num_params += params_layer_norm(model.transformer.ln_f)
    return num_params
