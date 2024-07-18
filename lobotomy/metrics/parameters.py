def compute_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def params_linear_layer(layer):
    params = layer.sub_network_in_features * layer.sub_network_out_features
    if layer.use_bias:
        params += layer.sub_network_out_features
    return params


def params_embedding_layer(embedding):
    return embedding.num_embeddings * embedding.sub_network_embedding_dim


def params_layer_norm(layer_norm):
    return 2 * layer_norm.sub_network_in_features


def params_attention_layer(attention):
    dmodel = attention.sub_network_n_embd
    dhead = attention.sub_network_head_size
    num_heads = attention.sub_network_n_head
    n_attention = (dmodel * dhead + dhead) * num_heads * 3
    n_attention += dmodel * dmodel + dmodel  # output
    print(num_heads, n_attention)
    return n_attention


def compute_parameters_sub_network(model):

    """
    Computes parameters of a selected sub-network.
    For formulas, see: https://towardsdatascience.com/how-to-estimate-the-number-of-parameters-in-transformer-models-ca0f57d8dff0

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

    return num_params
