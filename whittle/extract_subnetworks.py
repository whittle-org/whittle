from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn


class IdentityAttention(nn.Module):
    def __init__(self, model_type=None):
        self.model_type = model_type
        super().__init__()

    def forward(self, *input, **kwargs):
        if self.model_type == "distilbert-base-cased":
            return (kwargs["x"],)
        else:
            return input


class IdentityFFN(nn.Module):
    def __init__(self, model_type=None):
        self.model_type = model_type
        super().__init__()

    def forward(self, hidden_states):
        return hidden_states


class IdentityOutput(nn.Module):
    def __init__(self, model_type=None):
        self.model_type = model_type
        super().__init__()

    def forward(self, hidden_states, input_tensor):
        return hidden_states


def get_final_model(original_model, architecture_definition):
    new_model = deepcopy(original_model)

    for i in range(new_model.config.num_hidden_layers):
        if architecture_definition[f"layer_mha_{i}"] == 0:
            new_model.bert.encoder.layer[i].attention = IdentityAttention()
        if architecture_definition[f"layer_ffn_{i}"] == 0:
            new_model.bert.encoder.layer[i].intermediate = IdentityFFN()
            new_model.bert.encoder.layer[i].output = IdentityOutput()

    return new_model


def copy_linear_layer(new_layer, old_layer, weight_shape, bias_shape):
    old_state = old_layer.state_dict()
    new_state_dict = OrderedDict()
    new_state_dict["weight"] = old_state["weight"][: weight_shape[0], : weight_shape[1]]
    new_state_dict["bias"] = old_state["bias"][:bias_shape]
    new_layer.load_state_dict(new_state_dict)


def copy_layer_norm(new_layer, old_layer):
    old_state = old_layer.state_dict()
    new_state_dict = OrderedDict()
    new_state_dict["weight"] = old_state["weight"]
    new_state_dict["bias"] = old_state["bias"]
    new_layer.load_state_dict(new_state_dict)


def get_final_bert_model(original_model, new_model_config):
    original_model.eval()
    new_model = AutoModelForSequenceClassification.from_config(new_model_config)
    new_model.eval()

    new_model.bert.embeddings.load_state_dict(
        original_model.bert.embeddings.state_dict()
    )
    new_model.bert.pooler.load_state_dict(original_model.bert.pooler.state_dict())
    new_model.classifier.load_state_dict(original_model.classifier.state_dict())

    num_attention_heads = new_model_config.num_attention_heads
    attention_head_size = new_model_config.attention_head_size
    all_head_size = num_attention_heads * attention_head_size
    for li, layer in enumerate(new_model.bert.encoder.layer):
        attention = layer.attention
        attention.self.query = nn.Linear(new_model_config.hidden_size, all_head_size)
        attention.self.key = nn.Linear(new_model_config.hidden_size, all_head_size)
        attention.self.value = nn.Linear(new_model_config.hidden_size, all_head_size)
        attention.output.dense = nn.Linear(all_head_size, new_model_config.hidden_size)

        attention.self.all_head_size = all_head_size
        attention.self.attention_head_size = attention_head_size

        mha_original_model = original_model.bert.encoder.layer[li].attention
        copy_linear_layer(
            attention.self.query,
            mha_original_model.self.query,
            (all_head_size, new_model_config.hidden_size),
            (all_head_size),
        )

        copy_linear_layer(
            attention.self.key,
            mha_original_model.self.key,
            (all_head_size, new_model_config.hidden_size),
            (all_head_size),
        )

        copy_linear_layer(
            attention.self.value,
            mha_original_model.self.value,
            (all_head_size, new_model_config.hidden_size),
            (all_head_size),
        )

        copy_linear_layer(
            attention.output.dense,
            mha_original_model.output.dense,
            (new_model_config.hidden_size, all_head_size),
            (new_model_config.hidden_size),
        )

        copy_layer_norm(attention.output.LayerNorm, mha_original_model.output.LayerNorm)

        ffn_layer = layer.intermediate.dense
        ffn_original_model = original_model.bert.encoder.layer[li].intermediate.dense
        copy_linear_layer(
            ffn_layer,
            ffn_original_model,
            (new_model_config.intermediate_size, new_model_config.hidden_size),
            (new_model_config.intermediate_size),
        )

        ffn_layer = layer.output.dense
        ffn_original_model = original_model.bert.encoder.layer[li].output.dense
        copy_linear_layer(
            ffn_layer,
            ffn_original_model,
            (new_model_config.hidden_size, new_model_config.intermediate_size),
            (new_model_config.hidden_size),
        )

        copy_layer_norm(
            layer.output.LayerNorm,
            original_model.bert.encoder.layer[li].output.LayerNorm,
        )

    return new_model


if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.models.bert.modeling_bert import BertConfig

    from benchmarks.plm_pruning.mask import mask_bert

    model_type = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_type)

    head_mask = torch.ones(
        (model.config.num_hidden_layers, model.config.num_attention_heads)
    )
    head_mask[:, 6:] = 0
    head_mask[6:, :] = 0
    ffn_mask = torch.ones(
        (model.config.num_hidden_layers, model.config.intermediate_size)
    )
    ffn_mask[:, 1024:] = 0
    ffn_mask[6:, :] = 0
    config = BertConfig(
        num_hidden_layers=6, num_attention_heads=6, intermediate_size=1024
    )
    config.attention_head_size = int(
        config.hidden_size / model.config.num_attention_heads
    )

    new_model = get_final_bert_model(original_model=model, new_model_config=config)

    tokenizer = AutoTokenizer.from_pretrained(model_type)

    prompt = (
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )

    input = tokenizer(prompt, return_tensors="pt")

    handles = mask_bert(model, ffn_mask, head_mask)
    output = model(**input, head_mask=head_mask, output_hidden_states=True)

    print(output.logits)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_params)
    for handle in handles:
        handle.remove()
    output_2 = new_model(**input, output_hidden_states=True)
    print(output_2.logits)
    n_params = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
    print(n_params)
