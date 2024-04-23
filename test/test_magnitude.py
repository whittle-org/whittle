import torch

from transformers import AutoConfig, AutoModelForSequenceClassification

from nas_fine_tuning.magnitude import compute_magnitude

config = AutoConfig.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")


def test_magnitude():

    head_mask = torch.randint(
        0, 2, (model.config.num_hidden_layers, model.config.num_attention_heads)
    )

    ffn_mask = torch.randint(
        0, 2, (model.config.num_hidden_layers, model.config.intermediate_size)
    )
    mag = compute_magnitude(model.bert.encoder, head_mask, ffn_mask)

    assert mag > 0
