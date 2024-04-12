import numpy as np

from transformers import AutoModelForSequenceClassification

from nas_fine_tuning.estimate_efficency import compute_parameters


def test_compute_parameters():
    model_type = 'bert-base-cased'
    model = AutoModelForSequenceClassification.from_pretrained(model_type)
    n_params_enc = sum(p.numel() for p in model.bert.encoder.parameters() if p.requires_grad)

    n_heads = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    dmodel = model.config.hidden_size
    dffn = model.config.intermediate_size
    dhead = dmodel // n_heads
    num_heads_per_layer = np.ones(n_layers) * n_heads
    num_neurons_per_layer = np.ones(n_layers) * dffn
    n_params = compute_parameters(dmodel=dmodel, dhead=dhead,
                                  num_heads_per_layer=num_heads_per_layer,
                                  num_neurons_per_layer=num_neurons_per_layer)

    assert n_params == n_params_enc
