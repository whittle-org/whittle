import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.bert.modeling_bert import BertConfig, BertAttention, BertOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaAttention,
    RobertaOutput,
)

from nas_fine_tuning.mask import mask_bert, mask_gpt, mask_gpt_neox, mask_roberta
from nas_fine_tuning.sampling import SmallSearchSpace
from nas_fine_tuning.mask.mask_bert import (
    register_drop_attention_layer,
    register_mask_ffn,
)


def test_bert_masking():
    model_type = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_type)
    config = model.config
    seq_len = 128
    search_space = SmallSearchSpace(config)
    model.eval()

    arch = {"num_layers": 6, "num_heads": 6, "num_units": 1024}

    head_mask, neuron_mask = search_space.config_to_mask(arch)
    handles = mask_bert(model, neuron_mask=neuron_mask, head_mask=head_mask)

    input_tensor = torch.rand(1, seq_len, config.hidden_size)
    output = model.bert.encoder(input_tensor, output_hidden_states=True)

    hidden_states = output.hidden_states

    for i in range(7, config.num_hidden_layers):
        assert torch.all(hidden_states[6] == hidden_states[i])


def test_bert_attention():
    config = BertConfig()
    seq_len = 128
    attention = BertAttention(config)
    input = torch.rand(1, seq_len, config.hidden_size)
    handle = register_drop_attention_layer(attention)
    output = attention(input)
    assert torch.all(input == output[0])


def test_bert_ffn2():
    config = BertConfig()
    seq_len = 128
    output_layer = BertOutput(config)
    input = torch.rand(1, seq_len, config.hidden_size)
    hidden = torch.rand(1, seq_len, config.intermediate_size)

    nm = torch.ones(config.intermediate_size)
    nm[1024:] = 0
    handle = register_mask_ffn(output_layer, nm)
    output = output_layer(hidden, input)


def test_roberta_masking():
    model_type = "roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_type)
    config = model.config
    seq_len = 128
    search_space = SmallSearchSpace(config)
    model.eval()

    arch = {"num_layers": 6, "num_heads": 6, "num_units": 1024}

    head_mask, neuron_mask = search_space.config_to_mask(arch)
    handles = mask_roberta(model, neuron_mask=neuron_mask, head_mask=head_mask)

    input_tensor = torch.rand(1, seq_len, config.hidden_size)
    output = model.roberta.encoder(input_tensor, output_hidden_states=True)

    hidden_states = output.hidden_states

    for i in range(7, config.num_hidden_layers):
        assert torch.all(hidden_states[6] == hidden_states[i])


def test_roberta_attention():
    config = RobertaConfig()
    seq_len = 128
    attention = RobertaAttention(config)
    input = torch.rand(1, seq_len, config.hidden_size)
    handle = register_drop_attention_layer(attention)
    output = attention(input)
    assert torch.all(input == output[0])


def test_roberta_ffn2():
    config = RobertaConfig()
    seq_len = 128
    output_layer = RobertaOutput(config)
    input = torch.rand(1, seq_len, config.hidden_size)
    hidden = torch.rand(1, seq_len, config.intermediate_size)

    nm = torch.ones(config.intermediate_size)
    nm[1024:] = 0
    handle = register_mask_ffn(output_layer, nm)
    output = output_layer(hidden, input)


def xtest_gpt2_masking():
    model_type = "gpt2"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_type, use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    config = model.config
    search_space = SmallSearchSpace(config)
    model.eval()
    num_layers = 3
    arch = {"num_layers": num_layers, "num_heads": 3, "num_units": 1024}

    head_mask, neuron_mask = search_space.config_to_mask(arch)
    handles = mask_gpt_neox(model, neuron_mask=neuron_mask, head_mask=head_mask)

    prompt = (
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.forward(input_ids, output_hidden_states=True)

    hidden_states = output.hidden_states
    for i in range(num_layers, config.num_hidden_layers):
        assert torch.all(hidden_states[5] == hidden_states[i])


def test_gptneox_masking():
    model_type = "EleutherAI/pythia-70m"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_type, use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    config = model.config
    search_space = SmallSearchSpace(config)
    model.eval()
    num_layers = 3
    arch = {"num_layers": num_layers, "num_heads": 3, "num_units": 1024}

    head_mask, neuron_mask = search_space.config_to_mask(arch)
    handles = mask_gpt_neox(model, neuron_mask=neuron_mask, head_mask=head_mask)

    prompt = (
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.forward(input_ids, output_hidden_states=True)

    hidden_states = output.hidden_states
    for i in range(num_layers, config.num_hidden_layers):
        assert torch.all(hidden_states[5] == hidden_states[i])
