from transformers.models.bert import BertModel

from lobotomy.metrics.parameters import compute_parameters


def test_compute_parameters():
    model_type = "bert-base-cased"
    model = BertModel.from_pretrained(model_type)

    assert compute_parameters(model) == 108211810
