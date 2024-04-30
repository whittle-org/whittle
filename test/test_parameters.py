from lobotomy.metrics.parameters import compute_parameters

from test.test_training_strategies import MLP


def test_compute_parameters():
    model = MLP(8)

    assert compute_parameters(model) == 641
