from __future__ import annotations

from test.test_training_strategies import MLP
from whittle.metrics.parameters import compute_parameters


def test_compute_parameters():
    model = MLP(8)

    assert compute_parameters(model) == 641
