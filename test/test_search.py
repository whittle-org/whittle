from __future__ import annotations

import numpy as np
import pytest
from syne_tune.config_space import randint

from whittle.search import multi_objective_search
from whittle.search.baselines import methods


def objective(config, **kwargs):
    return np.random.rand() * config["a"], np.random.rand() * config["b"]


search_space = {"a": randint(0, 10), "b": randint(0, 100)}


@pytest.mark.parametrize("search_strategy", methods)
def test_multi_objective_search(search_strategy, num_samples=5):
    results = multi_objective_search(
        objective=objective,
        search_strategy=search_strategy,
        search_space=search_space,
        objective_kwargs={},
        num_samples=num_samples,
    )

    assert all(
        key in results for key in ["costs", "configs", "runtime", "is_pareto_optimal"]
    )

    assert results["costs"].shape == (num_samples, 2)
    assert len(results["configs"]) == num_samples
    assert len(results["runtime"]) == num_samples
    assert len(results["is_pareto_optimal"]) == num_samples

    if search_strategy != "nsga2":
        # check that first config are the initial design
        upper_bound = {hp_name: hp.upper for hp_name, hp in search_space.items()}
        assert results["configs"][0] == upper_bound

        lower_bound = {hp_name: hp.lower for hp_name, hp in search_space.items()}
        assert results["configs"][1] == lower_bound

        mid_point = {
            hp_name: (hp.upper - hp.lower) // 2 for hp_name, hp in search_space.items()
        }
        assert results["configs"][2] == mid_point
