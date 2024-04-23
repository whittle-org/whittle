import pytest
import numpy as np

from syne_tune.config_space import randint
from lobotomy.search import multi_objective_search
from lobotomy.search.baselines import methods


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
