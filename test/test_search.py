from __future__ import annotations

import numpy as np
import pytest
from syne_tune.config_space import randint

from whittle.search import multi_objective_search
from whittle.search.baselines import methods
from whittle.search.param_bins import ParamBins


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


bin_tolerance = [0, 1]
bin_size = [1, 2]
num_bins = [3, 10]

@pytest.mark.parametrize("bin_t", bin_tolerance)
@pytest.mark.parametrize("bin_s", bin_size)
@pytest.mark.parametrize("bin_n", num_bins)
def test_param_bins(bin_t, bin_s, bin_n):
    print(bin_t, bin_s, bin_n)
    def params_estimator(config):
        return config["a"]

    bin_width = 10
    min_config = {"a": 0}
    max_config = {"a": bin_n * bin_width}

    bins = ParamBins(
        min_config,
        max_config,
        params_estimator,
        num_bins=bin_n,
        log_bins=False,
        start_bin_size=bin_s,
        empty_bin_tolerance=bin_t,
    )

    # fill up to bin_n - 1 bins
    for j in range(bin_s):
        for i in range(bin_n - bin_t - 1):
            assert bins.put_in_bin({"a": 1 + i * bin_width})

        # fill the last one unless it'd be filled fully (leave 1 not full)
        if j < bin_s - 1:
            assert bins.put_in_bin({"a": 1 + (bin_n - bin_t - 1) * bin_width})
        
    assert bins.current_bin_length == bin_s

    # last bin is not filled fully -> this should be false
    assert not bins.put_in_bin({"a": 3})
    assert bins.current_bin_length == bin_s

    # last bin is filled fully -> this should be true
    assert bins.put_in_bin({"a": 2 + bin_width * (bin_n - bin_t - 1)})
    assert bins.put_in_bin({"a": 3})
    assert bins.current_bin_length == bin_s + 1
