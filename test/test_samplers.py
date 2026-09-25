from __future__ import annotations

from unittest.mock import Mock

import pytest
from syne_tune.config_space import randint

from whittle.sampling.grid_samplers import StratifiedRandomSampler

search_space = {"a": randint(0, 10)}


def test_stratified_random_sampler_stops_after_max_tries():
    sampler = StratifiedRandomSampler(search_space, seed=0, max_tries=5)
    sampler.param_bins = Mock()
    sampler.param_bins.put_in_bin.return_value = False

    with pytest.raises(ValueError, match="Could not find a valid configuration"):
        sampler.sample()

    assert sampler.param_bins.put_in_bin.call_count == 6


def test_stratified_random_sampler_returns_binned_config():
    sampler = StratifiedRandomSampler(search_space, seed=0, max_tries=5)
    sampler.param_bins = Mock()
    sampler.param_bins.put_in_bin.side_effect = [False, False, True]

    config = sampler.sample()

    assert 0 <= config["a"] <= 10
    assert sampler.param_bins.put_in_bin.call_count == 3
