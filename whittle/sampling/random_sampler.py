from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from syne_tune.config_space import Categorical, Domain

from whittle.args import ParamBinArgs
from whittle.sampling.param_bins import ParamBins


class RandomSampler:
    """
    RandomSampler samples configurations from a given search space using a random state.

    Args:
        config_space: The search space from which to sample.
        seed: Seed for the random number generator. Defaults to None.
    """

    def __init__(self, config_space: dict, seed: int | None = None):
        self.config_space = config_space
        self.rng = np.random.RandomState(seed)

    def sample(self) -> dict[str, Any]:
        """
        Gets a random sub-network configuration from the search space.

        Returns:
            A random sub-network configuration.
        """
        config = {}
        for hp_name, hparam in self.config_space.items():
            if isinstance(hparam, Domain):
                config[hp_name] = hparam.sample(random_state=self.rng)
        return config

    def get_smallest_sub_network(self) -> dict[str, Any]:
        """
        Gets the smallest sub-network configuration from the search space.

        Returns:
            The smallest sub-network configuration.
        """
        config = {}
        for k, v in self.config_space.items():
            if isinstance(v, Domain):
                if isinstance(v, Categorical):
                    if all(isinstance(e, (int, float)) for e in v.categories):
                        config[k] = min(v.categories)
                    else:
                        warnings.warn(
                            "Warning: Categoricals are non-integers, check if smallest network is as intended"
                        )
                        config[k] = v.categories[0]
                else:
                    config[k] = v.lower
        return config

    def get_largest_sub_network(self) -> dict[str, Any]:
        """
        gets the largest sub-network configuration from the search space.

        Returns:
            The largest sub-network configuration.
        """

        config = {}
        for k, v in self.config_space.items():
            if isinstance(v, Domain):
                if isinstance(v, Categorical):
                    if all(isinstance(e, (int, float)) for e in v.categories):
                        config[k] = max(v.categories)
                    else:
                        warnings.warn(
                            "Warning: Categoricals are non-integers, check if largest network is as intended"
                        )
                        config[k] = v.categories[-1]
                else:
                    config[k] = v.upper
        return config


class StratifiedRandomSampler(RandomSampler):
    """
    StratifiedRandomSampler samples configurations from a given search space using a random state.
    It maintains a set of bins to ensure that the configurations are sampled uniformly based on their parameter count.

    Args:
        config_space: The search space from which to sample.
        seed: Seed for the random number generator. Defaults to None.
        param_bins: The parameter bins that limit the sub-network params in the search.
    """

    def __init__(
        self,
        config_space: dict,
        params_estimator: Callable,
        seed: int | None = None,
        param_bins: ParamBinArgs | None = None,
    ):
        param_bins = param_bins if param_bins is not None else ParamBinArgs()
        super().__init__(config_space, seed=seed)
        self.param_bins = ParamBins(
            self.get_smallest_sub_network(),
            self.get_largest_sub_network(),
            params_estimator,
            num_bins=param_bins.num_bins,
            log_bins=param_bins.log_bins,
            start_bin_size=param_bins.start_bin_size,
            empty_bin_tolerance=param_bins.empty_bin_tolerance,
        )

    def sample(self) -> dict[str, Any]:
        """
        Gets the smallest sub-network configuration from the search space.

        Returns:
            The smallest sub-network configuration.
        """
        while True:
            config = super().sample()

            # find a bin for the config, if not found, continue sampling
            if self.param_bins.put_in_bin(config):
                break

        return config
