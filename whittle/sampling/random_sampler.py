from __future__ import annotations
import warnings

from typing import Any

import numpy as np
from syne_tune.config_space import Categorical, Domain


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
        Gets the smallest sub-network configuration from the search space.

        Returns:
            The smallest sub-network configuration.
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
