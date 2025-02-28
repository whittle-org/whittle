from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from syne_tune.config_space import Categorical, Domain

from whittle.search.search_spaces import SimpleSearchSpace


class BaseSampler:
    def sample(self):
        raise NotImplementedError

    def get_smallest_sub_network(self):
        raise NotImplementedError

    def get_medium_sub_network(self):
        raise NotImplementedError

    def get_largest_sub_network(self):
        raise NotImplementedError


class RandomSampler(BaseSampler):
    """
    RandomSampler samples configurations from a given search space using a random state.

    Args:
        search_space: The search space from which to sample.
        seed: Seed for the random number generator. Defaults to None.
        cast_search_space: Whether to cast the search space. Defaults to True.
    """

    def __init__(
        self,
        search_space: dict[str, Any] | Any,
        seed: int | None = None,
        cast_search_space: bool = True,
    ):
        self.search_space = (
            SimpleSearchSpace(search_space)
            if isinstance(search_space, dict)
            else search_space
        )
        self.rng = np.random.RandomState(seed)
        self.cast_search_space = cast_search_space

    def sample(self) -> dict[str, Any]:
        """
        Gets a random sub-network configuration from the search space.

        Returns:
            A random sub-network configuration.
        """
        config = {}
        for hp_name, hparam in self.search_space.config_space.items():
            if isinstance(hparam, Domain):
                config[hp_name] = hparam.sample(random_state=self.rng)

        return self.search_space.cast(config) if self.cast_search_space else config

    def get_smallest_sub_network(self) -> dict[str, Any]:
        """
        Gets the smallest sub-network configuration from the search space.

        Returns:
            The smallest sub-network configuration.
        """
        config = {}
        for k, v in self.search_space.config_space.items():
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

        return self.search_space.cast(config) if self.cast_search_space else config

    def get_largest_sub_network(self) -> dict[str, Any]:
        """
        gets the largest sub-network configuration from the search space.

        Returns:
            The largest sub-network configuration.
        """

        config = {}
        for k, v in self.search_space.config_space.items():
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

        return self.search_space.cast(config) if self.cast_search_space else config
