from __future__ import annotations

from collections.abc import Callable
from typing import Any

from whittle.args import ParamBinArgs
from whittle.metrics.parameters import (
    compute_parameters,
)
from whittle.sampling.param_bins import ParamBins
from whittle.sampling.random_sampler import RandomSampler


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
        search_space,
        params_estimator: Callable,
        seed: int | None = None,
        param_bins: ParamBinArgs | None = None,
        max_tries: int = 10000,
        cast_search_space: bool = True,
    ):
        param_bins = param_bins if param_bins is not None else ParamBinArgs()
        super().__init__(search_space, seed=seed, cast_search_space=cast_search_space)
        self.param_bins = ParamBins(
            self.get_smallest_sub_network(),
            self.get_largest_sub_network(),
            params_estimator,
            num_bins=param_bins.num_bins,
            log_bins=param_bins.log_bins,
            start_bin_size=param_bins.start_bin_size,
            empty_bin_tolerance=param_bins.empty_bin_tolerance,
        )

        self.max_tries = max_tries

    def sample(self) -> dict[str, Any]:
        """
        Gets the smallest sub-network configuration from the search space.

        Returns:
            The smallest sub-network configuration.
        """
        tries = 0
        while True:
            config = super().sample()

            # find a bin for the config, if not found, continue sampling
            if self.param_bins.put_in_bin(config):
                break

            if tries > self.max_tries:
                raise ValueError(
                    "Could not find a valid configuration in StratifiedRandomSampler. Try increasing max_tries or increasing empty_bin_tolerance."
                )

        return self.search_space.cast(config) if self.cast_search_space else config


class FixedParamGridSampler(RandomSampler):
    def __init__(
        self,
        search_space,
        num_configs: int = 21,
        n_trials: int = 5000,
        seed: int | None = None,
        cast_search_space: bool = True,
    ):
        super().__init__(search_space, seed=seed, cast_search_space=cast_search_space)
        self.search_space = search_space
        self.n_trials = n_trials
        self.seed = seed

        self.args = ParamBinArgs()
        self.args.empty_bin_tolerance = 0
        self.args.num_bins = num_configs
        self.args.start_bin_size = 1

        self.grid: list[dict[str, Any]] = []

    def initialize_grid(self, model):
        sampler = StratifiedRandomSampler(
            self.search_space,
            params_estimator=lambda config: self.get_parameters(model, config),
            seed=self.seed,
            param_bins=self.args,
            max_tries=self.n_trials,
            cast_search_space=self.cast_search_space,
        )

        # add smallest/largest sub-networks, update param bins from outside
        self.grid.append(self.get_smallest_sub_network())
        self.grid.append(self.get_largest_sub_network())

        sampler.param_bins.bins[0] += 1
        sampler.param_bins.bins[-1] += 1

        # fill in the intermediate parameter bins (1 network per bin)
        for _ in range(self.args.num_bins - 2):
            self.grid.append(sampler.sample())

        # sort grid by number of parameters
        self.grid_params = [self.get_parameters(model, config) for config in self.grid]
        self.grid = [config for _, config in sorted(zip(self.grid_params, self.grid))]
        self.grid_params.sort()

    from whittle.models.gpt import GPT

    def get_parameters(self, model: GPT, config):
        if self.cast_search_space:
            model.set_sub_network(**config)
        else:
            model.select_sub_network(config)

        params = compute_parameters(model)
        model.reset_super_network()
        return params

    def sample(self):
        return self.rng.choice(self.grid)

    def get_smallest_sub_network(self):
        return self.grid[0]

    def get_medium_sub_network(self):
        return self.grid[len(self.grid) // 2]

    def get_largest_sub_network(self):
        return self.grid[-1]
