from __future__ import annotations

from typing import Any

from whittle.args import ParamBinArgs
from whittle.sampling.param_bins import ParamBins
from whittle.sampling.random_sampler import RandomSampler


class StratifiedRandomSampler(RandomSampler):
    """
    StratifiedRandomSampler samples configurations from a given search space using a random state.
    It maintains a set of bins to ensure that the configurations are sampled uniformly based on their parameter count.

    Args:
        search_space: The search space from which to sample.
        seed: Seed for the random number generator. Defaults to None.
        param_bins: The parameter bins that limit the sub-network params in the search.
        cast_search_space: Whether to cast the search space to config aligned with arguments of GPT.set_sub_network(). Defaults to True.
    """

    def __init__(
        self,
        search_space: dict[str, Any] | Any,
        seed: int | None = None,
        param_bins: ParamBinArgs | None = None,
        max_tries: int = 10000,
        cast_search_space: bool = True,
    ):
        self.param_bins_args = param_bins if param_bins is not None else ParamBinArgs()
        super().__init__(search_space, seed=seed, cast_search_space=cast_search_space)

        self.param_bins = None
        self.max_tries = max_tries

    def initialize_param_bins(self, model):
        self.param_bins = ParamBins(
            self.get_smallest_sub_network(),
            self.get_largest_sub_network(),
            params_func=lambda config: self.get_parameters(model, config),
            num_bins=self.param_bins_args.num_bins,
            log_bins=self.param_bins_args.log_bins,
            start_bin_size=self.param_bins_args.start_bin_size,
            empty_bin_tolerance=self.param_bins_args.empty_bin_tolerance,
        )

    def sample(self) -> dict[str, Any]:
        """
        Gets the smallest sub-network configuration from the search space.

        Returns:
            The smallest sub-network configuration.
        """
        assert self.param_bins is not None, (
            "StratifiedRandomSampler.param_bins must be initialized first."
        )
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
        return config


class FixedParamGridSampler(RandomSampler):
    """
    FixedParamGridSampler creates a fixed grid of sampled configurations, each at a different parameter range. It samples
    uniformly just as StratifiedRandomSampler.

    Args:
        search_space: The search space from which to sample.
        num_configs: The number of configurations to sample for the grid.
        n_trials: The number of trials to sample before an error is thrown (not possible to find a network in a param range).
        seed: Seed for the random number generator. Defaults to None.
        cast_search_space: Whether to cast the search space. Defaults to True.
    """

    def __init__(
        self,
        search_space: dict[str, Any] | Any,
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
            seed=self.seed,
            param_bins=self.args,
            max_tries=self.n_trials,
            cast_search_space=self.cast_search_space,
        )

        sampler.initialize_param_bins(model)

        # add smallest/largest sub-networks, update param bins from outside
        self.grid.append(sampler.get_smallest_sub_network())
        self.grid.append(sampler.get_largest_sub_network())

        sampler.param_bins.bins[0] += 1
        sampler.param_bins.bins[-1] += 1

        # fill in the intermediate parameter bins (1 network per bin)
        for _ in range(self.args.num_bins - 2):
            self.grid.append(sampler.sample())

        # sort grid by number of parameters
        self.grid_params = [self.get_parameters(model, config) for config in self.grid]
        self.grid = [config for _, config in sorted(zip(self.grid_params, self.grid))]
        self.grid_params.sort()

    def sample(self):
        return self.rng.choice(self.grid)

    def get_smallest_sub_network(self):
        return self.grid[0]

    def get_medium_sub_network(self):
        return self.grid[len(self.grid) // 2]

    def get_largest_sub_network(self):
        return self.grid[-1]
