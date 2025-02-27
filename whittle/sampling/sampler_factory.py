from __future__ import annotations

import pickle

import numpy as np
from syne_tune.config_space import Categorical, Domain

from whittle.metrics.parameters import (
    compute_all_parameters,
    compute_parameters,
)
from whittle.sampling.random_sampler import WhittleRandomSampler


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
    def __init__(self, search_space, seed: int | None = None):
        self.search_space = search_space
        self.sampler = WhittleRandomSampler(self.search_space.config_space, seed=seed)

    def sample(self):
        return self.search_space.cast(self.sampler.sample())

    def get_smallest_sub_network(self):
        return self.search_space.cast(self.sampler.get_smallest_sub_network())

    def get_medium_sub_network(self):
        config = {}
        for hp_name, hp in self.search_space.config_space.items():
            if isinstance(hp, Categorical):
                config[hp_name] = hp.categories[len(hp.categories) // 2]
            else:
                upper = hp.upper
                lower = hp.lower
                config[hp_name] = int(0.5 * (upper - lower) + lower)
        return self.search_space.cast(config)

    def get_largest_sub_network(self):
        return self.search_space.cast(self.sampler.get_largest_sub_network())


class FixParamGridSampler(BaseSampler):
    def __init__(
        self,
        search_space,
        num_configs: int = 21,
        n_trials=5000,
        seed: int | None = None,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.rng = np.random.RandomState(seed)
        self.sampler = WhittleRandomSampler(self.search_space.config_space, seed=seed)
        self.values = [(i) / num_configs for i in range(num_configs + 1)]
        print(self.values)
        self.grid = []

    def add_max_config(self):
        config = {}
        for hp_name, hp in self.search_space.config_space.items():
            if isinstance(hp, Categorical):
                config[hp_name] = max(hp.categories)
            else:
                u = hp.upper
                l = hp.lower
                config[hp_name] = int(self.values[-1] * (u - l) + l)
        self.grid.append(config)

    def add_min_config(self):
        config = {}
        for hp_name, hp in self.search_space.config_space.items():
            if isinstance(hp, Categorical):
                config[hp_name] = min(hp.categories)
            else:
                u = hp.upper
                l = hp.lower
                config[hp_name] = int(self.values[0] * (u - l) + l)
        self.grid.append(config)

    def initialize_grid(self, model):
        model.reset_super_network()
        u = compute_all_parameters(model)
        l = self.get_smallest_params(model)
        params_min = l
        self.add_min_config()
        for value in self.values[1:]:
            params_max = int(value * (u - l) + l)
            config = self.constrained_search(params_min, params_max, model)
            if config is not None:
                self.grid.append(config)
            params_min = params_max
        self.add_max_config()

    def constrained_search(self, params_min, params_max, model):
        for _ in range(self.n_trials):
            config = self.sampler.sample()
            model.set_sub_network(**self.search_space.cast(config))
            params = compute_parameters(model)
            model.reset_super_network()
            if params >= params_min and params < params_max:
                # print(params)
                return config

    def get_smallest_params(self, model):
        config = {}
        for hp_name, hp in self.search_space.config_space.items():
            if isinstance(hp, Categorical):
                config[hp_name] = min(hp.categories)
            else:
                config[hp_name] = hp.lower
        model.set_sub_network(**self.search_space.cast(config))
        params = compute_parameters(model)
        model.reset_super_network()
        return params

    def sample(self):
        return self.search_space.cast(self.rng.choice(self.grid))

    def get_smallest_sub_network(self):
        return self.search_space.cast(self.grid[0])

    def get_medium_sub_network(self):
        return self.search_space.cast(self.grid[len(self.grid) // 2])

    def get_largest_sub_network(self):
        return self.search_space.cast(self.grid[-1])


def get_sampler(sampler_type, search_space, seed, num_configs, n_trials):
    if sampler_type == "random":
        return RandomSampler(search_space=search_space, seed=seed)
    elif sampler_type == "grid-params":
        return FixParamGridSampler(
            search_space=search_space,
            seed=42,
            n_trials=n_trials,
            num_configs=num_configs,
        )
    else:
        raise ValueError(f"Sampler type {sampler_type} not recognised")
