from __future__ import annotations

from whittle.sampling.grid_samplers import FixedParamGridSampler, StratifiedRandomSampler
from whittle.sampling.random_sampler import RandomSampler


class Samplers:
    RANDOM = "random"
    STRATIFIED_RANDOM = "stratified_random"
    GRID_PARAMS = "grid_params"


def get_sampler(sampler_type, search_space, seed, num_configs, n_trials, **kwargs):
    if sampler_type == Samplers.RANDOM:
        return RandomSampler(search_space=search_space, seed=seed, **kwargs)
    elif sampler_type == Samplers.STRATIFIED_RANDOM:
        return StratifiedRandomSampler(
            search_space=search_space,
            seed=seed,
            n_trials=n_trials,
            num_configs=num_configs,
            **kwargs,
        )
    elif sampler_type == Samplers.GRID_PARAMS:
        return FixedParamGridSampler(
            search_space=search_space,
            seed=seed,
            n_trials=n_trials,
            num_configs=num_configs,
            **kwargs,
        )
    else:
        raise ValueError(f"Sampler type {sampler_type} not recognised")
