from __future__ import annotations

from dataclasses import dataclass

from syne_tune.config_space import Categorical, Domain
from syne_tune.optimizer.baselines import (
    MOREA,
    NSGA2,
    MORandomScalarizationBayesOpt,
    RandomSearch,
)

# EHVI
from syne_tune.optimizer.schedulers.multiobjective.linear_scalarizer import (
    LinearScalarizedScheduler,
)

from whittle.search.local_search import LS


def get_random(config_space):
    config = {}
    for k, v in config_space.items():
        if isinstance(v, Domain):
            config[k] = v.sample()
    return config


def get_lower_bound(config_space):
    config = {}
    for k, v in config_space.items():
        if isinstance(v, Domain):
            if isinstance(v, Categorical):
                config[k] = v.categories[0]
            else:
                config[k] = v.lower
    return config


def get_upper_bound(config_space):
    config = {}
    for k, v in config_space.items():
        if isinstance(v, Domain):
            if isinstance(v, Categorical):
                config[k] = v.categories[-1]
            else:
                config[k] = v.upper
    return config


def get_mid_point(config_space):
    config = {}
    for i, (k, v) in enumerate(config_space.items()):
        if isinstance(v, Domain):
            if isinstance(v, Categorical):
                if i < len(config_space.keys()) // 2:
                    config[k] = v.categories[0]
                else:
                    config[k] = v.categories[-1]
            else:
                config[k] = (v.upper - v.lower) // 2
    return config


@dataclass
class MethodArguments:
    config_space: dict
    metrics: list
    mode: list
    random_seed: int


def initial_design(config_space):
    points_to_evaluate = []
    upper_bound = get_upper_bound(config_space)
    points_to_evaluate.append(upper_bound)
    lower_bound = get_lower_bound(config_space)
    points_to_evaluate.append(lower_bound)
    mid_point = get_mid_point(config_space)
    points_to_evaluate.append(mid_point)
    return points_to_evaluate


class Methods:
    RS = "random_search"
    MOREA = "morea"
    LS = "local_search"
    NSGA2 = "nsga2"
    LSBO = "lsbo"
    RSBO = "rsbo"
    EHVI = "ehvi"
    MOASHA = "moasha"


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics[0],
        mode=method_arguments.mode[0],
        random_seed=method_arguments.random_seed,
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    Methods.MOREA: lambda method_arguments: MOREA(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        sample_size=5,
        population_size=10,
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    Methods.LS: lambda method_arguments: LS(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        start_point=None,
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    Methods.NSGA2: lambda method_arguments: NSGA2(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        population_size=10,
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    Methods.LSBO: lambda method_arguments: LinearScalarizedScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        searcher="bayesopt",
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    Methods.RSBO: lambda method_arguments: MORandomScalarizationBayesOpt(
        config_space=method_arguments.config_space,
        metric=method_arguments.metrics,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    # Methods.MOASHA: lambda method_arguments: MOASHA(
    #     config_space=method_arguments.config_space,
    #     metrics=method_arguments.metrics,
    #     mode=method_arguments.mode,
    #     time_attr="epoch",
    #     max_t=method_arguments.config_space["num_train_epochs"],
    #     grace_period=1,
    #     reduction_factor=3,
    #     brackets=1,
    # random_seed=method_arguments.random_seed,
    # points_to_evaluate=initial_design(method_arguments.config_space),
    # ),
    # Methods.EHVI: lambda method_arguments: EHVI(
    #     config_space=method_arguments.config_space,
    #     metric=method_arguments.metrics,
    #     mode=method_arguments.mode,
    #     random_seed=method_arguments.random_seed,
    #     points_to_evaluate=initial_design(method_arguments.config_space),
    # ),
}
