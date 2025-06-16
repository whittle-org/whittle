from __future__ import annotations

from dataclasses import dataclass

from syne_tune.config_space import Categorical, Domain
from syne_tune.optimizer.baselines import (
    RandomSearch,
)
from syne_tune.optimizer.schedulers.multiobjective.multi_objective_regularized_evolution import MultiObjectiveRegularizedEvolution
from syne_tune.optimizer.schedulers.multiobjective.expected_hyper_volume_improvement import \
    ExpectedHyperVolumeImprovement
from syne_tune.optimizer.schedulers.multiobjective.linear_scalarizer import (
    LinearScalarizedScheduler,
)
from syne_tune.optimizer.schedulers.single_fidelity_scheduler import SingleFidelityScheduler
from syne_tune.optimizer.schedulers.searchers.conformal.conformal_quantile_regression_searcher import ConformalQuantileRegression

from whittle.sampling.param_bins import ParamBins
from whittle.search.local_search import LocalSearch
from whittle.search.stratified_search import StratifiedRandomSearcher


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
    param_bins: ParamBins | None = None


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
    LS = "local_search"
    LSBO = "lsbo"
    MOREA = "morea"
    EHVI = "ehvi"
    MOASHA = "moasha"
    SRS = "stratified_random_search"


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(
        config_space=method_arguments.config_space,
        metrics=method_arguments.metrics,
        do_minimize=method_arguments.mode[0]=='min',
        random_seed=method_arguments.random_seed,
        points_to_evaluate=initial_design(method_arguments.config_space),
    ),
    Methods.SRS: lambda method_arguments: SingleFidelityScheduler(
        config_space=method_arguments.config_space,
        metrics=method_arguments.metrics,
        do_minimize=method_arguments.mode == 'min',
        searcher=StratifiedRandomSearcher(config_space=method_arguments.config_space,
                                        random_seed=method_arguments.random_seed,
                                        param_bins=method_arguments.param_bins,
                                        points_to_evaluate=initial_design(method_arguments.config_space)
                                        ),
        random_seed=method_arguments.random_seed,
        searcher_kwargs=None,
    ),
    Methods.LS: lambda method_arguments: SingleFidelityScheduler(
        config_space=method_arguments.config_space,
        metrics=method_arguments.metrics,
        do_minimize=method_arguments.mode == 'min',
        searcher=LocalSearch(config_space=method_arguments.config_space,
                                                random_seed=method_arguments.random_seed,
                                                points_to_evaluate=initial_design(method_arguments.config_space)
                                                ),
        random_seed=method_arguments.random_seed,
        searcher_kwargs=None,
    ),
    Methods.LSBO: lambda method_arguments: LinearScalarizedScheduler(
        config_space=method_arguments.config_space,
        metrics=method_arguments.metrics,
        do_minimize=method_arguments.mode[0] == 'min',
        random_seed=method_arguments.random_seed,
        searcher=ConformalQuantileRegression(config_space=method_arguments.config_space,
                                             random_seed=method_arguments.random_seed,
                                             points_to_evaluate=initial_design(method_arguments.config_space)),
    ),
    Methods.EHVI: lambda method_arguments: SingleFidelityScheduler(
        config_space=method_arguments.config_space,
        metrics=method_arguments.metrics,
        do_minimize=method_arguments.mode == 'min',
        searcher=ExpectedHyperVolumeImprovement(config_space=method_arguments.config_space,
                                                random_seed=method_arguments.random_seed,
                                                points_to_evaluate=initial_design(method_arguments.config_space)
                                                ),
        random_seed=method_arguments.random_seed,
        searcher_kwargs=None,
    ),
    Methods.MOREA: lambda method_arguments: SingleFidelityScheduler(
        config_space=method_arguments.config_space,
        metrics=method_arguments.metrics,
        do_minimize=method_arguments.mode == 'min',
        searcher=MultiObjectiveRegularizedEvolution(config_space=method_arguments.config_space,
                                                random_seed=method_arguments.random_seed,
                                                points_to_evaluate=initial_design(method_arguments.config_space)
                                                ),
        random_seed=method_arguments.random_seed,
        searcher_kwargs=None,
    )
}
