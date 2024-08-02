from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from syne_tune.config_space import Domain
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.searchers import StochasticSearcher

logger = logging.getLogger(__name__)


MAX_SAMPLES = 1000


@dataclass
class PopulationElement:
    """Internal PBT state tracked per-trial."""

    trial_id: int
    config: dict
    result: dict


class LS(FIFOScheduler):
    """

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param population_size: See
        :class:`~syne_tune.optimizer.schedulers.searchers.RegularizedEvolution`.
        Defaults to 100
    :param sample_size: See
        :class:`~syne_tune.optimizer.schedulers.searchers.RegularizedEvolution`.
        Defaults to 10
    :param random_seed: Random seed, optional
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        metric: list[str],
        mode: list[str] | str = "min",
        start_point: dict[str, Any] | None = None,
        random_seed: int | None = None,
        points_to_evaluate: list[dict] | None = None,
        **kwargs,
    ):
        super().__init__(
            config_space=config_space,
            metric=metric,
            mode=mode,
            searcher=LocalSearch(
                config_space=config_space,
                metric=metric,
                start_point=start_point,
                mode=mode,
                random_seed=random_seed,
                points_to_evaluate=points_to_evaluate,
            ),
            random_seed=random_seed,
            **kwargs,
        )


class LocalSearch(StochasticSearcher):
    """ """

    def __init__(
        self,
        config_space,
        metric: list[str] | str,
        points_to_evaluate: list[dict] | None = None,
        start_point: dict | None = None,
        mode: list[str] | str = "min",
        **kwargs,
    ):
        if start_point is None:
            self.start_point = {
                k: v.sample() if isinstance(v, Domain) else v
                for k, v in config_space.items()
            }
        else:
            self.start_point = start_point

        self._pareto_front: list[PopulationElement] = []

        if points_to_evaluate is None:
            points_to_evaluate = [self.start_point]
        else:
            points_to_evaluate.append(self.start_point)

        super().__init__(
            config_space,
            metric,
            mode=mode,
            points_to_evaluate=points_to_evaluate,
            **kwargs,
        )
        if isinstance(self._mode, list):
            self._metric_op: dict[str, Any] = {
                metric: 1 if mode == "min" else -1
                for metric, mode in zip(metric, self._mode)
            }
        else:
            if self._mode == "min":
                self._metric_op = dict(zip(self._metric, [1.0] * len(self._metric)))
            elif self._mode == "max":
                self._metric_op = dict(zip(self._metric, [-1.0] * len(self._metric)))

    def _sample_random_neighbour(self, start_point):
        # get actual hyperparameters from the search space
        config = deepcopy(start_point)
        hypers = []
        for k, v in self.config_space.items():
            if isinstance(v, Domain):
                hypers.append(k)

        hp_name = np.random.choice(hypers)
        hp = self.config_space[hp_name]
        for i in range(MAX_SAMPLES):
            new_value = hp.sample()
            if new_value != start_point[hp_name]:
                config[hp_name] = new_value
                return config

        # mutation_name = np.random.choice(list(self._mutations.keys()))
        #
        # config = self._mutations[mutation_name](start_point)

        # sample mutation
        # name = np.random.choice(hypers)
        # mutation = self._mutations[name]

        # return mutation(config)

    def is_efficient(self, costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(
                np.any(costs[i + 1 :] > c, axis=1)
            )

        return is_efficient

    def dominates(self, incumbent, neighbour):
        return np.all(neighbour <= incumbent) * np.any(neighbour < incumbent)

    def get_config(self, **kwargs) -> dict | None:
        config = self._next_initial_config()
        if config is not None:
            return config

        if len(self._pareto_front) == 0:
            config = self._sample_random_neighbour(self.start_point)
        else:
            # we sample a random neighbour of one of the elements in the Pareto front
            element = self.random_state.choice(self._pareto_front)
            config = self._sample_random_neighbour(element.config)

        return config

    def _metric_dict(self, reported_results: dict) -> dict:
        return {
            metric: reported_results[metric] * self._metric_op[metric]
            for metric in self._metric
        }

    def _update(self, trial_id: int, config: dict[str, Any], result: dict[str, Any]):
        # assume that the new point is in the Pareto Front
        element = PopulationElement(
            trial_id=trial_id, config=config, result=self._metric_dict(result)
        )

        if len(self._pareto_front) == 0:
            self._pareto_front.append(element)
            return

        pareto_front = deepcopy(self._pareto_front)
        pareto_front.append(element)
        costs = np.array(
            [[v for v in element.result.values()] for element in pareto_front]
        )

        # check for Pareto efficiency
        is_efficient = self.is_efficient(costs)

        self._pareto_front = []
        for i, keep in enumerate(is_efficient):
            if keep:
                self._pareto_front.append(pareto_front[i])

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.scheduler_searcher import (
            TrialSchedulerWithSearcher,
        )

        assert isinstance(
            scheduler, TrialSchedulerWithSearcher
        ), "This searcher requires TrialSchedulerWithSearcher scheduler"
        super().configure_scheduler(scheduler)

    def clone_from_state(self, state: dict[str, Any]):
        raise NotImplementedError


if __name__ == "__main__":
    from nas_fine_tuning.sampling import SmallSearchSpace
    from syne_tune.config_space import Categorical
    from syne_tune.tuner import Trial
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("bert-base-cased")
    ss = SmallSearchSpace(config)

    start_point = {"num_layers": 12, "num_heads": 12, "num_units": 3072}

    ls = LS(
        ss.get_syne_tune_config_space(),
        start_point=start_point,
        metric=["a", "b"],
        random_seed=412,
        mode=["min", "min"],
    )

    def get_default(config_space):
        config = {}
        for k, v in config_space.items():
            if isinstance(v, Domain):
                if isinstance(v, Categorical):
                    config[k] = v.categories[0]
                else:
                    config[k] = v.upper
        return config

    print(get_default(ss.get_syne_tune_config_space()))

    for i in range(10):
        trial = ls.suggest(trial_id=i)
        print(trial)
        # ls._update(trial_id=i, config=config, result={'a': np.random.rand(), 'b':np.random.rand()})
        result = {"a": np.random.rand(), "b": np.random.rand()}
        ls.on_trial_result(
            Trial(trial_id=i, config=trial.config, creation_time=None), result=result
        )
