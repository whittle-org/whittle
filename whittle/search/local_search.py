from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from syne_tune.config_space import Domain
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)

logger = logging.getLogger(__name__)


MAX_SAMPLES = 1000


@dataclass
class PopulationElement:
    """Internal PBT state tracked per-trial."""

    trial_id: int
    config: dict
    result: dict


class LocalSearch(SingleObjectiveBaseSearcher):
    """
    Local Search algorithm for hyperparameter optimization.

    This searcher uses a local search strategy to explore the configuration space.
    It extends the StochasticSearcher and used searcher input parameter in LS scheduler.

    Args:
        config_space: Configuration space for the evaluation function.
        points_to_evaluate: Initial points to evaluate. Defaults to None.
        start_point: Starting point for the search. Defaults to None.
        random_seed: Seed for the random number generator.
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        points_to_evaluate: list[dict] | None = None,
        start_point: dict | None = None,
        random_seed: int | None = None,
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
            points_to_evaluate=points_to_evaluate,
            random_seed=random_seed,
        )
        self.random_state = np.random.RandomState(self.random_seed)

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

    def is_efficient(self, costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(
                np.any(costs[i + 1 :] > c, axis=1)
            )

        return is_efficient

    def dominates(self, incumbent, neighbour):
        return np.all(neighbour <= incumbent) * np.any(neighbour < incumbent)

    def suggest(self, **kwargs) -> dict | None:
        config = self._next_points_to_evaluate()

        if config is not None:
            return config

        if len(self._pareto_front) == 0:
            config = self._sample_random_neighbour(self.start_point)
        else:
            # we sample a random neighbour of one of the elements in the Pareto front
            element = self.random_state.choice(self._pareto_front)
            config = self._sample_random_neighbour(element.config)

        return config

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
