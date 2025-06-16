from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from lightning.fabric.loggers import Logger
from tqdm import tqdm

from syne_tune.optimizer.schedulers.ask_tell_scheduler import AskTellScheduler

from whittle.sampling.param_bins import ParamBins
from whittle.search.baselines import MethodArguments, methods
from whittle.search.multi_objective import get_pareto_optimal


def multi_objective_search(
    objective: Callable[..., Any],
    search_space: dict,
    search_strategy: str = "random_search",
    num_samples: int = 100,
    objective_kwargs: dict[str, Any] | None = None,
    logger: Logger | None = None,
    seed: int | None = None,
    param_bins: ParamBins | None = None,
    objective_1_name: str = "objective_1",
    objective_2_name: str = "objective_2",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Search for the Pareto-optimal sub-networks using the specified strategy.

    Args:
        objective: The objective function to optimize.
        search_space: The search space for the optimization.
        search_strategy: The search strategy to use.
            Defaults to "random_search".
        num_samples: The number of samples to evaluate.
            Defaults to 100.
        objective_kwargs: Keyword arguments for the objective function.
            Defaults to None.
        logger: The lightning logger to send metrics to.
            Defaults to None.
        seed: The random seed for reproducibility.
            Defaults to None.
        param_bins: The parameter bins that limit the sub-network params in the search.
            The configs from ask() are rejected if they fit into a bin that is full.
            The bin size is increased if all bins are full.
            Defaults to None.
        objective_1_name: The name of the first objective.
            Defaults to "objective_1".
        objective_2_name: The name of the second objective.
            Defaults to "objective_2".
        verbose: Whether to have a verbose tqdm output.
    Returns:
        The results of the search, including Pareto-optimal solutions.

    """
    metrics = ["objective_1", "objective_2"]
    if seed is None:
        seed = np.random.randint(0, 1000000)

    assert search_strategy in methods

    base_scheduler = methods[search_strategy](
        MethodArguments(
            config_space=search_space,
            metrics=metrics,
            mode=["min", "min"],
            random_seed=seed,
            param_bins=param_bins,
        )
    )

    scheduler = AskTellScheduler(base_scheduler=base_scheduler)

    costs = np.empty((num_samples, 2))
    runtime: list[float] = []
    configs: list[dict[str, Any]] = []
    start_time = time.time()

    for i in tqdm(range(num_samples), disable=not verbose):
        trial_suggestion = scheduler.ask()

        objective_1, objective_2 = objective(
            trial_suggestion.config, **(objective_kwargs or {})
        )

        scheduler.tell(
            trial_suggestion, {"objective_1": objective_1, "objective_2": objective_2}
        )

        # bookkeeping
        costs[i][0] = float(objective_1)
        costs[i][1] = float(objective_2)
        configs.append(trial_suggestion.config)

        runtime.append(time.time() - start_time)

        observation = {
            "iteration": i,
            objective_1_name: float(objective_1),
            objective_2_name: float(objective_2),
            "runtime": runtime[-1],
        }

        if logger is not None:
            logger.log_metrics(observation)
    idx = get_pareto_optimal(costs)

    results = {
        "costs": costs,
        "configs": configs,
        "runtime": runtime,
        "is_pareto_optimal": [bool(i) for i in idx],
    }
    return results
