import time
import numpy as np

from lobotomy.search.baselines import MethodArguments, methods
from lobotomy.search.ask_tell_scheduler import AskTellScheduler
from lobotomy.search.multi_objective import get_pareto_optimal


def multi_objective_search(
    objective,
    search_space: dict,
    search_strategy: str = "random_search",
    objective_kwargs: dict = None,
    num_samples: int = 100,
    seed: int = None,
):
    """
    Search for the Pareto optimal sub-networks.

    :param objective: the objective function to optimize.
    :param search_space: the search space.
    :param search_strategy: the search strategy.
    :param objective_kwargs: the keyword arguments for the objective function.
    :param num_samples: the number of samples to take.
    :param seed: the random seed.
    :return: the results of the search.
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
        )
    )

    scheduler = AskTellScheduler(base_scheduler=base_scheduler)

    costs = np.empty((num_samples, 2))
    runtime = []
    configs = []
    start_time = time.time()
    for i in range(num_samples):
        trial_suggestion = scheduler.ask()
        objective_1, objective_2 = objective(
            config=trial_suggestion.config, **objective_kwargs
        )

        scheduler.tell(
            trial_suggestion, {"objective_1": objective_1, "objective_2": objective_2}
        )

        # bookkeeping
        costs[i][0] = objective_1
        costs[i][1] = objective_2
        configs.append(trial_suggestion.config)

        runtime.append(time.time() - start_time)
        print(
            f"iteration {i}: objective_1={objective_1} ; objective_2={objective_2}; runtime = {runtime[-1]}"
        )
    idx = get_pareto_optimal(costs)

    results = {
        "costs": costs,
        "configs": configs,
        "runtime": runtime,
        "is_pareto_optimal": idx,
    }
    return results
