from __future__ import annotations

import numpy as np
import numpy.typing as npt


def get_pareto_optimal(costs: np.ndarray) -> npt.NDArray[np.bool_]:
    """
    Find the pareto-optimal point.

    Args:
        costs: array containing the costs for each objective asscoiated with each point (n_points, 2).

    Returns:
        (n_points, 1) indicator if point is on pareto front or not.

    """
    assert isinstance(costs, np.ndarray)
    assert costs.ndim == 2

    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self
    return is_pareto
