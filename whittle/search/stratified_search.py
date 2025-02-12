from __future__ import annotations

from typing import Any, Dict, Union, List, Optional
from syne_tune.config_space import Domain
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.searchers.random_grid_searcher import RandomSearcher

from whittle.sampling.param_bins import ParamBins


class StratifiedRandomSearch(FIFOScheduler):
    """
    Stratified Random Search (SRS) is a search strategy that samples configurations
    uniformly at random from the search space. It is a simple baseline for
    hyperparameter optimization.

    Args:
        config_space: The configuration space to sample from.
        metric: The metric to optimize.
        param_bins: The parameter bins that limit the sub-network params in the search.
            The configs from ask() are rejected if they fit into a bin that is full.
            The bin size is increased if all bins are full.
        mode: The optimization mode for the metric.
            Defaults to "min".
        start_point: Optional. The starting point for the search.
            Defaults to None.
        random_seed: Optional. The random seed for reproducibility.
            Defaults to None.
        points_to_evaluate: Optional. The initial configurations to evaluate.
            Defaults to None.
        **kwargs: Additional arguments for the scheduler.
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        metric: list[str],
        param_bins: ParamBins,
        mode: list[str] | str = "min",
        start_point: dict[str, Any] | None = None,
        random_seed: int | None = None,
        points_to_evaluate: list[dict] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config_space=config_space,
            metric=metric,
            mode=mode,
            searcher=StratifiedRandomSearcher(
                config_space=config_space,
                metric=metric,
                start_point=start_point,
                mode=mode,
                random_seed=random_seed,
                points_to_evaluate=points_to_evaluate,
                param_bins=param_bins,
            ),
            random_seed=random_seed,
            **kwargs,
        )


class StratifiedRandomSearcher(RandomSearcher):
    """
    Searcher which randomly samples configurations to try next.

    Additional arguments on top of parent class :class:`StochasticAndFilterDuplicatesSearcher`:

    :param debug_log: If ``True``, debug log printing is activated.
        Logs which configs are chosen when, and which metric values are
        obtained. Defaults to ``False``
    :param resource_attr: Optional. Key in ``result`` passed to :meth:`_update`
        for resource value (for multi-fidelity schedulers)
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: Union[List[str], str],
        param_bins: ParamBins,
        **kwargs,
    ):
        super().__init__(
            config_space,
            metric=metric,
            **kwargs,
        )
        self.param_bins = param_bins

    def _get_config(self, **kwargs) -> Optional[dict]:
        """Sample a new configuration at random. If it doesn't fit into bins of
        already sampled configurations, continue sampling until a valid config is found.

        If ``allow_duplicates == False``, this is done without replacement, so
        previously returned configs are not suggested again.

        :param trial_id: Optional. Used for ``debug_log``
        :return: New configuration, or None
        """
        while True:
            config = super()._get_config(**kwargs)

            # find a bin for the config, if not found, continue sampling
            if self.param_bins.put_in_bin(config):
                break

        return config

    def clone_from_state(self, state: Dict[str, Any]):
        new_searcher = StratifiedRandomSearcher(
            self.config_space,
            metric=self._metric,
            points_to_evaluate=[],
            debug_log=self._debug_log,
            allow_duplicates=self._allow_duplicates,
            param_bins=self.param_bins,
        )
        new_searcher._resource_attr = self._resource_attr
        new_searcher._restore_from_state(state)
        return new_searcher
