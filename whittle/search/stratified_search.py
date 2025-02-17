from __future__ import annotations

from typing import Any

from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.searchers import (
    LegacyRandomSearcher as RandomSearcher,
)

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
    Searcher which randomly samples configurations to try next. If a configuration
    gets in a full bin (we already sampled enough configurations with a similar parameter count),
    it is rejected and a new configuration is sampled.
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        metric: list[str] | str,
        param_bins: ParamBins,
        sample_patience: int = 10000,
        **kwargs,
    ):
        """
        Args:
            config_space: The configuration space to sample from.
            metric: The metric to optimize.
            param_bins: The parameter bins that limit the sub-network params in the search.
                The configs from ask() are rejected if they fit into a bin that is full.
                The bin size is increased if all bins are full.
            sample_patience: The number of rejected samples to try before raising an error.
                Defaults to 10000.
            **kwargs: Additional arguments for the searcher.
        """
        super().__init__(
            config_space,
            metric=metric,
            **kwargs,
        )
        self.param_bins = param_bins
        self.sample_patience = sample_patience

    def _get_config(self, **kwargs) -> dict | None:
        """Sample a new configuration at random. If it doesn't fit into bins of
        already sampled configurations, continue sampling until a valid config is found.

        If ``allow_duplicates == False``, this is done without replacement, so
        previously returned configs are not suggested again.

        :param trial_id: Optional. Used for ``debug_log``
        :return: New configuration, or None
        """
        i = 0
        while True:
            config = super()._get_config(**kwargs)

            # find a bin for the config, if not found, continue sampling
            if self.param_bins.put_in_bin(config):
                break
            i += 1
            if i >= self.sample_patience:
                raise ValueError(
                    f"Could not find a valid configuration after {self.sample_patience} samples. Try increasing the tolerance for parameter bins not filled to the max."
                )

        return config

    def clone_from_state(self, state: dict[str, Any]):
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
