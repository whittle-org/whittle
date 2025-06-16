from __future__ import annotations

from typing import Any, Optional, List, Dict


from whittle.sampling.param_bins import ParamBins

from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher

class StratifiedRandomSearcher(BaseSearcher):
    """
    Searcher which randomly samples configurations to try next. If a configuration
    gets in a full bin (we already sampled enough configurations with a similar parameter count),
    it is rejected and a new configuration is sampled.
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        param_bins: ParamBins,
        sample_patience: int = 10000,
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        random_seed: int = None,
    ):
        """
        Args:
            config_space: The configuration space to sample from.
            param_bins: The parameter bins that limit the sub-network params in the search.
                The configs from ask() are rejected if they fit into a bin that is full.
                The bin size is increased if all bins are full.
            sample_patience: The number of rejected samples to try before raising an error.
                Defaults to 10000.
            points_to_evaluate: Initial points to evaluate. Defaults to None.
            random_seed: Seed for the random number generator.
        """
        super().__init__(
            config_space,
            points_to_evaluate=points_to_evaluate,
            random_seed=random_seed,
        )
        self.param_bins = param_bins
        self.sample_patience = sample_patience

    def suggest(self) -> dict | None:
        """Sample a new configuration at random. If it doesn't fit into bins of
        already sampled configurations, continue sampling until a valid config is found.

        If ``allow_duplicates == False``, this is done without replacement, so
        previously returned configs are not suggested again.

        :return: New configuration
        """

        config = self._next_points_to_evaluate()
        if config is not None:
            return config
        i = 0
        while True:
            config =  {
                k: v.sample() if hasattr(v, "sample") else v for k, v in self.config_space.items()
            }

            # find a bin for the config, if not found, continue sampling
            if self.param_bins.put_in_bin(config):
                break
            i += 1
            if i >= self.sample_patience:
                raise ValueError(
                    f"Could not find a valid configuration after {self.sample_patience} samples. Try increasing the tolerance for parameter bins not filled to the max."
                )

        return config
