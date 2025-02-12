from __future__ import annotations

from typing import Any

import torch

from whittle.models.gpt import GPT
from whittle.pruning.pruners.base_pruner import Pruner


class MagnitudePruner(Pruner):
    def __call__(
        self,
        model: GPT,
        prune_n: int = 2,
        prune_m: int = 4,
        **kwargs: Any,
    ) -> float:
        model.reset_super_network()
        total_parameters = self.compute_parameters(model.transformer.h)
        self._prune(model, prune_n, prune_m)
        return self.count_sparse_parameters(model.transformer.h) / total_parameters

    def _prune(
        self,
        model: GPT,
        prune_n: int = 2,
        prune_m: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        Prunes a pre-trained model using magnitude-based structural pruning. For each
        structural component (e.g head) we compute a score based on the sum of the magnitudes of
        its weights.

        Args:
            model: The model to be pruned.
            prune_n: Number of weights to prune per group.
            prune_m: Total number of weights per group.
            **kwargs: Additional arguments specific to Wanda and SparseGPT.

        """

        layers = [model.transformer.h]
        for i in range(len(layers)):
            layer = layers[i]
            subset = self._find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                W_metric = torch.abs(W)
                W_mask = torch.zeros_like(W) == 1
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )

                W[W_mask] = 0
