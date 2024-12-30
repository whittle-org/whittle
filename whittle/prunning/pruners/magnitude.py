import torch

from whittle.models.gpt import GPT
from whittle.prunning.pruners.base_pruner import Pruner


class MagnitudePruner(Pruner):
    def _prune_magnitude(self, model: GPT, prune_n: int = 2, prune_m: int = 4) -> None:
        """
        Prune the model usign magnitude-based pruning.

        Args:
            model: The model to be pruned.
            prune_n: Number of weights to prune per group.
            prune_m: Total number of weights per group.

        """
        layers = [model.transformer, model.lm_head]
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
