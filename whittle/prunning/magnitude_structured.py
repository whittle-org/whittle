import torch
import torch.nn as nn

from whittle.modules.embedding import Embedding
from whittle.modules.linear import Linear


def find_layers(
    module: nn.Module,
    layers: list[type[nn.Module]] = [Linear, Embedding],
    name: str = "",
) -> dict[str, nn.Module]:
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module: The model with the layers to be found.
        layers: List of layer types to find.
        name: Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def prune_magnitude(model: nn.Module, prune_n: int = 0, prune_m: int = 0) -> None:
    """
    Prune the model using magnitude-based pruning.

    Args:
        model: The model to be pruned.
        prune_n: Number of weights to prune per group.
        prune_m: Total number of weights per group.
    """
    layers = [model.transformer, model.lm_head]

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            W_mask = torch.zeros_like(W) == 1
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:, ii : (ii + prune_m)].float()
                    W_mask.scatter_(
                        1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True
                    )

            W[W_mask] = 0
