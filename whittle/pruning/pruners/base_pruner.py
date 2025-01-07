from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from whittle.models.gpt import GPT
from whittle.modules.embedding import Embedding
from whittle.modules.linear import Linear
from whittle.pruning.utilis.catcher import Catcher


class Pruner:
    def __call__(
        self,
        model: GPT,
        prune_n: int = 2,
        prune_m: int = 4,
        **kwargs: Any,
    ) -> float:
        """
        Generic pruning interface that handles structural pruning methods, such as WANDA or magnitude-based pruning.

         Args:
             model: The model to be pruned.
             prune_n: Number of weights to prune per group.
             prune_m: Total number of weights per group.
             **kwargs: Additional arguments specific to Wanda and SparseGPT.

         Returns:
             float: The sparsity ratio of the pruned model.
        """

        model.reset_super_network()
        total_parameters = self.compute_parameters(model.transformer.h)
        self._prune(model, prune_n, prune_m, **kwargs)
        return self.count_sparse_parameters(model.transformer.h) / total_parameters

    def _find_layers(
        self,
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
                self._find_layers(
                    child,
                    layers=layers,
                    name=name + "." + name1 if name != "" else name1,
                )
            )
        return res

    @staticmethod
    def count_sparse_parameters(model: GPT) -> float:
        """
        Count the number of non-zero parameters in the model.
        """
        params = 0
        for _, p in model.named_parameters():
            params += torch.sum(p.data != 0).item()
        return float(params)

    @staticmethod
    def compute_parameters(model: GPT) -> int:
        """
        Compute the total number of parameters in the model.
        """
        return sum(p.numel() for p in model.parameters())

    def _prune(
        self,
        model: GPT,
        prune_n: int = 2,
        prune_m: int = 4,
        **kwargs: Any,
    ) -> None:
        pass

    @staticmethod
    def _prepare_calibration_input(
        model: GPT, dataloader: DataLoader, dev: str, nsamples: int
    ):
        """
        Prepare inputs for calibration during model pruning.
        """

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.transformer.h
        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (nsamples, model.max_seq_length, model.config.n_embd),
            dtype=dtype,
            device=dev,
        )
        inps.requires_grad = False
        cache = {"i": 0, "attention_mask": None, "position_ids": None}

        layers[0] = Catcher(layers[0], inps, cache)
        for batch in dataloader:
            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        model(batch[0].to(dev))
            except ValueError:
                pass
        layers[0] = layers[0].module

        outs = torch.zeros_like(inps)
        attention_mask = cache["attention_mask"]
        position_ids = cache["position_ids"]
        model.config.use_cache = use_cache

        return inps, outs, attention_mask, position_ids
