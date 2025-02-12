from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from whittle.models.gpt import GPT
from whittle.modules.linear import Linear
from whittle.pruning.pruners.base_pruner import Pruner
from whittle.pruning.utils.layerwrapper import WrappedGPT


class WandaPruner(Pruner):
    def _prune(
        self,
        model: GPT,
        prune_n: int = 2,
        prune_m: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        Prunes a pre-trained model using the WANDA [1] method.

        [1] Sun, M., Liu, Z., Bair, A., and Kolter, J. Z.
        A simple and effective pruning approach for large language models.
        In The Twelfth International Conference on Learning Representations, 2024

        Args:
            model: The model to be pruned.
            prune_n: Number of weights to prune per group.
            prune_m: Total number of weights per group.
            **kwargs: Additional arguments (e.g., 'nsamples' ,dev, dataloader).
        """

        nsamples = kwargs.get("nsamples", 32)
        dataloader: DataLoader = kwargs.get("dataloader")

        use_cache = model.config.use_cache
        model.config.use_cache = False

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = self._prepare_calibration_input(
                model=model,
                dataloader=dataloader,
                dev=kwargs.get("dev", "cuda"),
                nsamples=nsamples,
            )

        layers = model.transformer.h
        for i, layer in enumerate(layers):
            subset = self._find_layers(layer, layers=[Linear])

            wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = [
                subset[name].register_forward_hook(add_batch(name))
                for name in wrapped_layers
            ]

            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        mask=attention_mask,
                        cos=model.cos,
                        sin=model.sin,
                        input_pos=position_ids,
                    )

            for h in handles:
                h.remove()

            for name in subset:
                logging.info(f"Pruning layer {i} name {name}")
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )

                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
                subset[name].weight.data[W_mask] = 0

            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        mask=attention_mask,
                        cos=model.cos,
                        sin=model.sin,
                        input_pos=position_ids,
                    )[0]

            inps, outs = outs, inps

        model.config.use_cache = use_cache
        torch.cuda.empty_cache()
