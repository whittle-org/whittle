from __future__ import annotations

import logging
from typing import Any

import torch

from whittle.models.gpt import GPT
from whittle.modules.linear import Linear
from whittle.pruning.pruners.base_pruner import Pruner
from whittle.pruning.utils.sparsegpt import SparseGPT


class SparseGPTPruner(Pruner):
    def _prune(
        self,
        model: GPT,
        prune_n: int = 2,
        prune_m: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        Prunes a pre-trained model using the Sparse GPT method by Frantar an Alistarh.

        Frantar, E. and Alistarh, D.
        SparseGPT: Massive language models can be accurately pruned in one-shot.
        arXiv preprint arXiv:2301.00774,

        Args:
            model: The model to be pruned.
            prune_n: Number of weights to prune per group.
            prune_m: Total number of weights per group.
            kwargs: Additional arguments (e.g., 'nsamples',sparsity_ratio ,dev,dataloader).
        """

        nsamples = kwargs.get("nsamples", 32)
        sparsity_ratio = kwargs.get("sparsity_ratio", None)

        inps, outs, attention_mask, position_ids = self._prepare_calibration_input(
            model=model,
            dataloader=kwargs.get("dataloader"),
            dev=kwargs.get("dev", "cuda"),
            nsamples=nsamples,
        )

        layers = model.transformer.h

        for i in range(len(layers)):
            layer = layers[i]

            subset = self._find_layers(layer, layers=[Linear])
            gpts = {}

            for name in subset:
                gpts[name] = SparseGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in gpts:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        mask=attention_mask,
                        cos=model.cos,
                        sin=model.sin,
                        input_pos=position_ids,
                    )[0]

            for h in handles:
                h.remove()

            for name in subset:
                logging.info(f"pruning layer {i} name {name}")

                gpts[name].fasterprune(
                    sparsity_ratio,
                    prune_n=prune_n,
                    prune_m=prune_m,
                    percdamp=0.01,
                    blocksize=128,
                )
                gpts[name].free()

            for j in range(nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    mask=attention_mask,
                    cos=model.cos,
                    sin=model.sin,
                    input_pos=position_ids,
                )[0]

            layers[i] = layer
            torch.cuda.empty_cache()

            inps, outs = outs, inps
