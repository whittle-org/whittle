from argparse import Namespace
import logging

import torch
from torch.utils.data import DataLoader

from whittle.prunning.pruners.base_pruner import Pruner
from whittle.models.gpt import GPT
from whittle.modules.linear import Linear
from whittle.prunning.utilis.layerwrapper import WrappedGPT


class WandaPruner(Pruner):
    def _prune(
        self,
        args: Namespace,
        model: GPT,
        dataloader: DataLoader,
        prune_n: int = 2,
        prune_m: int = 4,
        dev: str = "cuda",
    ) -> None:
        """
        Prune the model using Sparse Prunning.

        Args:
            args: Namespace arguments containing nsamples, seed, and batch_size.
            model: The model to be pruned
            dataloader: Dataloader for WANDA pruning
            prune_n: Number of weights to prune per group
            prune_m: Total number of weights per group
            dev: Device to use for computation
        """
        use_cache = model.config.use_cache
        model.config.use_cache = False

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = self._prepare_calibration_input(
                args, model, dataloader, dev
            )

        layers = model.transformer.h
        for i in range(len(layers)):
            layer = layers[i]

            subset = self._find_layers(layer, layers=[Linear])

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(args.nsamples):
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
                logging.info(f"pruning layer {i} name {name}")
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )

                W_mask = torch.zeros_like(W_metric) == 1
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
                subset[name].weight.data[W_mask] = 0

            for j in range(args.nsamples):
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
