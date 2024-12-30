from argparse import Namespace
import logging

import torch
from torch.utils.data import DataLoader

from whittle.models.gpt import GPT
from whittle.modules.linear import Linear
from whittle.prunning.pruners.base_pruner import Pruner
from whittle.prunning.utilis.sparsegpt import SparseGPT
from whittle.prunning.utilis.catcher import Catcher


class SparseGptPruner(Pruner):
    def _prune_wanda_sparse(
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

        args.sparsity_ratio = None
        model.config.use_cache = False
        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.transformer.h

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (args.nsamples, model.max_seq_length, model.config.n_embd),
            dtype=dtype,
            device=dev,
        )
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
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache["attention_mask"]
        position_ids = cache["position_ids"]

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

            for j in range(args.nsamples):
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
                    args.sparsity_ratio,
                    prune_n=prune_n,
                    prune_m=prune_m,
                    percdamp=0.01,
                    blocksize=128,
                )
                gpts[name].free()
            for j in range(args.nsamples):
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
        model.config.use_cache = use_cache
        torch.cuda.empty_cache()
