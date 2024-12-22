from argparse import Namespace
from typing import Optional
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from whittle.models.gpt import GPT
from whittle.modules.embedding import Embedding
from whittle.modules.linear import Linear
from whittle.prunning.layerwrapper import WrappedGPT
from whittle.prunning.sparsegpt import SparseGPT
from whittle.prunning.catcher import Catcher


class Pruner:
    def __call__(
        self,
        model: GPT,
        prune_n: int = 2,
        prune_m: int = 4,
        args: Optional[Namespace] = None,
        dataloader: Optional[DataLoader] = None,
        dev: str = "cuda",
    ) -> float:
        """
        Generic pruning interface that handles both WANDA and magnitude-based pruning.

        Args:
            model: The model to be pruned
            prune_n: Number of weights to prune per group.
            prune_m: Total number of weights per group.
            args: Namespace for WANDA/SPARSE specific arguments.
            dataloader: Dataloader for WANDA/SPARSE pruning.
            dev: Device to use for computation.

        Returns:
            float: The sparsity ratio of the pruned model
        """
        model.reset_super_network()

        if isinstance(self, WandaPruner) or isinstance(self, SparseGptPruner):
            if args is None:
                raise ValueError("args must be provided for WANDA or SparseGPT pruning")
            total_parameters = self.compute_parameters(model.transformer.h)
            if dataloader is None:
                raise ValueError(
                    "dataloader must be provided for WANDA or SparseGPT pruning"
                )

            self._prune_wanda_sparse(args, model, dataloader, prune_n, prune_m, dev)
            return self.count_sparse_parameters(model.transformer.h) / total_parameters

        else:
            total_parameters = self.compute_parameters(model)
            self._prune_magnitude(model, prune_n, prune_m)
            return self.count_sparse_parameters(model) / total_parameters

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

    def _prune_magnitude(
        self,
        model: GPT,
        prune_n: int = 2,
        prune_m: int = 4,
    ) -> None:
        pass

    def _prune_wanda_sparse(
        self,
        args: Namespace,
        model: GPT,
        dataloader: DataLoader,
        prune_n: int = 2,
        prune_m: int = 4,
        dev: str = "cuda",
    ) -> None:
        pass

    @staticmethod
    def _prepare_calibration_input(args, model, dataloader, device):
        """
        Prepare inputs for calibration during model pruning.
        """
        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.transformer.h
        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (args.nsamples, model.max_seq_length, model.config.n_embd),
            dtype=dtype,
            device=device,
        )
        inps.requires_grad = False
        cache = {"i": 0, "attention_mask": None, "position_ids": None}

        layers[0] = Catcher(layers[0], inps, cache)
        for batch in dataloader:
            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        model(batch[0].to(device))
            except ValueError:
                pass
        layers[0] = layers[0].module

        outs = torch.zeros_like(inps)
        attention_mask = cache["attention_mask"]
        position_ids = cache["position_ids"]
        model.config.use_cache = use_cache

        return inps, outs, attention_mask, position_ids


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


class WandaPruner(Pruner):
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
