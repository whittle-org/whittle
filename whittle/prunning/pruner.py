from argparse import Namespace
from typing import Optional

import torch
from transformers import PreTrainedTokenizerBase

from whittle.models.gpt import GPT

from whittle.prunning.magnitude_structured import prune_magnitude
from whittle.prunning.prune_sparsegpt import prune_sparsegpt
from whittle.prunning.wanda_structured import prune_wanda


class Pruner:
    def __init__(self, args: Optional[Namespace] = None):
        """
        Initialize the Pruner with arguments, necessary.

        Args:
            args: Additional arguments for wanda or sparsed methods.
                  Arguments specified are nsamples, seed, batch_size.
        """
        self.args = args

    def __call__(
        self,
        model: GPT,
        prune_n: int = 2,
        prune_m: int = 4,
        prune_methode: str = "magnitude",
        dev: str = "cuda",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> float:
        """
        Prune the model and return the sparsity ratio.

        Args:
            model: The model to be pruned.
            prune_n: Number of weights to prune per group.
            prune_m: Total number of weights per group.
            prune_methode: Pruning method to use.
            tokenizer: Tokenizer for the model, used with sparse and wanda methods.

        Returns:
            float: The sparsity ratio of the pruned model.
        """
        total_parameters = self.compute_parameters(model)
        self.prune(
            model,
            prune_n,
            prune_m,
            prune_methode=prune_methode,
            tokenizer=tokenizer,
            dev=dev,
        )
        return self.count_sparse_parameters(model) / total_parameters

    def prune(
        self,
        model: GPT,
        prune_n: int = 2,
        prune_m: int = 4,
        prune_methode: str = "magnitude",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        dev: str = "cuda",
    ) -> None:
        """
        Prune the model using the specified method.
        """
        pass

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


class NMPruner(Pruner):
    def prune(
        self,
        model: GPT,
        prune_n: int = 2,
        prune_m: int = 4,
        prune_methode: str = "magnitude",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        dev: str = "cuda",
    ) -> None:
        """
        Prune the model using the specified method.

        Args:
            model: The model to be pruned.
            prune_n: Number of weights to prune per group.
            prune_m: Total number of weights per group.
            prune_methode : Pruning method to use.
            tokenizer: Tokenizer for the model, used with sparse and wanda methods.
            dev: Device to perform pruning on.
        """

        if prune_methode == "magnitude":
            prune_magnitude(model, prune_n, prune_m)

        elif prune_methode == "wanda":
            if self.args is None:
                raise ValueError("`args` must be provided for 'wanda' pruning.")
            prune_wanda(self.args, model, tokenizer, device=dev, prune_n=2, prune_m=4)

        elif prune_methode == "sparse":
            if self.args is None:
                raise ValueError("`args` must be provided for 'sparse' pruning.")
            prune_sparsegpt(self.args, model, tokenizer, dev=dev, prune_n=2, prune_m=4)

        else:
            raise NotImplementedError(
                f"Pruning method {prune_methode} not implemented!"
            )
