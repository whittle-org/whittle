import torch
from litgpt.model import GPT


from whittle.baselines.magnitude_structured import prune_magnitude
from whittle.baselines.prune_sparsegpt import prune_sparsegpt
from whittle.baselines.wanda_structured import prune_wanda


class Pruner:
    def __init__(self, args=None):
        self.args = args

    def __call__(
        self,
        model: GPT,
        prune_n=2,
        prune_m=4,
        prune_methode="magnitude",
        tokenizer=None,
    ) -> float:
        total_parameters = self.compute_parameters(model)
        self.prune(
            model, prune_n, prune_m, prune_methode=prune_methode, tokenizer=tokenizer
        )
        return self.count_sparse_parameters(model) / total_parameters

    def prune(
        self,
        model: GPT,
        prune_n=2,
        prune_m=4,
        prune_methode="magnitude",
        tokenizer=None,
        dev="cpu",
    ):
        pass

    @staticmethod
    def count_sparse_parameters(model: GPT):
        params = 0
        for _, p in model.named_parameters():
            params += torch.sum(p.data != 0).item()
        return float(params)

    @staticmethod
    def compute_parameters(model: GPT):
        return sum(p.numel() for p in model.parameters())


class NMPruner(Pruner):
    def prune(
        self,
        model: GPT,
        prune_n=2,
        prune_m=4,
        prune_methode="magnitude",
        tokenizer=None,
        dev="cpu",
    ):
        print(f"Pruning {prune_n} out of {prune_m} weights per group")
        print(f"Pruning method: {prune_methode}")

        if prune_methode == "magnitude":
            prune_magnitude(model, prune_n, prune_m)

        elif prune_methode == "wanda":
            prune_wanda(self.args, model, tokenizer, device=dev, prune_n=2, prune_m=4)

        elif prune_methode == "sparse":
            prune_sparsegpt(self.args, model, tokenizer, dev=dev, prune_n=2, prune_m=4)

        else:
            raise NotImplementedError(
                f"Pruning method {prune_methode} not implemented!"
            )
