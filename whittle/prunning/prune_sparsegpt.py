from argparse import Namespace

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase


from whittle.prunning.sparsegpt import SparseGPT
from whittle.prunning.data import get_loaders
from whittle.modules.linear import Linear


def find_layers(
    module: nn.Module,
    layers: list[type[nn.Module]] = [Linear],
    name: str = "",
) -> dict[str, nn.Module]:
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module: PyTorch module.
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


@torch.no_grad()
def prune_sparsegpt(
    args: Namespace,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase = None,
    dev: torch.device = torch.device("cuda:0"),
    prune_n: int = 0,
    prune_m: int = 0,
) -> None:
    """
    Prune the model using the WANDA method.
    SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa

    Args:
        args: Arguments for pruning.
        model : The model to be pruned.
        tokenizer: Tokenizer for the model.
        dev  : Device to perform pruning on.
        prune_n : Number of weights to prune per group.
        prune_m : Total number of weights per group.
    """

    print("Starting ...")
    args.sparsity_ratio = None
    dataloader, _ = get_loaders(
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.max_seq_length,
        tokenizer=tokenizer,
    )
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

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, cos, sin, mask, input_pos):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = mask
            cache["position_ids"] = input_pos
            raise ValueError

    layers[0] = Catcher(layers[0])
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

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

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

        for name in gpts:
            print(i, name)
            print("Pruning ...")

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
