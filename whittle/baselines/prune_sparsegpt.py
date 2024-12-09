import torch
import torch.nn as nn


from whittle.baselines.sparsegpt import SparseGPT
from whittle.baselines.data import get_loaders


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

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
def prune_sparsegpt(args, model, tokenizer, dev="cuda", prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print("Starting ...")
    args.sparsity_ratio = None
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.max_seq_length,
        tokenizer=tokenizer,
    )
    model.config.use_cache = False
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    # if "model.embed_tokens" in model.hf_device_map:
    #    dev = model.hf_device_map["model.embed_tokens"]

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
        # Only for llama
        # if f"model.layers.{i}" in model.hf_device_map:
        #    dev = model.hf_device_map[f"model.layers.{i}"]
        #    print(f"layer {i} device {dev}")
        #    inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

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
