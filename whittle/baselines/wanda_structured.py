import torch
import torch.nn as nn


from whittle.baselines.layerwrapper import WrappedGPT
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


def prepare_calibration_input(args, model, dataloader, device):
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
                    model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def prune_wanda(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0
):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.max_seq_length,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            args, model, dataloader, device
        )

    layers = model.transformer.h
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

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
                )  # attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:, ii : (ii + prune_m)].float()
                    W_mask.scatter_(
                        1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True
                    )

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

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
