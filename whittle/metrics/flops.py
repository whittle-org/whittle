from __future__ import annotations

import os
from typing import Literal

try:
    import deepspeed
except ImportError:
    raise ImportError(
        "DeepSpeed not installed. Please install whittle with the "
        "`distributed` dependency group: `pip install whittle[distributed]"
    )

import torch

from whittle.models.gpt import GPT


def compute_flops(
    model: GPT,
    batch_size: int = 1,
    sequence_length: int = 512,
    metric: Literal["flops", "macs"] = "flops",
    previous_device: str | None = None,
) -> float:
    """
    Estimates the number of floating-point operations (FLOPs) or multiply-accumulate operations (MACs) for a GPT model.

    This function uses DeepSpeed's FlopsProfiler to estimate the FLOPs or MACs of the model's forward pass.

    Args:
        model: The GPT model to profile.
        batch_size: The batch size for the input tensor. Defaults to 1.
        sequence_length: The sequence length for the input tensor. Defaults to 512.
        metric: The metric to return. Either "flops" for floating-point operations or "macs" for multiply-accumulate operations. Defaults to "flops".
        previous_device: The device to cast to after profiling. If None, the device is not changed. Defaults to None.

    Returns:
        The estimated number of floating-point operations (FLOPs) or multiply-accumulate operations (MACs) for the model's forward pass, depending on the specified metric.
    """

    from deepspeed.accelerator.cpu_accelerator import CPU_Accelerator
    from deepspeed.profiling.flops_profiler import get_model_profile

    input_tensor = torch.randint(
        0, model.config.padded_vocab_size, (batch_size, sequence_length)
    )

    model.eval()
    model.to("cpu")

    os.environ["DS_ACCELERATOR"] = "CPU"
    deepspeed.accelerator.set_accelerator(CPU_Accelerator())

    flops, macs, _ = get_model_profile(
        model=model,
        args=(input_tensor,),
        print_profile=False,
        detailed=False,
        warm_up=1,
        as_string=False,
    )

    if previous_device is not None:
        model.to(previous_device)

    if metric == "flops":
        return flops
    else:
        return macs
