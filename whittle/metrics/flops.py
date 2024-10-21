from __future__ import annotations

from typing import Literal

import torch
from deepspeed.profiling.flops_profiler import get_model_profile

from whittle.models.gpt import GPT


def estimate_flops(
    model: GPT,
    use_cuda: bool = False,
    batch_size: int = 1,
    sequence_length: int = 512,
    detailed: bool = False,
    metric: Literal["flops", "macs"] = "flops",
) -> float:
    """
    Estimates the number of floating-point operations (FLOPs) for a GPT model.

    This function uses DeepSpeed's FlopsProfiler to estimate the FLOPs of the model's forward pass.
    It supports both CPU and CUDA profiling.

    Args:
        model (GPT): The GPT model to profile.
        use_cuda (bool, optional): If True and CUDA is available, the model will be moved to the GPU for profiling. Defaults to False.
        batch_size (int, optional): The batch size for the input tensor. Defaults to 1.
        sequence_length (int, optional): The sequence length for the input tensor. Defaults to 512.
        detailed (bool, optional): If True, prints a detailed profile of the model. Defaults to False.

    Returns:
        float: The estimated number of floating-point operations (FLOPs) for the model's forward pass.
    """
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()

    elif use_cuda and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    input_tensor = torch.randint(
        0, model.config.padded_vocab_size, (batch_size, sequence_length)
    )

    if use_cuda and torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    model.eval()

    flops, macs, _ = get_model_profile(
        model=model,
        args=(input_tensor,),
        print_profile=detailed,
        detailed=detailed,
        warm_up=1,
        as_string=False,
    )

    if metric == "flops":
        return flops
    else:
        return macs
