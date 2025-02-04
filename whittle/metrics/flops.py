from __future__ import annotations

from typing import Literal
import torch
from torch.profiler import profile, ProfilerActivity
from litgpt.model import GPT


def _count_op_flops(event, debug: bool = False) -> float:
    """
    Calculate FLOPs for a single operator based on its name and input shapes.
    """
    flops = 0.0
    name = event.key

    try:
        # Get shapes from event's input argument list
        if hasattr(event, "input_shapes"):
            shapes = event.input_shapes
        else:
            return 0.0

        if "aten::linear" in name:
            # For linear layer: input @ weight.t() + bias
            # Shape: (N, in_features) @ (out_features, in_features).t()
            input_shape = shapes[0]  # (N, in_features)
            weight_shape = shapes[1]  # (out_features, in_features)
            if len(input_shape) >= 2 and len(weight_shape) == 2:
                batch_size = input_shape[0]
                in_features = input_shape[1]
                out_features = weight_shape[0]
                # multiply-adds for matrix multiplication, plus adds for bias
                flops = batch_size * (2 * in_features * out_features + out_features)

        elif "aten::matmul" in name or "aten::mm" in name or "aten::addmm" in name:
            if len(shapes) >= 2:
                shape1, shape2 = shapes[:2]
                if len(shape1) >= 2 and len(shape2) >= 2:
                    # For matmul: (B, M, K) @ (B, K, N) -> (B, M, N)
                    M = shape1[-2]
                    K = shape1[-1]
                    N = shape2[-1]
                    batch_size = (
                        torch.prod(torch.tensor(shape1[:-2])).item()
                        if len(shape1) > 2
                        else 1
                    )
                    flops = batch_size * (2 * M * N * K)  # multiply-adds

        elif "aten::native_layer_norm" in name:
            if shapes:
                # Count operations for mean, variance, normalization, scale, and bias
                numel = torch.prod(torch.tensor(shapes[0])).item()
                flops = 8 * numel  # 2 passes + normalization + scale & bias

        elif any(
            op in name for op in ["aten::add", "aten::sub", "aten::mul", "aten::div"]
        ):
            if shapes:
                # For elementwise ops, count one operation per element
                shape = shapes[0]
                numel = torch.prod(torch.tensor(shape)).item()
                flops = numel

        elif "aten::gelu" in name:
            if shapes:
                # Approximate GELU as 4 FLOPs per element (multiply, add, tanh/sigmoid)
                numel = torch.prod(torch.tensor(shapes[0])).item()
                flops = 4 * numel

    except Exception as e:
        if debug:
            print(f"Error processing {name}: {e}")
        return 0.0

    return flops


def compute_flops(
    model: GPT,
    batch_size: int = 1,
    sequence_length: int = 512,
    metric: Literal["flops", "macs"] = "flops",
    debug: bool = False,
) -> float:
    """
    Estimates the number of floating-point operations (FLOPs) or multiply-accumulate operations (MACs) for a GPT model.

    This function uses PyTorch's profiler to track operations and calculate their FLOPs based on input shapes.

    Args:
        model: The GPT model to profile.
        batch_size: The batch size for the input tensor. Defaults to 1.
        sequence_length: The sequence length for the input tensor. Defaults to 512.
        metric: The metric to return. Either "flops" for floating-point operations or "macs" for multiply-accumulate operations.
               For "macs", the function returns flops/2 as each MAC operation consists of one multiply and one add. Defaults to "flops".
        debug: If True, prints debug information about operations. Defaults to False.

    Returns:
        The estimated number of floating-point operations (FLOPs) or multiply-accumulate operations (MACs) for the model's forward pass.
    """
    input_tensor = torch.randint(
        0, model.config.padded_vocab_size, (batch_size, sequence_length)
    )

    model.eval()

    # Warm-up run
    with torch.no_grad():
        model(input_tensor)

    total_flops = 0.0

    # Profile run
    with (
        torch.no_grad(),
        profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof,
    ):
        model(input_tensor)

    # Calculate FLOPs for each operator
    events = prof.key_averages()
    if debug:
        print("\nOperations found:")
        for event in events:
            op_flops = _count_op_flops(event, debug)
            if op_flops > 0:
                print(
                    f"Op: {event.key}, Shapes: {event.input_shapes}, FLOPs: {op_flops}"
                )
            else:
                print(f"Op: {event.key}, Shapes: {event.input_shapes}")

    for event in events:
        flops = _count_op_flops(event, debug)
        if debug and flops > 0:
            print(f"Adding {flops} FLOPs from {event.key}")
        total_flops += flops

    if metric == "flops":
        return total_flops
    else:  # metric == "macs"
        return total_flops / 2  # Each MAC is one multiply and one add
