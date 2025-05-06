from __future__ import annotations

from collections.abc import Callable

import torch
from lightning.fabric.utilities.throughput import measure_flops

from whittle.models.gpt import GPT


def compute_flops(
    model: GPT,
    batch_size: int = 1,
    sequence_length: int = 512,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    device: str = "cpu",
    previous_device: str | None = None,
    verbose: bool = False,
) -> float:
    """
    Estimates the number of floating-point operations (FLOPs) for a GPT model using PyTorch Lightning.

    This function uses Lightning's measure_flops utility to estimate the FLOPs of the model's forward pass
    on a specified device. The model will be temporarily moved to the given device if not already there.
    After profiling, it will be moved back to the previous device if specified.

    Args:
        model: The GPT model to profile.
        batch_size: The batch size for the input tensor.
        sequence_length: The sequence length for the input tensor.
        loss_fn: Optional loss function that takes model output and target tensors.
        device: The device on which to run the FLOPs calculation ("cpu", "cuda", etc.).
        previous_device: Optional device to move the model back to after profiling.
        verbose: If True, prints debug information about profiling.

    Returns:
        The estimated number of floating-point operations (FLOPs) for the model's forward pass.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available.")

    original_device = next(model.parameters()).device
    if original_device.type == "meta":
        raise RuntimeError(
            "Model is on 'meta' device; cannot run FLOPs profiling without real weights."
        )

    if verbose:
        print(f"[FLOPs] Profiling on device: {device}")
        print(
            f"[FLOPs] Model: {model.__class__.__name__}, Batch size: {batch_size}, Seq length: {sequence_length}"
        )
        print(f"[FLOPs] Original device: {original_device}, Target device: {device}")

    if str(original_device) != device:
        model.to(device)

    input_tensor = torch.randint(
        0, model.config.padded_vocab_size, (batch_size, sequence_length), device=device
    )

    def forward_fn() -> torch.Tensor:
        return model(input_tensor)

    model_loss: Callable[[torch.Tensor], torch.Tensor] | None = None
    if loss_fn is not None:
        dummy_targets = torch.randint(
            0,
            model.config.padded_vocab_size,
            (batch_size, sequence_length),
            device=device,
        )

        def model_loss(y: torch.Tensor) -> torch.Tensor:
            return loss_fn(y, dummy_targets)

    try:
        flops = measure_flops(model=model, forward_fn=forward_fn, loss_fn=model_loss)
    except Exception as e:
        raise RuntimeError(f"Failed to compute FLOPs: {e}")

    if previous_device is not None:
        model.to(previous_device)
    elif str(original_device) != device:
        model.to(original_device)

    if verbose:
        print(f"[FLOPs] Estimated: {flops:.2e}")

    return flops
