from __future__ import annotations

import torch
import torch.nn as nn


class Catcher(nn.Module):
    """
    A helper module that intercepts inputs for calibration purposes.

    Args:
        module: The original layer to wrap.
        inps: Tensor to store intercepted inputs.
        cache: A dictionary to store intermediary states.
    """

    def __init__(self, module: nn.Module, inps: torch.Tensor, cache: dict):
        super().__init__()
        self.module = module
        self.inps = inps
        self.cache = cache

    def forward(
        self,
        inp: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> None:
        """
        Forward pass that captures the input data.

        Args:
            inp: Input tensor.
            cos: Rotary embeddings.
            sin: Rotary embeddings.
            mask: Attention mask tensor.
            input_pos: Position IDs.
        """
        self.inps[self.cache["i"]] = inp
        self.cache["i"] += 1
        self.cache["attention_mask"] = mask
        self.cache["position_ids"] = input_pos
        raise ValueError  # Trigger exception to exit
