from __future__ import annotations

import torch
import torch.nn as nn


class WrappedGPT:
    """

    GPT layer wrapper that enables the calculation and updating of scaling factors for pruning.
    Scaling factors are used to determine the importance of each row in the weight matrix during pruning.

    """

    def __init__(
        self, layer: nn.Module, layer_id: int = 0, layer_name: str = "none"
    ) -> None:
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """

        Computes and updates the L2 norms of the input tensor (row-wise) for each
        batch passed through the layer. These norms serve as scaling factors (`scaler_row`).

        Args:
            inp: Input tensor.
            out: Output tensor.

        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
