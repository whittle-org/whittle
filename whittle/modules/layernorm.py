from __future__ import annotations

import torch
import torch.nn.functional as F


class LayerNorm(torch.nn.LayerNorm):
    """An extension of PyTorch's `torch.nn.LayerNorm` with support  with support to sub-sample weights corresponding to the sub-network dimensionality."""

    def __init__(self, in_features: int, eps: float = 1e-5):
        super().__init__(in_features, eps)
        self.in_features = in_features

        # Set current sub-network to super-network
        self.sub_network_in_features = self.in_features
        self.sampled_ln_indices: list[int] | None = None

    def set_sub_network(
        self, sub_network_in_features: int, sampled_ln_indices: list[int] | None = None
    ):
        """Set the input dimensionality of the current sub-network."""
        self.sub_network_in_features = sub_network_in_features
        self.sampled_ln_indices = sampled_ln_indices

    def reset_super_network(self):
        """Reset the input dimensionality of the current sub-network to the super-network dimensionality."""
        self.sub_network_in_features = self.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sampled_ln_indices is None:
            weight = self.weight[: self.sub_network_in_features]
            bias = self.bias[: self.sub_network_in_features]
        else:
            weight = self.weight[self.sampled_ln_indices]
            bias = self.bias[self.sampled_ln_indices]

        return F.layer_norm(
            x,
            (self.sub_network_in_features,),
            weight=weight,
            bias=bias,
            eps=self.eps,
        )
