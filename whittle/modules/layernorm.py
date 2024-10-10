from __future__ import annotations

import torch
import torch.nn.functional as F


class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, in_features: int, eps: float = 1e-5):
        super().__init__(in_features, eps)
        self.in_features = in_features

        # Set current sub-network to super-network
        self.sub_network_in_features = self.in_features

    def set_sub_network(self, sub_network_in_features: int):
        self.sub_network_in_features = sub_network_in_features

    def reset_super_network(self):
        self.sub_network_in_features = self.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            (self.sub_network_in_features,),
            weight=self.weight[: self.sub_network_in_features],
            bias=self.bias[: self.sub_network_in_features],
            eps=self.eps,
        )
