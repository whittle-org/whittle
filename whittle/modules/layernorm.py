from __future__ import annotations

import torch
import torch.nn.functional as F


class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, in_features: int, eps: float = 1e-5):
        super().__init__(in_features, eps)
        self.in_features = in_features

        # Set current sub-network to super-network
        self.sub_network_in_features = self.in_features
        self.random_indices = torch.arange(self.sub_network_in_features)

    def set_sub_network(
        self, sub_network_in_features: int, sample_random_indices: bool = False
    ):
        self.sub_network_in_features = sub_network_in_features
        self.random_indices = torch.arange(self.sub_network_in_features)
        if sample_random_indices and self.sub_network_in_features < self.in_features:
            self.random_indices = torch.randperm(self.in_features)[
                : self.sub_network_in_features
            ]

    def reset_super_network(self):
        self.sub_network_in_features = self.in_features
        self.random_indices = torch.arange(self.sub_network_in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            (self.sub_network_in_features,),
            weight=self.weight[self.random_indices],
            bias=self.bias[self.random_indices],
            eps=self.eps,
        )
