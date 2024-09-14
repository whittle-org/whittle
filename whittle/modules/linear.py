from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """ """
        super().__init__(in_features, out_features, bias, device, dtype)

        # Set the current sub-network dimensions equal to super-network
        self.sub_network_in_features = in_features
        self.sub_network_out_features = out_features
        self.use_bias = bias
        self.random_indices_in_features = torch.arange(self.sub_network_in_features)
        self.random_indices_out_features = torch.arange(self.sub_network_out_features)

    def set_sub_network(
        self,
        sub_network_in_features: int,
        sub_network_out_features: int,
        sample_random_indices: bool = False,
    ):
        self.sub_network_in_features = sub_network_in_features
        self.sub_network_out_features = sub_network_out_features
        self.random_indices_in_features = torch.arange(self.sub_network_in_features)
        self.random_indices_out_features = torch.arange(self.sub_network_out_features)
        if sample_random_indices and self.sub_network_in_features < self.in_features:
            self.random_indices_in_features = torch.randperm(self.in_features)[
                : self.sub_network_in_features
            ]
        if sample_random_indices and self.sub_network_out_features < self.out_features:
            self.random_indices_out_features = torch.randperm(self.out_features)[
                : self.sub_network_out_features
            ]

    def reset_super_network(self):
        self.sub_network_in_features = self.in_features
        self.sub_network_out_features = self.out_features
        self.random_indices_in_features = torch.arange(self.sub_network_in_features)
        self.random_indices_out_features = torch.arange(self.sub_network_out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bias:
            return F.linear(
                x,
                self.weight[self.random_indices_out_features, :][
                    :, self.random_indices_in_features
                ],
                self.bias[self.random_indices_out_features],
            )
        else:
            return F.linear(
                x,
                self.weight[self.random_indices_out_features, :][
                    :, self.random_indices_in_features
                ],
            )
