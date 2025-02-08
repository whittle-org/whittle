from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    """An extension of PyTorch's torch.nn.Linear with flexible input and output dimensionality corresponding to sub-network"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)

        # Set the current sub-network dimensions equal to super-network
        self.sub_network_in_features = in_features
        self.sub_network_out_features = out_features
        self.use_bias = bias

    def set_sub_network(
        self, sub_network_in_features: int, sub_network_out_features: int
    ):
        """Set the linear transformation dimensions of the current sub-network."""
        self.sub_network_in_features = sub_network_in_features
        self.sub_network_out_features = sub_network_out_features

    def reset_super_network(self):
        """Reset the linear transformation dimensions of the current sub-network to the super-network dimensionality."""
        self.sub_network_in_features = self.in_features
        self.sub_network_out_features = self.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bias:
            return F.linear(
                x,
                self.weight[
                    : self.sub_network_out_features, : self.sub_network_in_features
                ],
                self.bias[: self.sub_network_out_features],
            )
        else:
            return F.linear(
                x,
                self.weight[
                    : self.sub_network_out_features, : self.sub_network_in_features
                ],
            )


class LinearQKV(nn.Linear):
    """An extension of Linear to support QKV Indexing"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)

        # Set the current sub-network dimensions equal to super-network
        self.sub_network_in_features = in_features
        self.sub_network_out_features = out_features
        self.use_bias = bias
        self.qkv_indices = None

    def set_sub_network(
        self,
        sub_network_in_features: int,
        sub_network_out_features: int,
        qkv_indices: torch.Tensor,
    ):
        """Set the linear transformation dimensions of the current sub-network."""
        self.sub_network_in_features = sub_network_in_features
        self.sub_network_out_features = sub_network_out_features
        self.qkv_indices = qkv_indices

    def reset_super_network(self):
        """Reset the linear transformation dimensions of the current sub-network to the super-network dimensionality."""
        self.sub_network_in_features = self.in_features
        self.sub_network_out_features = self.out_features
        self.qkv_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bias:
            if self.qkv_indices is not None:
                return F.linear(
                    x,
                    self.weight[self.qkv_indices, : self.sub_network_in_features],
                    self.bias[self.qkv_indices],
                )
            else:
                return F.linear(
                    x,
                    self.weight[:, : self.sub_network_in_features],
                    self.bias,
                )
        else:
            if self.qkv_indices is not None:
                return F.linear(
                    x,
                    self.weight[self.qkv_indices, : self.sub_network_in_features],
                )
            else:
                return F.linear(
                    x,
                    self.weight[:, : self.sub_network_in_features],
                )


class LinearProj(nn.Linear):
    """An extension of Linear to support Projection Indexing"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)

        # Set the current sub-network dimensions equal to super-network
        self.sub_network_in_features = in_features
        self.sub_network_out_features = out_features
        self.use_bias = bias
        self.proj_indices = None

    def set_sub_network(
        self,
        sub_network_in_features: int,
        sub_network_out_features: int,
        proj_indices: torch.Tensor,
    ):
        """Set the linear transformation dimensions of the current sub-network."""
        self.sub_network_in_features = sub_network_in_features
        self.sub_network_out_features = sub_network_out_features
        self.proj_indices = proj_indices

    def reset_super_network(self):
        """Reset the linear transformation dimensions of the current sub-network to the super-network dimensionality."""
        self.sub_network_in_features = self.in_features
        self.sub_network_out_features = self.out_features
        self.proj_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bias:
            if self.proj_indices is not None:
                return F.linear(
                    x,
                    self.weight[: self.sub_network_out_features, self.proj_indices],
                    self.bias[: self.sub_network_out_features],
                )
            else:
                return F.linear(
                    x,
                    self.weight[: self.sub_network_out_features, :],
                    self.bias[: self.sub_network_out_features],
                )
        else:
            if self.proj_indices is not None:
                return F.linear(
                    x,
                    self.weight[: self.sub_network_out_features, self.proj_indices],
                )
            else:
                return F.linear(
                    x,
                    self.weight[: self.sub_network_out_features, :],
                )
