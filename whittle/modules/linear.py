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
        self.sampled_in_indices: list[int] | None = None
        self.sampled_out_indices: list[int] | None = None

    def set_sub_network(
        self,
        sub_network_in_features: int,
        sub_network_out_features: int,
        sampled_in_indices: list[int] | None = None,
        sampled_out_indices: list[int] | None = None,
    ):
        """Set the linear transformation dimensions of the current sub-network."""
        self.sub_network_in_features = sub_network_in_features
        self.sub_network_out_features = sub_network_out_features
        self.sampled_in_indices = sampled_in_indices
        self.sampled_out_indices = sampled_out_indices

    def reset_super_network(self):
        """Reset the linear transformation dimensions of the current sub-network to the super-network dimensionality."""
        self.sub_network_in_features = self.in_features
        self.sub_network_out_features = self.out_features
        self.sampled_in_indices = None
        self.sampled_out_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_idx = (
            self.sampled_out_indices
            if self.sampled_out_indices is not None
            else slice(0, self.sub_network_out_features)
        )
        in_idx = (
            self.sampled_in_indices
            if self.sampled_in_indices is not None
            else slice(0, self.sub_network_in_features)
        )

        W = self.weight[out_idx][:, in_idx]
        b = self.bias[out_idx] if self.use_bias else None

        return F.linear(x, W, b)


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
        qkv_indices=None,
        sub_network_n_head=None,
        sub_network_query_groups=None,
        sub_network_head_size=None,
        sub_network_q_per_kv=None,
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
