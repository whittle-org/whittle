from __future__ import annotations


import torch


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(
        self,
        in_features: int,
        dim: int = -1,
        eps: float = 1e-6,
        add_unit_offset: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.weight = torch.nn.Parameter(torch.ones(in_features))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset
        self.sub_network_in_features: int | None = in_features

    def set_sub_network(self, sub_network_in_features: int):
        self.sub_network_in_features = sub_network_in_features

    def reset_super_network(self):
        self.sub_network_in_features = self.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            self.sub_network_in_features is not None
        ), "sub_network_in_features is not set"
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        weight = (
            (1 + self.weight[: self.sub_network_in_features])
            if self.add_unit_offset
            else self.weight[: self.sub_network_in_features]
        )
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
