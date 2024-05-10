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
        """

        """
        super().__init__(in_features, out_features, bias, device, dtype)

        # the current sub-network dimensions
        self.sub_network_in_features = None
        self.sub_network_out_features = None
        self.use_bias = bias

    def set_sub_network(
        self,
        sub_network_in_features: int = None,
        sub_network_out_features: int = None,
    ):
        assert sub_network_in_features is not None or sub_network_out_features is not None, "sub_network_in_features or sub_network_out_features must be set"
        self.sub_network_in_features = sub_network_in_features
        if sub_network_out_features is not None:
           self.sub_network_out_features = sub_network_out_features
        else:
            self.sub_network_out_features = self.out_features

    def reset_super_network(self):
        self.sub_network_in_features = self.in_features
        self.sub_network_out_features = self.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.sub_network_in_features is not None, "sub_network_in_features is not set"
        assert self.sub_network_out_features is not None, "sub_network_out_features is not set"
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
