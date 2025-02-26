from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from litgpt.lora import LoRALayer

from whittle.modules.linear import Linear, LinearProj


class LoRALinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs: Any,
    ):
        """LoRA wrapper around linear class.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        # call super for both
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.linear = Linear(in_features, out_features, **kwargs)

        self.use_bias = self.linear.use_bias
        self.in_features = in_features
        self.out_features = out_features
        self.sub_network_in_features = in_features
        self.sub_network_out_features = out_features
        self.merged: bool = False
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty((r, in_features)))
            self.lora_B = nn.Parameter(torch.empty((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def set_sub_network(
        self,
        sub_network_in_features: int,
        sub_network_out_features: int,
    ):
        self.sub_network_in_features = sub_network_in_features
        self.sub_network_out_features = sub_network_out_features
        self.linear.set_sub_network(sub_network_in_features, sub_network_out_features)

    def reset_super_network(self):
        self.sub_network_in_features = self.in_features
        self.sub_network_out_features = self.out_features
        self.linear.set_sub_network(
            self.sub_network_in_features, self.sub_network_out_features
        )

    def reset_parameters(self) -> None:
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def get_lora_AB(self) -> torch.Tensor:
        """Return merged lora_A and lora_B matrices with the same shape as the pretrained weights."""
        return (
            self.lora_B[: self.sub_network_out_features, :]
            @ self.lora_A[:, : self.sub_network_in_features]
        ) * self.scaling

    def merge(self) -> None:
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and not self.merged:
            pretrained_dtype = self.linear.weight.data.dtype
            lora_data = self.get_lora_AB()
            # if only the pretrained are in quantized form - dequantize, sum with LoRA and quantize the result
            if pretrained_dtype == torch.uint8:
                import bitsandbytes as bnb

                weight = self.linear.weight
                # dequantize the pretrained weights
                weight_data = bnb.functional.dequantize_4bit(
                    weight.data, weight.quant_state
                ).to(lora_data.dtype)
                # add pretrained and LoRA weights
                weight_data += lora_data
                # assign updated weights and quantize by moving to CUDA device
                self.linear.weight = bnb.nn.Params4bit(
                    weight_data, requires_grad=False, **weight.__dict__
                )
                self.linear.weight.cuda(weight.device)
            else:
                # self.linear might be on CPU and lora_data on CUDA
                # the inplace add will preserve the dtype of linear.weight
                self.linear.weight.data += lora_data.to(
                    device=self.linear.weight.data.device
                )
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if weights are merged or rank is less or equal to zero (LoRA is disabled) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or self.merged:
            return pretrained
        lora = (
            self.lora_dropout(x)
            @ self.lora_A[:, : self.sub_network_in_features].transpose(0, 1)
            @ self.lora_B[: self.sub_network_out_features, :].transpose(0, 1)
        ) * self.scaling
        return pretrained + lora


class LoRALinearProj(LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = LinearProj(in_features, out_features, **kwargs)
        self.use_bias = self.linear.use_bias
        self.in_features = in_features
        self.out_features = out_features
        self.sub_network_in_features = in_features
        self.sub_network_out_features = out_features
        self.proj_indices = None
        self.merged = False
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty((r, in_features)))
            self.lora_B = nn.Parameter(torch.empty((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def set_sub_network(
        self, sub_network_in_features, sub_network_out_features, proj_indices=None
    ):
        self.sub_network_in_features = sub_network_in_features
        self.sub_network_out_features = sub_network_out_features
        self.linear.set_sub_network(
            sub_network_in_features, sub_network_out_features, proj_indices
        )
        self.proj_indices = proj_indices

    def reset_super_network(self):
        self.sub_network_in_features = self.in_features
        self.sub_network_out_features = self.out_features
        self.linear.reset_super_network()
        self.proj_indices = None

    def reset_parameters(self) -> None:
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def get_lora_AB(self) -> torch.Tensor:
        """Return merged lora_A and lora_B matrices with the same shape as the pretrained weights."""
        return (
            self.lora_B[: self.sub_network_out_features, :]
            @ self.lora_A[:, : self.sub_network_in_features]
        ) * self.scaling

    def merge(self) -> None:
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and not self.merged:
            pretrained_dtype = self.linear.weight.data.dtype
            lora_data = self.get_lora_AB()
            # if only the pretrained are in quantized form - dequantize, sum with LoRA and quantize the result
            if pretrained_dtype == torch.uint8:
                import bitsandbytes as bnb

                weight = self.linear.weight
                # dequantize the pretrained weights
                weight_data = bnb.functional.dequantize_4bit(
                    weight.data, weight.quant_state
                ).to(lora_data.dtype)
                # add pretrained and LoRA weights
                weight_data += lora_data
                # assign updated weights and quantize by moving to CUDA device
                self.linear.weight = bnb.nn.Params4bit(
                    weight_data, requires_grad=False, **weight.__dict__
                )
                self.linear.weight.cuda(weight.device)
            else:
                # self.linear might be on CPU and lora_data on CUDA
                # the inplace add will preserve the dtype of linear.weight
                self.linear.weight.data += lora_data.to(
                    device=self.linear.weight.data.device
                )
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if weights are merged or rank is less or equal to zero (LoRA is disabled) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or self.merged:
            return pretrained
        if self.proj_indices is not None:
            lora = (
                self.lora_dropout(x)
                @ self.lora_A[:, self.proj_indices].transpose(0, 1)
                @ self.lora_B[: self.sub_network_out_features, :].transpose(0, 1)
            ) * self.scaling
        else:
            lora = (
                self.lora_dropout(x)
                @ self.lora_A[:, : self.sub_network_in_features].transpose(0, 1)
                @ self.lora_B[: self.sub_network_out_features, :].transpose(0, 1)
            ) * self.scaling
        return pretrained + lora
