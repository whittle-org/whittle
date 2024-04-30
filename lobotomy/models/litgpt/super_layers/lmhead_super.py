import torch
import torch.nn as nn
import torch.nn.functional as F

class LMHeadSuper(nn.Linear):
    def __init__(self, super_dim_in:int, output_dim:int, bias:bool=True):
        super().__init__(super_dim_in, output_dim, bias=bias)

        # the largest embed dim
        self.super_dim_in = super_dim_in
        self.output_dim = output_dim

        # the current sampled embed dim
        self.sample_dim_in = None

    def set_sample_config(self, sample_dim_in:int):
        self.sample_dim_in = sample_dim_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias!=None:
            return F.linear(x, self.weight[:, :self.sample_dim_in], self.bias)
        else:
            return F.linear(x, self.weight[:, :self.sample_dim_in])