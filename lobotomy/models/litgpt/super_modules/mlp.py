import torch
import torch.nn as nn
from typing import Any, Optional, Tuple
from lobotomy.models.litgpt.config import Config
from lobotomy.models.litgpt.super_layers.linear_super import SuperLinear


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = SuperLinear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = SuperLinear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.config = config

    def set_sample_config(self, sample_embed_dim: int, sample_intermediate_size: int) -> None:
        self.fc.set_sample_config(sample_embed_dim, sample_intermediate_size)
        self.proj.set_sample_config(sample_intermediate_size, sample_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(self.fc.sample_dim_in)
        #print(self.fc.sample_dim_out)
        x = self.fc(x)
        #print("after fc",torch.sum(x))
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        #print("After gelu",torch.sum(x))
        return self.proj(x)
    

class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = SuperLinear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = SuperLinear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = SuperLinear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def set_sample_config(self, sample_embed_dim: int, sample_intermediate_size: int) -> None:
        self.fc_1.set_sample_config(sample_embed_dim, sample_intermediate_size)
        self.fc_2.set_sample_config(sample_embed_dim, sample_intermediate_size)
        self.proj.set_sample_config(sample_intermediate_size, sample_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)
    

class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x)