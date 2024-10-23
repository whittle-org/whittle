from __future__ import annotations


import litgpt
import torch
from litgpt import Config

from whittle.modules import Linear
from whittle.models.gpt.blocks import GptNeoxMLP, LLaMAMLP, GemmaMLP


class GptNeoxMLPFlex(GptNeoxMLP):
    def __init__(self, config: Config, intermediate_size: int | None = None) -> None:
        isize = config.intermediate_size
        config.intermediate_size = 5
        super().__init__(config)
        config.intermediate_size = isize
        
        intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.fc = Linear(config.n_embd, intermediate_size, bias=config.bias)
        self.proj = Linear(intermediate_size, config.n_embd, bias=config.bias)
        self.config = config
        self.in_features = config.n_embd
        self.intermediate_size = intermediate_size

        # Set current sub-network to super-network
        self.sub_network_n_embd = self.in_features
        self.sub_network_intermediate_size = self.intermediate_size

    def set_sub_network(
        self, sub_network_n_embd: int, sub_network_intermediate_size: int
    ):
        self.sub_network_n_embd = sub_network_n_embd
        self.sub_network_intermediate_size = sub_network_intermediate_size

        self.fc.set_sub_network(
            self.sub_network_n_embd, self.sub_network_intermediate_size
        )
        self.proj.set_sub_network(
            self.sub_network_intermediate_size, self.sub_network_n_embd
        )

    def reset_super_network(self):
        self.sub_network_n_embd = self.in_features
        self.sub_network_intermediate_size = self.intermediate_size

        self.fc.reset_super_network()
        self.proj.reset_super_network()


class LLaMAMLPFlex(LLaMAMLP):
    def __init__(self, config: Config, intermediate_size: int | None = None) -> None:
        super().__init__(config)
        intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.fc_1 = Linear(config.n_embd, intermediate_size, bias=config.bias)
        self.fc_2 = Linear(config.n_embd, intermediate_size, bias=config.bias)
        self.proj = Linear(intermediate_size, config.n_embd, bias=config.bias)
        self.in_features = config.n_embd
        self.intermediate_size = intermediate_size
        self.sub_network_n_embd: int | None = None
        self.sub_network_intermediate_size: int | None = None
        self.config = config

    def set_sub_network(
        self, sub_network_n_embd: int, sub_network_intermediate_size: int
    ):
        self.sub_network_n_embd = sub_network_n_embd
        self.sub_network_intermediate_size = sub_network_intermediate_size

        self.fc_1.set_sub_network(
            self.sub_network_n_embd, self.sub_network_intermediate_size
        )
        self.fc_2.set_sub_network(
            self.sub_network_n_embd, self.sub_network_intermediate_size
        )
        self.proj.set_sub_network(
            self.sub_network_intermediate_size, self.sub_network_n_embd
        )

    def reset_super_network(self):
        self.sub_network_n_embd = self.in_features
        self.sub_network_intermediate_size = self.intermediate_size

        self.fc_1.reset_super_network()
        self.fc_2.reset_super_network()
        self.proj.reset_super_network()


class GemmaMLPFlex(GemmaMLP):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = (
            torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate)
            * x_fc_2
        )
        return self.proj(x)
