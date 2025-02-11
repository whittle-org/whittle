from whittle.models.gpt.blocks.mlp import (
    GptNeoxMLP as GptNeoxMLPBase,
    LLaMAMLP as LLaMAMLPBase,
)
from whittle.lora.lora_linear import LoRALinear
from litgpt.config import Config
from typing import Any
import torch
from litgpt.utils import map_old_state_dict_weights


class LoRAGptNeoxMLP(GptNeoxMLPBase):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.in_features = config.n_embd
        self.intermediate_size = config.intermediate_size
        self.fc = LoRALinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            r=(config.lora_r if config.lora_mlp else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        self.proj = LoRALinear(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
            r=(config.lora_r if config.lora_mlp else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

        self.config = config

    def _load_from_state_dict(
        self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc.weight": "fc.linear.weight",
            "fc.bias": "fc.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class LoRALLaMAMLP(LLaMAMLPBase):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.in_features = config.n_embd
        self.intermediate_size = config.intermediate_size
        self.fc_1 = LoRALinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            r=(config.lora_r if config.lora_mlp else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        self.fc_2 = LoRALinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            r=(config.lora_r if config.lora_mlp else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        self.proj = LoRALinear(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
            r=(config.lora_r if config.lora_mlp else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

        self.config = config

    def _load_from_state_dict(
        self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc_1.weight": "fc_1.linear.weight",
            "fc_1.bias": "fc_1.linear.bias",
            "fc_2.weight": "fc_2.linear.weight",
            "fc_2.bias": "fc_2.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class LoRAGemmaMLP(LoRALLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = (
            torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate)
            * x_fc_2
        )
        return self.proj(x)
