from typing import Union
import torch.nn as nn
from whittle.models.gpt.blocks import Block as BaseBlock
from whittle.lora.lora_attention import CausalSelfAttention
from whittle.modules.rmsnorm import RMSNorm
from whittle.modules.layernorm import LayerNorm
from whittle.lora.lora_mlps import (
    LoRAGptNeoxMLP as GptNeoxMLP,
    LoRALLaMAMLP as LLaMAMLP,
    LoRAGemmaMLP as GemmaMLP,
)
from whittle.lora.config import LoRAConfig as Config


class LoRABlock(BaseBlock):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__(config, block_idx)
        self.config = config
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = self.norm_class()(config.n_embd, eps=config.norm_eps)
        self.post_attention_norm = (
            self.norm_class()(config.n_embd, eps=config.norm_eps)
            if config.post_attention_norm
            else nn.Identity()
        )
        self.attn = CausalSelfAttention(config, block_idx)
        self.norm_2: Union[LayerNorm, RMSNorm, None] = (
            None
            if config.shared_attention_norm
            else self.norm_class()(config.n_embd, eps=config.norm_eps)
        )
        self.mlp = self.mlp_class()(config)
        self.post_mlp_norm = (
            self.norm_class()(config.n_embd, eps=config.norm_eps)
            if config.post_mlp_norm
            else nn.Identity()
        )
        # Set current sub-network to super-network
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_intermediate_size = self.config.intermediate_size
        self.sub_network_num_heads = self.config.n_head

    def mlp_class(self):
        # `self._mlp_class` cannot be the type to keep the config json serializable
        if self.config.mlp_class_name == "LLaMAMLP":
            return LLaMAMLP
        elif self.config.mlp_class_name == "GemmaMLP":
            return GemmaMLP
        elif self.config.mlp_class_name == "GptNeoxMLP":
            return GptNeoxMLP
        else:
            raise ValueError(f"Unknown MLP class: {self.config._mlp_class}")
