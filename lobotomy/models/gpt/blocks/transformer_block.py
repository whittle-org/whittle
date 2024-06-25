from typing import Optional, Union

import litgpt
from litgpt import Config
from lobotomy.models.gpt.blocks.causal_self_attention import CausalSelfAttention
from lobotomy.modules.rmsnorm import RMSNorm
from lobotomy.modules.layernorm import LayerNorm
from lobotomy.models.gpt.blocks.mlp import GptNeoxMLP, LLaMAMLP, GemmaMLP


class Block(litgpt.model.Block):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = self.norm_class()(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2: Optional[Union[LayerNorm, RMSNorm]] = (
            None
            if config.shared_attention_norm
            else self.norm_class()(config.n_embd, eps=config.norm_eps)
        )
        self.mlp = self.mlp_class()(config)

        # Set current sub-network to super-network
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_intermediate_size = self.config.intermediate_size
        self.sub_network_num_heads = self.config.n_head

    def norm_class(self):
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self.config.norm_class_name == "RMSNorm":
            return RMSNorm
        return LayerNorm

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

    def set_sub_network(
        self,
        sub_network_n_embd: int,
        sub_network_intermediate_size: int,
        sub_network_num_heads: int,
    ) -> None:
        self.sub_network_n_embd = sub_network_n_embd
        self.sub_network_intermediate_size = sub_network_intermediate_size
        self.sub_network_num_heads = sub_network_num_heads
        self.norm_1.set_sub_network(self.sub_network_n_embd)
        self.attn.set_sub_network(self.sub_network_n_embd, self.sub_network_num_heads)
        if not self.config.shared_attention_norm and self.norm_2 is not None:
            self.norm_2.set_sub_network(self.sub_network_n_embd)
        self.mlp.set_sub_network(
            self.sub_network_n_embd, self.sub_network_intermediate_size
        )

    def reset_super_network(self):
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_intermediate_size = self.config.intermediate_size
        self.sub_network_num_heads = self.config.n_head
        self.norm_1.reset_super_network()
        self.attn.reset_super_network()
        if not self.config.shared_attention_norm:
            self.norm_2.reset_super_network()
        self.mlp.reset_super_network()
