from __future__ import annotations


import litgpt
from litgpt import Config

from whittle.models.gpt.blocks.causal_self_attention import CausalSelfAttention
from whittle.models.gpt.blocks.mlp import GemmaMLP, GptNeoxMLP, LLaMAMLP
from whittle.modules.layernorm import LayerNorm
from whittle.modules.rmsnorm import RMSNorm


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
        self.norm_2: LayerNorm | RMSNorm | None = (
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
        sub_network_query_groups=None,
        sub_network_head_size=None,
        sample_random_indices: bool = False,
    ) -> None:
        self.sub_network_n_embd = sub_network_n_embd
        self.sub_network_intermediate_size = sub_network_intermediate_size
        self.sub_network_num_heads = sub_network_num_heads
        self.norm_1.set_sub_network(self.sub_network_n_embd, sample_random_indices)
        self.attn.set_sub_network(
            self.sub_network_n_embd,
            self.sub_network_num_heads,
            sub_network_query_groups,
            sub_network_head_size,
            sample_random_indices,
        )
        if not self.config.shared_attention_norm and self.norm_2 is not None:
            self.norm_2.set_sub_network(self.sub_network_n_embd, sample_random_indices)
        self.mlp.set_sub_network(
            self.sub_network_n_embd,
            self.sub_network_intermediate_size,
            sample_random_indices,
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
