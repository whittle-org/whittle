from __future__ import annotations

from functools import partial

import litgpt
import torch.nn as nn
from litgpt import Config

from whittle.models.gpt.blocks.causal_self_attention import CausalSelfAttention
from whittle.models.gpt.blocks.mlp import GemmaMLP, GptNeoxMLP, LLaMAMLP
from whittle.modules.layernorm import LayerNorm
from whittle.modules.rmsnorm import RMSNorm


class Block(litgpt.model.Block):
    """An extension of litgpt's Transformer Block with support to adapt to sub-network dimensionality."""

    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__(config, block_idx)
        self.config = config
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = self.norm_class()(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, block_idx)
        self.post_attention_norm = (
            self.norm_class()(config.n_embd, eps=config.norm_eps)
            if config.post_attention_norm
            else nn.Identity()
        )
        self.norm_2: LayerNorm | RMSNorm | None = (
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

    def norm_class(self):
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self.config.norm_class_name == "RMSNorm":
            return partial(RMSNorm, add_unit_offset="Gemma" in self.config.name)
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
        sub_network_query_groups: int,
        sub_network_head_size: int,
        sampled_intermediate_indices: list[int] | None = None,
        sampled_head_indices: list[int] | None = None,
        sampled_query_group_indices: list[int] | None = None,
        sampled_head_size_indices: list[int] | None = None,
        sampled_embd_indices: list[int] | None = None,
    ) -> None:
        """
        Set the Block to the specified sub-network dimensionality.

        Args:
            sub_network_n_embd: Embedding dimension of the sub-network.
            sub_network_intermediate_size: Intermediate size of the sub-network.
            sub_network_num_heads: Number of attention heads in the sub-network.
            sub_network_query_groups: Number of query groups in the sub-network.
            sub_network_head_size: Size of each attention head in the sub-network.
        """
        self.sub_network_n_embd = sub_network_n_embd
        self.sub_network_intermediate_size = sub_network_intermediate_size
        self.sub_network_num_heads = sub_network_num_heads
        self.norm_1.set_sub_network(self.sub_network_n_embd, sampled_embd_indices)
        self.attn.set_sub_network(
            self.sub_network_n_embd,
            self.sub_network_num_heads,
            sub_network_query_groups,
            sub_network_head_size,
            sampled_head_indices,
            sampled_embd_indices,
            sampled_head_size_indices,
            sampled_query_group_indices,
        )
        if isinstance(self.post_attention_norm, LayerNorm) or isinstance(
            self.post_attention_norm, RMSNorm
        ):
            self.post_attention_norm.set_sub_network(
                self.sub_network_n_embd, sampled_embd_indices
            )
        if not self.config.shared_attention_norm and self.norm_2 is not None:
            self.norm_2.set_sub_network(self.sub_network_n_embd, sampled_embd_indices)
        self.mlp.set_sub_network(
            self.sub_network_n_embd,
            self.sub_network_intermediate_size,
            sampled_intermediate_indices,
            sampled_embd_indices,
        )
        if isinstance(self.post_mlp_norm, LayerNorm) or isinstance(
            self.post_mlp_norm, RMSNorm
        ):
            self.post_mlp_norm.set_sub_network(
                self.sub_network_n_embd, sampled_embd_indices
            )

    def reset_super_network(self):
        """
        Resets the layers in the Block to it's original super-network dimensionality.
        """
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_intermediate_size = self.config.intermediate_size
        self.sub_network_num_heads = self.config.n_head
        self.norm_1.reset_super_network()
        self.attn.reset_super_network()
        if not self.config.shared_attention_norm:
            self.norm_2.reset_super_network()
        self.mlp.reset_super_network()
        if isinstance(self.post_attention_norm, LayerNorm) or isinstance(
            self.post_attention_norm, RMSNorm
        ):
            self.post_attention_norm.reset_super_network()
        if isinstance(self.post_mlp_norm, LayerNorm) or isinstance(
            self.post_mlp_norm, RMSNorm
        ):
            self.post_mlp_norm.reset_super_network()
