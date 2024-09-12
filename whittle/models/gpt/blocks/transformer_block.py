from __future__ import annotations
from functools import partial
import torch.nn as nn

import litgpt
from litgpt import Config

from whittle.models.gpt.blocks.causal_self_attention import CausalSelfAttention
from whittle.models.gpt.blocks.mlp import GemmaMLP, GptNeoxMLP, LLaMAMLP
from whittle.modules.layernorm import LayerNorm
from whittle.modules.rmsnorm import RMSNorm


class Block(litgpt.model.Block):
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
        sub_network_query_groups=None,
        sub_network_head_size=None,
        sample_random_indices: bool = False,
        index_mlp=None,
        index_head=None,
        index_emb=None,
        index_block=None,
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
        if isinstance(self.post_attention_norm, LayerNorm) or isinstance(
            self.post_attention_norm, RMSNorm
        ):
            self.post_attention_norm.set_sub_network(
                self.sub_network_n_embd, sample_random_indices
            )
        if not self.config.shared_attention_norm and self.norm_2 is not None:
            self.norm_2.set_sub_network(self.sub_network_n_embd, sample_random_indices)
        self.mlp.set_sub_network(
            self.sub_network_n_embd,
            self.sub_network_intermediate_size,
            sample_random_indices,
            index=index_mlp,
        )
        if isinstance(self.post_mlp_norm, LayerNorm) or isinstance(
            self.post_mlp_norm, RMSNorm
        ):
            self.post_mlp_norm.set_sub_network(
                self.sub_network_n_embd, sample_random_indices
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
        if isinstance(self.post_attention_norm, LayerNorm) or isinstance(
            self.post_attention_norm, RMSNorm
        ):
            self.post_attention_norm.reset_super_network()
        if isinstance(self.post_mlp_norm, LayerNorm) or isinstance(
            self.post_mlp_norm, RMSNorm
        ):
            self.post_mlp_norm.reset_super_network()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ──────────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓                   ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───────►    norm_2
           │  ↓                     │  ↓                   ↓
           │  attn                  │  attn                MLP
           │  ↓                     │  ↓                   ↓
           |  post_attn_norm        |  post_attn_norm      post_mlp_norm
           |  ↓                     |  ↓                   ↓
        ┌─ └► +                     └► + ◄─────────────────┘
        |     ↓
        │     norm_2
        │     ↓
        │     MLP
        │     ↓
        |     post_mlp_norm
        |     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output_base = self.attn(x_normed, cos, sin, mask, input_pos)
        attention_output = self.post_attention_norm(attention_output_base)

        if self.config.parallel_residual:
            x_normed = x_normed if self.config.shared_attention_norm else self.norm_2(x)
            mlp_out = self.mlp(x_normed)
            x = mlp_out + attention_output + x
        else:
            x = attention_output + x
            mlp_out = self.mlp(self.norm_2(x))
            x = self.post_mlp_norm(mlp_out) + x
        return x, attention_output_base, mlp_out
