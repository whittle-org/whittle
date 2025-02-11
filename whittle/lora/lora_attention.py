from whittle.models.gpt.blocks.causal_self_attention import (
    CausalSelfAttention as BaseCausalSelfAttention,
)
from whittle.lora.lora_qkv_linear import LoRAQKVLinear
from whittle.lora.lora_linear import LoRALinearProj
from whittle.lora.config import LoRAConfig as Config
from typing import Any, Optional
import torch
from litgpt.utils import map_old_state_dict_weights
from litgpt.model import KVCache


class CausalSelfAttention(BaseCausalSelfAttention):
    def __init__(self, config: Config, block_idx: int) -> None:
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        super().__init__(config=config, block_idx=block_idx)
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = LoRAQKVLinear(
            config=config,
            in_features=config.n_embd,
            out_features=shape,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            enable_lora=(config.lora_query, config.lora_key, config.lora_value),
            bias=config.bias,
            # for MQA/GQA support
            head_size=config.head_size,
            n_head=config.n_head,
            n_query_groups=config.n_query_groups,
            fix_head_size=config.fix_head_size,
        )
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = LoRALinearProj(
            config.head_size * config.n_head,
            config.n_embd,
            bias=config.bias,
            r=(config.lora_r if config.lora_projection else 0),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None
            and block_idx % config.sliding_window_layer_placing == 0
        )

        # Set current sub-network to super-network
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_n_head = self.config.n_head
        self.sub_network_head_size = self.config.head_size
        self.sub_network_qkv_shape = (
            self.config.n_head + 2 * self.config.n_query_groups
        ) * self.config.head_size
        self.sub_network_query_groups = self.config.n_query_groups
        self.sub_network_q_per_kv = (
            self.sub_network_n_head // self.sub_network_query_groups
        )
        self.sub_attention_scaler = self.config.attention_scores_scalar
        self.q_per_kv = self.config.n_head // self.config.n_query_groups
        self.qkv_indices = None
        self.proj_indices = None

    def set_sub_network(
        self,
        sub_network_n_embd: int,
        sub_network_n_head: int,
        sub_network_query_groups: int,
        sub_network_head_size: int,
    ):
        """
        Sets the CausalSelfAttention block to the specified sub-network dimensionality.

        Args:
            sub_network_n_embd: Embedding dimension of the sub-network
            sub_network_n_head: Number of attention heads in the sub-network
            sub_network_query_groups: Number of query groups for grouped-query attention (GQA).
            sub_network_head_size: Size of each attention head in the sub-network.
        """
        self.sub_network_n_embd = (
            sub_network_n_embd if sub_network_n_embd else self.config.n_embd
        )
        self.sub_network_n_head = (
            sub_network_n_head if sub_network_n_head else self.config.n_head
        )
        self.sub_network_query_groups = (
            sub_network_query_groups
            if sub_network_query_groups
            else self.config.n_query_groups
        )
        self.sub_network_head_size = (
            sub_network_head_size if sub_network_head_size else self.config.head_size
        )
        if self.config.n_query_groups == 1:
            q_per_kv = self.sub_network_n_head
            self.sub_network_query_groups = 1
        elif (
            self.config.n_head != self.config.n_query_groups
            and self.config.n_query_groups != 1
        ):
            self.sub_network_query_groups = (
                sub_network_query_groups
                if sub_network_query_groups
                else self.config.n_query_groups
            )
            q_per_kv = self.sub_network_n_head // self.config.n_query_groups
        elif self.config.n_head == self.config.n_query_groups:
            q_per_kv = 1
            self.sub_network_query_groups = self.sub_network_n_head
        self.sub_network_qkv_shape = (
            (q_per_kv + 2) * self.sub_network_head_size * self.sub_network_query_groups
        )
        self.sub_network_q_per_kv = int(q_per_kv)
        self.qkv_indices = self.get_qkv_indices()
        self.attn.set_sub_network(
            self.sub_network_n_embd,
            self.sub_network_qkv_shape,
            qkv_indices=self.qkv_indices,
            sub_network_n_head=self.sub_network_n_head,
            sub_network_query_groups=self.sub_network_query_groups,
            sub_network_head_size=self.sub_network_head_size,
            sub_network_q_per_kv=self.q_per_kv,
        )
        self.proj_indices = self.get_proj_indices()
        self.proj.set_sub_network(
            self.sub_network_head_size
            * self.sub_network_query_groups
            * self.sub_network_q_per_kv,
            self.sub_network_n_embd,
            self.proj_indices,
        )
        if self.config.attention_scores_scalar:
            self.sub_attention_scaler = (
                self.sub_network_n_embd // self.sub_network_n_head
            )
        else:
            self.sub_attention_scaler = self.config.attention_scores_scalar

    def _load_from_state_dict(
        self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "attn.weight": "attn.linear.weight",
            "attn.bias": "attn.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
