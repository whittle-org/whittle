from __future__ import annotations

from typing import Any

from litgpt.model import KVCache
from litgpt.scripts.convert_hf_checkpoint import qkv_reassemble
from litgpt.utils import map_old_state_dict_weights

from whittle.lora_model.config import LoRAConfig as Config
from whittle.lora_model.lora_linear import LoRALinearProj
from whittle.lora_model.lora_qkv_linear import LoRAQKVLinear
from whittle.models.gpt.blocks.causal_self_attention import (
    CausalSelfAttention as BaseCausalSelfAttention,
)


class CausalSelfAttention(BaseCausalSelfAttention):
    def __init__(self, config: Config, block_idx: int) -> None:
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        super().__init__(config=config, block_idx=block_idx)
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.qkv = LoRAQKVLinear(
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
        self.kv_cache: KVCache | None = None

        self.config = config
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None
            and block_idx % config.sliding_window_layer_stride == 0
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

    def _load_from_state_dict(
        self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        """For compatibility with base and/or legacy checkpoints."""
        mapping = {
            "qkv.weight": "qkv.linear.weight",
            "qkv.bias": "qkv.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)

        for attr in ("weight", "bias"):
            legacy_key = f"{prefix}attn.linear.{attr}"
            current_key = f"{prefix}qkv.linear.{attr}"
            if legacy_key in state_dict:
                state_dict[current_key] = qkv_reassemble(
                    state_dict.pop(legacy_key), self.config
                )

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
