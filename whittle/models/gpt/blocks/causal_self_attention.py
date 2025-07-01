from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from litgpt import Config
from litgpt.model import KVCache, apply_rope, do_softcapping
from litgpt.scripts.convert_hf_checkpoint import qkv_reassemble

from whittle.modules import LinearProj, LinearQKV


class CausalSelfAttention(nn.Module):
    """Extension of litgpt's `litgpt.model.CausalSelfAttention` with support to adapt to sub-network dimensionality."""

    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.qkv = LinearQKV(config.n_embd, shape, bias=config.bias or config.attn_bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = LinearProj(
            config.head_size * config.n_head, config.n_embd, bias=config.bias
        )
        # disabled by default
        self.kv_cache: KVCache | None = None
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None
            and block_idx % config.sliding_window_layer_stride == 0
        )
        self.config = config
        self.block_idx = block_idx
        if config.norm_qk:
            self.norm_q = config.norm_class(
                config.head_size * config.n_head, eps=config.norm_eps
            )
            self.norm_k = config.norm_class(
                config.head_size * config.n_query_groups, eps=config.norm_eps
            )
        else:
            self.norm_q = self.norm_k = None
        # Set current sub-network to super-network
        self.q_per_kv = config.n_head // config.n_query_groups
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_n_head = self.config.n_head
        self.sub_network_head_size = self.config.head_size
        self.sub_network_qkv_shape = (
            (self.q_per_kv + 2) * self.config.head_size * self.config.n_query_groups
        )
        self.sub_network_query_groups = self.config.n_query_groups
        self.sub_network_q_per_kv = int(
            self.sub_network_n_head // self.sub_network_query_groups
        )
        self.sub_attention_scaler = self.config.attention_scores_scalar

    def get_qkv_indices(self):
        head_size = self.config.head_size
        n_head = self.config.n_head
        n_query_groups = self.config.n_query_groups
        sub_n_head = self.sub_network_n_head
        sub_head_size = self.sub_network_head_size
        sub_q_groups = self.sub_network_query_groups
        sub_q_per_kv = self.sub_network_q_per_kv

        heads_per_group = n_head // n_query_groups

        # Count heads per section
        if n_head == n_query_groups:
            num_q = num_k = n_head
        elif n_query_groups == 1:
            num_q = n_head
            num_k = 1
        else:
            num_q = n_head
            num_k = n_query_groups

        # Compute block start offsets
        q_block_start = 0
        k_block_start = num_q * head_size
        v_block_start = k_block_start + num_k * head_size

        q_parts, k_parts, v_parts = [], [], []

        if sub_n_head == sub_q_groups:
            for i in range(sub_n_head):
                q_start = q_block_start + i * head_size
                k_start = k_block_start + i * head_size
                v_start = v_block_start + i * head_size
                q_parts.append(torch.arange(q_start, q_start + sub_head_size))
                k_parts.append(torch.arange(k_start, k_start + sub_head_size))
                v_parts.append(torch.arange(v_start, v_start + sub_head_size))

        elif sub_q_groups == 1:
            for i in range(sub_n_head):
                q_start = q_block_start + i * head_size
                q_parts.append(torch.arange(q_start, q_start + sub_head_size))

            k_parts.append(torch.arange(k_block_start, k_block_start + sub_head_size))
            v_parts.append(torch.arange(v_block_start, v_block_start + sub_head_size))

        else:
            for g in range(sub_q_groups):
                for h in range(sub_q_per_kv):
                    q_head_index = g * heads_per_group + h
                    q_start = q_block_start + q_head_index * head_size
                    q_parts.append(torch.arange(q_start, q_start + sub_head_size))

                k_start = k_block_start + g * head_size
                k_parts.append(torch.arange(k_start, k_start + sub_head_size))

                v_start = v_block_start + g * head_size
                v_parts.append(torch.arange(v_start, v_start + sub_head_size))

        qkv = torch.cat(q_parts + k_parts + v_parts)
        return qkv

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
            q_per_kv = self.sub_network_n_head // self.sub_network_query_groups
        elif self.config.n_head == self.config.n_query_groups:
            q_per_kv = 1
            self.sub_network_query_groups = self.sub_network_n_head
        self.sub_network_qkv_shape = (
            (q_per_kv + 2) * self.sub_network_head_size * self.sub_network_query_groups
        )
        self.sub_network_q_per_kv = int(q_per_kv)
        if self.sub_network_n_head != self.sub_network_query_groups:
            assert self.sub_network_n_head % self.sub_network_query_groups == 0, (
                f"Number of heads {self.sub_network_n_head} must be divisible by number of query groups {self.sub_network_query_groups} for GQA"
            )
        self.qkv_indices = self.get_qkv_indices()
        self.proj_indices = self.qkv_indices[
            : torch.searchsorted(
                self.qkv_indices, sub_network_n_head * self.config.head_size, right=False
            )
        ]
        self.qkv.set_sub_network(
            self.sub_network_n_embd, self.sub_network_qkv_shape, self.qkv_indices
        )
        self.proj.set_sub_network(
            self.sub_network_head_size
            * self.sub_network_query_groups
            * self.sub_network_q_per_kv,
            self.sub_network_n_embd,
            self.proj_indices,
        )
        if self.config.norm_qk:
            self.norm_q.set_sub_network(
                self.sub_network_head_size
                * self.sub_network_n_query_groups
                * self.sub_network_q_per_kv
            )
            self.norm_k.set_sub_network(
                self.sub_network_head_size * self.sub_network_query_groups
            )
        if self.config.attention_scores_scalar:
            self.sub_attention_scaler = self.sub_network_n_embd // self.sub_network_n_head
        else:
            self.sub_attention_scaler = self.config.attention_scores_scalar

    def reset_super_network(self):
        """Resets the dimensionality of the current sub-network to the super-network dimensionality."""
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_n_head = self.config.n_head
        self.sub_network_head_size = self.config.head_size
        self.sub_network_qkv_shape = (
            self.config.n_head + 2 * self.config.n_query_groups
        ) * self.config.head_size
        self.sub_network_query_groups = self.config.n_query_groups
        self.sub_network_q_per_kv = int(
            self.sub_network_n_head // self.sub_network_query_groups
        )
        self.qkv.reset_super_network()
        self.proj.reset_super_network()
        if self.config.norm_qk:
            self.norm_q.reset_super_network()
            self.norm_k.reset_super_network()
        self.sub_attention_scaler = self.config.attention_scores_scalar

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        input_pos_maxp1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.sub_network_n_embd is not None, (
            "You need to call `gpt.set_sub_network()"
        )
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # Perform a single multiplication operation using a combined QKV matrix to calculate `query`, `key`, and `value`
        # instead of individually multiplying the input `x` with the respective weight matrices.
        qkv = self.qkv(x)  # (B, T, 3xC*)
        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        query_size = (
            self.sub_network_q_per_kv
            * self.sub_network_query_groups
            * self.sub_network_head_size
        )
        key_size = value_size = self.sub_network_query_groups * self.sub_network_head_size
        # Split qkv into query, key and value matrices.
        q, k, v = qkv.split((query_size, key_size, value_size), dim=-1)  # 3x(B, T, C*)

        if self.config.norm_qk:
            q = self.norm_q(q)
            k = self.norm_k(k)

        q = q.view(
            B,
            T,
            self.sub_network_q_per_kv * self.sub_network_query_groups,
            self.sub_network_head_size,
        )  # (B, T, nh_q, hs)
        k = k.view(B, T, self.sub_network_query_groups, self.sub_network_head_size)
        v = v.view(B, T, self.sub_network_query_groups, self.sub_network_head_size)
        # The tensors `query`, `key`, and `value` are now accurately structured: within each batch element (B), there are
        # multiple heads (nh), and within each head, there is a sequence of elements (T), each represented by a vector
        # of size `hs`.
        q = q.transpose(1, 2)  # (B, nh_q, T, hs)
        k = k.transpose(1, 2)  # (B, nh_k, T, hs)
        v = v.transpose(1, 2)  # (B, nh_v, T, hs)
        rope_n_elem = int(self.sub_network_head_size * self.config.rotary_percentage)
        # apply rope to the first `rope_n_elem` elements of the query and key tensors
        q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)  # (B, nh_q, T, hs)
        k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)  # (B, nh_k, T, hs)

        # Apply kv-cache during inference.
        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)
            if input_pos_maxp1 is not None:
                # Subselect along sequence dimension
                k = k[..., :input_pos_maxp1, :]
                v = v[..., :input_pos_maxp1, :]
            # k, v: (B, nh_k, input_pos_maxp1, hs)
            # If input_pos_maxp1 is None -> max_seq_length
        # Grouped queries: balance the number of heads across all three matrices.
        # NOTE: flash attention requires it in training mode.
        # Multi-query: this step can be skipped since there is only 1 head, allowing us to use broadcasting.
        if self.sub_network_query_groups != self.sub_network_n_head and (
            input_pos is None or self.sub_network_query_groups != 1
        ):
            q_per_kv = self.sub_network_q_per_kv
            k = k.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)
            v = v.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)

        if self.apply_sliding_window_attention:
            """
                  Global Window              Sliding window             Sliding window
                  attention mask      +            bias          =      attention mask
            ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
            │ True False False False │  │ True  True  True True │  │ True  False False False │
            │ True True  False False │  │ True  True  True True │  │ True  True  False False │
            │ True True  True  False │  │ False True  True True │  │ False True  True  False │
            │ True True  True  True  │  │ False False True True │  │ False False True  True  │
            └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
            """
            if mask is None:
                mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), float("-inf"))
                mask = mask.view(1, 1, *mask.shape)
            sliding_window_bias = torch.ones_like(mask).tril(
                diagonal=-self.config.sliding_window_size
            )
            sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            mask += sliding_window_bias
        # Efficient attention using Flash Attention CUDA kernels.
        # NOTE: efficient implementation is disabled if `mask` is not None or softcapping is enabled.
        # ↓ (B, nh, T, hs) @ (B, nh, T, hs).mT --> (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        y = self.scaled_dot_product_attention(q, k, v, mask)
        y = y.reshape(
            B,
            T,
            self.sub_network_head_size
            * self.sub_network_q_per_kv
            * self.sub_network_query_groups,
        )  # re-assemble all head outputs side by side
        return self.proj(y)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.sub_attention_scaler or self.sub_network_head_size)
        # with softcapping we cannot use SDPA
        if self.config.attention_logit_softcapping is not None:
            scale = 1.0 / math.sqrt(
                self.sub_attention_scaler or self.sub_network_head_size
            )
            scores = q @ k.mT * scale
            scores = do_softcapping(scores, self.config.attention_logit_softcapping)
            if mask is None:
                mask = torch.ones(
                    q.size(2), q.size(2), dtype=q.dtype, device=q.device
                ).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            scores = scores + mask
            scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(
                dtype=q.dtype
            )
            y = scores @ v
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                scale=scale,
                is_causal=mask is None,
            )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope_n_elem: int | None = None,
    ) -> KVCache:
        v_shape = (
            batch_size,
            self.sub_network_query_groups,
            max_seq_length,
            self.sub_network_head_size,
        )
        rope_n_elem = int(self.sub_network_head_size * self.config.rotary_percentage)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length=gpt.cos.size(-1)` value"
                )
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                self.sub_network_query_groups,
                max_seq_length,
                rope_cache_length + self.sub_network_head_size - rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)

    def _load_from_state_dict(
        self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        """For compatibility with legacy checkpoints."""

        for attr in ("weight", "bias"):
            legacy_key = f"{prefix}attn.{attr}"
            current_key = f"{prefix}qkv.{attr}"
            if legacy_key in state_dict:
                state_dict[current_key] = qkv_reassemble(
                    state_dict.pop(legacy_key), self.config
                )

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


if __name__ == "__main__":

    def check_shapes(n_query_groups, n_head, subnet_query_groups, subnet_n_head):
        config = Config()
        config.n_query_groups = n_query_groups
        config.n_head = n_head
        config.head_size = 10
        config.n_embd = 16

        config.n_layer = 1
        config.attention_scores_scalar = 1
        config.rotary_percentage = 0.25
        config.max_seq_len = 8
        config.bias = True
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.dtype = torch.float32
        attention = CausalSelfAttention(config, 0)

        def forward_pass(attention):
            B, T = 32, 100
            input = torch.rand(B, T, config.n_embd)
            h = int(config.head_size * config.rotary_percentage)
            cos = torch.rand(1, T, h)
            sin = torch.rand(1, T, h)
            attention(input, cos, sin)

        print("supernet:")
        forward_pass(attention)
        attention.set_sub_network(
            sub_network_n_embd=config.n_embd,
            sub_network_n_head=subnet_n_head,
            sub_network_query_groups=subnet_query_groups,
            sub_network_head_size=6,
        )
        print("subnet:")
        forward_pass(attention)

    supernet_attn_configs = {
        # "mha": (8, 8),  # (num groups, num_query_heads)
        "gqa": (4, 16),
        # "mqa": (1, 8),
    }

    subnet_attn_configs = {
        "mha": (1, 4),  # (num groups, num_query_heads)
        # "gqa": (4, 2),
        # "mqa": (1, 4),
    }

    print("\nShape is (batch, num_heads, sequence_length, head_embed_dim)\n")

    def format_config(conf):
        return (
            f"({conf[0]} groups, {conf[1]} heads, {conf[1] / conf[0]} query heads/group)"
        )

    for supernet_attn, supernet_config in supernet_attn_configs.items():
        for subnet_attn, subnet_config in subnet_attn_configs.items():
            try:
                print(
                    f"{supernet_attn} {format_config(supernet_config)} -> {subnet_attn} {format_config(subnet_config)}"
                )
                check_shapes(*supernet_config, *subnet_config)
            except Exception as e:
                print("Failed!")
                print(e)
            print()
