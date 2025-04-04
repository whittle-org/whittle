from __future__ import annotations

import math

import torch
import torch.nn as nn
from litgpt import Config
from litgpt.model import KVCache, apply_rope

from whittle.modules import LinearProj, LinearQKV


class CausalSelfAttention(nn.Module):
    """Extension of litgpt's `litgpt.model.CausalSelfAttention` with support to adapt to sub-network dimensionality."""

    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = LinearQKV(config.n_embd, shape, bias=config.bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = LinearProj(
            config.head_size * config.n_head, config.n_embd, bias=config.bias
        )
        # disabled by default
        self.kv_cache: KVCache | None = None
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None
            and block_idx % config.sliding_window_layer_placing == 0
        )
        self.config = config
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
        qkv_indices = []
        heads_per_group = self.config.n_head // self.config.n_query_groups
        if self.config.n_head == self.config.n_query_groups:
            for h in range(self.sub_network_n_head):
                # append q
                start_q = 3 * h * self.config.head_size
                end_q = start_q + self.sub_network_head_size
                qkv_indices.extend([i for i in range(start_q, end_q)])
                # append k
                start_k = (3 * h + 1) * self.config.head_size
                end_k = start_k + self.sub_network_head_size
                qkv_indices.extend([i for i in range(start_k, end_k)])
                # append v
                start_v = (3 * h + 2) * self.config.head_size
                end_v = start_v + self.sub_network_head_size
                qkv_indices.extend([i for i in range(start_v, end_v)])
        elif self.config.n_query_groups == 1:
            for h in range(self.sub_network_n_head):
                start_q = h * self.config.head_size
                end_q = start_q + self.sub_network_head_size
                qkv_indices.extend([i for i in range(start_q, end_q)])
            end_queries = self.config.n_head * self.config.head_size
            qkv_indices.extend(
                [i for i in range(end_queries, end_queries + self.sub_network_head_size)]
            )
            end_keys = end_queries + self.config.head_size
            qkv_indices.extend(
                [i for i in range(end_keys, end_keys + self.sub_network_head_size)]
            )
        else:
            for g in range(self.sub_network_query_groups):
                start_q = g * (heads_per_group + 2) * self.config.head_size
                for h in range(self.sub_network_q_per_kv):
                    qkv_indices.extend(
                        [
                            i
                            for i in range(
                                start_q + h * self.config.head_size,
                                start_q
                                + h * self.config.head_size
                                + self.sub_network_head_size,
                            )
                        ]
                    )
                start_k = start_q + heads_per_group * self.config.head_size
                qkv_indices.extend(
                    [i for i in range(start_k, start_k + self.sub_network_head_size)]
                )
                start_v = start_k + self.config.head_size
                qkv_indices.extend(
                    [i for i in range(start_v, start_v + self.sub_network_head_size)]
                )
        return qkv_indices

    def get_proj_indices(self):
        n_head = self.config.n_head
        n_query_groups = self.config.n_query_groups
        sub_network_n_head = self.sub_network_n_head
        heads_per_group = self.config.n_head // self.config.n_query_groups
        sub_network_query_groups = self.sub_network_query_groups
        sub_network_head_size = self.sub_network_head_size
        head_size = self.config.head_size
        proj_indices = []
        if n_head == n_query_groups:
            for i in range(sub_network_n_head):
                proj_indices.extend(
                    i for i in range(i * head_size, i * head_size + sub_network_head_size)
                )
        else:
            for g in range(sub_network_query_groups):
                start = g * heads_per_group * head_size
                for h in range(self.sub_network_q_per_kv):
                    proj_indices.extend(
                        i
                        for i in range(
                            start + h * head_size,
                            start + h * head_size + sub_network_head_size,
                        )
                    )
        return proj_indices

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
            self.sub_network_n_embd, self.sub_network_qkv_shape, self.qkv_indices
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
        self.attn.reset_super_network()
        self.proj.reset_super_network()
        self.sub_attention_scaler = self.config.attention_scores_scalar
        self.qkv_indices = None
        self.proj_indices = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.sub_network_n_embd is not None, (
            "You need to call `gpt.set_sub_network()"
        )
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.attn(x)
        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        total_qkv = (
            self.sub_network_q_per_kv + 2
        )  # each group has 1+ queries, 1 key, and 1 value

        qkv = qkv.view(
            B,
            T,
            self.sub_network_query_groups,
            total_qkv,
            self.sub_network_head_size,
        )

        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((self.sub_network_q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.sub_network_query_groups != (
            self.sub_network_query_groups * self.sub_network_q_per_kv
        ) and (input_pos is None or self.sub_network_query_groups != 1):
            k = k.expand(
                B,
                self.sub_network_query_groups,
                self.sub_network_q_per_kv,
                T,
                self.sub_network_head_size,
            )
            v = v.expand(
                B,
                self.sub_network_query_groups,
                self.sub_network_q_per_kv,
                T,
                self.sub_network_head_size,
            )
        q = q.reshape(B, -1, T, self.sub_network_head_size)
        k = k.reshape(B, -1, T, self.sub_network_head_size)
        v = v.reshape(B, -1, T, self.sub_network_head_size)
        rope_n_elem = int(self.sub_network_head_size * self.config.rotary_percentage)
        # cos, sin = build_rope_cache(seq_len=T, n_elem=rope_n_elem,device=q.device)
        q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)
        k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)
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
            sliding_window_bias = torch.ones_like(mask).tril(
                diagonal=-self.config.sliding_window_size
            )
            sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            mask += sliding_window_bias
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
            scores = (
                torch.tanh(scores / self.config.attention_logit_softcapping)
                * self.config.attention_logit_softcapping
            )
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
        heads = 1 if self.sub_network_query_groups == 1 else self.sub_network_n_head
        v_shape = (batch_size, heads, max_seq_length, self.sub_network_head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length=gpt.cos.size(-1)` value"
                )
            k_shape = v_shape
        else:
            rope_n_elem = (
                rope_n_elem if rope_n_elem is not None else self.config.rope_n_elem
            )
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.sub_network_head_size - rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)
