"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

from __future__ import annotations
from functools import partial

from typing import Any
from typing_extensions import Self

import torch
import torch.nn as nn
from litgpt import Config
from litgpt.model import build_rope_cache
from litgpt.model import batched_index_select
from whittle.models.gpt.blocks import Block
from whittle.modules.embedding import Embedding
from whittle.modules.layernorm import LayerNorm
from whittle.modules.linear import Linear
from whittle.modules.rmsnorm import RMSNorm


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lm_head = Linear(
            config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(
                    Block(config, block_idx) for block_idx in range(config.n_layer)
                ),
                ln_f=self.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_layer = config.n_layer
        self.max_seq_length = self.config.block_size
        self.mask_cache: torch.Tensor | None = None

        # Set current sub-network to super-network
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_intermediate_size = self.config.intermediate_size
        self.sub_network_num_heads = self.config.n_head
        self.sub_network_n_layers = self.config.n_layer
        self.cos: torch.Tensor
        self.sin: torch.Tensor
        self.config.is_encoder_decoder = False
        self.main_input_name = "input_pos"
        self._supports_cache_class = True
        self.sub_network_head_size = None
        # self.transformer.wte.weight = self.lm_head.weight # weight tying: TODO: where does litgpt do this?

    @property
    def norm_class(self):
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self.config.norm_class_name == "RMSNorm":
            return partial(RMSNorm, add_unit_offset="Gemma" in self.config.name)
        return LayerNorm

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}"
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache(self._max_seq_length, self.config.rope_n_elem)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(
                seq_len=self._max_seq_length,
                n_elem=self.config.rope_n_elem,
                device=self.cos.device,
            )
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(
            seq_len=self._max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=self.cos.device,
        )

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def tie_weights(self) -> None:
        if self.config.tie_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

    def rope_cache(
        self, seq_len: int, n_elem: int, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_adjustments is None:
            extra_config = None

        else:
            adjusted_params_required = [
                "factor",
                "low_freq_factor",
                "high_freq_factor",
                "original_max_seq_len",
            ]
            params_present = [
                param in self.config.rope_adjustments
                for param in adjusted_params_required
            ]
            num_params_present = sum(params_present)

            if num_params_present == 0:
                extra_config = None  # uses standard RoPE
            elif num_params_present == 4:
                # These parameters should always be used together so that we don't interfere with standard rope
                extra_config = {
                    "original_max_seq_len": self.config.rope_adjustments[
                        "original_max_seq_len"
                    ],
                    "factor": self.config.rope_adjustments["factor"],
                    "low_freq_factor": self.config.rope_adjustments["low_freq_factor"],
                    "high_freq_factor": self.config.rope_adjustments[
                        "high_freq_factor"
                    ],
                }
            else:
                # Some but not all parameters are specified; raise an error
                missing_params = [
                    param
                    for param, present in zip(adjusted_params_required, params_present)
                    if not present
                ]
                raise ValueError(
                    f"The following adjusted RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                    "All adjusted RoPE parameters must be specified together."
                )

        return build_rope_cache(
            seq_len=seq_len,
            n_elem=n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            extra_config=extra_config,
        )

    def set_sub_network(
        self,
        sub_network_n_embd: int,
        sub_network_intermediate_size: list,
        sub_network_num_heads: list,
        sub_network_n_layers: int,
        sub_network_query_groups=None,
        sub_network_head_size=None,
    ) -> None:
        self.sub_network_head_size = sub_network_head_size
        self.sub_network_n_embd = sub_network_n_embd
        self.sub_network_intermediate_size = sub_network_intermediate_size
        self.sub_network_num_heads = sub_network_num_heads
        self.sub_network_n_layers = sub_network_n_layers
        self.transformer.wte.set_sub_network(self.sub_network_n_embd)
        self.transformer.ln_f.set_sub_network(self.sub_network_n_embd)
        for i in range(self.sub_network_n_layers):
            block = self.transformer.h[i]
            block.set_sub_network(
                sub_network_n_embd,
                sub_network_intermediate_size[i],
                sub_network_num_heads[i],
                sub_network_query_groups,
                sub_network_head_size,
            )
        self.lm_head.set_sub_network(sub_network_n_embd, self.config.padded_vocab_size)

    def select_sub_network(self, config):
        self.set_sub_network(
            config["embed_dim"],
            [config["mlp_ratio"] * config["embed_dim"] for i in range(config["depth"])],
            [config["num_heads"] for i in range(config["depth"])],
            config["depth"],
        )

    def reset_super_network(self):
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_intermediate_size = self.config.intermediate_size
        self.sub_network_num_heads = self.config.n_head
        self.sub_network_n_layers = self.config.n_layer
        self.transformer.wte.reset_super_network()
        self.transformer.ln_f.reset_super_network()
        for i in range(self.config.n_layer):
            block = self.transformer.h[i]
            block.reset_super_network()
        self.lm_head.reset_super_network()

    def process_rope_cache(self, cos, sin, input_pos, T):
        if input_pos is not None:  # use the kv cache
            cos = batched_index_select(self.cos, 0, input_pos)
            sin = batched_index_select(self.sin, 0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = batched_index_select(self.mask_cache, 2, input_pos)
            if mask.dim() > 4:
                # the mask cache has a batch dim of 1 in addition to the one
                # we get if input_pos has a batch dimension
                mask = mask.squeeze(1)
        else:
            cos = cos[:T]
            sin = sin[:T]
            mask = None
        return cos, sin, mask

    def forward(
        self, idx: torch.Tensor, input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)
        for i in range(self.sub_network_n_layers):
            block = self.transformer.h[i]
            if not self.config.fix_head_size:
                if isinstance(self.sub_network_num_heads, list):
                    cos, sin = self.rope_cache(
                        seq_len=self.max_seq_length,
                        n_elem=int(
                            self.config.rotary_percentage
                            * (self.sub_network_n_embd // self.sub_network_num_heads[i])
                        ),
                        device=self.device,
                    )
                else:
                    cos, sin = self.rope_cache(
                        seq_len=self.max_seq_length,
                        n_elem=int(
                            self.config.rotary_percentage
                            * (self.sub_network_n_embd // self.sub_network_num_heads)
                        ),
                        device=self.device,
                    )
            else:
                if self.sub_network_head_size is None:
                    cos, sin = self.rope_cache(
                        seq_len=self.max_seq_length,
                        n_elem=int(
                            self.config.rotary_percentage * (self.config.head_size)
                        ),
                        device=self.device,
                    )
                else:
                    cos, sin = self.rope_cache(
                        seq_len=self.max_seq_length,
                        n_elem=int(
                            self.config.rotary_percentage * (self.sub_network_head_size)
                        ),
                        device=self.device,
                    )

            cos, sin, mask = self.process_rope_cache(cos, sin, input_pos, T)

            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)  # (b, t, vocab_size)
        if self.config.final_logit_softcapping is not None:
            x = (
                torch.tanh(x / self.config.final_logit_softcapping)
                * self.config.final_logit_softcapping
            )
        return x

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def set_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int | None = None,
        rope_cache_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != self.max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(self.max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None


def build_mask_cache(
    max_seq_length: int, device: torch.device | None = None
) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)
