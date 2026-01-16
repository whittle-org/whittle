"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any
from typing_extensions import Self

import torch
import torch.nn as nn
from litgpt import Config  # type: ignore
from litgpt.model import (  # type: ignore
    batched_index_select,
    build_rope_cache,
    do_softcapping,
)

from whittle.exceptions import IllegalSubNetworkError
from whittle.models.gpt.blocks import Block
from whittle.modules.embedding import Embedding
from whittle.modules.layernorm import LayerNorm
from whittle.modules.linear import Linear
from whittle.modules.rmsnorm import RMSNorm


class GPT(nn.Module):
    """An extension of litgpt's GPT model with support to adapt to sub-network dimensionality."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
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
        self.sub_network_head_size: int | None = self.config.head_size
        self.sub_network_query_groups: int | None = self.config.n_query_groups
        self.sub_network_rope_n_elem = self.config.rope_n_elem
        self.cos: torch.Tensor
        self.sin: torch.Tensor
        self.config.is_encoder_decoder = False
        self.main_input_name = "input_pos"
        self._supports_cache_class = True
        self.sampled_intermediate_indices: list[int] | list[list[int]] | None = None
        self.sampled_head_indices: list[int] | list[list[int]] | None = None
        self.sampled_query_group_indices: list[int] | list[list[int]] | None = None
        self.sampled_head_size_indices: list[int] | list[list[int]] | None = None
        self.sampled_layer_indices: list[int] | None = None
        self.sampled_embd_indices: list[int] | None = None
        self.cos_list = None
        self.sin_list = None
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
            cos, sin = self.rope_cache(
                self._max_seq_length, self.config.rope_n_elem, device=torch.device("cpu")
            )
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(
                seq_len=self._max_seq_length,
                n_elem=self.sub_network_rope_n_elem,
                device=self.cos.device,
            )

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(
            seq_len=self._max_seq_length,
            n_elem=self.sub_network_rope_n_elem,
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
            self.transformer.wte.weight = self.lm_head.weight  # type: ignore

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
                    name: self.config.rope_adjustments[name]
                    for name in adjusted_params_required
                }
            elif "factor" in self.config.rope_adjustments:
                # linear RoPE
                adjusted_params_required = ["factor"]
                extra_config = {
                    name: self.config.rope_adjustments[name]
                    for name in adjusted_params_required
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
            rope_local_base_freq=self.config.rope_local_base_freq,
        )

    def set_block_config(self, block, i):
        def is_none_or_list_of_ints(obj):
            if obj is None:
                return True
            return isinstance(obj, list) and all(isinstance(x, int) for x in obj)

        def get_val(val, _type):
            if _type is int:
                return val if isinstance(val, int) else val[i]
            elif _type is list:
                return val if is_none_or_list_of_ints(val) else val[i]

        block.set_sub_network(
            sub_network_n_embd=self.sub_network_n_embd,
            sub_network_intermediate_size=get_val(
                self.sub_network_intermediate_size, int
            ),
            sub_network_num_heads=get_val(self.sub_network_num_heads, int),
            sub_network_query_groups=get_val(self.sub_network_query_groups, int),
            sub_network_head_size=get_val(self.sub_network_head_size, int),
            sampled_intermediate_indices=get_val(self.sampled_intermediate_indices, list),
            sampled_head_indices=get_val(self.sampled_head_indices, list),
            sampled_query_group_indices=get_val(self.sampled_query_group_indices, list),
            sampled_head_size_indices=get_val(self.sampled_head_size_indices, list),
            sampled_embd_indices=self.sampled_embd_indices,
        )

    def _verify_config(
        self,
        super_n_dim: int,
        sub_n_dim: int,
        sampled_dim_indices: list[int] | list[list[int]] | None,
        n_layers: int,
    ):
        if sampled_dim_indices is None:
            if isinstance(sub_n_dim, list) or isinstance(sub_n_dim, tuple):
                for dim in sub_n_dim:
                    if dim > super_n_dim:
                        raise IllegalSubNetworkError(
                            "Dimension of subnet cannot be greater than the supernet's."
                        )

            else:
                if sub_n_dim > super_n_dim:
                    raise IllegalSubNetworkError(
                        "Dimension of subnet cannot be greater than the supernet's."
                    )
            return
        elif isinstance(sampled_dim_indices, list):
            if len(sampled_dim_indices) == 0:
                raise IllegalSubNetworkError("List of indices cannot be empty!")
            if isinstance(sampled_dim_indices[0], list):  # list of lists
                if len(sampled_dim_indices) != n_layers:
                    raise IllegalSubNetworkError(
                        f"The number of lists of indices {len(sampled_dim_indices)} must"
                        " match the number of layers in the subnet ({n_layers})!"
                    )
                for i, list_of_indices in enumerate(sampled_dim_indices):
                    if isinstance(sub_n_dim, int):
                        sub_dim = sub_n_dim
                    else:
                        sub_dim = sub_n_dim[i]
                    if len(list_of_indices) != sub_dim:  # type: ignore
                        raise IllegalSubNetworkError(
                            f"Number of indices in {list_of_indices} does not match the"
                            " dimensions of the subnet."
                        )
            elif isinstance(sampled_dim_indices[0], int):
                if len(sampled_dim_indices) != sub_n_dim:
                    raise IllegalSubNetworkError(
                        f"Number of indices in {sampled_dim_indices} does not match the"
                        " dimensions of the subnet."
                    )
        else:
            raise IllegalSubNetworkError(
                f"f{sampled_dim_indices} is not a valid value for the list of indices."
            )

    def _infer_sub_network_sizes_from_indices(
        self,
        sub_network_n_embd: int | None = None,
        sub_network_intermediate_size: int | None = None,
        sub_network_num_heads: int | None = None,
        sub_network_n_layers: int | None = None,
        sub_network_query_groups: int | None = None,
        sub_network_head_size: int | None = None,
        sampled_intermediate_indices: list[int] | list[list] | None = None,
        sampled_head_indices: list[int] | list[list] | None = None,
        sampled_query_group_indices: list[int] | list[list] | None = None,
        sampled_head_size_indices: list[int] | list[list] | None = None,
        sampled_layer_indices: list[int] | None = None,
        sampled_embd_indices: list[int] | None = None,
    ):
        def infer_size(indices):
            if isinstance(indices[0], list):
                return len(indices[0])
            else:
                return len(indices)

        if sub_network_n_embd is None and sampled_embd_indices is not None:
            sub_network_n_embd = infer_size(sampled_embd_indices)

        if (
            sub_network_intermediate_size is None
            and sampled_intermediate_indices is not None
        ):
            sub_network_intermediate_size = infer_size(sampled_intermediate_indices)

        if sub_network_query_groups is None and sampled_query_group_indices is not None:
            if isinstance(sampled_query_group_indices[0], list):
                sub_network_query_groups: list[int] = []  # type: ignore
                for indices in sampled_query_group_indices:
                    sub_network_query_groups.append(len(indices))  # type: ignore
            else:
                sub_network_query_groups = len(sampled_query_group_indices)

        if sub_network_head_size is None and sampled_head_size_indices is not None:
            sub_network_head_size = infer_size(sampled_head_size_indices)

        if sub_network_n_layers is None and sampled_layer_indices is not None:
            sub_network_n_layers = infer_size(sampled_layer_indices)

        if sub_network_num_heads is None and sampled_head_indices is not None:
            if sub_network_query_groups is not None:
                sub_network_num_heads = (
                    len(sampled_head_indices) * sub_network_query_groups
                )
            else:
                sub_network_num_heads = (
                    len(sampled_head_indices) * self.config.n_query_groups
                )
        if sub_network_n_embd is None:
            sub_network_n_embd = self.config.n_embd

        if sub_network_intermediate_size is None:
            sub_network_intermediate_size = self.config.intermediate_size

        if sub_network_num_heads is None:
            sub_network_num_heads = self.config.n_head

        if sub_network_n_layers is None:
            sub_network_n_layers = self.config.n_layer

        if sub_network_query_groups is None:
            # sub_network_query_groups = sub_network_num_heads
            if (self.config.n_query_groups == self.config.n_head) and (
                sub_network_num_heads is not None
            ):
                sub_network_query_groups = sub_network_num_heads
            else:
                sub_network_query_groups = self.config.n_query_groups
        if sub_network_head_size is None:
            sub_network_head_size = self.config.head_size

        return {
            "sub_network_n_embd": sub_network_n_embd,
            "sub_network_intermediate_size": sub_network_intermediate_size,
            "sub_network_num_heads": sub_network_num_heads,
            "sub_network_n_layers": sub_network_n_layers,
            "sub_network_query_groups": sub_network_query_groups,
            "sub_network_head_size": sub_network_head_size,
        }

    def _verify_sub_network(
        self,
        sub_network_n_embd: int,
        sub_network_intermediate_size: int,
        sub_network_num_heads: int,
        sub_network_n_layers: int,
        sub_network_query_groups: int,
        sub_network_head_size: int,
        sampled_intermediate_indices: list[int] | list[list] | None = None,
        sampled_head_indices: list[int] | list[list] | None = None,
        sampled_query_group_indices: list[int] | list[list] | None = None,
        sampled_head_size_indices: list[int] | list[list] | None = None,
        sampled_layer_indices: list[int] | None = None,
        sampled_embd_indices: list[int] | None = None,
    ):
        if sampled_layer_indices is not None:
            if len(sampled_layer_indices) != sub_network_n_layers:
                raise IllegalSubNetworkError(
                    f"Number of layer indices ({len(sampled_layer_indices)}) do not "
                    f"match sub_network_n_layers ({sub_network_n_layers})"
                )
            elif max(sampled_layer_indices) >= self.config.n_layer:
                raise IllegalSubNetworkError(
                    f"The layer indices ({sampled_layer_indices}) must not be greater "
                    f"than the number of layers in the supernet ({self.config.n_layer})"
                )

        self._verify_config(
            self.config.n_embd,
            sub_network_n_embd,
            sampled_embd_indices,
            sub_network_n_layers,
        )
        self._verify_config(
            self.config.n_query_groups,
            sub_network_query_groups,
            sampled_query_group_indices,
            sub_network_n_layers,
        )
        self._verify_config(
            self.config.n_head,
            sub_network_num_heads,
            sampled_head_indices,
            sub_network_n_layers,
        )
        self._verify_config(
            self.config.head_size,
            sub_network_head_size,
            sampled_head_size_indices,
            sub_network_n_layers,
        )
        self._verify_config(
            self.config.intermediate_size,
            sub_network_intermediate_size,
            sampled_intermediate_indices,
            sub_network_n_layers,
        )

    def set_sub_network(
        self,
        sub_network_n_embd: int,
        sub_network_intermediate_size: int,
        sub_network_num_heads: int,
        sub_network_n_layers: int,
        sub_network_query_groups: int | None = None,
        sub_network_head_size: int | None = None,
        sampled_intermediate_indices: list[int] | list[list] | None = None,
        sampled_head_indices: list[int] | list[list] | None = None,
        sampled_query_group_indices: list[int] | list[list] | None = None,
        sampled_head_size_indices: list[int] | list[list] | None = None,
        sampled_layer_indices: list[int] | None = None,
        sampled_embd_indices: list[int] | None = None,
    ) -> None:
        """
        Sets the GPT model to the specified sub-network dimensionality.
        Input arguments are set to the specified sub-network dimensionality.

        Args:
            sub_network_n_embd: Embedding dimension of the sub-network.
            sub_network_intermediate_size: Intermediate size of the sub-network.
            sub_network_num_heads: Number of attention heads in the sub-network.
            sub_network_n_layers: Number of layers in the sub-network.
            sub_network_query_groups: Number of query groups in the sub-network. Defaults to None.
            sub_network_head_size: Size of each attention head in the sub-network. Defaults to None.
        """
        self.reset_super_network()
        sub_network_sizes = self._infer_sub_network_sizes_from_indices(
            sub_network_n_embd=sub_network_n_embd,
            sub_network_intermediate_size=sub_network_intermediate_size,
            sub_network_num_heads=sub_network_num_heads,
            sub_network_n_layers=sub_network_n_layers,
            sub_network_query_groups=sub_network_query_groups,
            sub_network_head_size=sub_network_head_size,
            sampled_intermediate_indices=sampled_intermediate_indices,
            sampled_head_indices=sampled_head_indices,
            sampled_query_group_indices=sampled_query_group_indices,
            sampled_head_size_indices=sampled_head_size_indices,
            sampled_layer_indices=sampled_layer_indices,
            sampled_embd_indices=sampled_embd_indices,
        )
        self._verify_sub_network(
            **sub_network_sizes,
            sampled_intermediate_indices=sampled_intermediate_indices,
            sampled_head_indices=sampled_head_indices,
            sampled_query_group_indices=sampled_query_group_indices,
            sampled_head_size_indices=sampled_head_size_indices,
            sampled_layer_indices=sampled_layer_indices,
            sampled_embd_indices=sampled_embd_indices,
        )

        self.sub_network_n_embd = sub_network_sizes["sub_network_n_embd"]
        self.sub_network_intermediate_size = sub_network_sizes[
            "sub_network_intermediate_size"
        ]
        self.sub_network_num_heads = sub_network_sizes["sub_network_num_heads"]
        self.sub_network_n_layers = sub_network_sizes["sub_network_n_layers"]
        self.sampled_intermediate_indices = sampled_intermediate_indices
        self.sampled_head_indices = sampled_head_indices
        self.sampled_query_group_indices = sampled_query_group_indices
        self.sampled_head_size_indices = sampled_head_size_indices
        self.sampled_layer_indices = sampled_layer_indices
        self.sampled_embd_indices = sampled_embd_indices

        self.transformer.wte.set_sub_network(  # type: ignore
            self.sub_network_n_embd, sampled_embd_indices
        )
        self.transformer.ln_f.set_sub_network(  # type: ignore
            self.sub_network_n_embd, sampled_embd_indices
        )

        self.sub_network_query_groups = (
            sub_network_sizes["sub_network_query_groups"]
            if sub_network_sizes["sub_network_query_groups"] is not None
            else self.config.n_query_groups
        )

        if self.config.fix_head_size:  # TODO: Deprecate. Not used (always set to True)
            if sub_network_sizes["sub_network_head_size"] is None:
                self.sub_network_head_size = self.config.head_size
            else:
                self.sub_network_head_size = sub_network_sizes["sub_network_head_size"]
        else:
            if sub_network_sizes["sub_network_head_size"] is not None:
                self.sub_network_head_size = sub_network_sizes["sub_network_head_size"]
            else:
                self.sub_network_head_size = (
                    self.sub_network_n_embd // self.sub_network_num_heads
                )

        if sampled_layer_indices is not None:
            for i, j in enumerate(sampled_layer_indices):
                block = self.transformer.h[j]  # type: ignore
                self.set_block_config(block, i)
        else:
            for i in range(self.sub_network_n_layers):
                block = self.transformer.h[i]  # type: ignore
                self.set_block_config(block, i)

        # these change inside causal_self_attention
        if self.sub_network_n_layers > 0:
            self.sub_network_query_groups = block.attn.sub_network_query_groups  # type: ignore

        self.lm_head.set_sub_network(
            self.sub_network_n_embd,
            self.config.padded_vocab_size,
            sampled_embd_indices,
        )

        self.sub_network_rope_n_elem = (
            math.ceil(self.config.rotary_percentage * self.sub_network_head_size)
            if isinstance(self.sub_network_head_size, int)
            else [
                math.ceil(self.config.rotary_percentage * head_size)
                for head_size in self.sub_network_head_size  # type: ignore
            ]
        )

        if isinstance(self.sub_network_rope_n_elem, int):
            self.cos, self.sin = self.rope_cache(
                seq_len=self._max_seq_length,
                n_elem=self.sub_network_rope_n_elem,
                device=self.cos.device,
            )
        else:
            self.cos_list = [None for _ in range(self.sub_network_n_layers)] # type: ignore
            self.sin_list = [None for _ in range(self.sub_network_n_layers)] # type: ignore
            for i, n_elem in enumerate(self.sub_network_rope_n_elem):
                self.cos_list[i], self.sin_list[i] = self.rope_cache( # type: ignore
                    seq_len=self._max_seq_length,
                    n_elem=n_elem,
                    device=self.cos.device,
                )

    def select_sub_network(self, config: dict[str, Any]) -> None:
        """
        Selects and sets the sub-network configuration based on the provided configuration.
        """
        self.set_sub_network(
            sub_network_n_embd=config["embed_dim"],
            sub_network_intermediate_size=int(config["mlp_ratio"] * config["embed_dim"]),
            sub_network_num_heads=config["num_heads"],
            sub_network_n_layers=config["depth"],
            sub_network_head_size=config.get("head_size", None),
            sub_network_query_groups=config.get("n_query_groups", None),
        )

    def reset_super_network(self):
        """
        Resets the GPT model to the original super-network dimensionality.
        """
        rebuild_rope = self.sub_network_rope_n_elem != self.config.rope_n_elem

        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_intermediate_size = self.config.intermediate_size
        self.sub_network_num_heads = self.config.n_head
        self.sub_network_n_layers = self.config.n_layer
        self.sub_network_head_size: int | None = self.config.head_size  # type: ignore
        self.sub_network_query_groups: int | None = self.config.n_query_groups  # type: ignore
        self.sub_network_rope_n_elem = self.config.rope_n_elem
        self.transformer.wte.reset_super_network()
        self.transformer.ln_f.reset_super_network()
        for i in range(self.config.n_layer):
            block = self.transformer.h[i]
            block.reset_super_network()
        self.lm_head.reset_super_network()
        self.sampled_intermediate_indices: list[int] | list[list[int]] | None = None
        self.sampled_head_indices: list[int] | list[list[int]] | None = None
        self.sampled_query_group_indices: list[int] | list[list[int]] | None = None
        self.sampled_head_size_indices: list[int] | list[list[int]] | None = None
        self.sampled_layer_indices: list[int] | None = None
        self.sampled_embd_indices: list[int] | None = None
        self.cos_list = None
        self.sin_list = None

        # rebuild the rope cache
        if rebuild_rope:
            self.reset_parameters()

    def process_rope_cache(self, cos, sin, input_pos, input_pos_maxp1, T):
        if input_pos is not None:  # use the kv cache
            if input_pos.dim() > 2:
                # otherwise, things go wrong in `apply_rope`
                raise ValueError(
                    f"input_pos must have 1 or 2 dimensions, input_pos.shape = {input_pos.shape}"
                )
            if input_pos.shape[-1] != T:
                raise ValueError(
                    f"input_pos.shape[-1] = {input_pos.shape[-1]} != {T} = idx.shape[1], must be the same"
                )
            cos = batched_index_select(cos, 0, input_pos)
            sin = batched_index_select(sin, 0, input_pos)
            if input_pos.dim() == 1:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = batched_index_select(self.mask_cache, 2, input_pos)
            if mask.dim() > 4:
                # the mask cache has a batch dim of 1 in addition to the one
                # we get if input_pos has a batch dimension
                mask = mask.view(*(mask.shape[0:1] + mask.shape[2:]))
            if input_pos_maxp1 is not None:
                # Shorten final dimension so it just covers all `input_pos` entries
                if input_pos_maxp1 > self.max_seq_length:
                    raise ValueError(
                        f"Positions in 'input_pos' must be in [0,{self.max_seq_length})"
                    )
                mask = mask[..., :input_pos_maxp1]
        else:
            # unsqueeze to have a batch dimension
            cos = cos[:T].unsqueeze(0)
            sin = sin[:T].unsqueeze(0)
            # `cos`, `sin` have shape (1, T, config.rope_n_elem)
            mask = None  # defaults to causal mask
            input_pos_maxp1 = None
        return cos, sin, mask, input_pos_maxp1

    def process_blocks(self, x, idx, input_pos, j, i, input_pos_maxp1, T):
        block = self.transformer.h[j]
        cos, sin, mask, input_pos_maxp1_block = self.process_rope_cache(
            self.cos, self.sin, input_pos, input_pos_maxp1, T
        )

        if isinstance(self.cos_list, list):
            cos, sin = self.cos_list[i].to(idx.device), self.sin_list[i].to(idx.device)
        else:
            cos, sin = self.cos.to(idx.device), self.sin.to(idx.device)
        cos, sin, mask, input_pos_maxp1_block = self.process_rope_cache(
            cos, sin, input_pos, input_pos_maxp1, T
        )
        x = block(x, cos, sin, mask, input_pos, input_pos_maxp1_block)

        return x

    def forward(
        self,
        idx: torch.Tensor,
        input_pos: torch.Tensor | None = None,
        input_pos_maxp1: torch.Tensor | None = None,
        lm_head_chunk_size: int = 0,
    ) -> torch.Tensor | list[torch.Tensor]:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )

        x = self.transformer.wte(idx)  # type: ignore  # token embeddings of shape (b, t, n_embd)
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.sub_network_n_embd**0.5, dtype=x.dtype)

        if self.sampled_layer_indices is not None:
            for i, j in enumerate(self.sampled_layer_indices):
                x = self.process_blocks(x, idx, input_pos, j, i, input_pos_maxp1, T)

        else:
            for i in range(self.sub_network_n_layers):
                x = self.process_blocks(x, idx, input_pos, i, i, input_pos_maxp1, T)

        x = self.transformer.ln_f(x)  # type: ignore
        clamp_head = (
            partial(do_softcapping, thresh=self.config.final_logit_softcapping)
            if self.config.final_logit_softcapping is not None
            else nn.Identity()
        )
        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            return [
                clamp_head(self.lm_head(x_i))
                for x_i in x.split(lm_head_chunk_size, dim=1)
            ]
        else:
            return clamp_head(self.lm_head(x))  # (B, T, padded_vocab_size)

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
            rope_cache_length = self.sub_network_rope_n_elem
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        # initialize the kv cache for all blocks
        for block in self.transformer.h:  # type: ignore
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size,
                max_seq_length,
                rope_cache_length=rope_cache_length,
                device=device,
                dtype=dtype,
                rope_n_elem=self.sub_network_rope_n_elem,
            )

        if self.mask_cache is None or self.mask_cache.size(3) != self.max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(self.max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:  # type: ignore
            block.attn.kv_cache = None


def build_mask_cache(
    max_seq_length: int, device: torch.device | None = None
) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)
