from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from litgpt.lora import LoRALayer

from whittle.modules.linear import LinearQKV


class LoRAQKVLinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        config,
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        head_size: int,
        n_head: int,
        n_query_groups: int,
        fix_head_size: bool,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: bool | tuple[bool, bool, bool] = False,
        **kwargs: Any,
    ):
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.config = config
        assert out_features == (n_head + 2 * n_query_groups) * head_size
        self.linear = LinearQKV(in_features, out_features, **kwargs)
        self.use_bias = self.linear.use_bias
        self.head_size = head_size
        self.fix_head_size = fix_head_size
        self.n_head = n_head
        self.in_features = in_features
        self.out_features = out_features
        self.n_query_groups = n_query_groups
        if isinstance(enable_lora, bool):
            enable_lora = (enable_lora, enable_lora, enable_lora)
        assert len(enable_lora) == 3
        self.enable_lora = enable_lora
        self.sub_network_in_features = in_features
        self.sub_network_out_features = out_features
        self.sub_network_head_size = head_size
        self.sub_network_n_head = n_head
        self.sub_network_query_groups = n_query_groups
        self.q_per_kv = n_head // n_query_groups
        self.sub_network_q_per_kv = self.q_per_kv
        self.qkv_indices = None
        self.merged = False
        # Actual trainable parameters
        # To better understand initialization let's imagine that we have such parameters:
        # ⚬ in_features: 128 (embeddings_size)
        # ⚬ out_features: 384 (3 * embedding_size)
        # ⚬ r: 2
        # ⚬ enable_lora: [True, False, True]
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                torch.empty((r * sum(enable_lora), in_features))
            )  # (4, 128)
            self.enable_q, self.enable_k, self.enable_v = enable_lora
            # qkv_shapes will be used to split a tensor with weights correctly
            qkv_shapes = (
                # if `head_size` is explicitly specified in the config, `n_embd` (or `in_features`)
                # might not be equal to `head_size * n_head`, thus we use it directly here
                self.sub_network_head_size
                * self.sub_network_query_groups
                * self.sub_network_q_per_kv,
                head_size * n_query_groups,
                head_size * n_query_groups,
            )
            self.qkv_shapes = [s for s in qkv_shapes if s]
            self.lora_B = nn.Parameter(torch.empty(sum(self.qkv_shapes), r))  # (256, 2))
            # Notes about shapes above
            # - self.lora_A has shape (4, 128): 4 because rank is 2 and LoRA is applied only to two matrices;
            # 128 is the input size of the x (embedding size). (4, 128) and not (128, 4) because later on in
            # F.linear function weights are automatically transposed. In addition conv1d requires channels to
            # be before seq length
            # - self.lora_B has shape (256, 2): 256 because LoRA is applied only to two matrices, so the output is
            # 128*2; 2 tells to have two channels per group for group convolution

            # Scaling:
            # This balances the pretrained model`s knowledge and the new task-specific adaptation
            # https://lightning.ai/pages/community/tutorial/lora-llm/
            # So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set
            # alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can
            # tune these values to your needs. This value can be even slightly greater than 1.0!
            # https://github.com/cloneofsimo/lora
            self.scaling = self.lora_alpha / self.r

            self.reset_parameters()

    def set_sub_network(
        self,
        sub_network_in_features: int,
        sub_network_out_features: int,
        qkv_indices=None,
        sub_network_n_head=None,
        sub_network_query_groups=None,
        sub_network_head_size=None,
        sub_network_q_per_kv=None,
    ):
        self.sub_network_in_features = sub_network_in_features
        self.sub_network_out_features = sub_network_out_features
        self.sub_network_n_head = sub_network_n_head
        self.sub_network_query_groups = sub_network_query_groups
        self.sub_network_head_size = sub_network_head_size
        self.sub_network_q_per_kv = sub_network_q_per_kv
        self.linear.set_sub_network(
            sub_network_in_features, sub_network_out_features, qkv_indices
        )
        self.qkv_indices = qkv_indices

        # trigger resetting the indices for LoRA
        self._set_lora_ind()

    def reset_super_network(self):
        """Resets the dimensionality of the current sub-network to the super-network dimensionality."""
        self.sub_network_in_features = self.in_features
        self.sub_network_out_features = self.out_features
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_n_head = self.config.n_head
        self.q_per_kv = self.config.n_head // self.config.n_query_groups
        self.sub_network_head_size = self.config.head_size
        self.sub_network_qkv_shape = (
            self.config.n_head + 2 * self.config.n_query_groups
        ) * self.config.head_size
        self.sub_network_query_groups = self.config.n_query_groups
        self.sub_network_q_per_kv = self.q_per_kv

        self.linear.reset_super_network()
        self.sub_attention_scaler = self.config.attention_scores_scalar
        self.qkv_indices = None

        # trigger resetting the indices for LoRA
        self._set_lora_ind()

    @property
    def q_ind(self) -> list[int]:
        # compared to litgpt, we need these indices as properties to ensure correct indexing of sub-networks
        if not hasattr(self, "_q_ind"):
            self._set_lora_ind()

        return self._q_ind

    @property
    def k_ind(self) -> list[int]:
        if not hasattr(self, "_k_ind"):
            self._set_lora_ind()

        return self._k_ind

    @property
    def v_ind(self) -> list[int]:
        if not hasattr(self, "_v_ind"):
            self._set_lora_ind()

        return self._v_ind

    @property
    def q_target(self) -> list[int]:
        if not hasattr(self, "_q_target"):
            self._set_lora_ind()

        return self._q_target

    @property
    def k_target(self) -> list[int]:
        if not hasattr(self, "_k_target"):
            self._set_lora_ind()

        return self._k_target

    @property
    def v_target(self) -> list[int]:
        if not hasattr(self, "_v_target"):
            self._set_lora_ind()

        return self._v_target

    def _set_lora_ind(self) -> None:
        """Set the indices for LoRA."""
        enable_q, enable_k, enable_v = self.enable_lora
        qkv_group_size = self.sub_network_q_per_kv + 2
        candidate_indices = (
            range(self.sub_network_out_features)
            if self.qkv_indices is None
            else self.qkv_indices
        )

        q_ind: list[int] = []
        k_ind: list[int] = []
        v_ind: list[int] = []
        q_target: list[int] = []
        k_target: list[int] = []
        v_target: list[int] = []
        if enable_q:
            q_indices: list[tuple[int, int]] = [
                (x, i)
                for i, x in enumerate(candidate_indices)
                if (i // self.sub_network_head_size) % qkv_group_size < qkv_group_size - 2
            ]
            q_ind, q_target = zip(*q_indices)  # type: ignore[assignment]
        if enable_k:
            k_indices: list[tuple[int, int]] = [
                (x, i)
                for i, x in enumerate(candidate_indices)
                if (i // self.sub_network_head_size) % qkv_group_size
                == qkv_group_size - 2
            ]
            k_ind, k_target = zip(*k_indices)  # type: ignore[assignment]
        if enable_v:
            v_indices: list[tuple[int, int]] = [
                (x, i)
                for i, x in enumerate(candidate_indices)
                if (i // self.sub_network_head_size) % qkv_group_size
                == qkv_group_size - 1
            ]
            v_ind, v_target = zip(*v_indices)  # type: ignore[assignment]

        # *_ind indices are the same as self.qkv_indices, only splitted into 3 parts (for q, k and v)
        # *_target indices serve for populating the resulting tensor -> since we do not index in super-network
        # anymore, we need indices relative to the sub-network tensor (e.g. for out_features == 16)
        # and sub-network out_features == 3, if qkv_indices are [0, 14, 15], our target indices will be [0, 1, 2]
        all_indices = [
            (q_ind, "_q_ind"),
            (k_ind, "_k_ind"),
            (v_ind, "_v_ind"),
            (q_target, "_q_target"),
            (k_target, "_k_target"),
            (v_target, "_v_target"),
        ]

        # lazy creation of a buffer with LoRA indices to overcome the limitation when FSDP with meta device is used
        for index_value, index_name in all_indices:
            index_tensor = torch.tensor(index_value, device=self.linear.weight.device)
            if not hasattr(self, index_name):
                self.register_buffer(index_name, index_tensor, persistent=False)
            else:
                setattr(self, index_name, index_tensor)

    def reset_parameters(self) -> None:
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Properly pad the last dimension of weight updates with zeros.

        If, based on `self.enable_lora`, we want to fine-tune queries and values, but not keys,
        then the weights update should be:

        [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
         [....................................],
         [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            ↑              ↑            ↑
        ________________________________________
        | query         | key       | value    |
        ----------------------------------------
        For Llama2's GQA support, Q, K, and V weights are interleaved, so that weights for grouped
        queries are adjacent to their associated key and value weights.
        For example, suppose we have n_head = 12 with 3 query groups.
        Then along the embedding dimension the interleaved weights would look like

        [Q, Q, Q, Q, K, V, Q, Q, Q, Q, K, V, Q, Q, Q, Q, K, V],

        where each Q, K, and V has size head_size.

        In this case, the previously-described weight update applies separately to each
        individual block, so the update will take the form

        [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW, ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW, ...],
         [.............................................................................],
         [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW, ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW, ...]]
             ↑              ↑            ↑        ↑             ↑            ↑
        ________________________________________________________________________________
        | q block 1 | k block 1  | v block 1 | q block 2 |  k block 2 |  v block 2 | ...
        --------------------------------------------------------------------------------
        Note that in the above diagram, the size of each q block will equal q_per_kv
        times the size of each k and v block.

        Args:
            x: tensor with weights update that will be padded with zeros if necessary

        Returns:
            A tensor with weight updates and zeros for deselected q, k or v
        """
        # we need to scatter the indices every time because sub-network needs zero padding
        # and super-network lora updates have to align with the other weights

        # In litgpt, there's a bug where [Q1, K1, V1, Q2, K2, V2] += [ΔQ1, ΔQ1, ΔK1, ΔK2, ΔKV1 ΔV2]
        # (this occurs for all versions that still support interleaving).
        # For super-network training, we need this to be correct.

        # Let's image that:
        # ⚬ input x has shape (64, 64, 256): (batch_size, sequence_length, embeddings_size)
        # ⚬ embeddings_size: 128
        # ⚬ self.linear.out_features: 384 (3 * embeddings_size)
        # ⚬ enable_lora: [True, False, True]
        # Then x has embeddings_size of 256 (2 * 128 as enable_lora only for query and value, not keys) and expected
        # embeddings_size is 384 (self.linear.out_features), so that means that we need to pad from 256 to 384 with zeros, but
        # only for key updates (this is where self.lora_ind comes in handy)

        # ensures the same device and shape as the weights
        result = x[0].new_zeros(
            *x[0].shape[:-1], self.sub_network_out_features
        )  # (64, 64, 384)

        active_inds = [self.q_target, self.k_target, self.v_target]
        active_inds = [ind for ind in active_inds if len(ind) > 0]

        for ind, weight in zip(active_inds, x):
            result = result.index_copy_(dim=-1, index=ind, source=weight)  # (64, 64, 384)

        return result

    def conv1d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """An extension of the `torch.nn.functional.conv1d` function with a logic specific to grouped queries.

        Lora in litgpt is applied separately to keys, queries and values. Because of sub-network selection,
        we need flexible indexing to select parts of the lora_B matrix that correspond to active queries, keys and values.
        Since QKV are interleaved (i.e. QQKV QQKV ... QQKV), we need to select them in `_set_lora_ind`, and then call `_split_lora_B`
        to get the corresponding parts of the weight matrix. Then, we apply each part of the weight matrix to the
        corresponding input's part and concatenate the result.

        Compared to litgpt, we don't use `groups` in case the number of heads is equal to the number of query groups
        (grouped queries are disabled). We still need the more complex indexing because of sub-network selection.

        Args:
            input: input matrix of shape (B, C, T)
            weight: weight matrix of shape (C_output, rank, 1).
                "C_output" is defined as a sum of embedding sizes for each enabled LoRA layer (see init method of the class).

        Returns:
            A tensor with a shape (B, C_output, T)

        """

        # Notation:
        # ⚬ N: number of enabled LoRA layers (self.enable_lora)
        # ⚬ C_output': embeddings size for each LoRA layer (not equal in size)
        # ⚬ r: rank of all LoRA layers (equal in size)

        input_splitted = input.chunk(sum(self.enable_lora), dim=1)  # N * (B, C // N, T)
        # (256, 2) -> (256, 2, 1) -> split to (q, 2, 1), (k, 2, 1), (v, 2, 1)
        # (k + v + q = 256, k = v, q = self.q_per_kv * k)

        active_inds = [self.q_ind, self.k_ind, self.v_ind]
        active_inds = [ind for ind in active_inds if len(ind) > 0]

        weight_splitted = [weight[ind].data.unsqueeze(-1) for ind in active_inds]
        return [F.conv1d(a, b) for a, b in zip(input_splitted, weight_splitted)]

    def get_lora_AB(self) -> torch.Tensor:
        """Return merged lora_A and lora_B matrices with the same shape as the pretrained weights."""
        # Let's assume that:
        # ⚬ self.linear.weight.data: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)
        lora = self.conv1d(
            self.lora_A[:, : self.sub_network_in_features].data.unsqueeze(
                0
            ),  # (4, 128) -> (1, 4, 128)
            self.lora_B,  # (256, 2) -> (256, 2, 1) -> splitted (inside conv1d)
        )  # (1, 4, 128) @ (256, 2, 1) -> (1, 256, 128) -> (256, 128)
        return self.zero_pad(
            [lora_w.squeeze(0).T * self.scaling for lora_w in lora.T]
        ).T  # (256, 128) after zero_pad (384, 128)

    def merge(self) -> None:
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and any(self.enable_lora) and not self.merged:
            pretrained_dtype = self.linear.weight.data.dtype
            lora_data = self.get_lora_AB()
            # if only the pretrained are in quantized form - dequantize, sum with LoRA and quantize the result
            if pretrained_dtype == torch.uint8:
                import bitsandbytes as bnb

                weight = self.linear.weight
                # dequantize the pretrained weights
                weight_data = bnb.functional.dequantize_4bit(
                    weight.data, weight.quant_state
                ).to(lora_data.dtype)
                # add pretrained and LoRA weights
                weight_data += lora_data
                # assign updated weights and quantize by moving to CUDA device
                self.linear.weight = bnb.nn.Params4bit(
                    weight_data, requires_grad=False, **weight.__dict__
                )
                self.linear.weight.cuda(weight.device)
            else:
                # self.linear might be on CPU and lora_data on CUDA
                # the inplace add will preserve the dtype of linear.weight
                self.linear.weight.data += lora_data.to(
                    device=self.linear.weight.data.device
                )
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass.

        If LoRA's weights are merged with pretrained ones then it's a simple matrix multiplication.
        If not, then multiply pretrained weights with input, apply LoRA on input and do summation.

        Args:
            x: input tensor of shape (batch_size, context_length, embedding_size)

        Returns:
            Output tensor of shape (batch_size, context_length, 3 * embedding_size)
        """

        # Let's assume that:
        # ⚬ x: (64, 64, 128) or (batch_size, context_length, embedding_size)
        # ⚬ self.linear.weight: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)

        # if weights are merged or LoRA is disabled (r <= 0 or all `enable_lora` are False) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or not any(self.enable_lora) or self.merged:
            return pretrained
        after_A = F.linear(
            self.lora_dropout(x), self.lora_A[:, : self.sub_network_in_features]
        )  # (64, 64, 128) @ (4, 128) -> (64, 64, 4)
        # For F.conv1d:
        # ⚬ input: input tensor of shape (mini-batch, in_channels, iW)
        # ⚬ weight: filters of shape (out_channels, in_channels/groups, kW)

        # changes compared to litgpt - we need more flexible indexing because of sub-network selection
        after_B = self.conv1d(
            after_A.transpose(-2, -1),  # (64, 64, 4) -> (64, 4, 64)
            self.lora_B,  # (256, 2) -> (256, 2, 1) -> split to (q, 2, 1), (k, 2, 1), (v, 2, 1)
        )  # (64, 4, 64) @ (256, 2, 1) -> (64, 256, 64)

        lora = self.zero_pad(
            [a.transpose(-2, -1) * self.scaling for a in after_B]
        )  # (64, 64, 256) after zero_pad (64, 64, 384)
        return pretrained + lora
