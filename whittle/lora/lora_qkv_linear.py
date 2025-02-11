from whittle.lora.lora_linear import LoRALayer, LoRALinearQKV
from typing import Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        enable_lora: Union[bool, tuple[bool, bool, bool]] = False,
        **kwargs: Any,
    ):
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.config = config
        assert out_features == (n_head + 2 * n_query_groups) * head_size
        self.linear = LoRALinearQKV(in_features, out_features, **kwargs)
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
                head_size * n_head * self.enable_q,
                head_size * n_query_groups * self.enable_k,
                head_size * n_query_groups * self.enable_v,
            )
            self.qkv_shapes = [s for s in qkv_shapes if s]
            self.lora_B = nn.Parameter(
                torch.empty(sum(self.qkv_shapes), r)
            )  # (256, 2))
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

    @property
    def lora_ind(self) -> torch.Tensor:
        """Lazy creation of a buffer with LoRA indices to overcome the limitation when FSDP with meta device is used."""
        # Indices are needed to properly pad weight updates with zeros.
        if not hasattr(self, "_lora_ind"):
            enable_q, enable_k, enable_v = self.enable_lora
            qkv_group_size = self.sub_network_q_per_kv + 2
            candidate_indices = (
                range(self.sub_network_out_features)
                if self.qkv_indices is None
                else self.qkv_indices
            )
            lora_ind = []
            if enable_q:
                q_ind = [
                    x
                    for x in candidate_indices
                    if (x // self.sub_network_head_size) % qkv_group_size
                    < qkv_group_size - 2
                ]
                lora_ind.extend(q_ind)
            if enable_k:
                k_ind = [
                    x
                    for x in candidate_indices
                    if (x // self.sub_network_head_size) % qkv_group_size
                    == qkv_group_size - 2
                ]
                lora_ind.extend(k_ind)
            if enable_v:
                v_ind = [
                    x
                    for x in candidate_indices
                    if (x // self.sub_network_head_size) % qkv_group_size
                    == qkv_group_size - 1
                ]
                lora_ind.extend(v_ind)
            self.register_buffer(
                "_lora_ind",
                torch.tensor(lora_ind, device=self.linear.weight.device),
                persistent=False,
            )

        return self._lora_ind

    def zero_pad(self, x: torch.Tensor) -> torch.Tensor:
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
        # we need to do zero padding only if LoRA is disabled for one of QKV matrices
        if all(self.enable_lora):
            return x

        # Let's image that:
        # ⚬ input x has shape (64, 64, 256): (batch_size, sequence_length, embeddings_size)
        # ⚬ embeddings_size: 128
        # ⚬ self.linear.out_features: 384 (3 * embeddings_size)
        # ⚬ enable_lora: [True, False, True]
        # Then x has embeddings_size of 256 (2 * 128 as enable_lora only for query and value, not keys) and expected
        # embeddings_size is 384 (self.linear.out_features), so that means that we need to pad from 256 to 384 with zeros, but
        # only for key updates (this is where self.lora_ind comes in handy)
        result = x.new_zeros(
            *x.shape[:-1], self.sub_network_out_features
        )  # (64, 64, 384)
        return result.index_copy_(
            dim=-1, index=self.lora_ind, source=x
        )  # (64, 64, 384)

    def conv1d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """An extension of the `torch.nn.functional.conv1d` function with a logic specific to grouped queries.

        If the number of heads is equal to the number of query groups - grouped queries are disabled
        (see scheme in `litgpt/config.py:Config`). In this case the combined QKV matrix consists of equally sized
        query, key and value parts, which means we can utilize `groups` argument from `conv1d`: with this argument the
        input and weight matrices will be splitted in equally sized parts and applied separately (like having multiple
        conv layers side by side).

        Otherwise QKV matrix consists of unequally sized parts and thus we have to split input and weight matrices manually,
        apply each part of the weight matrix to the corresponding input's part and concatenate the result.

        Args:
            input: input matrix of shape (B, C, T)
            weight: weight matrix of shape (C_output, rank, 1).
                "C_output" is defined as a sum of embedding sizes for each enabled LoRA layer (see init method of the class).

        Returns:
            A tensor with a shape (B, C_output, T)

        """
        if self.config.n_head == self.config.n_query_groups:
            return F.conv1d(
                input, weight, groups=sum(self.enable_lora)
            )  # (B, C_output, T)

        # Notation:
        # ⚬ N: number of enabled LoRA layers (self.enable_lora)
        # ⚬ C_output': embeddings size for each LoRA layer (not equal in size)
        # ⚬ r: rank of all LoRA layers (equal in size)

        input_splitted = input.chunk(sum(self.enable_lora), dim=1)  # N * (B, C // N, T)
        qkv_shapes = [
            # if `head_size` is explicitly specified in the config, `n_embd` (or `in_features`)
            # might not be equal to `head_size * n_head`, thus we use it directly here
            self.sub_network_head_size
            * self.sub_network_query_groups
            * self.sub_network_q_per_kv
            * self.enable_q,
            self.sub_network_head_size * self.sub_network_query_groups * self.enable_k,
            self.sub_network_head_size * self.sub_network_query_groups * self.enable_v,
        ]
        qkv_shapes = [s for s in qkv_shapes if s]
        weight_splitted = weight.split(qkv_shapes)  # N * (C_output', r, 1)
        return torch.cat(
            [F.conv1d(a, b) for a, b in zip(input_splitted, weight_splitted)],
            dim=1,  # (B, C_output', T)
        )  # (B, C_output, T)

    def get_lora_AB(self) -> torch.Tensor:
        """Return merged lora_A and lora_B matrices with the same shape as the pretrained weights."""
        # Let's assume that:
        # ⚬ self.linear.weight.data: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)
        qkv_shapes = [
            # if `head_size` is explicitly specified in the config, `n_embd` (or `in_features`)
            # might not be equal to `head_size * n_head`, thus we use it directly here
            self.sub_network_head_size * self.sub_network_n_head * self.enable_q,
            self.sub_network_head_size * self.sub_network_query_groups * self.enable_k,
            self.sub_network_head_size * self.sub_network_query_groups * self.enable_v,
        ]
        qkv_shapes = [s for s in qkv_shapes if s]
        if self.qkv_indices is not None:
            lora = self.conv1d(
                self.lora_A[:, : self.sub_network_in_features].data.unsqueeze(
                    0
                ),  # (4, 128) -> (1, 4, 128)
                self.lora_B[self.qkv_indices, :].data.unsqueeze(
                    -1
                ),  # (256, 2) -> (256, 2, 1)
            ).squeeze(0)
        else:
            lora = self.conv1d(
                self.lora_A[:, : self.sub_network_in_features].data.unsqueeze(
                    0
                ),  # (4, 128) -> (1, 4, 128)
                self.lora_B[: sum(qkv_shapes), :].data.unsqueeze(
                    -1
                ),  # (256, 2) -> (256, 2, 1)
            ).squeeze(0)  # (1, 4, 128) @ (256, 2, 1) -> (1, 256, 128) -> (256, 128)
        return self.zero_pad(
            lora.T * self.scaling
        ).T  # (256, 128) after zero_pad (384, 128)

    def merge(self) -> None:
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and any(self.enable_lora) and not self.merged:
            super().merge()

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
        qkv_shapes = [
            # if `head_size` is explicitly specified in the config, `n_embd` (or `in_features`)
            # might not be equal to `head_size * n_head`, thus we use it directly here
            self.sub_network_head_size
            * self.sub_network_query_groups
            * self.sub_network_q_per_kv
            * self.enable_q,
            self.sub_network_head_size * self.sub_network_query_groups * self.enable_k,
            self.sub_network_head_size * self.sub_network_query_groups * self.enable_v,
        ]
        qkv_shapes = [s for s in qkv_shapes if s]
        if self.qkv_indices is not None:
            after_B = self.conv1d(
                after_A.transpose(-2, -1),  # (64, 64, 4) -> (64, 4, 64)
                self.lora_B[self.qkv_indices, :].unsqueeze(
                    -1
                ),  # (256, 2) -> (256, 2, 1)
            ).transpose(
                -2, -1
            )  # (64, 4, 64) @ (256, 2, 1) -> (64, 256, 64) -> (64, 64, 256)
        else:
            after_B = self.conv1d(
                after_A.transpose(-2, -1),  # (64, 64, 4) -> (64, 4, 64)
                self.lora_B[: sum(qkv_shapes), :].unsqueeze(
                    -1
                ),  # (256, 2) -> (256, 2, 1)
            ).transpose(-2, -1)
        lora = (
            self.zero_pad(after_B) * self.scaling
        )  # (64, 64, 256) after zero_pad (64, 64, 384)
        return pretrained + lora
