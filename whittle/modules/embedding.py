from __future__ import annotations

import torch
import torch.nn.functional as F


class Embedding(torch.nn.Embedding):
    "An extension of PyTorch's torch.nn.Embedding with support to sub-sample weights corresponding to the sub-network dimensionality"

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            device,
            dtype,
        )

        # the embedding dimensionality of the current sub-network
        self.sub_network_embedding_dim: int | None = embedding_dim
        self.sampled_embd_dim_indices: list[int] | None = None

    def set_sub_network(
        self,
        sub_network_embedding_dim: int,
        sampled_embd_dim_indices: list[int] | None = None,
    ):
        """Set the embedding dimensionality of the current sub-network."""
        self.sub_network_embedding_dim = sub_network_embedding_dim
        self.sampled_embd_dim_indices = sampled_embd_dim_indices

    def reset_super_network(self):
        """Reset the embedding dimensionality of the current sub-network to the super-network dimensionality"""
        self.sub_network_embedding_dim = self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = (
            self.weight[:, self.sampled_embd_dim_indices]
            if self.sampled_embd_dim_indices is not None
            else self.weight[:, : self.sub_network_embedding_dim]
        )

        return F.embedding(
            x,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
