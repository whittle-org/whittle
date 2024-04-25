import torch
import torch.nn.functional as F

from typing import Optional


class Embedding(torch.nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 device=None, dtype=None) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, device, dtype)

        # the embedding dimensionality of the current sub-network
        self.sub_network_embedding_dim = None

    def set_sub_network(self, sub_network_embedding_dim: int):
        self.sub_network_embedding_dim = sub_network_embedding_dim

    def reset_super_network(self):
        self.sub_network_embedding_dim = self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            x,
            self.weight[:, : self.sample_embed_dim],
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
