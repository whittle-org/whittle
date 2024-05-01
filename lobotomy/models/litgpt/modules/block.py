import torch
import torch.nn as nn
from lobotomy.models.litgpt.config import Config
from typing import Optional
from lobotomy.models.litgpt.modules.attention import CausalSelfAttention

class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓             ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───►  norm_2
           │  ↓                     │  ↓             ↓
           │  attn                  │  attn          mlp
           │  ↓                     │  ↓             │
        ┌─ └► +                     └► + ◄───────────┘
        │     norm_2
        │     ↓
        │     mlp
        │     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output = self.attn(x_normed, cos, sin, mask, input_pos)

        if self.config.parallel_residual:
            x_normed = x_normed if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(x_normed) + attention_output + x
        else:
            x = attention_output + x
            x = self.mlp(self.norm_2(x)) + x
        return x