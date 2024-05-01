import torch
import torch.nn as nn
from lobotomy.models.litgpt.config import Config
from typing import Optional
from lobotomy.models.litgpt.super_modules.attention import CausalSelfAttention

class Block(nn.Module):
    def __init__(self, config: Config, rotary_emb: nn.Module) -> None:
        super().__init__()
        self.config = config
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = self.norm_class()(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, rotary_emb)
        self.norm_2 = None if config.shared_attention_norm else self.norm_class()(config.n_embd, eps=config.norm_eps)
        self.mlp = self.mlp_class()(config)

        

    def norm_class(self):
        # `self._norm_class` cannot be the type to keep the config json serializable
        from lobotomy.models.litgpt.super_layers.rmsnorm_super import RMSNormSuper
        from lobotomy.models.litgpt.super_layers.layernorm_super import LayerNormSuper
        if self.config.norm_class_name == "RMSNorm":

            return RMSNormSuper
        return LayerNormSuper    

    def mlp_class(self):
        # `self._mlp_class` cannot be the type to keep the config json serializable
        from lobotomy.models.litgpt.super_modules.mlp import GptNeoxMLP, LLaMAMLP, GemmaMLP
        if self.config.mlp_class_name == "LLaMAMLP":
            return LLaMAMLP
        elif self.config.mlp_class_name == "GemmaMLP":
            return GemmaMLP
        elif self.config.mlp_class_name == "GptNeoxMLP":
            return GptNeoxMLP
        else:
            raise ValueError(f"Unknown MLP class: {self.config._mlp_class}")
        
    def set_sample_config(self, sample_embed_dim: int, sample_intermediate_size:int, sample_num_heads:int) -> None:
        self.norm_1.set_sample_config(sample_embed_dim)
        self.attn.set_sample_config(sample_embed_dim, sample_num_heads)
        if not self.config.shared_attention_norm:
            self.norm_2.set_sample_config(sample_embed_dim)
        self.mlp.set_sample_config(sample_embed_dim, sample_intermediate_size)
    
    def forward(
        self,
        x: torch.Tensor,
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
        attention_output = self.attn(x_normed, mask, input_pos)

        if self.config.parallel_residual:
            x_normed = x_normed if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(x_normed) + attention_output + x
        else:
            x = attention_output + x
            x = self.mlp(self.norm_2(x)) + x
        return x