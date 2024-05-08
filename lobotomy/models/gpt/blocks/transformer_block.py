import torch
import torch.nn as nn
from typing import Optional
import litgpt
from litgpt import Config
from lobotomy.models.gpt.blocks.causal_self_attention import CausalSelfAttention
from lobotomy.modules.rmsnorm import RMSNorm
from lobotomy.modules.layernorm import LayerNorm
from lobotomy.models.gpt.blocks.mlp import GptNeoxMLP, LLaMAMLP, GemmaMLP

class Block(litgpt.model.Block):
    def __init__(self, config: Config, rotary_emb: nn.Module) -> None:
        super().__init__(config)
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

        self.sub_network_n_embd = None
        self.sub_network_intermediate_size = None
        self.sub_network_num_heads = None


    def norm_class(self):
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self.config.norm_class_name == "RMSNorm":

            return RMSNorm
        return LayerNorm

    def mlp_class(self):
        # `self._mlp_class` cannot be the type to keep the config json serializable
        if self.config.mlp_class_name == "LLaMAMLP":
            return LLaMAMLP
        elif self.config.mlp_class_name == "GemmaMLP":
            return GemmaMLP
        elif self.config.mlp_class_name == "GptNeoxMLP":
            return GptNeoxMLP
        else:
            raise ValueError(f"Unknown MLP class: {self.config._mlp_class}")
    
    def set_sub_network(
            self,
            sub_network_n_embd: int,
            sub_network_intermediate_size: int,
            sub_network_num_heads: int,
        ) -> None:
        self.sub_network_n_embd = sub_network_n_embd
        self.sub_network_intermediate_size = sub_network_intermediate_size
        self.sub_network_num_heads = sub_network_num_heads
        self.norm_1.set_sub_network(self.sub_network_n_embd)
        self.attn.set_sub_network(self.sub_network_n_embd, self.sub_network_num_heads)
        if not self.config.shared_attention_norm:
            self.norm_2.set_sub_network(self.sub_network_n_embd)
        self.mlp.set_sub_network(self.sub_network_n_embd, self.sub_network_intermediate_size)

    def reset_super_network(self):
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_intermediate_size = self.config.intermediate_size
        self.sub_network_num_heads = self.config.n_head
        self.norm_1.reset_super_network()
        self.attn.reset_super_network()
        if not self.config.shared_attention_norm:
            self.norm_2.reset_super_network()
        self.mlp.reset_super_network()

    
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
        #print("Sum after norm_1: ", x_normed.sum())
        attention_output = self.attn(x_normed, mask, input_pos)
        #print("Sum after attention: ", attention_output.sum())

        if self.config.parallel_residual:
            x_normed = x_normed if self.config.shared_attention_norm else self.norm_2(x)
            #print("sum after norm_2: ", x_normed.sum())
            x = self.mlp(x_normed) + attention_output + x
            #print("Sum after mlp: ", x.sum())
        else:
            x = attention_output + x
            #print("Sum after attention: ", x.sum())
            x = self.mlp(self.norm_2(x)) + x
            #print("Sum after mlp: ", x.sum())
        return x