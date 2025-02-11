from litgpt.config import Config as BaseConfig
from dataclasses import dataclass
from whittle.lora.lora_mlps import (
    LoRAGptNeoxMLP as GptNeoxMLP,
    LoRALLaMAMLP as LLaMAMLP,
    LoRAGemmaMLP as GemmaMLP,
)


@dataclass
class LoRAConfig(BaseConfig):
    """
    Args:
        lora_r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        lora_alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        lora_*: whether to apply LoRA to the specified weights or not
    """

    lora_r: int = 4
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    lora_query: bool = False
    lora_key: bool = False
    lora_value: bool = False
    lora_projection: bool = False
    lora_mlp: bool = False
    lora_head: bool = False
    lora_emb: bool = False

    @property
    def mlp_class(self) -> type:
        if self.mlp_class_name == "GptNeoxMLP":
            return GptNeoxMLP
        elif self.mlp_class_name == "LLaMAMLP":
            return LLaMAMLP
        elif self.mlp_class_name == "GemmaMLP":
            return GemmaMLP
        else:
            raise ValueError(f"Unknown MLP class: {self.mlp_class_name}")
