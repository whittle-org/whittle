import litgpt
from litgpt import Config

from lobotomy.modules import Linear


class GptNeoxMLP(litgpt.model.GptNeoxMLP):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = Linear(config.n_embd, config.intermediate_size)
        self.proj = Linear(config.intermediate_size, config.n_embd)
        self.config = config

    def set_sub_network(
        self,
        sub_network_n_embd: int,
        sub_network_intermediate_size: int,
        use_bias: bool = True,
    ):
        self.sub_network_n_embd = sub_network_n_embd
        self.sub_network_intermediate_size = sub_network_intermediate_size
        self.use_bias = use_bias

        self.fc.set_sub_network(self.sub_network_n_embd, self.sub_network_intermediate_size)
        self.proj.set_sub_network(self.sub_network_intermediate_size, self.sub_network_n_embd)

    def reset_super_network(self):
        self.sub_network_n_embd = self.config.n_embd
        self.sub_network_intermediate_size = self.config.intermediate_size

        self.fc.reset_super_network()
        self.proj.reset_super_network()

#
# class LLaMAMLP(nn.Module):
#     def __init__(self, config: Config) -> None:
#         super().__init__()
#         self.fc_1 = Linear(config.n_embd, config.intermediate_size)
#         self.fc_2 = Linear(config.n_embd, config.intermediate_size)
#         self.proj = Linear(config.intermediate_size, config.n_embd)
#         self.act = nn.SiLU()
#
#     def set_sample_config(
#         self,
#         sample_embed_dim: int,
#         sample_intermediate_size: int,
#         sample_bias_flag: bool,
#     ) -> None:
#         self.fc_1.set_sample_config(
#             sample_embed_dim, sample_intermediate_size, sample_bias_flag
#         )
#         self.fc_2.set_sample_config(
#             sample_embed_dim, sample_intermediate_size, sample_bias_flag
#         )
#         self.proj.set_sample_config(
#             sample_intermediate_size, sample_embed_dim, sample_bias_flag
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_fc_1 = self.fc_1(x)
#         x_fc_2 = self.fc_2(x)
#         x = checkpoint(self.act, x_fc_1) * x_fc_2
#         return self.proj(x)
