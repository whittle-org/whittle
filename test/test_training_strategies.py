import pytest
import torch.nn as nn
import torch.nn.functional
from litgpt import Config
from syne_tune.config_space import randint, choice
from lobotomy.models.gpt import GPT
from lobotomy.training_strategies import (
    SandwichStrategy,
    SandwichStrategyKD,
    RandomStrategy,
    StandardStrategy,
    RandomLinearStrategy,
    ATS,
)
from lobotomy.sampling.random_sampler import RandomSampler
from lobotomy.modules.linear import Linear


methods = [
    SandwichStrategy,
    SandwichStrategyKD,
    RandomStrategy,
    StandardStrategy,
    RandomLinearStrategy,
    ATS,
]

search_space_mlp = {"num_units": randint(1, 64)}

sampler_mlp = RandomSampler(config_space=search_space_mlp, seed=42)

loss_function = torch.nn.functional.mse_loss

search_space_gpt = {
    "embed_dim": randint(1, 64),
    "num_heads": choice([2, 4, 8]),
    "mlp_ratio": randint(1, 4),
    "depth": randint(1, 4),
}
sampler_gpt = RandomSampler(config_space=search_space_gpt, seed=42)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = 64
        self.input = Linear(input_dim, self.hidden_dim)
        self.output = Linear(self.hidden_dim, 1)

    def forward(self, x):
        x_ = self.input(x)
        x_ = self.output(x_)
        return x_

    def select_sub_network(self, config):
        self.input.set_sub_network(self.input_dim, config["num_units"])
        self.output.set_sub_network(config["num_units"], 1)

    def reset_super_network(self):
        self.output.reset_super_network()


@pytest.mark.parametrize("strategy", methods)
def test_integration_training_strategies_mlp(strategy):
    update_op = strategy(
        sampler=sampler_mlp,
        loss_function=loss_function,
        device="cpu",
        total_number_of_steps=1,
    )

    model = MLP(5)
    inputs = torch.rand((8, 5))
    outputs = torch.rand((8, 1))
    loss = update_op(model, inputs, outputs)
    assert isinstance(loss, float)


@pytest.mark.parametrize("strategy", methods)
def test_integration_training_strategies_gpt(strategy):

    update_op = strategy(
        sampler=sampler_gpt,
        loss_function=torch.nn.CrossEntropyLoss(),
        device="cpu",
        total_number_of_steps=1,
    )

    config = Config()
    config.padded_vocab_size = 512
    config.n_embd = 64
    config.intermediate_size = 64 * 4
    config.n_head = 8
    config.n_query_groups = 4
    config.head_size = 8
    config.n_layer = 8
    config.block_size = 512
    config.norm_class_name = "RMSNorm"
    config.mlp_class_name = "LLaMAMLP"
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)
    config.norm_eps = 1e-5
    config.lm_head_bias = True
    config.fix_head_size = True
    gpt = GPT(config)
    inputs = torch.randint(0, 512, (8, 512))
    outputs = torch.randn([8, 512, 512])
    loss = update_op(gpt, inputs, outputs)
    assert isinstance(loss, float)
