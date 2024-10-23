from __future__ import annotations

import pytest
import torch.nn as nn
import torch.nn.functional
from litgpt import Config
from syne_tune.config_space import choice, randint

from whittle.loss import DistillLoss
from whittle.models.gpt import GPT
from whittle.modules.linear import Linear
from whittle.sampling.random_sampler import RandomSampler
from whittle.training_strategies import (
    ATS,
    RandomLinearStrategy,
    RandomStrategy,
    SandwichStrategy,
    StandardStrategy,
)

methods = [
    SandwichStrategy,
    RandomStrategy,
    StandardStrategy,
    RandomLinearStrategy,
    ATS,
]

search_space_mlp = {"num_units": randint(1, 64)}

sampler_mlp = RandomSampler(config_space=search_space_mlp, seed=42)

loss_function = torch.nn.functional.mse_loss
search_space_gpt = {
    "embed_dim": randint(1, 32),
    "num_heads": choice([2, 4]),
    "mlp_ratio": randint(1, 2),
    "depth": randint(1, 2),
}
sampler_gpt = RandomSampler(config_space=search_space_gpt, seed=42)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    update_op = strategy(
        sampler=sampler_mlp,
        loss_function=loss_function,
        device=device,
        total_number_of_steps=1,
    )

    model = MLP(5).to(device)
    inputs = torch.rand((8, 5)).to(device)
    outputs = torch.rand((8, 1)).to(device)
    loss = update_op(model, inputs, outputs)
    assert isinstance(loss, float)


@pytest.mark.parametrize("strategy", methods)
@pytest.mark.parametrize("kd_loss", [None, DistillLoss(0.5, 0.5)])
def test_integration_training_strategies_gpt(strategy, kd_loss):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    update_op = strategy(
        sampler=sampler_gpt,
        loss_function=torch.nn.CrossEntropyLoss(),
        kd_loss=kd_loss,
        device=device,
        total_number_of_steps=1,
    )

    config = Config()
    config.padded_vocab_size = 128
    config.n_embd = 32
    config.intermediate_size = 32 * 2
    config.n_head = 4
    config.n_query_groups = 4
    config.head_size = 8
    config.n_layer = 2
    config.block_size = 128
    config.norm_class_name = "RMSNorm"
    config.mlp_class_name = "LLaMAMLP"
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)
    config.norm_eps = 1e-5
    config.lm_head_bias = True
    config.fix_head_size = True
    gpt = GPT(config).to(device)
    inputs = torch.randint(0, 128, (1, 128)).to(device)
    outputs = torch.randn([1, 128, 128]).to(device)
    loss = update_op(gpt, inputs, outputs)
    assert isinstance(loss, float)
