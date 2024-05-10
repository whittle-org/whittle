import pytest
import torch.nn as nn
import torch.nn.functional

from syne_tune.config_space import randint

from lobotomy.training_strategies import SandwichStrategy, RandomStrategy, StandardStrategy, RandomLinearStrategy, ATS
from lobotomy.sampling.random_sampler import RandomSampler
from lobotomy.modules.linear import Linear


methods = [SandwichStrategy, RandomStrategy, StandardStrategy, RandomLinearStrategy, ATS]

search_space = {"num_units": randint(1, 64)}

sampler = RandomSampler(config_space=search_space, seed=42)

loss_function = torch.nn.functional.mse_loss


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
        self.input.set_sub_network(self.input_dim, config['num_units'])
        self.output.set_sub_network(config["num_units"], 1)

    def reset_super_network(self):
        self.output.reset_super_network()


@pytest.mark.parametrize("strategy", methods)
def test_integration_training_strategies(strategy):

    update_op = strategy(
        sampler=sampler,
        loss_function=loss_function,
        device='cpu',
        total_number_of_steps=1,
    )

    model = MLP(5)
    inputs = torch.rand((8, 5))
    outputs = torch.rand((8, 1))
    loss = update_op(model, inputs, outputs)
    assert isinstance(loss, float)
