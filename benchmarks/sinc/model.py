import torch
import torch.nn as nn

from syne_tune.config_space import randint

from lobotomy.modules import Linear

search_space = {
    'num_units':  randint(1, 512),
}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, device="cuda"):

        super(MLP, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_layer = Linear(input_dim, hidden_dim)
        self.hidden_layer = Linear(hidden_dim, hidden_dim)
        self.output_layer = Linear(hidden_dim, 1)

    def forward(self, x):
        x_ = self.input_layer(x)
        x_ = torch.tanh(x_)
        x_ = self.hidden_layer(x_)
        x_ = torch.tanh(x_)
        x_ = self.output_layer(x_)
        return x_

    def select_sub_network(self, config):
        self.hidden_layer.set_sub_network(self.hidden_dim, config['num_units'])
        self.output_layer.set_sub_network(config['num_units'], 1)

    def reset_super_network(self):
        self.hidden_layer.reset_super_network()
        self.output_layer.reset_super_network()
