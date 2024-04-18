import torch
import torch.nn as nn

from syne_tune.config_space import randint

search_space = {
    'num_units':  randint(1, 512),
}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, device="cuda"):
        super(MLP, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x_ = self.input_layer(x)
        x_ = torch.tanh(x_)
        x_ = self.hidden_layer(x_)
        x_ = torch.tanh(x_)
        x_ = self.output_layer(x_)
        return x_


def select_sub_network(model, config):
    hidden_layer = model.hidden_layer
    mask = torch.ones(hidden_layer.out_features)
    mask[config['num_units']:] = 0

    def hook(module, inputs, outputs):
        outputs = outputs * mask
        return outputs

    handle = hidden_layer.register_forward_hook(hook)
    return handle
