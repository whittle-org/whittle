from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from whittle.modules import Linear

# TODO: move this later to sth like train_fashion_mnist.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.fc_base = Linear(28 * 28, 256, bias=True)
        self.fc1 = Linear(256, 120, bias=True)  # 5x5 image dimension
        self.fc2 = Linear(120, 84, bias=True)
        self.fc3 = Linear(84, 10, bias=True)

    def forward_dense_lottery(self, x, fc1_out, fc2_out):
        x = x.reshape(x.shape[0], -1)
        x = self.fc_base(x)
        x = F.relu(x)
        self.fc1.set_sub_network(
            sub_network_in_features=256, sub_network_out_features=fc1_out
        )
        x = self.fc1(x)
        self.fc1.reset_super_network()
        x = F.relu(x)

        self.fc2.set_sub_network(
            sub_network_in_features=fc1_out,
            sub_network_out_features=fc2_out,
        )
        x = self.fc2(x)
        self.fc2.reset_super_network()
        x = F.relu(x)

        self.fc3.set_sub_network(
            sub_network_in_features=fc2_out, sub_network_out_features=10
        )
        x = self.fc3(x)
        self.fc3.reset_super_network()
        return x

    @staticmethod
    def sample_ids(max_id: int, num_ids: int):
        return np.random.choice(np.arange(max_id), num_ids)

    @staticmethod
    def sample_f_in():
        return np.random.randint(1, 121), np.random.randint(1, 85)

    @staticmethod
    def sample_dense_dims():
        fc1_out = np.random.randint(1, 121)
        fc2_out = np.random.randint(1, 85)
        return fc1_out, fc2_out

    @staticmethod
    def sample_max():
        return 120, 84

    @staticmethod
    def sample_min():
        return 1, 1

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc_base(x)
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
