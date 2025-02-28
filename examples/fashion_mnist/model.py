from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from whittle.modules import Linear


class LeNet(nn.Module):
    def __init__(self, fc1_out: int, fc2_out: int, fc_base_out: int | None = 128):
        super().__init__()
        # FIXME: (fixup comment) 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.fc_base_out = fc_base_out
        self.fc_base = Linear(28 * 28, fc_base_out, bias=True)
        self.fc1 = Linear(fc_base_out, fc1_out, bias=True)  # 5x5 image dimension
        self.fc2 = Linear(fc1_out, fc2_out, bias=True)
        self.fc3 = Linear(fc2_out, 10, bias=True)

    def forward(self, x: torch.Tensor):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc_base(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def set_sub_network(self, config: dict[Literal["fc1_out", "fc2_out"], int]):
        fc1_out = config["fc1_out"]
        fc2_out = config["fc2_out"]

        self.fc1.set_sub_network(
            sub_network_in_features=self.fc_base_out, sub_network_out_features=fc1_out
        )
        self.fc2.set_sub_network(
            sub_network_in_features=fc1_out,
            sub_network_out_features=fc2_out,
        )
        self.fc3.set_sub_network(
            sub_network_in_features=fc2_out, sub_network_out_features=10
        )

    def reset_super_network(self):
        self.fc_base.reset_super_network()
        self.fc1.reset_super_network()
        self.fc2.reset_super_network()
        self.fc3.reset_super_network()
