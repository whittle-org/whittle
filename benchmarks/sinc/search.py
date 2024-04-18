# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import os

import numpy as np
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from pathlib import Path
from torch.utils.data import DataLoader


from lobotomy.search import multi_objective_search
from lobotomy.estimate_efficency import compute_mac_linear_layer

from sinc_nas import validate, f
from model import MLP,  select_sub_network, search_space

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--training_strategy", type=str, default='sandwich')
    parser.add_argument("--do_plot", type=bool, default=False)
    parser.add_argument("--st_checkpoint_dir", type=str, default="./checkpoints")

    args, _ = parser.parse_known_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.RandomState(42)
    num_data_points = 500
    x = rng.rand(num_data_points)
    print(x[:10])
    y = f(x)

    data = np.empty((num_data_points, 2))
    # data[:, 0] = (x - np.mean(x)) / np.std(x)
    # data[:, 1] = (y - np.mean(y)) / np.std(y)
    data[:, 0] = x
    data[:, 1] = y

    data = torch.tensor(data, dtype=torch.float)
    n_train = int(data.shape[0] * 0.7)
    train_data = data[:n_train]
    valid_data = data[n_train:]

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)

    model = MLP(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    path = Path(args.st_checkpoint_dir) / f'{args.training_strategy}_model_{args.hidden_dim}.pt'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state'])
    model.eval()

    def objective(config):
        handle = select_sub_network(model, config)
        loss = validate(model, valid_dataloader, device)
        handle.remove()

        mac = 0
        mac += compute_mac_linear_layer(model.input_layer.in_features, model.input_layer.out_features)
        mac += compute_mac_linear_layer(model.hidden_layer.in_features, config['num_units'])
        mac += compute_mac_linear_layer(model.output_layer.in_features, model.output_layer.out_features)

        return mac, loss

    results = multi_objective_search(objective, search_space, objective_kwargs={})

    print(results)
    costs = np.array(results['costs'])
    plt.scatter(costs[:, 0], costs[:, 1], color='black', label='sub-networks')

    idx = np.array(results['is_pareto_optimal'])
    plt.scatter(costs[idx, 0], costs[idx, 1], color='red', label='Pareto optimal')

    # plt.scatter(np.arange(1, num_subnets), grid)
    # plt.title(args.training_strategy)
    #
    # grid = json.load(open('grid.json'))
    # plt.scatter(grid['hidden_dim'], grid['val_loss'],
    #             color='black', label='from-scratch')
    #
    plt.xlabel('mac')
    plt.ylabel('Validation loss')
    # plt.ylim(0.005, 0.08)
    plt.xscale('log')
    plt.grid(linewidth="1", alpha=0.4)
    plt.show()


