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

import numpy as np
import torch
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from pathlib import Path
from torch.utils.data import DataLoader


def compute_mag(model, mask):
    magnitude = 0
    for params in model.input_layer.parameters():
        magnitude += torch.sum(torch.abs(params))

    binary_mask = torch.tensor(mask, dtype=torch.bool)
    for params in model.input_layer.parameters():
            if len(params.shape) == 1:
                p = params[binary_mask]
            else:
                p = params[binary_mask, :]
            magnitude += torch.sum(torch.abs(p))
    for params in model.output_layer.parameters():
        magnitude += torch.sum(torch.abs(params))

    return float(magnitude)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--training_strategy", type=str, default='standard')
    parser.add_argument("--do_plot", type=bool, default=False)
    parser.add_argument("--st_checkpoint_dir", type=str, default="./checkpoints")

    args, _ = parser.parse_known_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.RandomState(42)
    num_data_points = 500
    x = rng.rand(num_data_points)
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

    # path = Path(args.st_checkpoint_dir) / f'{args.training_strategy}_model_{128}.pt'
    path = Path(args.st_checkpoint_dir) / f'{args.training_strategy}_model.pt'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state'])
    model.eval()

    num_subnets = model.hidden_dim
    grid = []
    grid_mag = []
    for i in range(1, num_subnets):
        mask = torch.zeros(model.hidden_dim, )
        mask[:i] = 1
        loss = validate(model, valid_dataloader, mask=mask, device=device)
        grid.append(float(loss))
        mag = compute_mag(model, mask)
        grid_mag.append(mag)

    plt.scatter(grid_mag, grid)

    plt.title(args.training_strategy)

    # grid = json.load(open('grid.json'))
    # plt.scatter(grid['hidden_dim'], grid['val_loss'],
    #             color='black', label='from-scratch')

    plt.xlabel('weight magnitude')
    plt.ylabel('Validation loss')
    # plt.ylim(0.01, 0.12)
    plt.grid(linewidth="1", alpha=0.4)
    plt.show()



