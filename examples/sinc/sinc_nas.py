import os

import numpy as np
import json
import torch
import torch.nn as nn

from argparse import ArgumentParser
from pathlib import Path
from torch.utils.data import DataLoader
from syne_tune.config_space import randint
from syne_tune.report import Reporter
from lobotomy.training_strategies import SandwichStrategy
from lobotomy.sampling.random_sampler import RandomSampler

from model import MLP

report = Reporter()


def f(x):
    return np.sinc(x * 10 - 5)
    # return (x - 0.5) ** 2 + np.random.randn() * 1e-3


def validate(model, valid_loader, device):
    model.eval()
    overall_loss = 0

    for batch_idx, batch in enumerate(valid_loader):
        x = batch[:, 0].reshape(-1, 1)
        y = batch[:, 1].reshape(-1, 1)
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = nn.functional.mse_loss(y, y_hat)

        overall_loss += loss.item()

    return overall_loss / (batch_idx + 1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--training_strategy", type=str, default="sandwich")
    parser.add_argument("--do_plot", type=bool, default=False)
    parser.add_argument("--st_checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    search_space = {"num_units": randint(1, args.hidden_dim)}
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.st_checkpoint_dir, exist_ok=True)
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

    model = MLP(input_dim=1, hidden_dim=args.hidden_dim, device=device).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    current_best = None
    # if args.st_checkpoint_dir is not None:
    #     os.makedirs(args.st_checkpoint_dir, exist_ok=True)
    lc_valid = []
    lc_train = []

    sampler = RandomSampler(search_space, seed=args.seed)
    training_strategies = {
        # 'standard': train_epoch,
        "sandwich": SandwichStrategy(
            sampler=sampler, loss_function=nn.functional.mse_loss
        ),
        # 'sandwich_kd': train_sandwich_kd,
        # 'random': train_random,
        # 'ats': train_ats,
        # 'random_linear': partial(train_random_linear,  total_number_of_steps=args.epochs * 500 // args.batch_size),
    }
    update_op = training_strategies[args.training_strategy]

    model.train()
    for epoch in range(args.epochs):
        train_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            x = batch[:, 0].reshape(-1, 1)
            y = batch[:, 1].reshape(-1, 1)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            loss = update_op(model, x, y)

            train_loss += loss

            optimizer.step()

            scheduler.step()

        valid_loss = validate(model, valid_dataloader, device=device)
        print(
            "\tEpoch",
            epoch + 1,
            "\tTraining Loss: ",
            train_loss,
            "\tValidation Loss: ",
            valid_loss,
            "\tLearning Rate: ",
            scheduler.get_last_lr(),
        )
        lc_train.append(float(train_loss))
        lc_valid.append(float(valid_loss))
        if np.isnan(valid_loss):
            valid_loss = 1000000
        report(
            epoch=epoch + 1,
            train_loss=train_loss,
            valid_loss=valid_loss,
            num_params=n_params,
        )
        if current_best is None or current_best >= valid_loss:
            current_best = valid_loss
            if args.st_checkpoint_dir is not None:
                os.makedirs(args.st_checkpoint_dir, exist_ok=True)
                checkpoint = {
                    "state": model.state_dict(),
                    "config": {"hidden_dim": args.hidden_dim},
                }
                torch.save(
                    checkpoint,
                    Path(args.st_checkpoint_dir)
                    / f"{args.training_strategy}_model_{args.hidden_dim}.pt",
                )

                history = {"train_loss": lc_train, "valid_loss": lc_valid}
                json.dump(
                    history,
                    open(
                        Path(args.st_checkpoint_dir)
                        / f"{args.training_strategy}_history_{args.hidden_dim}.json",
                        "w",
                    ),
                )

    #
    # # num_layers, num_units, num_heads
    #
    if args.do_plot:
        import matplotlib.pyplot as plt

        plt.scatter(train_data[:, 0], train_data[:, 1], label="Train")
        plt.scatter(valid_data[:, 0], valid_data[:, 1], label="Valid")
        model.eval()
        y_hat = model(valid_data[:, 0].reshape(-1, 1)).detach().numpy()
        plt.scatter(valid_data[:, 0], y_hat, label="predicted")
        plt.legend()
        plt.show()
        plt.plot(lc_train, label="train")
        plt.plot(lc_valid, label="valid")
        plt.yscale("log")
        plt.legend()
        plt.show()
