from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from syne_tune.config_space import randint
from syne_tune.report import Reporter
from tabulate import tabulate
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import trange

from examples.fashion_mnist.model import LeNet
from whittle.sampling.samplers import RandomSampler
from whittle.training_strategies import (
    RandomStrategy,
    SandwichStrategy,
    StandardStrategy,
)


def correct(output: torch.Tensor, target: torch.Tensor) -> int:
    """Returns the number of correct predictions."""
    predicted_digits = output.argmax(1)
    correct_ones = (predicted_digits == target).type(torch.float32)
    return correct_ones.sum().item()


def validate(
    model: LeNet, test_loader: DataLoader, criterion: Callable, device: torch.device
):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device)

            output = model.forward(x=data)
            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()

            # Count number of correct digits
            total_correct += correct(output, target)

    test_loss = test_loss / num_batches
    accuracy = total_correct / num_items

    # print(f"Testset accuracy: {100 * accuracy:>0.1f}%, average loss: {test_loss:>7f}")
    return 100 * accuracy, loss.cpu().item()


if __name__ == "__main__":
    report = Reporter()

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fc1_out", type=int, default=32)
    parser.add_argument("--fc2_out", type=int, default=16)
    parser.add_argument("--training_strategy", type=str, default="sandwich")
    parser.add_argument("--st_checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_train_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = datasets.FashionMNIST(
        ".", train=True, download=True, transform=ToTensor()
    )

    # Split the dataset
    train_dataset, _ = random_split(
        full_dataset,
        [args.num_train_samples, len(full_dataset) - args.num_train_samples],
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=datasets.FashionMNIST(".", train=False, transform=ToTensor()),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = LeNet(fc1_out=args.fc1_out, fc2_out=args.fc2_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    search_space = {
        "fc1_out": randint(1, args.fc1_out),
        "fc2_out": randint(1, args.fc2_out),
    }

    os.makedirs(args.st_checkpoint_dir, exist_ok=True)
    current_best = None
    lc_valid = []
    lc_train = []

    sampler = RandomSampler(search_space, seed=args.seed)
    training_strategies = {
        "standard": StandardStrategy(
            sampler=sampler, loss_function=nn.functional.cross_entropy
        ),
        "sandwich": SandwichStrategy(
            sampler=sampler, loss_function=nn.functional.cross_entropy
        ),
        "random": RandomStrategy(
            random_samples=2, sampler=sampler, loss_function=nn.functional.cross_entropy
        ),
    }
    update_op = training_strategies[args.training_strategy]

    for epoch in trange(args.epochs, desc=f"Training with {args.training_strategy}"):
        train_loss = 0
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss = update_op(model, x, y)
            train_loss += loss
            optimizer.step()
            scheduler.step()

        valid_acc, valid_loss = validate(
            model=model,
            test_loader=test_loader,
            criterion=torch.nn.functional.cross_entropy,
            device=device,
        )

        lc_train.append(float(train_loss))
        lc_valid.append(float(valid_loss))
        if np.isnan(valid_loss):
            valid_loss = 1000000

        report(
            epoch=epoch + 1,
            train_loss=train_loss,
            valid_loss=valid_loss,
            valid_acc=valid_acc,
            num_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        if current_best is None or current_best >= valid_loss:
            current_best = valid_loss
            if args.st_checkpoint_dir is not None:
                os.makedirs(args.st_checkpoint_dir, exist_ok=True)
                checkpoint = {
                    "state": model.state_dict(),
                    "config": {"fc1_out": args.fc1_out, "fc2_out": args.fc2_out},
                }
                torch.save(
                    checkpoint,
                    Path(args.st_checkpoint_dir)
                    / f"{args.training_strategy}_model_{args.fc1_out}_{args.fc2_out}.pt",
                )

                history = {"train_loss": lc_train, "valid_loss": lc_valid}
                json.dump(
                    history,
                    open(
                        Path(args.st_checkpoint_dir)
                        / f"{args.training_strategy}_history_{args.fc1_out}_{args.fc2_out}.json",
                        "w",
                    ),
                )

    lottery_grid = [[4, 4], [8, 8], [32, 16]]
    df = pd.DataFrame(
        {"fc1_out": [], "fc2_out": [], "accuracy": [], "loss": []},
    )
    for i, k in enumerate(lottery_grid):
        config = {"fc1_out": k[0], "fc2_out": k[1]}
        model.select_sub_network(config=config)
        acc, loss = validate(
            model=model,
            test_loader=test_loader,
            criterion=torch.nn.functional.cross_entropy,
            device=device,
        )
        df.loc[i] = [*list(config.values()), acc, loss]
    df = df.astype(
        {"fc1_out": "str", "fc2_out": "str", "accuracy": "float", "loss": "float"}
    )
    print(tabulate(df, headers="keys", tablefmt="pipe", floatfmt=".2f"))
