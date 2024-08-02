from typing import Literal
from argparse import ArgumentParser

import pandas as pd
from tabulate import tabulate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from examples.fashion_mnist.model import LeNet


def correct(output: torch.Tensor, target: torch.Tensor) -> int:
    """Returns the number of correct predictions."""
    predicted_digits = output.argmax(1)
    correct_ones = (predicted_digits == target).type(torch.float32)
    return correct_ones.sum().item()


def sandwich_update(
    model: LeNet,
    batch: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
    target: torch.Tensor,
):
    """Performs one sandwich rule update."""
    f1, f2 = model.sample_max()
    output = model.forward_dense_lottery(batch, f1, f2)
    loss = criterion(output, target)
    loss.backward()

    f1, f2 = model.sample_min()
    output = model.forward_dense_lottery(batch, f1, f2)
    loss = criterion(output, target)
    loss.backward()

    f1, f2 = model.sample_dense_dims()
    output = model.forward_dense_lottery(batch, f1, f2)
    loss = criterion(output, target)
    loss.backward()

    f1, f2 = model.sample_dense_dims()
    output = model.forward_dense_lottery(batch, f1, f2)
    loss = criterion(output, target)
    loss.backward()

    return output, loss


def train(
    dataloader: DataLoader,
    model: LeNet,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training_scheme: Literal[
        "sampling",
        "standard",
        "sandwich",
    ] = "sampling",
):
    """Trains the model for one epoch with the given training scheme."""
    model.train()

    num_batches = len(dataloader)
    num_items = len(dataloader.dataset)

    total_loss = 0.0
    total_correct = 0
    for data, target in dataloader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)
        f1, f2 = model.sample_f_in()
        # Do a forward pass

        if training_scheme == "standard":
            output = model(data)
        elif training_scheme == "sampling":
            output = model.forward_dense_lottery(data, f1, f2)
        elif training_scheme == "sandwich":
            output, loss = sandwich_update(model, data, criterion, target)
        else:
            raise ValueError(f"Unknown train_scheme: {training_scheme}")

        # Calculate the loss
        if training_scheme != "sandwich":
            loss = criterion(output, target)
        total_loss += loss

        # Count number of correct digits
        total_correct += correct(output, target)

        # Backpropagation
        if training_scheme != "sandwich":
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss / num_batches
    accuracy = total_correct / num_items
    print(
        f"Training scheme: {training_scheme}, Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}"
    )


def test(
    test_loader: DataLoader,
    model: LeNet,
    criterion: nn.CrossEntropyLoss,
    f1_out: int,
    f2_out: int,
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

            output = model.forward_dense_lottery(x=data, fc1_out=f1_out, fc2_out=f2_out)
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
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    model_standard = LeNet().to(device)
    model_sampling = LeNet().to(device)
    model_sandwich = LeNet().to(device)

    optimizer_standard = torch.optim.Adam(model_standard.parameters(), lr=args.lr)
    optimizer_sampling = torch.optim.Adam(model_sampling.parameters(), lr=args.lr)
    optimizer_sandwich = torch.optim.Adam(model_sandwich.parameters(), lr=args.lr)

    batch_size = args.batch_size

    train_loader = DataLoader(
        dataset=datasets.FashionMNIST(
            ".", train=True, download=True, transform=ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=datasets.FashionMNIST(".", train=False, transform=ToTensor()),
        batch_size=batch_size,
        shuffle=False,
    )

    for epoch in range(args.epochs):
        print(f"Training epoch: {epoch + 1}")
        train(
            dataloader=train_loader,
            model=model_standard,
            criterion=criterion,
            optimizer=optimizer_standard,
            device=device,
            training_scheme="standard",
        )
        train(
            dataloader=train_loader,
            model=model_sampling,
            criterion=criterion,
            optimizer=optimizer_sampling,
            device=device,
            training_scheme="sampling",
        )
        train(
            dataloader=train_loader,
            model=model_sandwich,
            criterion=criterion,
            optimizer=optimizer_sandwich,
            device=device,
            training_scheme="sandwich",
        )

    lottery_grid = [[25, 25], [50, 50], [120, 84]]
    training_schemes = ["standard", "sampling", "sandwich"]
    lottery_dense = {scheme: {} for scheme in training_schemes}
    for k in lottery_grid:
        for scheme in training_schemes:
            if scheme == "standard":
                model = model_standard
            elif scheme == "sampling":
                model = model_sampling
            else:
                model = model_sandwich

            acc, loss = test(
                test_loader,
                model,
                criterion,
                f1_out=k[0],
                f2_out=k[1],
            )
            lottery_dense[scheme][f"{str(k[0] * k[1])}_accuracy"] = acc
            lottery_dense[scheme][f"{str(k[0] * k[1])}_loss"] = loss

    df = pd.DataFrame(lottery_dense)
    print(tabulate(df, headers="keys", tablefmt="pipe", floatfmt=".4f"))
