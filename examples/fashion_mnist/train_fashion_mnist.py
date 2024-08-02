from typing import Literal

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from examples.fashion_mnist.model import LeNet


def correct(output: torch.Tensor, target: torch.Tensor) -> int:
    # pick digit with largest network output
    predicted_digits = output.argmax(1)

    # 1.0 for correct, 0.0 for incorrect
    correct_ones = (predicted_digits == target).type(torch.float32)
    return correct_ones.sum().item()


def sandwich_update_random_id(model: LeNet, batch, criterion, target: torch.Tensor):
    f1, f2 = model.sample_max()
    output_largest = model.forward_dense_random_neurons(batch, f1, f2)
    loss = criterion(output_largest, target)
    loss.backward()

    f1, f2 = model.sample_min()
    seed = np.random.randint(0, 10000)
    output = model.forward_dense_random_neurons(batch, f1, f2, seed=seed)
    loss = criterion(output, target)
    loss.backward()

    f1, f2 = model.sample_dense_dims()
    seed = np.random.randint(0, 10000)
    output = model.forward_dense_random_neurons(batch, f1, f2, seed=seed)
    loss = criterion(output, target)
    loss.backward()

    f1, f2 = model.sample_dense_dims()
    seed = np.random.randint(0, 10000)
    output = model.forward_dense_random_neurons(batch, f1, f2, seed=seed)
    loss = criterion(output, target)
    loss.backward()

    # for name,param in model.named_parameters():
    #  param.grad = param.grad/4

    return output, loss


def sandwich_update(model, batch, criterion, target):
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
    # for name,param in model.named_parameters():
    #  param.grad = param.grad/4
    return output, loss


def sparse_dense_update(model, batch, criterion, target):
    f1, f2 = model.sample_max()
    output = model.forward_sparse_dense_lottery(batch, f1, f2)
    loss = criterion(output, target)
    loss.backward()
    f1, f2 = model.sample_min()
    output = model.forward_sparse_dense_lottery(batch, f1, f2)
    loss = criterion(output, target)
    loss.backward()
    f1, f2 = model.sample_dense_dims()
    output = model.forward_sparse_dense_lottery(batch, f1, f2)
    loss = criterion(output, target)
    loss.backward()
    f1, f2 = model.sample_dense_dims()
    output = model.forward_sparse_dense_lottery(batch, f1, f2)
    loss = criterion(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        param.grad = param.grad / 4
    return output, loss


def train(
    data_loader: DataLoader,
    model: LeNet,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_scheme: Literal[
        "sampling", "sparsify", "sparse_dense", "standard", "sandwich", "sandwich_rand"
    ] = "sampling",
):
    model.train()

    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_loss = 0.0
    total_correct = 0
    print(train_scheme)
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)
        f1, f2 = model.sample_f_in()
        # Do a forward pass

        match train_scheme:
            case "standard":
                output = model(data)
            case "sampling":
                output = model.forward_dense_lottery(data, f1, f2)
            case "sparsify":
                output = model.forward_sparse_lottery(data, f1, f2)
            case "sandwich":
                output, loss = sandwich_update(model, data, criterion, target)
            case "sandwich_rand":
                output, loss = sandwich_update_random_id(model, data, criterion, target)
            case "sparse_dense":
                output, loss = sparse_dense_update(model, data, criterion, target)
            case _:
                raise ValueError(f"Unknown train_scheme: {train_scheme}")

        # Calculate the loss
        if train_scheme not in ["sandwich", "sandwich_rand", "sparse_dense"]:
            loss = criterion(output, target)
        total_loss += loss

        # Count number of correct digits
        total_correct += correct(output, target)

        # Backpropagation
        if train_scheme not in ["sandwich", "sandwich_rand", "sparse_dense"]:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss / num_batches
    accuracy = total_correct / num_items
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")


def test(
    test_loader: DataLoader,
    model: LeNet,
    criterion: nn.CrossEntropyLoss,
    sparsity_type: Literal["sparse", "dense", "sparse-dense", "dense-rand"] = "sparse",
    f1_out=10,
    f2_out=10,
):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0
    if sparsity_type == "dense-rand":
        accuracies = []
        for i in range(10):
            total_correct = 0
            total_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    output = model.forward_dense_random_neurons(
                        data, f1_out, f2_out, seed=i
                    )
                    total_correct += correct(output, target)
                    loss = criterion(output, target)
                    test_loss += loss.item()
            test_loss = test_loss / num_batches
            accuracy = total_correct / num_items
            accuracies.append(accuracy)
        return max(accuracies)

    else:
        with torch.no_grad():
            for data, target in test_loader:
                # Copy data and targets to GPU
                data = data.to(device)
                target = target.to(device)

                # Do a forward pass
                if sparsity_type == "sparse":
                    output = model.forward_sparse_lottery(data, f1_out, f2_out)
                if sparsity_type == "dense":
                    output = model.forward_dense_lottery(data, f1_out, f2_out)
                if sparsity_type == "sparse-dense":
                    output = model.forward_sparse_dense_lottery(data, f1_out, f2_out)
                # Calculate the loss
                loss = criterion(output, target)
                test_loss += loss.item()

                # Count number of correct digits
                total_correct += correct(output, target)

    test_loss = test_loss / num_batches
    accuracy = total_correct / num_items

    print(f"Testset accuracy: {100 * accuracy:>0.1f}%, average loss: {test_loss:>7f}")
    return 100 * accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    model_standard = LeNet().to(device)
    model_random_id = LeNet().to(device)
    model_sampling = LeNet().to(device)
    model_sparse_dense = LeNet().to(device)
    model_sandwich = LeNet().to(device)

    optimizer_standard = torch.optim.Adam(model_standard.parameters(), lr=1e-3)
    optimizer_random_id = torch.optim.Adam(model_random_id.parameters(), lr=1e-3)
    optimizer_sampling = torch.optim.Adam(model_sampling.parameters(), lr=1e-3)
    optimizer_sparse_dense = torch.optim.Adam(model_sparse_dense.parameters(), lr=1e-3)
    optimizer_sandwich = torch.optim.Adam(model_sandwich.parameters(), lr=1e-3)

    batch_size = 1024
    train_dataset = datasets.FashionMNIST(
        ".", train=True, download=True, transform=ToTensor()
    )
    test_dataset = datasets.FashionMNIST(".", train=False, transform=ToTensor())

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    epochs = 5
    for epoch in range(epochs):
        print(f"Training epoch: {epoch + 1}")
        train(
            train_loader,
            model_standard,
            criterion,
            optimizer_standard,
            device=device,
            train_scheme="standard",
        )
        train(
            train_loader,
            model_sampling,
            criterion,
            optimizer_sampling,
            device=device,
            train_scheme="sampling",
        )
        train(
            train_loader,
            model_sparse_dense,
            criterion,
            optimizer_sparse_dense,
            device=device,
            train_scheme="sparse_dense",
        )
        train(
            train_loader,
            model_random_id,
            criterion,
            optimizer_random_id,
            device=device,
            train_scheme="sandwich_rand",
        )
        train(
            train_loader,
            model_sandwich,
            criterion,
            optimizer_sandwich,
            device=device,
            train_scheme="sandwich",
        )

    lottery_grid = [[25, 25], [50, 50], [120, 84]]
    training_schemes = ["standard", "sampling", "sandwich", "sandwich_rand"]
    lottery_sparse = {}
    lottery_dense = {}
    lottery_sparse_dense = {}
    lottery_dense_rand = {}
    for scheme in training_schemes:
        lottery_sparse[scheme] = {}
        lottery_dense[scheme] = {}
        lottery_sparse[scheme] = {}
        lottery_dense_rand[scheme] = {}
    for k in lottery_grid:
        for scheme in training_schemes:
            if scheme == "standard":
                model = model_standard
            elif scheme == "sampling":
                model = model_sampling
            elif scheme == "sparse-dense":
                model = model_sparse_dense
            elif scheme == "sandwich_rand":
                model = model_random_id
            else:
                model = model_sandwich
            lottery_sparse[scheme][str(k[0] * k[1])] = test(
                test_loader,
                model,
                criterion,
                sparsity_type="sparse",
                f1_out=k[0],
                f2_out=k[1],
            )
            lottery_dense[scheme][str(k[0] * k[1])] = test(
                test_loader,
                model,
                criterion,
                sparsity_type="dense",
                f1_out=k[0],
                f2_out=k[1],
            )
            lottery_dense_rand[scheme][str(k[0] * k[1])] = test(
                test_loader,
                model,
                criterion,
                sparsity_type="dense-rand",
                f1_out=k[0],
                f2_out=k[1],
            )
            lottery_sparse_dense[scheme][str(k[0]) + "_" + str(k[1])] = test(
                test_loader,
                model,
                criterion,
                sparsity_type="sparse-dense",
                f1_out=k[0],
                f2_out=k[1],
            )
