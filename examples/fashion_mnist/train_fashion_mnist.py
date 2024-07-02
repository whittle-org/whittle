from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import LeNet


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
        data_loader,
        model: LeNet,
        criterion,
        optimizer: torch.optim.Optimizer,
        device: str,
        train_scheme: Optional[str] = "sampling"
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

        if train_scheme == "sampling":
            output = model.forward_dense_lottery(data, f1, f2)
        elif train_scheme == "sparsify":
            output = model.forward_sparse_lottery(data, f1, f2)
        elif train_scheme == "sparse_dense":
            output, loss = sparse_dense_update(model, data, criterion, target)
        elif train_scheme == "standard":
            output = model(data)
        elif train_scheme == "sandwich":
            output, loss = sandwich_update(model, data, criterion, target)
        elif train_scheme == "sandwich_rand":
            output, loss = sandwich_update_random_id(model, data, criterion, target)
        # Calculate the loss
        if "sandwich" not in train_scheme and train_scheme != "sparse_dense":
            loss = criterion(output, target)
        total_loss += loss

        # Count number of correct digits
        total_correct += correct(output, target)

        # Backpropagation
        if "sandwich" not in train_scheme and train_scheme != "sparse_dense":
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss / num_batches
    accuracy = total_correct / num_items
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    model_standard = LeNet().to(device)
    model_random_id = LeNet().to(device)
    model_sampling = LeNet().to(device)
    model_sparse_dense = LeNet().to(device)
    # model_sparse = LeNet().to(device)
    model_sandwich = LeNet().to(device)
    # model_sandwich_kd = LeNet().to(device)
    # model_sandwich_last = LeNet().to(device)
    optimizer_standard = torch.optim.Adam(model_standard.parameters(), lr=1e-3)
    optimizer_sampling = torch.optim.Adam(model_sampling.parameters(), lr=1e-3)
    # optimizer_sparse = torch.optim.Adam(model_sparse.parameters(),lr=1e-3)
    # optimizer_sparse_dense = torch.optim.Adam(model_sparse_dense.parameters(),lr=1e-3)
    optimizer_sandwich = torch.optim.Adam(model_sandwich.parameters(), lr=1e-3)
    optimizer_random_id = torch.optim.Adam(model_random_id.parameters(), lr=1e-3)
    # optimizer_sandwich_kd = torch.optim.Adam(model_sandwich_kd.parameters(),lr=1e-4)
    # optimizer_sandwich_last = torch.optim.Adam(model_sandwich_last.parameters(),lr=1e-3)

    batch_size = 1024
    train_dataset = datasets.FashionMNIST(".", train=True, download=True, transform=ToTensor())
    test_dataset = datasets.FashionMNIST(".", train=False, transform=ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    epochs = 5
    for epoch in range(epochs):
        print(f"Training epoch: {epoch + 1}")
        train(train_loader, model_standard, criterion, optimizer_standard, device=device, train_scheme="standard")
        train(train_loader, model_sampling, criterion, optimizer_sampling, device=device, train_scheme="sampling")
        # train(train_loader, model_sparse_dense, criterion, optimizer_sparse_dense, train_scheme="sparse_dense")
        # train(train_loader, model_sandwich_last, criterion, optimizer_sandwich_last, train_scheme="sandwich_last")
        train(
            train_loader,
            model_random_id,
            criterion,
            optimizer_random_id,
            device=device,
            train_scheme="sandwich_rand"
        )
        train(train_loader, model_sandwich, criterion, optimizer_sandwich, device=device, train_scheme="sandwich")
