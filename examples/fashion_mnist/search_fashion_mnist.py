from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from syne_tune.config_space import randint
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from examples.fashion_mnist.model import LeNet
from examples.fashion_mnist.train_fashion_mnist import validate
from whittle.search import multi_objective_search


def compute_mac_linear_layer(in_features: int, out_features: int):
    return in_features * out_features


def objective(
    config: dict[Literal["fc1_out", "fc2_out"], int], model: LeNet, device: torch.device
) -> tuple[int, float]:
    model.select_sub_network(config=config)

    _, loss = validate(
        test_loader=test_loader,
        model=model,
        criterion=torch.nn.functional.cross_entropy,
        device=device,
    )

    mac = 0
    mac += compute_mac_linear_layer(
        model.fc_base.in_features, model.fc_base.out_features
    )
    mac += compute_mac_linear_layer(model.fc1.in_features, config["fc1_out"])
    mac += compute_mac_linear_layer(config["fc1_out"], config["fc2_out"])
    mac += compute_mac_linear_layer(config["fc2_out"], 10)

    return mac, loss


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fc1_out", type=int, default=120)
    parser.add_argument("--fc2_out", type=int, default=84)
    parser.add_argument("--training_strategy", type=str, default="sandwich")
    parser.add_argument("--search_strategy", type=str, default="random_search")
    parser.add_argument("--do_plot", type=bool, default=True)
    parser.add_argument("--st_checkpoint_dir", type=str, default="./checkpoints")
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = DataLoader(
        dataset=datasets.FashionMNIST(".", train=False, transform=ToTensor()),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = LeNet(fc1_out=args.fc1_out, fc2_out=args.fc2_out)
    path = (
        Path(args.st_checkpoint_dir)
        / f"{args.training_strategy}_model_{args.fc1_out}_{args.fc2_out}.pt"
    )
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state"])
    model = model.to(device)
    model.eval()

    search_space = {
        "fc1_out": randint(1, args.fc1_out),
        "fc2_out": randint(1, args.fc2_out),
    }

    results = multi_objective_search(
        objective,
        search_space,
        objective_kwargs={"model": model, "device": device},
        search_strategy=args.search_strategy,
        num_samples=100,
        seed=42,
    )

    costs = np.array(results["costs"])
    plt.scatter(costs[:, 0], costs[:, 1], color="black", label="sub-networks")

    idx = np.array(results["is_pareto_optimal"])
    if args.do_plot:
        plt.scatter(
            costs[idx, 0],
            costs[idx, 1],
            color="red",
            label="Pareto optimal",
            s=100,
        )

        plt.xlabel("MACs")
        plt.ylabel("Validation Loss")
        plt.xscale("log")
        plt.grid(linewidth="1", alpha=0.4)
        plt.title("Pareto front for Fashion MNIST")
        plt.legend()
        plt.tight_layout()
        plt.savefig("pareto_front.png", dpi=300)
