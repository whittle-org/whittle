from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lobotomy.search import multi_objective_search
from syne_tune.config_space import randint
from torch.utils.data import DataLoader

from .estimate_efficiency import compute_mac_linear_layer
from .model import MLP
from .sinc_nas import f, validate

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument(
        "--training_strategy", type=str, default="sandwich"
    )
    parser.add_argument(
        "--search_strategy", type=str, default="random_search"
    )
    parser.add_argument("--do_plot", type=bool, default=False)
    parser.add_argument(
        "--st_checkpoint_dir", type=str, default="./checkpoints"
    )

    args, _ = parser.parse_known_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    search_space = {"num_units": randint(1, args.hidden_dim)}
    rng = np.random.RandomState(42)
    num_data_points = 500
    x = rng.rand(num_data_points)
    y = f(x)

    data = np.empty((num_data_points, 2))
    data[:, 0] = x
    data[:, 1] = y

    data = torch.tensor(data, dtype=torch.float)
    n_train = int(data.shape[0] * 0.7)
    train_data = data[:n_train]
    valid_data = data[n_train:]

    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False
    )

    model = MLP(input_dim=1, hidden_dim=args.hidden_dim, device=device)

    path = (
            Path(args.st_checkpoint_dir)
            / f"{args.training_strategy}_model_{args.hidden_dim}.pt"
    )
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state"])
    model = model.to(device)
    model.eval()


    def objective(config):
        model.select_sub_network(config)
        loss = validate(model, valid_dataloader, device)

        mac = 0
        mac += compute_mac_linear_layer(
            model.input_layer.in_features, model.input_layer.out_features
        )
        mac += compute_mac_linear_layer(
            model.hidden_layer.in_features, config["num_units"]
        )
        mac += compute_mac_linear_layer(
            model.output_layer.in_features,
            model.output_layer.out_features,
        )

        model.reset_super_network()

        return mac, loss


    results = multi_objective_search(
        objective,
        search_space,
        objective_kwargs={},
        search_strategy=args.search_strategy,
        num_samples=100,
        seed=42,
    )

    costs = np.array(results["costs"])
    plt.scatter(
        costs[:, 0], costs[:, 1], color="black", label="sub-networks"
    )

    idx = np.array(results["is_pareto_optimal"])
    if args.do_plot:
        plt.scatter(
            costs[idx, 0],
            costs[idx, 1],
            color="red",
            label="Pareto optimal",
        )

        plt.xlabel("mac")
        plt.ylabel("Validation loss")
        plt.xscale("log")
        plt.grid(linewidth="1", alpha=0.4)
        plt.title("Pareto front for Sinc NAS")
        plt.savefig("pareto_front.png")
