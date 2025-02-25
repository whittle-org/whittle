import logging
import os
import time

import torch
import argparse
import numpy as np

from torch.nn import KLDivLoss
import torch.nn.functional as F

from pathlib import Path
import pathlib
import json
from litgpt.tokenizer import Tokenizer
from litgpt import Config
from litgpt.utils import chunked_cross_entropy

from whittle.models.gpt import GPT
from whittle.search import multi_objective_search
from whittle.metrics.parameters import (
    compute_parameters,
    compute_parameters_sub_network_gpt,
)
from whittle.metrics.mag import weight_magnitude
from whittle.eval.utils import convert_and_evaluate
from downstream_datasets import HellaSWAG, ARCEasy, ARCChallenge, RTE, SciQ
from search_spaces import search_spaces

logging.basicConfig(level=logging.INFO)


datasets_cls = {
    "hellaswag": HellaSWAG,
    "arc_easy": ARCEasy,
    "arc_challenge": ARCChallenge,
    "rte": RTE,
    "sciq": SciQ,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--search_strategy", type=str, default="random_search")
    parser.add_argument("--task", type=str, default="arc_easy")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--model_id", type=str, default="EleutherAI/pythia-70m/")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./results_search")
    parser.add_argument("--objective", type=str, default="kl")
    parser.add_argument("--search_space", type=str, default="hw_gpt_bench")
    parser.add_argument("--calib_objective", type=str, default="mag")
    parser.add_argument("--importance_objective", type=str, default="mean")
    parser.add_argument("--data_fraction", type=float, default=0.1)
    parser.add_argument("--random_order", type=bool, default=False)
    parser.add_argument("--n_trials", type=int, default=10000)
    parser.add_argument("--resume", type=str, required=True)
    args, _ = parser.parse_known_args()

    if args.seed is None:
        seed = np.random.randint(2**16 - 1)
    else:
        seed = args.seed

    num_tokens = args.num_tokens
    batch_size = args.batch_size
    temperature = 1
    random_order_indices = args.random_order
    search_strategy = args.search_strategy
    checkpoint_dir = args.resume
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = Tokenizer(checkpoint_dir=f"checkpoints/{args.model_id}")
    config = Config.from_file(
        str(f"checkpoints/{args.model_id}/" + "model_config.yaml")
    )
    config.fix_head_size = True

    search_space = search_spaces[args.search_space](config)

    config.model_type = "gpt"
    model = GPT(config)
    with open(str(f"checkpoints/{args.model_id}/" + "config.json")) as f:
        hf_config = json.load(f)
    config.tie_embeddings = hf_config["tie_word_embeddings"]
    model.name_or_path = f"checkpoints/{args.model_id}/"
    model.set_kv_cache(batch_size=batch_size, device=device)
    model = model.to(device)
    model.to(torch.bfloat16)
    model.load_state_dict(torch.load(str(checkpoint_dir + "/lit_model.pth"))["model"])
    model.eval()

    # Load dataset
    # data = datasets_cls[args.task](val_split_fraction=1 - args.data_fraction)
    # data.connect(tokenizer=tokenizer, batch_size=batch_size)
    # data.setup()
    # dataloader = data.train_dataloader()

    params_super = compute_parameters(model)
    mag_super = weight_magnitude(model)

    def objective_kl(config):
        loss_function = KLDivLoss(reduction="batchmean", log_target=True)
        sub_network_config = search_space.cast(config)
        loss = 0
        counter = 0
        model.set_sub_network(
            **sub_network_config, sample_random_indices=random_order_indices
        )
        params_sub = compute_parameters_sub_network_gpt(model)
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                model.reset_super_network()
                original_output = model(batch["input_ids"])

                model.set_sub_network(
                    **sub_network_config, sample_random_indices=random_order_indices
                )
                sub_network_output = model(batch["input_ids"])

                loss += loss_function(
                    F.log_softmax(sub_network_output / temperature, dim=1),
                    F.log_softmax(original_output / temperature, dim=1),
                ).item()
            counter += 1
        loss /= counter

        return loss, params_sub / params_super

    def objective_mag(config):
        sub_network_config = search_space.cast(config)

        model.set_sub_network(
            **sub_network_config, sample_random_indices=random_order_indices
        )
        params_sub = compute_parameters_sub_network_gpt(model)

        loss = 1 - (weight_magnitude(model) / mag_super)

        return loss, params_sub / params_super

    def objective_perplexity(config):
        sub_network_config = search_space.cast(config)
        loss = 0
        counter = 0
        model.set_sub_network(
            **sub_network_config, sample_random_indices=random_order_indices
        )
        params_sub = compute_parameters_sub_network_gpt(model)
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                model.set_sub_network(
                    **sub_network_config, sample_random_indices=random_order_indices
                )
                logits = model(batch["input_ids"])
                targets = batch["labels"]
                loss = torch.exp(
                    chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
                ).item()

            counter += 1
        loss /= counter

        return loss, params_sub / params_super

    def objective_accuracy(config):
        if "grid" in args.search_strategy:
            sub_network_config = search_space.cast(config)
        else:
            sub_network_config = config
        model.set_sub_network(
            **sub_network_config, sample_random_indices=random_order_indices
        )
        params_sub = compute_parameters_sub_network_gpt(model)
        out_dir = pathlib.Path("out_dir/")
        convert_and_evaluate(
            model,
            out_dir=out_dir,
            device=None,
            dtype=torch.float32,
            tasks=args.task,
            batch_size=args.batch_size,  # Test for non-positive integer
        )
        with open(str(out_dir / "results.json"), "r") as f:
            results = json.load(f)
        if args.task == "winogrande":
            acc = results["results"][args.task]["acc,none"]
        else:
            acc = results["results"][args.task]["acc_norm,none"]
        return 1 - acc, params_sub / params_super

    if args.objective == "kl":
        objective = objective_kl
    elif args.objective == "acc":
        objective = objective_accuracy
    elif args.objective == "mag":
        objective = objective_mag
    elif args.objective == "perplexity":
        objective = objective_perplexity
    else:
        logging.error(f"Objective {args.objective} is not supported.")
        exit()

    if (
        search_strategy == "grid"
        or search_strategy == "grid-params"
        or search_strategy == "calibrate"
        or "importance" in search_strategy
    ):
        from collections import defaultdict
        from sampler import (
            RandomSampler,
            FixGridSampler,
            FixParamGridSampler,
            CalibFixGridSampler,
            ImportanceSampler,
            ImportanceParamGridSampler,
            ImportanceCalibFixGridSampler,
        )

        if search_strategy == "grid":
            sampler = FixGridSampler(search_space=search_space)
        elif search_strategy == "grid-params":
            sampler = FixParamGridSampler(
                search_space=search_space, seed=42, n_trials=args.n_trials
            )
            print("Initializing params grid....")
            sampler.initialize_grid(model)
        elif search_strategy == "calibrate":
            sampler = CalibFixGridSampler(
                checkpoint_dir=f"checkpoints/{args.model_id}",
                search_space_type=args.search_space,
                search_space=search_space,
                objective=args.calib_objective,
            )
        elif search_strategy == "importance-grid-params":
            sampler = ImportanceParamGridSampler(
                sorted_ids_path=os.path.join(
                    f"checkpoints/{args.model_id}", "sorted_ids.pkl"
                ),
                search_space=search_space,
                seed=42,
                n_trials=args.n_trials,
            )
            sampler.initialize_grid(model)
        elif search_strategy == "importance-calibrate":
            sampler = ImportanceCalibFixGridSampler(
                objective=args.calib_objective,
                importance_objective=args.importance_objective,
                sorted_ids_path=os.path.join(
                    f"checkpoints/{args.model_id}", "sorted_ids.pkl"
                ),
                checkpoint_dir=f"checkpoints/{args.model_id}",
                search_space_type=args.search_space,
                search_space=search_space,
                seed=seed,
            )
        search_results = defaultdict(list)

        t = time.time()
        for config in sampler.grid[::-1]:
            err, params = objective(config)
            search_results["configs"].append(config)
            search_results["costs"].append([err, params])
            search_results["runtime"].append(time.time() - t)
            search_results["is_pareto_optimal"].append(True)

        search_results["costs"] = np.array(search_results["costs"])

    else:
        search_results = multi_objective_search(
            objective,
            search_space.config_space,
            objective_kwargs={},
            search_strategy=search_strategy,
            num_samples=args.iterations,
            seed=seed,
        )

    results = dict()
    results["dataset"] = args.task
    results["configs"] = search_results["configs"]
    results["error"] = list(search_results["costs"][:, 0])
    results["params"] = list(search_results["costs"][:, 1])
    results["args"] = vars(args)
    results["runtime"] = list(search_results["runtime"])
    results["is_pareto_optimal"] = list(search_results["is_pareto_optimal"])

    os.makedirs(args.output_dir, exist_ok=True)
    fh = open(str(Path(args.output_dir) / "results.json"), "w")
    json.dump(results, fh)
