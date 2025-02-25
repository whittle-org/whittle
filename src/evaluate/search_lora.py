import logging
import os
import time
import torch
import argparse
import numpy as np
import gc
from torch.nn import KLDivLoss
import torch.nn.functional as F
from pathlib import Path
import json
from litgpt.tokenizer import Tokenizer
from litgpt import Config
from litgpt.utils import chunked_cross_entropy
from lora_whittle.lora_gpt import GPT
from whittle.search import multi_objective_search
from whittle.metrics.parameters import (
    compute_parameters,
    compute_parameters_sub_network_gpt,
)
from whittle.metrics.mag import weight_magnitude
from whittle.eval.utils import convert_and_evaluate
from downstream_datasets import HellaSWAG, ARCEasy, ARCChallenge, RTE, SciQ
from search_spaces import search_spaces
import pathlib
from lora_whittle.lora_block import Block

logging.basicConfig(level=logging.INFO)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, StateDictType
from lightning.fabric import Fabric  # Fabric import
from lightning.fabric.strategies import FSDPStrategy
import lightning as L

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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--search_strategy", type=str, default="random_search")
    parser.add_argument("--task", type=str, default="sciq")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./results_search")
    parser.add_argument("--objective", type=str, default="kl")
    parser.add_argument("--calib_objective", type=str, default="mag")
    parser.add_argument("--importance_objective", type=str, default="mean")
    parser.add_argument("--search_space", type=str, default="llama")
    parser.add_argument("--data_fraction", type=float, default=0.1)
    parser.add_argument("--random_order", type=bool, default=False)
    parser.add_argument("--n_trials", type=int, default=10000)
    parser.add_argument("--resume", type=str, default=".")
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
    checkpoint_dir = "checkpoints/" + args.model_id
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = Tokenizer(checkpoint_dir="checkpoints/" + args.model_id)
    config = Config.from_file(str(checkpoint_dir + "/model_config.yaml"))
    config.lora_head = True
    config.lora_query = True
    config.lora_key = True
    config.lora_value = True
    config.lora_projection = True
    config.lora_mlp = True
    config.lora_r = 32
    config.lora_alpha = 16
    config.fix_head_size = True
    config.lora_dropout = 0.05

    search_space = search_spaces[args.search_space](config)

    config.model_type = "gpt"
    model = GPT(config)
    with open(str(checkpoint_dir + "/config.json")) as f:
        hf_config = json.load(f)
    config.tie_embeddings = hf_config["tie_word_embeddings"]
    model.name_or_path = Path("checkpoints") / Path(args.model_id)
    model.max_seq_length = 653
    # model.set_kv_cache(batch_size=batch_size, device=device)
    # model = model.to(device)

    if "importance" in args.search_strategy:
        base_dict = torch.load(
            checkpoint_dir + "/lit_model_permuted_mean.pth", map_location="cpu"
        )
    else:
        base_dict = torch.load(checkpoint_dir + "/lit_model.pth", map_location="cpu")

    lora_dict = torch.load(args.resume, map_location="cpu")["model"]
    merged_dict = {**base_dict, **lora_dict}

    model.load_state_dict(merged_dict)
    strategy = FSDPStrategy(
        auto_wrap_policy={Block},
        activation_checkpointing_policy={Block},
        state_dict_type="full",
        limit_all_gathers=True,
        cpu_offload=False,
    )
    fabric = L.Fabric(accelerator="gpu", devices=4, strategy=strategy)
    fabric.launch()
    model = fabric.setup_module(model)
    model.eval()
    model.eval()

    # Load dataset
    data = datasets_cls[args.task](val_split_fraction=1 - args.data_fraction)
    data.connect(tokenizer=tokenizer, batch_size=batch_size)
    data.setup()
    dataloader = data.train_dataloader()

    params_super = compute_parameters(model)

    # Memory management improvements
    torch.cuda.empty_cache()
    gc.collect()

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
            torch.cuda.empty_cache()  # Clear cache after each batch to manage memory
        loss /= counter

        return loss, params_sub / params_super

    def objective_mag(config):
        sub_network_config = search_space.cast(config)
        model.set_sub_network(
            **sub_network_config, sample_random_indices=random_order_indices
        )
        params_sub = compute_parameters_sub_network_gpt(model)
        loss = 1 - (weight_magnitude(model) / mag_super)
        torch.cuda.empty_cache()  # Clear cache after each run
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
                loss += torch.exp(
                    chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
                ).item()
            counter += 1
            torch.cuda.empty_cache()  # Clear cache after each batch to manage memory
        loss /= counter

        return loss, params_sub / params_super

    def objective_accuracy(config):
        gc.collect()
        torch.cuda.empty_cache()  # Clear GPU memory before evaluation
        sub_network_config = search_space.cast(config)
        model.set_sub_network(
            **sub_network_config, sample_random_indices=random_order_indices
        )
        params_sub = compute_parameters_sub_network_gpt(model)
        out_dir = pathlib.Path("out_dir/")
        model.eval()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                convert_and_evaluate(
                    model,
                    out_dir=out_dir,
                    device=None,
                    tasks=args.task,
                    dtype=torch.float32,
                    batch_size=4,
                )
        with open(str(out_dir / "results.json"), "r") as f:
            results = json.load(f)
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
                checkpoint_dir=checkpoint_dir,
                search_space_type=args.search_space,
                search_space=search_space,
            )
        elif search_strategy == "importance-grid-params":
            sampler = ImportanceParamGridSampler(
                sorted_ids_path=os.path.join(checkpoint_dir, "sorted_ids.pkl"),
                search_space=search_space,
                seed=42,
                n_trials=args.n_trials,
            )
            sampler.initialize_grid(model)
        elif search_strategy == "importance-calibrate":
            sampler = ImportanceCalibFixGridSampler(
                objective=args.calib_objective,
                importance_objective=args.importance_objective,
                sorted_ids_path=os.path.join(checkpoint_dir, "sorted_ids.pkl"),
                checkpoint_dir=checkpoint_dir,
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
