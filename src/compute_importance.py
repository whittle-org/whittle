import os
import argparse
import json
import torch
import pickle
import transformers
import numpy as np
from whittle.models.gpt.model import GPT
from litgpt import Config
from whittle.sampling.random_sampler import RandomSampler
from whittle.metrics.parameters import compute_parameters
from modules.block_importance import compute_order_block_importance
from modules.drop_layer import compute_order_layers_ppl
from modules.embd_size import compute_order_embd
from modules.intermediate_size import compute_order_intermediate_dims
from modules.num_heads import compute_order_heads, compute_order_head_groups
from modules.utils import evaluate_wikitext, permute_model
from src.utils.search_spaces import search_spaces

import hashlib

# Serialize and hash the config
def get_config_hash(config):
    json_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()

def save_checkpoint(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def get_unique_configs(sampler, n, existing_configs):
    configs = []
    seen_hashes = {hash(json.dumps(c, sort_keys=True)) for c in existing_configs}

    while len(configs) < n:
        config = sampler.sample()
        config_hash = hash(json.dumps(config, sort_keys=True))
        if config_hash not in seen_hashes:
            configs.append(config)
            seen_hashes.add(config_hash)

    return configs

def evaluate_config_largest(model, configs, search_space, layer_order=None):
    ppls = []
    params = []
    archs = []
    for c in configs:
        print(f"Evaluating config: {c}")
        if layer_order is None:
            model.set_sub_network(**c)
        else:
            layer_order_top_k = sorted(layer_order[: int(c["sub_network_n_layers"])])
            model.set_sub_network(**c, sampled_layer_indices=layer_order_top_k)
        
        param = compute_parameters(model)
        ppl = evaluate_wikitext(args.max_seq_len, model, tokenizer, batch_size, num_batches)
        ppls.append(ppl)
        params.append(param)
        archs.append(c)
    
    return ppls, params, archs

def evaluate_configs(model, configs, search_space, layer_order=None):
    ppls = []
    params = []
    archs = []
    for c in configs:
        print(f"Evaluating config: {c}")
        if layer_order is None:
            model.set_sub_network(**search_space.cast(c))
        else:
            layer_order_top_k = sorted(layer_order[: int(c["depth"])])
            model.set_sub_network(**search_space.cast(c), sampled_layer_indices=layer_order_top_k)
        
        param = compute_parameters(model)
        ppl = evaluate_wikitext(args.max_seq_len, model, tokenizer, batch_size, num_batches)
        ppls.append(ppl)
        params.append(param)
        archs.append(c)
    
    return ppls, params, archs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-grained Model Importance Computation and Permutation")
    parser.add_argument("--model_id", type=str, required=True, help="ID of the model to be used")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches for evaluation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--objective", type=str, default="norm")
    parser.add_argument("--space", type=str, default="hw_gpt_bench")
    parser.add_argument("--layer_scheme", type=str, default="block_importance")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=21512)
    parser.add_argument("--n_configs", type=int, default=5)
    args = parser.parse_args()

    print(args)
    model_id = args.model_id
    num_batches = args.num_batches
    batch_size = args.batch_size

    # Create checkpoint directory
    checkpoint_dir = os.path.join("checkpoints", model_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Unique identifier for checkpointing
    identifier = f"{args.space}_{args.objective}_{args.layer_scheme}"

    # Paths for checkpointing
    sampled_configs_path = os.path.join(checkpoint_dir, f"sampled_configs_{identifier}.pkl")
    eval_results_before_path = os.path.join(checkpoint_dir, f"eval_results_before_{identifier}.pkl")
    eval_results_after_path = os.path.join(checkpoint_dir, f"eval_results_after_{identifier}.pkl")
    orders_path = os.path.join(checkpoint_dir, f"importance_orders_{identifier}.pkl")

    config_path = os.path.join(checkpoint_dir, "model_config.yaml")
    config_path_hf = os.path.join(checkpoint_dir, "config.json")
    model_path = os.path.join(checkpoint_dir, "lit_model.pth")

    config = Config.from_file(config_path)
    with open(config_path_hf) as f:
        hf_config = json.load(f)
    config.tie_embeddings = hf_config["tie_word_embeddings"]
    config.fix_head_size = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(torch.bfloat16)
    model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
    space = search_spaces[args.space](config)
    sampler = RandomSampler(space.config_space, seed=args.seed)

    # Resume or initialize config sampling
    if os.path.exists(sampled_configs_path):
        configs = load_checkpoint(sampled_configs_path)
    else:
        configs = []

    new_configs = get_unique_configs(sampler, args.n_configs - len(configs), configs)
    configs.extend(new_configs)
    save_checkpoint(sampled_configs_path, configs)

    # Pre-Permutation Evaluation with Fine-Grained Checkpointing
    if os.path.exists(eval_results_before_path):
        eval_results_before = load_checkpoint(eval_results_before_path)
    else:
        eval_results_before = {}
    largest_model_config = {
        "sub_network_n_embd": config.n_embd,
        "sub_network_intermediate_size": config.intermediate_size,
        "sub_network_num_heads": config.n_head,
        "sub_network_n_layers": config.n_layer,
        "sub_network_head_size": config.head_size,
    }
    config_hash = get_config_hash(largest_model_config)
    if config_hash not in eval_results_before:
        print(f"Evaluating largest model config before permutation: {largest_model_config}")
        model.reset_super_network()
        ppls, params, _ = evaluate_config_largest(model, [largest_model_config], space)
        
        eval_results_before[config_hash] = {
            "config": largest_model_config,
            "perplexity": ppls[0],
            "parameters": params[0]
        }
        save_checkpoint(eval_results_before_path, eval_results_before)
    for sampled_config in configs:
        config_hash = get_config_hash(sampled_config)
        if config_hash in eval_results_before:
            print(f"Config {config_hash} already evaluated before permutation.")
            continue

        print(f"Evaluating config before permutation: {sampled_config}")
        model.reset_super_network()
        ppls, params, _ = evaluate_configs(model, [sampled_config], space)
        
        eval_results_before[config_hash] = {
            "config": sampled_config,
            "perplexity": ppls[0],
            "parameters": params[0]
        }
        save_checkpoint(eval_results_before_path, eval_results_before)

    # Compute Importance Orders
    if not os.path.exists(orders_path):
        orders = {
            "embedding_order": compute_order_embd(args.max_seq_len, args.objective, model, tokenizer, batch_size, num_batches),
            "mlp_order": compute_order_intermediate_dims(args.max_seq_len, args.objective, model, tokenizer, batch_size, num_batches),
        }

        #orders["layer_order"] = compute_order_block_importance(args.max_seq_len, args.objective, model, tokenizer, batch_size, num_batches)
        if args.layer_scheme == "block_importance":
            # compute layer order perplexity scheme
            layer_order = compute_order_block_importance(
                args.max_seq_len, args.objective, model, tokenizer, batch_size, num_batches
            )
        elif args.layer_scheme == "perplexity":
            # compute layer order perplexity scheme
            layer_order = compute_order_layers_ppl(
                args.max_seq_len,
                model,
                tokenizer,
                largest_model_config,
                batch_size,
                num_batches,
            )
        orders["layer_order"] = layer_order
        orders["head_order"] = compute_order_head_groups(args.max_seq_len, args.objective, model, tokenizer, batch_size, num_batches)
        
        save_checkpoint(orders_path, orders)
    else:
        orders = load_checkpoint(orders_path)

    # Permute Model with Head Order
    model.reset_super_network()
    model = permute_model(orders["embedding_order"], orders["head_order"], orders["mlp_order"], model, model_id)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"permuted_model_{identifier}.pth"))

    # Post-Permutation Evaluation
    if os.path.exists(eval_results_after_path):
        eval_results_after = load_checkpoint(eval_results_after_path)
    else:
        eval_results_after = {}
    largest_model_config = {
        "sub_network_n_embd": config.n_embd,
        "sub_network_intermediate_size": config.intermediate_size,
        "sub_network_num_heads": config.n_head,
        "sub_network_n_layers": config.n_layer,
        "sub_network_head_size": config.head_size,
    }
    config_hash = get_config_hash(largest_model_config)
    if config_hash not in eval_results_after:
        print(f"Evaluating largest model config after permutation: {largest_model_config}")
        model.reset_super_network()
        ppls, params, _ = evaluate_config_largest(model, [largest_model_config], space, layer_order=orders["layer_order"])
        
        eval_results_after[config_hash] = {
            "config": largest_model_config,
            "perplexity": ppls[0],
            "parameters": params[0]
        }
        save_checkpoint(eval_results_after_path, eval_results_after
        )
    for sampled_config in configs:
        config_hash = get_config_hash(sampled_config)
        if config_hash in eval_results_after:
            print(f"Config {config_hash} already evaluated after permutation.")
            continue

        print(f"Evaluating config after permutation: {sampled_config}")
        model.reset_super_network()
        ppls, params, _ = evaluate_configs(model, [sampled_config], space, layer_order=orders["layer_order"])
        
        eval_results_after[config_hash] = {
            "config": sampled_config,
            "perplexity": ppls[0],
            "parameters": params[0]
        }
        save_checkpoint(eval_results_after_path, eval_results_after)
