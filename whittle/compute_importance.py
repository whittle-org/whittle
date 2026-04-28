import os
import argparse
import json
import torch
import pickle
import transformers
import numpy as np
from models.gpt.model import GPT
from litgpt import Config
from litgpt.scripts.download import download_from_hub
from litgpt import LLM
from sampling.random_sampler import RandomSampler
from metrics.parameters import compute_parameters

from importance.block_importance import compute_order_block_importance, compute_block_importance
from importance.drop_layer import compute_order_layers_ppl
from importance.embd_size import compute_order_embd, compute_importance_embd
from importance.intermediate_size import compute_importance_intermediate_size, compute_order_intermediate_dims
from importance.num_heads import compute_order_heads, compute_order_head_groups, compute_importance_heads, compute_importance_head_groups
from importance.utils import evaluate_wikitext, permute_model
from search.search_spaces import search_spaces

def get_configs(sampler, n=5):
    configs = []
    for _ in range(n):
        configs.append(sampler.sample())
    return configs


def compute_avg_decrease(ppl_before, ppl_after):
    return np.mean(ppl_before - ppl_after)


def evaluate_configs(model, configs, search_space, layer_order=None):
    ppls = []
    params = []
    for c in configs:
        #print(c)
        if layer_order is None:
            model.set_sub_network(**space.cast(c))
            param = compute_parameters(model)
            ppl = evaluate_wikitext(
                args.max_seq_len, model, tokenizer, batch_size, num_batches
            )
            ppls.append(ppl)
            params.append(param)
        else:
            layer_order_top_k = sorted(layer_order[: int(c["depth"])])
            model.set_sub_network(**space.cast(c), sampled_layer_indices=layer_order_top_k)
            param = compute_parameters(model)
            ppl = evaluate_wikitext(
                args.max_seq_len, model, tokenizer, batch_size, num_batches
            )
            ppls.append(ppl)
            params.append(param)
    return ppls, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model Importance Computation and Permutation"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, help="ID of the model to be used"
    )
    parser.add_argument(
        "--num_batches", type=int, default=10, help="Number of batches for evaluation"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--objective", type=str, default="norm")
    parser.add_argument("--space", type=str, default="hw_gpt_bench")
    parser.add_argument(
        "--layer_scheme", type=str, default="block_importance"
    )  # can be block importance, or perplexity scheme
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=21512)
    parser.add_argument("--n_configs", type=int, default=5)
    args = parser.parse_args()
    print(args)
    model_id = args.model_id
    num_batches = args.num_batches
    batch_size = args.batch_size

    # download model and set model config params
    # download_from_hub(repo_id=model_id)

    config_path = os.path.join("checkpoints", model_id, "model_config.yaml")
    config_path_hf = os.path.join("checkpoints", model_id, "config.json")
    model_path = os.path.join("checkpoints", model_id, "lit_model.pth")

    config = Config.from_file(config_path)
    config.fix_head_size = True
    config.model_type = "gpt"
    with open(config_path_hf) as f:
        hf_config = json.load(f)
    config.tie_embeddings = hf_config["tie_word_embeddings"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config,compute_importance=True)

    model.name_or_path = os.path.join("checkpoints", model_id)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(torch.bfloat16)
    model.to(device)

    # initialize model specific tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join("checkpoints", model_id)
    )
    # get search space
    space = search_spaces[args.space](config)

    # initialize sampler to sample a random architecture
    sampler = RandomSampler(space.config_space, seed=args.seed)
    configs = get_configs(sampler=sampler, n=args.n_configs)
    # the full large model or the teacher architecture
    largest_model_config = {
        "sub_network_n_embd": config.n_embd,
        "sub_network_intermediate_size": [config.intermediate_size] * config.n_layer,
        "sub_network_num_heads": [config.n_head] * config.n_layer,
        "sub_network_n_layers": config.n_layer,
    }
    # set random architecture
    model.reset_super_network()
    # compute params and perplexity before importance sorting
    before_sorting = evaluate_wikitext(
        args.max_seq_len, model, tokenizer, batch_size, num_batches
    )
    ppls_before, params_before = evaluate_configs(
        model, configs, space, layer_order=None
    )
    model.reset_super_network()

    # compute embedding order with layer norm
    embedding_order = compute_order_embd(
        compute_importance_embd, args.max_seq_len, args.objective, model, tokenizer, batch_size, num_batches
    )
    #print("Embedding order:", embedding_order)
    assert len(set(embedding_order)) == config.n_embd, "Embedding order length mismatch with model config"
    # compute intermediate_dim order
    mlp_order = compute_order_intermediate_dims(
        compute_importance_intermediate_size, args.max_seq_len, args.objective, model, tokenizer, batch_size, num_batches
    )
    #print("MLP order:", mlp_order)
    assert len(set(mlp_order)) == config.intermediate_size, "MLP order length mismatch with model config"
    if args.layer_scheme == "block_importance":
        # compute layer order perplexity scheme
        layer_order = compute_order_block_importance(compute_block_importance, args.max_seq_len, args.objective, model, tokenizer, batch_size, num_batches)
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
    assert len(set(layer_order)) == config.n_layer, "Layer order length mismatch with model config"
    #print("Layer order:", layer_order)
    if "pythia" in args.model_id:
        head_order = compute_order_heads(
            compute_importance_heads, args.max_seq_len, args.objective, model, tokenizer, batch_size, num_batches
        )
    else:
        head_order = compute_order_head_groups(
            compute_importance_head_groups, args.max_seq_len, args.objective, model, tokenizer, batch_size, num_batches
        )
    #print("Head order:", head_order)
    assert len(set(head_order)) == config.n_head, "Head order length mismatch with model config"
    #model = permute_model(embedding_order, head_order, mlp_order, model, model_id)
    model.reset_super_network()
    print(config)
    largest_model_config = {
        "sub_network_n_embd": config.n_embd,
        "sub_network_intermediate_size": config.intermediate_size,
        "sub_network_n_layers": config.n_layer,
        'sub_network_num_heads': config.n_head,
        "sampled_embd_indices": embedding_order,
        "sampled_intermediate_indices": mlp_order,
        "sampled_query_group_indices": head_order,
    }
    #print(largest_model_config)
    model.set_sub_network(**largest_model_config)
    # evaluate sampled architecture after importance sorting
    #model.reset_super_network()
    # compute params and perplexity after importance sorting, usually leads to drop in perplexity
    after_sorting = evaluate_wikitext(
        args.max_seq_len, model, tokenizer, batch_size, num_batches
    )
    ppls_after, params_after = evaluate_configs(
        model, configs, space, layer_order=layer_order
    )
    ppls_after.append(after_sorting)
    ppls_before.append(before_sorting)
    print("PPL of full network before", before_sorting)
    print("PPL of full network after", after_sorting)
    from models.gpt.extract import extract_current_sub_network
    permuted_model = extract_current_sub_network(model)
    permuted_model.to(torch.bfloat16)
    after_sorting = evaluate_wikitext(
        args.max_seq_len, permuted_model, tokenizer, batch_size, num_batches
    )
    print("PPL of sampled architectures before", after_sorting)

    print(
        "Average decrease",
        compute_avg_decrease(np.array(ppls_before), np.array(ppls_after)),
    )
    sorted_ids = {
        "embedding_order": embedding_order,
        "mlp_order": mlp_order,
        "layer_order": layer_order,
        "head_order": head_order,
    }
    sorted_ids_save_path = os.path.join(
        "checkpoints", model_id, f"sorted_ids_{args.objective}.pkl"
    )
    with open(sorted_ids_save_path, "wb") as f:
        pickle.dump(sorted_ids, f)

    # compute params and perplexity after importance sorting, usually leads to drop in perplexity
    after_sorting = evaluate_wikitext(
        args.max_seq_len, model, tokenizer, batch_size, num_batches
    )
    ppls_after, params_after = evaluate_configs(
        model, configs, space, layer_order=layer_order
    )
    ppls_after.append(after_sorting)
    ppls_before.append(before_sorting)
    print("PPL of full network before", before_sorting)
    print("PPL of full network after", after_sorting)
    #print(
    #    "Average decrease",
    #    compute_avg_decrease(np.array(ppls_before), np.array(ppls_after)),
    #)
    sorted_ids = {
        "embedding_order": embedding_order,
        "mlp_order": mlp_order,
        "layer_order": layer_order,
        "head_order": head_order,
    }
    sorted_ids_save_path = os.path.join(
        "checkpoints", model_id, f"sorted_ids_{args.objective}.pkl"
    )
    with open(sorted_ids_save_path, "wb") as f:
        pickle.dump(sorted_ids, f)
