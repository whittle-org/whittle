from __future__ import annotations

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

from importance.utils import get_dataloader

global dataset_path
dataset_path = "/work/dlclarge2/sukthank-whittle/dense-lotteries/dataloaders/wikitext/"


def compute_softmaxed_scores(scores_dict, dim):
    max_val = max(scores_dict.values())  # Get the maximum value
    normalization_factor = sum([np.exp(v - max_val) for v in scores_dict.values()])
    return {
        str(k): np.exp(v - max_val) / normalization_factor for k, v in scores_dict.items()
    }


def compute_ppl_block(max_length, model, tokenizer, batch_size=32, num_batches=10):
    mixed_dataset = load_from_disk(dataset_path)
    dataloader = get_dataloader(tokenizer, mixed_dataset, max_length, batch_size)
    nlls = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["labels"].to(device)
        with torch.no_grad():
            outputs = model(input_ids)  # output with a dropped layer
            neg_log_likelihood = torch.nn.CrossEntropyLoss()(
                outputs.view(-1, outputs.size(-1)), target_ids.view(-1)
            )
        nlls.append(neg_log_likelihood)
        if count + 1 == num_batches:
            break

    return torch.exp(torch.stack(nlls).mean()).item()  # compute perplextity


def compute_order_layers_ppl(
    max_seq_len, objective, model, tokenizer, batch_size=32, num_batches=10
):
    model.reset_super_network()
    layer_importance_scores = {}
    largest_model_config = {}
    largest_model_config["sub_network_n_embd"] = model.config.n_embd
    largest_model_config["sub_network_intermediate_size"] = model.config.intermediate_size
    largest_model_config["sub_network_num_heads"] = model.config.n_head
    largest_model_config["sub_network_n_layers"] = model.config.n_layer
    sub_network_n_layers = largest_model_config["sub_network_n_layers"] - 1
    del largest_model_config["sub_network_n_layers"]
    for layer in range(model.config.n_layer):
        model.set_sub_network(
            **largest_model_config,
            sampled_layer_indices=[i for i in range(model.config.n_layer) if i != layer],
            sub_network_n_layers=sub_network_n_layers,
        )  # drop layer corresponding to index_block
        score = compute_ppl_block(max_seq_len, model, tokenizer, batch_size, num_batches)
        layer_importance_scores[str(layer)] = score
    # compute mean over batches

    # extract scores
    keys = list(layer_importance_scores.keys())
    scores = np.array([layer_importance_scores[k] for k in keys])

    # get descending ranks: higher score → higher rank
    order = np.argsort(scores)  # descending order
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)  # rank starts at 1

    # normalize ranks to [0, 1]
    norm_ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())

    # replace scores with normalized ranks
    scores = {k: norm_ranks[i] for i, k in enumerate(keys)}
    if isinstance(scores, list):
        return scores
    return [
        int(i) for i in sorted(scores, key=scores.get, reverse=True)
    ]  # higher the score more important the block
