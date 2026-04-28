from __future__ import annotations

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

# from whittle.loss.loss_factory import LossFactory
from modules.utils import (
    aggregate_by_scheme,
    get_dataloader,
)


def compute_softmaxed_scores(scores_dict, dim):
    max_val = max(scores_dict.values())  # Get the maximum value
    normalization_factor = sum([np.exp(v - max_val) for v in scores_dict.values()])
    return {
        str(k): np.exp(v - max_val) / normalization_factor for k, v in scores_dict.items()
    }


def compute_importance_head_size(
    max_length,
    objective,
    model,
    tokenizer,
    batch_size=32,
    num_batches=10,
    dataset_path="dataloaders/wikitext/",
):
    mixed_dataset = load_from_disk(dataset_path)
    dataloader = get_dataloader(tokenizer, mixed_dataset, max_length, batch_size)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    model.reset_super_network()
    head_dict = {}
    # initialize zero score for all batches, all heads
    for i in range(model.config.head_size):
        head_dict[str(i)] = [0 for _ in range(num_batches)]
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            _ = model(input_ids)
        for i in range(model.config.n_layer):  # iterate and sum up scores across layers
            k = f"block_{i}_attn_out"
            q, k, v = model.intermediate_outputs[k]
            for j in range(model.config.head_size):
                head_size_q_act = q[:, :, :, j].reshape(-1)
                head_size_k_act = k[:, :, :, j].reshape(-1)
                head_size_v_act = v[:, :, :, j].reshape(-1)
                head_size_score = (
                    aggregate_by_scheme(head_size_q_act, objective)
                    + aggregate_by_scheme(head_size_k_act, objective)
                    + aggregate_by_scheme(head_size_v_act, objective)
                )
                # aggregate across all layers for each batch and for each head j
                head_dict[str(j)][count] += head_size_score
        if count + 1 == num_batches:
            break

    for i in range(model.config.head_size):
        head_dict[str(i)] = np.mean(np.array(head_dict[str(i)]))
    # compute mean over batches
    head_scores = head_dict

    # extract scores
    keys = list(head_scores.keys())
    scores = np.array([head_scores[k] for k in keys])

    # get descending ranks: higher score → higher rank
    order = np.argsort(-scores)  # descending order
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)  # rank starts at 1

    # normalize ranks to [0, 1]
    norm_ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())

    # replace scores with normalized ranks
    head_ranks = {k: norm_ranks[i] for i, k in enumerate(keys)}

    return head_ranks


def compute_order_head_size(
    function, max_seq_len, objective, model, tokenizer, batch_size=32, num_batches=10
):
    model.reset_super_network()
    head_size_importance_scores = function(
        max_seq_len, objective, model, tokenizer, batch_size, num_batches
    )
    if isinstance(head_size_importance_scores, list):
        return head_size_importance_scores
    return [
        int(i)
        for i in sorted(
            head_size_importance_scores, key=head_size_importance_scores.get, reverse=True
        )
    ]
