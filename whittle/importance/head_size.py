from __future__ import annotations

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

from whittle.importance.utils import (
    aggregate_by_scheme,
    get_dataloader,
    sort_keys_by_score,
)


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

    # mean over batches
    for i in range(model.config.head_size):
        head_dict[str(i)] = np.mean(np.array(head_dict[str(i)]))

    return head_dict


def compute_order_head_size(
    function, max_seq_len, objective, model, tokenizer, batch_size=32, num_batches=10
):
    model.reset_super_network()
    head_size_importance_scores = function(
        max_seq_len, objective, model, tokenizer, batch_size, num_batches
    )
    return sort_keys_by_score(head_size_importance_scores)
