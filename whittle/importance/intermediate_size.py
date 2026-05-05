from __future__ import annotations

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

from whittle.importance.utils import (
    aggregate_by_scheme,
    get_dataloader,
    normalize_ranks,
    sort_keys_by_score,
)


def compute_importance_intermediate_size(
    max_length,
    objective,
    model,
    tokenizer,
    batch_size=32,
    num_batches=10,
    dataset_path="/work/dlclarge2/sukthank-whittle/dense-lotteries/dataloaders/wikitext/",
):
    mixed_dataset = load_from_disk(dataset_path)
    dataloader = get_dataloader(tokenizer, mixed_dataset, max_length, batch_size)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.reset_super_network()
    model.eval()
    mlp_scores = {}
    # initialize zero score for all batches, all neurons
    for i in range(model.config.intermediate_size):
        mlp_scores[f"{i}"] = [0 for i in range(num_batches)]
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        with torch.no_grad():  # don't compute grad, save memory
            _ = model(input_ids)  # save activations in forward
            # intermediate_out is a dictionary saving intermediate activations
            for k in model.intermediate_outputs:
                if "mlp" in k:
                    for i in range(model.config.intermediate_size):
                        matrix_x_fc = model.intermediate_outputs[k].reshape(
                            -1, model.config.intermediate_size
                        )[:, i]  # extract the output corresponding to ith neuron
                        mlp_scores[f"{i}"][count] += aggregate_by_scheme(
                            matrix_x_fc, objective
                        )
        if count + 1 == num_batches:
            break
    # mean over batches
    for i in range(model.config.intermediate_size):
        mlp_scores[str(i)] = np.mean(np.array(mlp_scores[str(i)]))
    return normalize_ranks(mlp_scores, descending=True)


def compute_order_intermediate_dims(
    function,
    max_seq_len,
    objective,
    model,
    tokenizer,
    batch_size=32,
    num_batches=10,
    group_size=128,
):
    model.reset_super_network()
    if objective in [
        "forward_kl",
        "reverse_kl",
        "symmetric_kl",
        "js_distance",
        "l1",
        "l2",
        "cosine_similarity",
        "mmd",
    ]:
        mlp_importance_scores = function(
            max_seq_len,
            objective,
            model,
            tokenizer,
            batch_size,
            num_batches,
            group_size,
        )
    else:
        mlp_importance_scores = function(
            max_seq_len, objective, model, tokenizer, batch_size, num_batches
        )
    return sort_keys_by_score(mlp_importance_scores)
