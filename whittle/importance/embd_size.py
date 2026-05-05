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


def aggregate_across_batches(n_embd, embd_scores):
    # mean over batches per embedding dimension
    return {str(i): np.mean(embd_scores[str(i)]) for i in range(n_embd)}


def compute_importance_embd(
    max_length,
    objective,
    model,
    tokenizer,
    batch_size,
    num_batches,
    dataset_path="/work/dlclarge2/sukthank-whittle/dense-lotteries/dataloaders/wikitext/",
):
    mixed_dataset = load_from_disk(dataset_path)
    dataloader = get_dataloader(tokenizer, mixed_dataset, max_length, batch_size)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    model.reset_super_network()
    embd_scores = {}
    for i in range(model.config.n_embd):
        embd_scores[f"{i}"] = [0 for _ in range(num_batches)]
    # if we have multiple batches sum scores over batches take mean over batch scores
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)

        with torch.no_grad():  # don't compute grad, save memory
            _ = model(input_ids)  # save activations in forward
            # intermediate_out is a dictionary saving intermediate activations
            for k in model.intermediate_outputs:
                NORM_KEY = ["norm_f"]
                if any(
                    substr in k for substr in NORM_KEY
                ):  # since we compute emb importance, only consider norm layers
                    for i in range(model.config.n_embd):
                        matrix_x_fc = model.intermediate_outputs[k].reshape(
                            -1, model.config.n_embd
                        )[:, i]  # extract the output corresponding to ith neuron
                        importance_agg = aggregate_by_scheme(matrix_x_fc, objective)
                        embd_scores[f"{i}"][count] += importance_agg
        if count + 1 == num_batches:
            break
    return aggregate_across_batches(model.config.n_embd, embd_scores)


def compute_order_embd(
    function,
    max_seq_len,
    objective,
    model,
    tokenizer,
    batch_size=32,
    num_batches=10,
):
    embd_importance_scores = function(
        max_seq_len, objective, model, tokenizer, batch_size, num_batches
    )
    return sort_keys_by_score(embd_importance_scores)
