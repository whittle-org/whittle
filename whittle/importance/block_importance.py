from __future__ import annotations

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

from whittle.importance.utils import (
    get_dataloader,
    sort_keys_by_score,
)


def compute_block_importance(
    max_length,
    model,
    tokenizer,
    dataset_path,
    batch_size=32,
    num_batches=10,
):
    mixed_dataset = load_from_disk(dataset_path)
    dataloader = get_dataloader(tokenizer, mixed_dataset, max_length, batch_size)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    bi_dict = {}
    model.reset_super_network()
    # initialize zero score for all batches, all layers
    for i in range(model.config.n_layer):
        bi_dict[str(i)] = [0 for _ in range(num_batches)]
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            _ = model(input_ids)
        for i in range(model.config.n_layer):
            mat_1 = model.intermediate_outputs[f"block_{i}"].reshape(
                -1, model.config.n_embd
            )  # layer input, shape B*S, Emb
            mat_2 = model.intermediate_outputs[f"block_{i + 1}"].reshape(
                -1, model.config.n_embd
            )  # layer output, shape B*S, Emb

            for j in range(mat_1.shape[0]):  # index over features of each word i.e. B*S
                xtx = torch.sum(mat_1[j, :] * mat_2[j, :])
                norm1 = torch.norm(mat_1[j, :])
                norm2 = torch.norm(mat_2[j, :])
                bi_dict[str(i)][count] += xtx / (
                    norm1 * norm2
                )  # sum over word feature similarity
            bi_dict[str(i)][count] = (
                bi_dict[str(i)][count].item() / mat_1.shape[0]
            )  # divide by total number of words
        if count + 1 == num_batches:
            break

    # Calculate the inverted block importance aggregating over batches.
    # Higher 1 - cos(input, output) means the block transformed the
    # representation more, i.e. the block is more important.
    return {
        str(i): 1 - np.mean(np.array(bi_dict[str(i)]))
        for i in range(model.config.n_layer)
    }


def compute_order_block_importance(
    function, max_seq_len, model, tokenizer, batch_size=32, num_batches=10
):
    model.reset_super_network()
    scores = function(max_seq_len, model, tokenizer, batch_size, num_batches)
    # higher score → more important block
    return sort_keys_by_score(scores)
