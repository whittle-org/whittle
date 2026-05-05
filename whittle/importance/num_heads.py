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


def compute_importance_heads(
    max_length,
    objective,
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
    model.reset_super_network()
    head_dict = {}
    # initialize zero score for all batches, all heads
    for i in range(model.config.n_head):
        head_dict[str(i)] = [0 for _ in range(num_batches)]
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            _ = model(input_ids)
        for i in range(model.config.n_layer):  # iterate and sum up scores across layers
            k = f"block_{i}_attn_out"
            q, k, v, mask = model.intermediate_outputs[k]
            for j in range(model.config.n_head):
                head_act = model.transformer.h[i].attn.scaled_dot_product_attention(
                    q[:, j, :, :].unsqueeze(1),
                    k[:, j, :, :].unsqueeze(1),
                    v[:, j, :, :].unsqueeze(1),
                    mask,
                )  # attention computation for a single head
                head_act = head_act.reshape(head_act.shape[0], max_length, -1)
                head_act = torch.norm(head_act, dim=-1)
                head_act_norm = aggregate_by_scheme(head_act, objective)
                # aggregate across all layers for each batch and for each head j
                head_dict[str(j)][count] += head_act_norm
        if count + 1 == num_batches:
            break

    for i in range(model.config.n_head):
        head_dict[str(i)] = np.mean(np.array(head_dict[str(i)])).item()
    return head_dict


def compute_order_heads(
    function, max_seq_len, objective, model, tokenizer, batch_size=32, num_batches=10
):
    model.reset_super_network()
    head_importance_scores = function(
        max_seq_len, objective, model, tokenizer, batch_size, num_batches
    )
    return sort_keys_by_score(head_importance_scores)


def compute_importance_head_groups(
    max_length,
    objective,
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
    model.reset_super_network()
    head_dict = {}
    # initialize zero score for all batches, all heads
    for i in range(1, model.config.n_query_groups + 1):
        head_dict[str(i)] = {}
        for n_query_groups in range(
            1, model.config.n_head // model.config.n_query_groups + 1
        ):
            head_dict[f"{i}"][f"{n_query_groups}"] = [0 for _ in range(num_batches)]
    n_heads_per_group = model.config.n_head // model.config.n_query_groups
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            _ = model(input_ids)
        for i in range(model.config.n_layer):  # iterate and sum up scores across layers
            k = f"block_{i}_attn_out"
            q, k, v, mask = model.intermediate_outputs[k]
            for n_query_groups in range(1, model.config.n_query_groups + 1):
                for h in range(1, n_heads_per_group + 1):
                    head_indices = torch.arange(h)
                    group_start_indices = torch.arange(
                        0, n_query_groups * n_heads_per_group, n_heads_per_group
                    )
                    q_indices = (
                        head_indices.view(1, -1) + group_start_indices.view(-1, 1)
                    ).flatten()

                    act_q = q[:, q_indices, :, :]
                    act_k = k[:, q_indices, :, :]
                    act_v = v[:, q_indices, :, :]
                    head_act = model.transformer.h[i].attn.scaled_dot_product_attention(
                        act_q,
                        act_k,
                        act_v,
                        mask,
                    )
                    head_act = head_act.reshape(batch_size, max_length, -1)
                    head_act = torch.norm(head_act, dim=-1)
                    head_act_norm = aggregate_by_scheme(head_act, objective)
                    # aggregate across all layers for each batch and for each head j
                    head_dict[str(n_query_groups)][str(h)][count] += head_act_norm

        if count + 1 == num_batches:
            break

    for i in range(1, model.config.n_query_groups + 1):
        for n_query_groups in range(1, n_heads_per_group + 1):
            head_dict[str(i)][str(n_query_groups)] = torch.mean(
                torch.tensor(head_dict[str(i)][str(n_query_groups)])
            )
    return head_dict


def compute_order_head_groups(
    function, max_seq_len, objective, model, tokenizer, batch_size=32, num_batches=10
):
    model.reset_super_network()
    head_importance_scores = function(
        max_seq_len, objective, model, tokenizer, batch_size, num_batches
    )
    group_importance_scores = {}
    for group in head_importance_scores:
        group_importance_scores[group] = sum(head_importance_scores[group].values())
    return sort_keys_by_score(group_importance_scores)
