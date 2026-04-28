from importance.utils import get_dataloader
from tqdm import tqdm
import torch
import numpy as np
#from whittle.loss.loss_factory import LossFactory
from importance.utils import (
    aggregate_by_scheme,
)
from datasets import load_from_disk

global dataset_path
dataset_path = "/work/dlclarge2/sukthank-whittle/dense-lotteries/dataloaders/wikitext/"

def compute_softmaxed_scores(scores_dict, dim):
    max_val = max(scores_dict.values())  # Get the maximum value
    normalization_factor = sum([np.exp(v-max_val) for v in scores_dict.values()])
    return {str(k): np.exp(v-max_val) / normalization_factor for k, v in scores_dict.items()}


def compute_importance_heads(
    max_length, objective, model, tokenizer, batch_size=32, num_batches=10
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
    # extract scores
    keys = list(head_dict.keys())
    scores = np.array([head_dict[k] for k in keys])

    # get descending ranks: higher score → higher rank
    order = np.argsort(-scores)  # descending order
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)  # rank starts at 1

    # normalize ranks to [0, 1]
    norm_ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())

    # replace scores with normalized ranks
    head_ranks = {k: norm_ranks[i] for i, k in enumerate(keys)}
    return head_ranks


def compute_order_heads(
    function, max_seq_len, objective, model, tokenizer, batch_size=32, num_batches=10
):
    model.reset_super_network()
    head_importance_scores = function(
        max_seq_len, objective, model, tokenizer, batch_size, num_batches
    )
    if isinstance(head_importance_scores, list):
        return head_importance_scores
    return [
        int(i)
        for i in sorted(
            head_importance_scores, key=head_importance_scores.get, reverse=True
        )
    ]

def compute_importance_head_groups(
    max_length, objective, model, tokenizer, batch_size=32, num_batches=10
):
    mixed_dataset = load_from_disk(dataset_path)
    dataloader = get_dataloader(tokenizer, mixed_dataset, max_length, batch_size)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    model.reset_super_network()
    head_dict = {}
    # initialize zero score for all batches, all heads
    for i in range(model.config.n_query_groups):
        head_dict[str(i)] = {}
        for j in range(model.config.n_head//model.config.n_query_groups):
            head_dict[f"{i}"][f"{j}"] = [0 for _ in range(num_batches)]
    n_heads_per_group = model.config.n_head // model.config.n_query_groups
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            _ = model(input_ids)
        for i in range(model.config.n_layer):  # iterate and sum up scores across layers
            k = f"block_{i}_attn_out"
            q, k, v, mask = model.intermediate_outputs[k]
            for j in range(model.config.n_query_groups):
                for h in range(n_heads_per_group):
                    act_q = q[
                        :, j * n_heads_per_group + h, :, :
                    ].unsqueeze(1)
                    act_k = k[
                        :, j, :, :
                    ].unsqueeze(1)
                    act_v = v[
                        :, j, :, :
                    ].unsqueeze(1)
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
                    head_dict[str(j)][str(h)][count] += head_act_norm
        
        if count + 1 == num_batches:
            break

    for i in range(model.config.n_query_groups):
        for j in range(n_heads_per_group):
            head_dict[str(i)][str(j)] = torch.mean(
                torch.tensor(head_dict[str(i)][str(j)])
            )
        #head_dict[str(i)] = compute_softmaxed_scores(head_dict[str(i)], 0)
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
    if isinstance(group_importance_scores, list):
        return group_importance_scores
    return [
        int(i)
        for i in sorted(
            group_importance_scores, key=group_importance_scores.get, reverse=True
        )
    ]


