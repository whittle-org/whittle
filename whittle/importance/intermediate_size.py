from importance.utils import get_dataloader
from tqdm import tqdm
import torch
import numpy as np
#from whittle.loss.loss_factory import LossFactory
from importance.utils import (
    aggregate_by_scheme
)
from datasets import load_from_disk

global dataset_path
dataset_path = "/work/dlclarge2/sukthank-whittle/dense-lotteries/dataloaders/wikitext/"

def compute_softmaxed_scores(scores_dict, dim):
    max_val = max(scores_dict.values())  # Get the maximum value
    normalization_factor = sum([np.exp(v-max_val) for v in scores_dict.values()])
    return {str(k): np.exp(v-max_val) / normalization_factor for k, v in scores_dict.items()}

def compute_importance_intermediate_size(
    max_length, objective, model, tokenizer, batch_size=32, num_batches=10, dataset_path="/work/dlclarge2/sukthank-whittle/dense-lotteries/dataloaders/wikitext/"
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
    # compute mean over batches
    for i in range(model.config.intermediate_size):
        mlp_scores[str(i)] = np.mean(np.array(mlp_scores[str(i)]))
    # extract scores
    keys = list(mlp_scores.keys())
    scores = np.array([mlp_scores[k] for k in keys])

    # get descending ranks: higher score → higher rank
    order = np.argsort(-scores)  # descending order
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)  # rank starts at 1

    # normalize ranks to [0, 1]
    norm_ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())

    # replace scores with normalized ranks
    mlp_ranks = {k: norm_ranks[i] for i, k in enumerate(keys)}
    return mlp_ranks

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
    if isinstance(mlp_importance_scores, list):
        return mlp_importance_scores
    return [
        int(i)
        for i in sorted(
            mlp_importance_scores, key=mlp_importance_scores.get, reverse=True
        )
    ]