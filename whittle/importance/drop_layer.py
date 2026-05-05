from __future__ import annotations

import torch
from datasets import load_from_disk
from tqdm import tqdm

from whittle.importance.utils import (
    get_dataloader,
    normalize_ranks,
    sort_keys_by_score,
)


def compute_ppl_block(
    max_length,
    model,
    tokenizer,
    batch_size=32,
    num_batches=10,
    dataset_path="/work/dlclarge2/sukthank-whittle/dense-lotteries/dataloaders/wikitext/",
):
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

    return torch.exp(torch.stack(nlls).mean()).item()  # compute perplexity


def compute_order_layers_ppl(
    max_seq_len,
    model,
    tokenizer,
    batch_size=32,
    num_batches=10,
    dataset_path="/work/dlclarge2/sukthank-whittle/dense-lotteries/dataloaders/wikitext/",
):
    model.reset_super_network()
    layer_importance_scores = {}
    largest_model_config = {
        "sub_network_n_embd": model.config.n_embd,
        "sub_network_intermediate_size": model.config.intermediate_size,
        "sub_network_num_heads": model.config.n_head,
    }
    sub_network_n_layers = model.config.n_layer - 1
    for layer in range(model.config.n_layer):
        model.set_sub_network(
            **largest_model_config,
            sampled_layer_indices=[i for i in range(model.config.n_layer) if i != layer],
            sub_network_n_layers=sub_network_n_layers,
        )  # drop layer corresponding to index_block
        score = compute_ppl_block(
            max_seq_len, model, tokenizer, batch_size, num_batches, dataset_path
        )
        layer_importance_scores[str(layer)] = score
    # higher PPL after dropping → more important layer, so rank ascending
    scores = normalize_ranks(layer_importance_scores, descending=False)
    return sort_keys_by_score(scores)
