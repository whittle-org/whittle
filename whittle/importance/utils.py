# os.environ['HF_DATASETS_OFFLINE'] = "1"
from __future__ import annotations

import copy

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from whittle.models.gpt.blocks import GemmaMLP, GptNeoxMLP, LLaMAMLP
from whittle.modules.layernorm import LayerNorm
from whittle.modules.rmsnorm import RMSNorm


def normalize_ranks(scores_dict, descending: bool = True):
    """Convert a {key: score} dict to {key: rank-normalised-to-[0, 1]}.

    Highest-scoring key gets rank 1 when ``descending=True``; the resulting
    rank is then min-max scaled into [0, 1].
    """
    keys = list(scores_dict.keys())
    scores = np.array([scores_dict[k] for k in keys])
    order = np.argsort(-scores) if descending else np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    norm_ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    return {k: norm_ranks[i] for i, k in enumerate(keys)}


def sort_keys_by_score(scores_dict):
    """Return integer keys of ``scores_dict`` sorted by descending score."""
    return [int(i) for i in sorted(scores_dict, key=scores_dict.get, reverse=True)]


def aggregate_by_scheme_distance(features, objective):
    if objective == "euclidean":
        return pairwise_euclidean_distance(features)
    elif objective == "cosine":
        return pairwise_cosine_similarity(features)
    else:
        raise ValueError("Invalid objective")


def aggregate_by_scheme(feature, scheme="mean"):
    if scheme == "mean":
        return torch.mean(torch.abs(feature.reshape(-1))).item()
    elif scheme == "norm":
        return torch.norm(feature.reshape(-1)).item()
    elif scheme == "mean-norm":
        return torch.mean(torch.norm(feature, dim=-1), dim=-1).item()
    elif scheme == "norm-mean":
        return torch.norm(torch.mean(torch.abs(feature), dim=-1), dim=-1).item()
    elif scheme == "variance":
        return torch.var(torch.var(feature, dim=-1), dim=-1).item()
    elif scheme == "variance-norm":
        return torch.var(torch.norm(feature, dim=-1), dim=-1).item()
    elif scheme == "variance-mean":
        return torch.var(torch.mean(torch.absolute(feature), dim=-1), dim=-1).item()
    elif scheme == "mean-variance":
        return torch.mean(torch.var(feature, dim=-1), dim=-1).item()
    elif scheme == "norm-variance":
        return torch.norm(torch.var(feature, dim=-1), dim=-1).item()
    else:
        raise ValueError("Invalid objective")


def pairwise_euclidean_distance(features):
    device = "cpu"
    with torch.no_grad():  # Disable gradient tracking to save memory
        feature_matrix = torch.stack(list(features.values())).to(device)
        if feature_matrix.dim() == 1:
            feature_matrix = feature_matrix.unsqueeze(0)

        diffs = feature_matrix.unsqueeze(1) - feature_matrix.unsqueeze(0)
        distance_matrix = torch.norm(diffs, dim=2)
        del diffs  # Free up memory
        torch.cuda.empty_cache()

    return distance_matrix


def pairwise_cosine_similarity(features):
    device = "cpu"
    with torch.no_grad():
        feature_matrix = torch.stack(list(features.values())).to(device)
        if feature_matrix.dim() == 1:
            feature_matrix = feature_matrix.unsqueeze(0)

        norms = torch.norm(feature_matrix, dim=1, keepdim=True).clamp(min=1e-10)
        normalized_features = feature_matrix / norms
        similarity_matrix = 1 - torch.mm(normalized_features, normalized_features.T)

        del norms, normalized_features
        torch.cuda.empty_cache()

    return similarity_matrix


def sort_by_average_distance(distance_matrix):
    with torch.no_grad():
        avg_distances = distance_matrix.sum(dim=1) / distance_matrix.shape[1]
        sorted_indices = torch.argsort(avg_distances, descending=True)

    return sorted_indices.tolist()


# Tokenize each example with dynamic sequence length
def tokenize_function(examples, tokenizer, seq_len):
    encodings = tokenizer("\n\n".join(examples["text"]), return_tensors="pt")
    seq_len_orig = encodings.input_ids.size(1)
    input_ids_li = []
    target_ids_li = []

    for begin_loc in range(0, seq_len_orig, seq_len):
        end_loc = min(begin_loc + seq_len, seq_len_orig)
        if end_loc != begin_loc + seq_len:  # ignore last batch
            break
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = torch.zeros_like(input_ids)
        target_ids[:, 0:-1] = input_ids[:, 1:]
        target_ids[:, -1] = -100
        # target_ids[:, -1] = tokenizer.pad_token_id  # Target padding
        input_ids_li.append(torch.squeeze(input_ids))
        target_ids_li.append(torch.squeeze(target_ids))

    return {
        "input_ids": input_ids_li,
        "labels": target_ids_li,
    }


# Dataset class
class TextDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }


# Dataloader function
# Tokenization and DataLoader Creation
def get_dataloader(tokenizer, mixed_dataset, seq_len=512, batch_size=8):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token if tokenizer.eos_token else tokenizer.eos_token_id
        )

    # Apply tokenization
    tokenized_data = tokenize_function(mixed_dataset, tokenizer, seq_len)
    dataset = TextDataset(tokenized_data["input_ids"], tokenized_data["labels"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_dataloader_dataset(tokenizer, mixed_dataset, seq_len=512, batch_size=8):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token if tokenizer.eos_token else tokenizer.eos_token_id
        )

    # Apply tokenization
    tokenized_data = tokenize_function(mixed_dataset, tokenizer, seq_len)
    dataset = TextDataset(tokenized_data["input_ids"], tokenized_data["labels"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset


def evaluate_wikitext(
    max_length,
    model,
    tokenizer,
    batch_size,
    num_batches,
    dataset_path="/work/dlclarge2/sukthank-whittle/dense-lotteries/dataloaders/wikitext/",
):
    mixed_dataset = load_from_disk(dataset_path)
    dataloader = get_dataloader(tokenizer, mixed_dataset, max_length, batch_size)
    nlls = []
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    model.eval()
    torch.manual_seed(2809)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["labels"].to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            neg_log_likelihood = torch.nn.CrossEntropyLoss()(
                outputs.view(-1, outputs.size(-1)), target_ids.view(-1)
            )
        nlls.append(neg_log_likelihood)
        if count + 1 == num_batches:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors="pt").input_ids[
        0
    ]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len) : ((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)


def evaluate_wikitext_correct(max_length, model, tokenizer, batch_size=1):
    # Load WikiText-2 dataset from Hugging Face
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = process_data(dataset, tokenizer, max_length, "text")
    test_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    nlls = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.to(torch.bfloat16)
    model.eval()
    torch.manual_seed(2809)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    for count, batch in enumerate(tqdm(test_dataloader, desc="Processing batches")):
        batch = batch.to(device)

        with torch.no_grad():
            outputs = model(batch)
            logits = outputs
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss = loss_fn(
                shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        nlls.append(loss)

        # if count + 1 == num_batches:
        #    break

    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()


@torch.no_grad()
def evaluate(search_loader, model, device, sampled_config, num_batches):
    nlls = []

    # switch to evaluation mode
    model.eval()
    model_module = model.module if hasattr(model, "module") else model
    model_module.set_sub_network(**sampled_config)

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        print(f"Rank {rank}: sampled model config: {sampled_config}")
    else:
        print(f"sampled model config: {sampled_config}")
    print(f"sampled model config: {sampled_config}")

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    for count, batch in enumerate(tqdm(search_loader, desc="Processing batches")):
        batch["input_ids"] = batch["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(batch["input_ids"])
            logits = outputs
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()

            loss = loss_fn(
                shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        nlls.append(loss)
    # Stack NLLs and compute mean
    local_mean_nll = torch.stack(nlls).mean()

    # Synchronize across processes
    if torch.distributed.is_initialized():
        # Aggregate results across processes
        global_sum = torch.tensor([local_mean_nll.item()], device=device)
        torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM)
        global_mean_nll = global_sum.item() / world_size
    else:
        global_mean_nll = local_mean_nll.item()

    # Compute perplexity
    ppl = torch.exp(torch.tensor(global_mean_nll))

    # Log perplexity
    if not torch.distributed.is_initialized() or rank == 0:
        print(f"* PPL: {ppl.item():.3f}")
    model.module.reset_super_network()
    return {"ppl": ppl.item()}


def permute_norm(embedding_order, norm):
    if norm is None:
        return
    if isinstance(norm, LayerNorm):
        norm.weight.data = norm.weight.data[embedding_order]
        norm.bias.data = norm.bias.data[embedding_order]
    elif isinstance(norm, RMSNorm):
        norm.weight.data = norm.weight.data[embedding_order]
    else:
        raise ValueError("Norm not supported")


def permute_mlp(embedding_order, mlp_order, mlp):
    if isinstance(mlp, GptNeoxMLP):
        mlp.fc.weight.data = mlp.fc.weight.data[mlp_order, :][:, embedding_order]
        if mlp.fc.bias is not None:
            mlp.fc.bias.data = mlp.fc.bias.data[mlp_order]
    elif isinstance(mlp, (LLaMAMLP, GemmaMLP)):
        mlp.fc_1.weight.data = mlp.fc_1.weight.data[mlp_order, :][:, embedding_order]
        if mlp.fc_1.bias is not None:
            mlp.fc_1.bias.data = mlp.fc_1.bias.data[mlp_order]
        mlp.fc_2.weight.data = mlp.fc_2.weight.data[mlp_order, :][:, embedding_order]
        if mlp.fc_2.bias is not None:
            mlp.fc_2.bias.data = mlp.fc_2.bias.data[mlp_order]
    else:
        raise ValueError("MLP not supported")
    mlp.proj.weight.data = mlp.proj.weight.data[:, mlp_order][embedding_order, :]
    if mlp.proj.bias is not None:
        mlp.proj.bias.data = mlp.proj.bias.data[embedding_order]


def permute_attention(embedding_order, indices_attn, indices_proj, attn):
    attn.qkv.set_sub_network(
        attn.sub_network_n_embd,
        attn.sub_network_qkv_shape,
        sampled_in_indices=embedding_order,
        sampled_out_indices=indices_attn,
    )
    attn.proj.set_sub_network(
        attn.sub_network_head_size
        * attn.sub_network_query_groups
        * attn.sub_network_q_per_kv,
        attn.sub_network_n_embd,
        sampled_in_indices=indices_proj,
        sampled_out_indices=embedding_order,
    )


def get_indices_pythia(head_order, n_head, head_size):
    indices_attn = []
    for h in head_order:
        start = 3 * h * head_size
        end = 3 * (h + 1) * head_size
        indices_attn.extend(range(start, end))
    return indices_attn


def permute_model(embedding_order, head_order, mlp_order, model, name):
    # print("head_order", head_order)
    model.transformer.wte.weight.data = model.transformer.wte.weight.data[
        :, embedding_order
    ]
    # print(head_order)
    model.lm_head.weight.data = model.lm_head.weight.data[:, embedding_order]
    permute_norm(embedding_order, model.transformer.ln_f)

    for block in model.transformer.h:
        permute_norm(embedding_order, block.norm_1)
        permute_norm(embedding_order, block.norm_2)
        permute_mlp(embedding_order, mlp_order, block.mlp)
        indices_attn = block.attn.get_qkv_indices(
            sampled_head_indices=head_order,
            sampled_query_groups_indices=None,
            sampled_head_size_indices=list(range(model.config.head_size)),
        )
        indices_proj = indices_attn[
            : torch.searchsorted(
                indices_attn, model.config.n_head * model.config.head_size, right=False
            )
        ]
        permute_attention(embedding_order, indices_attn, indices_proj, block.attn)
    return model


def permute_model_layerwise(embedding_order, head_order, mlp_order, model_base, name):
    model = copy.deepcopy(model_base)
    model.transformer.wte.weight.data = model.transformer.wte.weight.data[
        :, embedding_order
    ]
    model.lm_head.weight.data = model.lm_head.weight.data[:, embedding_order]
    permute_norm(embedding_order, model.transformer.ln_f)
    for i, block in enumerate(model.transformer.h):
        if "pythia" in name:
            (
                model.config.n_head + 2 * model.config.n_query_groups
            ) * model.config.head_size
            indices_attn = get_indices_pythia(
                head_order[f"{i}"], model.config.n_head, model.config.head_size
            )
            indices_proj = [
                k
                for h in head_order[f"{i}"]
                for k in range(
                    h * model.config.head_size, (h + 1) * model.config.head_size
                )
            ]
        else:
            heads_per_group = model.config.n_head // model.config.n_query_groups
            head_size = model.config.head_size
            indices_attn = []
            # print(head_order[f"{i}"])
            for ids in head_order[f"{i}"]:
                start = ids * (heads_per_group + 2) * head_size
                end = (ids + 1) * (heads_per_group + 2) * head_size
                indices_attn.extend(range(start, end))
            indices_proj = []
            for id in head_order[f"{i}"]:
                start = id * (heads_per_group) * head_size
                end = (id + 1) * (heads_per_group) * head_size
                indices_proj.extend(range(start, end))

        mlp_order_i = mlp_order[f"{i}"]
        permute_norm(embedding_order, block.norm_1)
        permute_norm(embedding_order, block.norm_2)
        permute_mlp(embedding_order, mlp_order_i, block.mlp)
        permute_attention(embedding_order, indices_attn, indices_proj, block.attn)

    return model
