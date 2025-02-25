import json
import torch

from litgpt.utils import chunked_cross_entropy

from whittle.eval.utils import convert_and_evaluate


def update_config(
    config,
    sub_network_n_embd: int,
    sub_network_intermediate_size: int,
    sub_network_num_heads: int,
    sub_network_n_layers: int,
    sub_network_query_groups=None,
    sub_network_head_size=None,
):
    config.n_embd = sub_network_n_embd
    config.intermediate_size = sub_network_intermediate_size
    config.n_head = sub_network_num_heads
    if sub_network_query_groups is not None:
        config.n_query_groups = sub_network_query_groups
    if sub_network_head_size is not None:
        config.head_size = sub_network_head_size
    return config


def evaluate_grid(model, sampler, dataset, sampling_strategy, checkpoint_dir):
    params = []
    acc_norms = []
    model.max_seq_length = 1205

    from whittle.metrics.parameters import (
        compute_parameters,
        compute_parameters_sub_network_gpt,
    )

    for config in sampler.grid[::-1]:
        if "grid-params" in sampling_strategy:
            sub_network_config = sampler.search_space.cast(config)
        else:
            sub_network_config = config
        model.set_sub_network(**sub_network_config)
        params_sub = compute_parameters_sub_network_gpt(model)
        acc = compute_accuracy(model, dataset, checkpoint_dir)
        params.append(params_sub)
        acc_norms.append(acc)
    return params, acc_norms


def get_task_metric_map(dataset):
    if dataset == "winogrande":
        return "acc"
    elif dataset == "arc_challenge":
        return "acc_norm"
    elif dataset == "mmlu":
        return "acc"
    elif dataset == "hellaswag":
        return "acc_norm"
    elif dataset == "gsm8k":
        return "acc"
    elif dataset == "truthfulqa_mc2":
        return "acc"
    else:
        return "acc_norm"


def evaluate_grid_lambada(model, sampler, dataset, sampling_strategy, checkpoint_dir):
    params = []
    acc_norms = []
    model.max_seq_length = 653
    from whittle.metrics.parameters import (
        compute_parameters,
        compute_parameters_sub_network_gpt,
    )

    for config in sampler.grid[::-1]:
        if "grid-params" in sampling_strategy:
            sub_network_config = sampler.search_space.cast(config)
        else:
            sub_network_config = config
        model.set_sub_network(**sub_network_config)
        params_sub = compute_parameters_sub_network_gpt(model)
        acc = compute_accuracy_lambada(model, dataset, checkpoint_dir)
        params.append(params_sub)
        acc_norms.append(acc)
    return params, acc_norms


def compute_accuracy_lambada(model, dataset, checkpoint_dir):
    convert_and_evaluate(
        model,
        out_dir=checkpoint_dir,
        device=None,
        dtype=torch.float32,
        tasks=dataset,
        batch_size=16,  # Test for non-positive integer
    )
    with open(str(checkpoint_dir / "results.json")) as f:
        results = json.load(f)
    acc = results["results"]["lambada_openai"]["acc,none"]
    return acc


def plot_validation_metrics(model, val_dataloader, eval, sampler):
    # compute loss for superent
    model.eval()
    model.reset_super_network()
    val_loss_largest = compute_loss(model, val_dataloader, eval)
    middle_config = sampler.get_medium_sub_network()
    model.set_sub_network(**middle_config)
    val_loss_medium = compute_loss(model, val_dataloader, eval)
    model.reset_super_network()
    smallest_config = sampler.get_smallest_sub_network()
    model.set_sub_network(**smallest_config)
    val_loss_smallest = compute_loss(model, val_dataloader, eval)
    model.reset_super_network()
    return val_loss_largest, val_loss_medium, val_loss_smallest


def plot_validation_metrics_custom(model, val_dataloader, eval, sampler):
    # compute loss for superent
    model.eval()
    model.reset_super_network()
    val_loss_largest = compute_loss(model, val_dataloader, eval)

    return val_loss_largest, 0, 0


def plot_accuracies(model, sampler, dataset, checkpoint_dir):
    model.eval()
    model.reset_super_network()
    val_loss_largest = compute_accuracy(model, dataset, checkpoint_dir)
    middle_config = sampler.get_medium_sub_network()
    model.set_sub_network(**middle_config)
    val_loss_medium = compute_accuracy(model, dataset, checkpoint_dir)
    model.reset_super_network()
    smallest_config = sampler.get_smallest_sub_network()
    model.set_sub_network(**smallest_config)
    val_loss_smallest = compute_accuracy(model, dataset, checkpoint_dir)
    model.reset_super_network()
    return val_loss_largest, val_loss_medium, val_loss_smallest


def plot_accuracies_custom(model, sampler, dataset, checkpoint_dir):
    model.eval()
    model.reset_super_network()
    val_loss_largest = compute_accuracy(model, dataset, checkpoint_dir)
    return val_loss_largest, 0.0, 0.0

def plot_accuracies_slm(model, sampler, dataset, checkpoint_dir, subnet_config):
    model.eval()
    model.set_sub_network(**subnet_config)
    val_loss = compute_accuracy(model, dataset, checkpoint_dir)
    return val_loss

def plot_validation_metrics_slm(model, val_dataloader, eval, sampler, subnet_config):

    model.eval()
    model.set_sub_network(**subnet_config)
    val_loss = compute_loss(model, val_dataloader, eval)
    return val_loss

def compute_loss(model, val_dataloader, eval):
    losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(
            logits[..., :-1, :], targets[..., 1:], chunk_size=0
        )

    val_loss = losses.mean()
    return val_loss


def compute_accuracy(model, dataset, checkpoint_dir):
    metric = get_task_metric_map(dataset)
    convert_and_evaluate(
        model,
        out_dir=checkpoint_dir,
        device=None,
        dtype=torch.float32,
        tasks=dataset,
        batch_size=16,  # Test for non-positive integer
    )
    with open(str(checkpoint_dir / "results.json")) as f:
        results = json.load(f)
    acc = results["results"][dataset][f"{metric},none"]
    return acc
