import json
import torch

from litgpt.utils import chunked_cross_entropy

from whittle.eval.utils import convert_and_evaluate
import pickle
import os


def evaluate_grid(
    results_path, model, sampler, dataset, sampling_strategy, checkpoint_dir
):
    params = []
    acc_norms = []
    model.max_seq_length = 1205
    archs_evaluated = []
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results = pickle.load(f)
            archs_evaluated = results["grid"]
            acc_norms = results["accs"]
            params = results["params"]
    from whittle.metrics.parameters import (
        compute_all_parameters as compute_parameters,
        compute_parameters as compute_parameters_sub_network_gpt,
    )

    for config in sampler.grid[::-1]:
        if config not in archs_evaluated:
            if "grid-params" in sampling_strategy:
                sub_network_config = sampler.search_space.cast(config)
            else:
                sub_network_config = config
            model.set_sub_network(**sub_network_config)
            params_sub = compute_parameters_sub_network_gpt(model)
            acc = compute_accuracy(model, dataset, checkpoint_dir)
            params.append(params_sub)
            acc_norms.append(acc)
            archs_evaluated.append(config)
            with open(results_path, "wb") as f:
                pickle.dump(
                    {"accs": acc_norms, "params": params, "grid": archs_evaluated}, f
                )
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
    archs_evaluated = []
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results = pickle.load(f)
            archs_evaluated = results["grid"]
            acc_norms = results["accs"]
            params = results["params"]

    model.max_seq_length = 653
    from whittle.metrics.parameters import (
        compute_parameters,
        compute_parameters_sub_network_gpt,
    )

    for config in sampler.grid[::-1]:
        if config not in archs_evaluated:
            if "grid-params" in sampling_strategy:
                sub_network_config = sampler.search_space.cast(config)
            else:
                sub_network_config = config
            model.set_sub_network(**sub_network_config)
            params_sub = compute_parameters_sub_network_gpt(model)
            acc = compute_accuracy_lambada(model, dataset, checkpoint_dir)
            params.append(params_sub)
            acc_norms.append(acc)
            archs_evaluated.append(config)
            with open(results_path, "wb") as f:
                pickle.dump(
                    {"accs": acc_norms, "params": params, "grid": archs_evaluated}, f
                )
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
        batch_size=2,  # Test for non-positive integer
    )
    with open(str(checkpoint_dir / "results.json")) as f:
        results = json.load(f)
    acc = results["results"][dataset][f"{metric},none"]
    return acc
