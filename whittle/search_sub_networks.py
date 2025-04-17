from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from litgpt import Tokenizer
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import Alpaca, DataModule, TinyStories
from litgpt.finetune.lora import validate as finetune_validate
from litgpt.model import Config
from litgpt.pretrain import get_dataloaders, validate
from litgpt.utils import (
    auto_download_checkpoint,
    check_nvlink_connectivity,
    check_valid_checkpoint_dir,
    choose_logger,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    load_checkpoint,
    parse_devices,
)
from torch.utils.data import DataLoader

from whittle.args import ParamBinArgs, SearchArgs
from whittle.metrics import compute_flops, compute_latency, compute_parameters
from whittle.models.gpt import GPT, Block
from whittle.models.gpt.checkpoint import save_sub_network
from whittle.pretrain_super_network import get_search_space
from whittle.sampling.param_bins import ParamBins, ParamsEstimator
from whittle.search import multi_objective_search
from whittle.search.baselines import Methods


def setup(
    checkpoint_dir: Path,
    out_dir: Path | None = Path("out/finetune/full"),
    precision: str | None = None,
    devices: int | str | None = 1,
    num_nodes: int = 1,
    resume: bool | Literal["auto"] | Path | None = False,
    data: DataModule | None = None,
    search: SearchArgs = SearchArgs(
        iterations=100,
        log_interval=1,
    ),
    train: TrainArgs = TrainArgs(
        max_seq_length=512,
    ),
    eval: EvalArgs | None = EvalArgs(),
    logger_name: Literal["wandb", "tensorboard", "csv"] | None = "csv",
    seed: int | None = 1337,
    access_token: str | None = None,
    param_bins: ParamBinArgs = ParamBinArgs(),
    performance_metric: str | None = "val_loss",
    efficiency_metric: str | None = "parameters",
    log_objective_names: bool | None = True,
    save_checkpoints: bool = True,
    fine_tuned: bool = False,
    copy_config_files: bool = True,
    verbose: bool = True,
    num_workers: int = 4,
) -> None:
    """
    Multi-objective search to select Pareto optimal set of sub-networks from trained super-network.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        devices: How many devices/GPUs to use
        num_nodes: How many nodes the code is being run on.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``. An error will be raised if no checkpoint is found. Passing
            ``'auto'`` will resume from the latest checkpoint but not error if no checkpoint exists.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyStories`` or ``litgpt.data.Alpaca`` for fine-tuned models.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        search: Search-related arguments. See ``whittle.args.SearchArgs`` for details.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
        param_bins: The parameter bins that limit the sub-network params in the search.
        performance_metric: The name of the first objective to optimize (possible - val_loss, perplexity). Defaults to "val_loss".
        efficiency_metric: The name of the second objective to optimize (possible - parameters, latency, flops). Defaults to "parameters".
        log_objective_names: Whether to log the names of the objectives in the logger, or log as objective_1 and objective_2. Defaults to True.
        save_checkpoints: Whether to save checkpoints of the sub-networks, or config + path to super-network. Defaults to True.
            If False, `lit_model.pth` will have the following format:
            `{'sub_network_config': sub_network_config, 'parent_dir': checkpoint_dir}`
        fine_tuned: Whether the model is fine-tuned. Defaults to False.
            This flag determines the dataset to use if `data` is not provided. Additionally, it changes the validation function to use for evaluation.
            fine_tuned=True: litgpt.finetune.lora.validate, fine_tuned=False: litgpt.pretrain.validate.
        copy_config_files: Whether to copy the config files from the super-network to the sub-networks. Defaults to True.
            If set to False, we save `parent_dir` to `lit_model.pth`. If save_checkpoints is False, this argument is ignored.
        verbose: Whether to print verbose output. Defaults to True.
        num_workers: Number of workers to use for data loading. Defaults to 4.
    """
    assert performance_metric in [
        "val_loss",
        "perplexity",
    ], f"Invalid objective_1: {performance_metric}, must be 'val_loss' or 'perplexity'"
    assert efficiency_metric in [
        "parameters",
        "latency",
        "flops",
    ], (
        f"Invalid objective_2: {efficiency_metric}, must be 'parameters', 'latency' or 'flops'"
    )

    checkpoint_dir = auto_download_checkpoint(
        model_name=checkpoint_dir, access_token=access_token
    )

    if data is None:
        # import sys
        # sys.path.append('../do-not-touch/compressing_llms')
        # from datasets_custom.llamamini import LLaMaMini
        # data = LLaMaMini() if fine_tuned else TinyStories()
        data = (
            Alpaca(num_workers=num_workers)
            if fine_tuned
            else TinyStories(num_workers=num_workers)
        )

    num_devices = int(parse_devices(devices))
    out_dir = init_out_dir(out_dir)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    config.fix_head_size = True

    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"search-{config.name}",
        resume=bool(resume),
        log_interval=search.log_interval,
    )

    if num_devices * num_nodes > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=num_devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        loggers=logger,
    )

    if torch.cuda.is_available() and num_devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch(
        main,
        num_devices,
        resume,
        seed,
        config,
        data,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        search,
        param_bins if search.search_strategy == Methods.SRS else None,
        performance_metric,
        efficiency_metric,
        log_objective_names,
        save_checkpoints,
        fine_tuned,
        copy_config_files,
        verbose,
    )


# FSDP has issues with `inference_mode`
@torch.no_grad()
def _objective(
    config: dict,
    fabric: L.Fabric,
    model: GPT,
    val_dataloader: DataLoader,
    eval: EvalArgs,
    verbose: bool | None = True,
    objective_1: str = "val_loss",
    objective_2: str = "parameters",
    fine_tuned: bool = False,
) -> tuple[float, float]:
    model.select_sub_network(config)

    if fine_tuned:
        val_loss = finetune_validate(fabric, model, val_dataloader, eval, verbose=verbose)
    else:
        val_loss = validate(
            fabric, model, val_dataloader, max_iters=eval.max_iters, verbose=verbose
        )

    if objective_1 == "perplexity":
        val_loss = torch.exp(val_loss)

    if objective_2 == "parameters":
        obj_2 = compute_parameters(model)
    elif objective_2 == "latency":
        obj_2 = compute_latency(model, device=model.lm_head.weight.device)
    elif objective_2 == "flops":
        obj_2 = compute_flops(model, previous_device=model.lm_head.weight.device)
    else:
        raise ValueError(f"Invalid objective_2: {objective_2}")

    return float(val_loss), obj_2


def main(
    fabric: L.Fabric,
    devices: int,
    resume: bool | Literal["auto"] | Path,
    seed: int,
    config: Config,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    search: SearchArgs,
    param_bins: ParamBinArgs | None = None,
    performance_metric: str = "val_loss",
    efficiency_metric: str = "parameters",
    log_objective_names: bool = True,
    save_checkpoints: bool = True,
    fine_tuned: bool = False,
    copy_config_files: bool = True,
    verbose: bool = True,
) -> None:
    fabric.seed_everything(seed)

    tokenizer = Tokenizer(checkpoint_dir)

    train_dataloader, val_dataloader = get_dataloaders(
        fabric, data, tokenizer, train, train.max_seq_length
    )

    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)

    model = fabric.setup(model)

    state = {"model": model, "iter_num": 0, "step_count": 0}

    resume = find_resume_path(resume, out_dir)
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)
    else:
        load_checkpoint(fabric, model, checkpoint_path)

    train_time = time.perf_counter()

    longest_seq_length = len(val_dataloader.dataset)
    model.max_seq_length = min(longest_seq_length, train.max_seq_length or int("inf"))

    search_space = get_search_space(config)

    fabric.print("Start multi-objective search")

    bins = None
    if param_bins is not None:
        from whittle.sampling.random_sampler import RandomSampler

        sampler = RandomSampler(search_space, seed=seed)

        # get bins limited by the smallest/largest config
        params_estimator = ParamsEstimator(model)
        bins = ParamBins(
            sampler.get_smallest_sub_network(),
            sampler.get_largest_sub_network(),
            params_estimator,
            num_bins=param_bins.num_bins,
            log_bins=param_bins.log_bins,
            start_bin_size=param_bins.start_bin_size,
        )

    # fabric.is_global_zero
    search_results = multi_objective_search(
        _objective,
        search_space,
        objective_kwargs={
            "fabric": fabric,
            "model": model,
            "val_dataloader": val_dataloader,
            "eval": eval,
            "objective_1": performance_metric,
            "objective_2": efficiency_metric,
            "fine_tuned": fine_tuned,
        },
        search_strategy=search.search_strategy,
        num_samples=search.iterations,
        seed=seed,
        logger=fabric.logger,
        param_bins=bins,
        objective_1_name=performance_metric if log_objective_names else "objective_1",
        objective_2_name=efficiency_metric if log_objective_names else "objective_2",
        verbose=verbose and fabric.is_global_zero,
    )
    training_time = time.perf_counter() - train_time

    fabric.print(f"Total search time: {training_time:.02f}.")
    fabric.print(
        f"Found {len(search_results['configs'])} sub-networks ({sum(i for i in search_results['is_pareto_optimal'])} Pareto optimal). Save checkpoints to {out_dir}."
    )

    pareto_optimal_paths = []
    for i, sub_network_dict in enumerate(search_results["configs"]):
        save_path = out_dir / f"sub_network_{i}" / "lit_model.pth"

        save_sub_network(
            model,
            checkpoint_dir,
            save_path.parent,
            sub_network_config=sub_network_dict,
            save_checkpoints=save_checkpoints,
            copy_config_files=copy_config_files,
            fabric=fabric,
        )

        if search_results["is_pareto_optimal"][i]:
            pareto_optimal_paths.append(str(save_path.absolute()))

    # save all paths to pareto optimal sub-networks
    with open(out_dir / "pareto_optimal_paths.json", "w") as f:
        json.dump(pareto_optimal_paths, f)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
