# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import dataclasses
import math
import os
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Literal, Optional, Tuple, Union
import warnings
from syne_tune.config_space import randint, choice
import lightning as L
import torch
import pickle
from sampler import (
    RandomSampler,
    FixGridSampler,
    FixParamGridSampler,
    CalibFixGridSampler,
    ImportanceSampler,
    ImportanceParamGridSampler,
    ImportanceCalibFixGridSampler,
)
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics import RunningMean
from whittle.training_strategies.sandwich import SandwichStrategy
from whittle.training_strategies.base_strategy import BaseTrainingStrategy
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import Alpaca, DataModule
from litgpt.generate.base import generate

from litgpt.lora import Config, lora_filter, mark_only_lora_as_trainable
from litgpt.prompts import save_prompt_style
from distillation_loss import KDLoss
from sandwich import SandwichStrategy
from standard import StandardStrategy
from train_args import FineTuningArgs
from search_spaces import search_spaces

# from litgpt.scripts.merge_lora import merge_lora
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    auto_download_checkpoint,
    check_nvlink_connectivity,
    CycleIterator,
    check_valid_checkpoint_dir,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    get_default_supported_precision,
    load_checkpoint,
    init_out_dir,
    instantiate_torch_optimizer,
    instantiate_bnb_optimizer,
    num_parameters,
    parse_devices,
    save_hyperparameters,
)
from whittle.models.gpt import GPT
from lora_whittle.lora_block import Block
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""This script merges the LoRA weights with the base model"""
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
import yaml
from litgpt.lora import Config, lora_filter, merge_lora_weights
from litgpt.utils import check_valid_checkpoint_dir, extend_checkpoint_dir

from utils_eval import (
    plot_validation_metrics,
    plot_accuracies,
    evaluate_grid,
    evaluate_grid_lambada,
)


def merge_lora(
    sampling_strategy: str,
    importance_objective: str,
    checkpoint_dir: Path,
    pretrained_checkpoint_dir: Optional[Path] = None,
    precision: Optional[str] = None,
) -> None:
    """Merges the LoRA weights with the base model.

    See ``litgpt finetune lora``.

    Creates a new ``lit_model.pth`` file by merging the LoRA weights (``lit_model.pth.lora``)
    with the original checkpoint weights.

    Arguments:
        checkpoint_dir: Path to the checkpoint directory with trained LoRA weights, which is the output of
            ``litgpt finetune lora``.
        pretrained_checkpoint_dir: Optional path to the checkpoint directory with the weights of the base model
            corresponding to the LoRA checkpoint. By default, this will automatically be inferred from the metadata
            in the given `checkpoint_dir` directory. Only set this if the base model's checkpoint directory
            has moved or was renamed.
        precision: Optional precision setting to instantiate the model weights in. By default, this will
            automatically be inferred from the metadata in the given ``checkpoint_dir`` directory.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    if pretrained_checkpoint_dir is not None:
        pretrained_checkpoint_dir = extend_checkpoint_dir(pretrained_checkpoint_dir)
    pprint(locals())

    check_valid_checkpoint_dir(checkpoint_dir, model_filename="lit_model.pth.lora")
    if pretrained_checkpoint_dir is not None:
        check_valid_checkpoint_dir(pretrained_checkpoint_dir)
    if (checkpoint_dir / "lit_model.pth").is_file():
        print("LoRA weights have already been merged in this checkpoint.")
        return

    lora_params, meta_pretrained_checkpoint_dir, lora_precision = load_lora_metadata(
        checkpoint_dir
    )
    precision = precision if precision is not None else lora_precision

    if pretrained_checkpoint_dir is None:
        pretrained_checkpoint_dir = meta_pretrained_checkpoint_dir
        pretrained_checkpoint_dir = extend_checkpoint_dir(pretrained_checkpoint_dir)

    fabric = L.Fabric(devices=1, precision=precision, accelerator="cpu")
    config = Config.from_file(checkpoint_dir / "model_config.yaml", **lora_params)
    config.fix_head_size = True
    with fabric.init_module(), torch.device("meta"):
        model = GPT(config)
        # we don't care about these to perform merging
        model.cos = None
        model.sin = None

    lora_path = checkpoint_dir / "lit_model.pth.lora"
    if "importance" in sampling_strategy:
        pretrained_checkpoint = torch.load(
            str(
                pretrained_checkpoint_dir
                / f"lit_model_permuted_{importance_objective}.pth"
            ),
            mmap=True,
        )
    else:
        pretrained_checkpoint = torch.load(
            str(pretrained_checkpoint_dir / "lit_model.pth"), mmap=True
        )
    lora_checkpoint = torch.load(str(lora_path), mmap=True)
    lora_checkpoint = lora_checkpoint.get("model", lora_checkpoint)

    # Merge LoRA weights into the base model
    pretrained_checkpoint.update(lora_checkpoint)
    model.load_state_dict(pretrained_checkpoint, assign=True)
    # since LoRA finetuning only saves the LoRA weights, we treat the lora weights dtype as the expected dtype
    lora_dtype = next(iter(lora_checkpoint.values())).dtype
    model.to(dtype=lora_dtype, device="cpu")
    merge_lora_weights(model)

    # Remove LoRA parameters and the LoRA linear substring
    state_dict = {
        k.replace("linear.", ""): v
        for k, v in model.state_dict().items()
        if not lora_filter(k, v)
    }
    save_path = checkpoint_dir / "lit_model.pth"
    torch.save(state_dict, save_path)

    fabric.print(f"Saved merged weights to {str(checkpoint_dir / 'lit_model.pth')!r}")


def load_lora_metadata(
    checkpoint_dir: Path,
) -> Tuple[Dict[str, Any], Path, Optional[str]]:
    hparams_file = checkpoint_dir / "hyperparameters.yaml"
    if not hparams_file.is_file():
        raise FileNotFoundError(
            f"The path {str(hparams_file)!r} is not a valid checkpoint directory. It is missing a"
            f" `hyperparameters.yaml` file. Please point to the checkpoint directory that was produced by"
            f" the `litgpt/finetune/lora.py` script."
        )

    with open(hparams_file, "r", encoding="utf-8") as file:
        hparams = yaml.safe_load(file)

    lora_params = {k: v for k, v in hparams.items() if k.startswith("lora_")}
    pretrained_checkpoint_dir = Path(hparams["checkpoint_dir"])
    precision = hparams.get("precision")
    return lora_params, pretrained_checkpoint_dir, precision


def find_resume_path(
    resume: Union[bool, Literal["auto"], Path], out_dir: Path
) -> Optional[Path]:
    resume_path = out_dir / "final" / "lit_model.pth.lora"
    if not resume_path.exists():
        return False
    return resume_path


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path("out/finetune/lora"),
    precision: Optional[str] = None,
    quantize: Optional[
        Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]
    ] = None,
    devices: Union[int, str] = 1,
    num_nodes: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_query: bool = True,
    lora_key: bool = True,
    lora_value: bool = True,
    lora_projection: bool = True,
    lora_mlp: bool = True,
    lora_head: bool = True,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=100,
        log_interval=1,
        global_batch_size=64,
        micro_batch_size=1,
        lr_warmup_steps=100,
        epochs=1,
        max_seq_length=None,
    ),
    train_strategy: str = "sandwich",
    search_space_type: str = "hw_gpt_bench",
    sampling_strategy: str = "random",
    eval: EvalArgs = EvalArgs(interval=10, max_new_tokens=100, max_iters=100),
    optimizer: Union[str, Dict] = "AdamW",
    logger_name: Literal["wandb", "tensorboard", "csv"] = "csv",
    seed: int = 1337,
    access_token: Optional[str] = None,
    n_trials: int = 10000,
    downstream_test_iters: int = 500,
    downstream_dataset: str = "arc_easy",
    importance_objective: str = "norm",
    objective: str = "ppl",
    resume: Union[bool, Literal["auto"], Path] = True,
    model_path: str = "",
) -> None:
    """Finetune a model using the LoRA method.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        quantize: If set, quantize the model with this algorithm. See ``tutorials/quantize.md`` for more information.
        devices: How many devices/GPUs to use.
        num_nodes: How many nodes the code is being run on.
        lora_r: The LoRA rank.
        lora_alpha: The LoRA alpha.
        lora_dropout: The LoRA dropout value.
        lora_query: Whether to apply LoRA to the query weights in attention.
        lora_key: Whether to apply LoRA to the key weights in attention.
        lora_value: Whether to apply LoRA to the value weights in attention.
        lora_projection: Whether to apply LoRA to the output projection in the attention block.
        lora_mlp: Whether to apply LoRA to the weights of the MLP in the attention block.
        lora_head: Whether to apply LoRA to output head in GPT.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.Alpaca``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
    """
    checkpoint_dir = auto_download_checkpoint(
        model_name=checkpoint_dir, access_token=access_token
    )
    pprint(locals())
    data = Alpaca() if data is None else data
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(
        checkpoint_dir / "model_config.yaml",
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_query=True,
        lora_key=True,
        lora_value=True,
        lora_projection=True,
        lora_mlp=True,
        lora_head=True,
    )
    config.fix_head_size = True
    config.tie_embeddings = False
    config.model_type = "gpt"
    search_space = search_spaces[search_space_type](config)
    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"finetune-{config.name}",
        log_interval=train.log_interval,
    )

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        if RequirementCache("bitsandbytes != 0.42.0"):
            warnings.warn(
                "LitGPT only supports bitsandbytes v0.42.0. "
                "This may result in errors when using quantization."
            )
        dtype = {
            "16-true": torch.float16,
            "bf16-true": torch.bfloat16,
            "32-true": torch.float32,
        }[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    if devices * num_nodes > 1:
        if quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 and num_nodes=1"
                " when using the --quantize flag."
            )
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
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        loggers=logger,
        plugins=plugins,
    )
    if sampling_strategy == "random":
        sampler = RandomSampler(search_space=search_space, seed=seed)
    elif sampling_strategy == "grid":
        sampler = FixGridSampler(search_space=search_space, seed=seed)
    elif sampling_strategy == "grid-params":
        sampler = FixParamGridSampler(
            search_space=search_space, seed=42, n_trials=n_trials
        )
    elif sampling_strategy == "calibrate":
        sampler = CalibFixGridSampler(
            checkpoint_dir=checkpoint_dir,
            search_space_type=search_space_type,
            search_space=search_space,
            seed=seed,
        )
    elif sampling_strategy == "importance-random":
        sampler = ImportanceSampler(
            os.path.join(checkpoint_dir, "sorted_ids.pkl"), search_space, seed=seed
        )
    elif sampling_strategy == "importance-grid-params":
        sampler = ImportanceParamGridSampler(
            sorted_ids_path=os.path.join(checkpoint_dir, "sorted_ids.pkl"),
            search_space=search_space,
            seed=42,
            n_trials=n_trials,
        )
    elif sampling_strategy == "importance-calibrate":
        sampler = ImportanceCalibFixGridSampler(
            objective=objective,
            importance_objective=importance_objective,
            sorted_ids_path=os.path.join(checkpoint_dir, "sorted_ids.pkl"),
            checkpoint_dir=checkpoint_dir,
            search_space_type=search_space_type,
            search_space=search_space,
            seed=seed,
        )
    if train_strategy == "sandwich":
        strategy = SandwichStrategy(
            loss_function=chunked_cross_entropy,
            lora=True,
            sampler=sampler,
        )
    elif train_strategy == "kl_sandwich":
        strategy = SandwichStrategy(
            loss_function=chunked_cross_entropy,
            kd_loss=KDLoss(temperature=10, distillation_weight=0.5),
            lora=True,
            sampler=sampler,
        )
    elif train_strategy == "standard":
        strategy = StandardStrategy(
            loss_function=chunked_cross_entropy,
            lora=True,
            sampler=sampler,
        )

    strategy.fabric = fabric
    strategy.gradient_accumulation_step = train.gradient_accumulation_iters(devices)
    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch(
        main,
        devices,
        seed,
        config,
        data,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        optimizer,
        sampling_strategy,
        strategy,
        downstream_dataset,
        downstream_test_iters,
        importance_objective,
        resume,
        model_path,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    config: Config,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: Union[str, Dict],
    sampling_strategy: str,
    strategy: BaseTrainingStrategy,
    downstream_dataset: str,
    downstream_test_iters: int,
    importance_objective: str,
    resume: bool,
    model_path: str,
) -> None:
    validate_args(train, eval)

    tokenizer = Tokenizer(checkpoint_dir)
    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train)
    steps_per_epoch = len(train_dataloader) // train.gradient_accumulation_iters(
        devices
    )
    lr_max_steps = min(
        train.epochs * steps_per_epoch, (train.max_steps or float("inf"))
    )

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    if "importance" in sampling_strategy:
        checkpoint_path = (
            checkpoint_dir / f"lit_model_permuted_{importance_objective}.pth"
        )
    else:
        checkpoint_path = checkpoint_dir / f"lit_model.pth"
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
        if "grid-params" in sampling_strategy:
            print("Initializing params grid....")
            strategy.sampler.initialize_grid(model)
            print("Requested configs:", len(strategy.sampler.values))
            print("Grid Size", len(strategy.sampler.grid))
    model.name_or_path = checkpoint_dir
    mark_only_lora_as_trainable(model)

    fabric.print(
        f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}"
    )
    fabric.print(
        f"Number of non-trainable parameters: {num_parameters(model, requires_grad=False):,}"
    )

    model = fabric.setup_module(model)

    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        optimizer = instantiate_bnb_optimizer(optimizer, model.parameters())
    else:
        optimizer = instantiate_torch_optimizer(optimizer, model.parameters())

    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(
        optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps
    )
    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "iter_num": 0,
        "step_count": 0,
    }
    # strict=False because missing keys due to LoRA weights not contained in state dict
    model_ckpt_path = f"{model_path}/final/lit_model.pth"
    load_checkpoint(fabric, model_ckpt_path, model_path, strict=True)
    # load_checkpoint(fabric, model, "/p/project/projectnucleus/sukthanker1/importance/llm_compression_experiments/llama2_standard/final/lit_model.pth.lora", strict=False)
    train_time = time.perf_counter()
    fit(
        fabric,
        state,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        devices,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        data,
        strategy,
        downstream_dataset,
        downstream_test_iters,
        sampling_strategy,
        resume,
        model_path,
    )
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def fit(
    fabric: L.Fabric,
    state: Dict,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    devices: int,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    data: DataModule,
    train_strategy: BaseTrainingStrategy,
    downstream_dataset: str,
    downstream_test_iters: int,
    sampling_strategy: str,
    resume: bool,
    model_path: str,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(
        ConcatDataset([train_dataloader.dataset, val_dataloader.dataset])
    )
    model.max_seq_length = (
        653  # min(longest_seq_length, train.max_seq_length or float("inf"))
    )
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )
    path = f"{model_path}/grid_eval_{downstream_dataset}.pkl"
    if downstream_dataset == "lambada":
        params, accs = evaluate_grid_lambada(
            path,
            model,
            train_strategy.sampler,
            downstream_dataset,
            sampling_strategy,
            out_dir,
        )
    else:
        params, accs = evaluate_grid(
            path,
            model,
            train_strategy.sampler,
            downstream_dataset,
            sampling_strategy,
            out_dir,
        )


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: GPT,
    val_dataloader: DataLoader,
    eval: EvalArgs,
    verbose: bool = True,
    sampler=None,
) -> torch.Tensor:
    if verbose:
        fabric.print("Validating ...")
    model.eval()
    val_loss_largest, val_loss_middle, val_loss_smallest = plot_validation_metrics(
        model, val_dataloader, eval, sampler
    )
    model.train()
    return val_loss_largest, val_loss_middle, val_loss_smallest


@torch.no_grad()
def test_downstream(
    fabric: L.Fabric,
    model: GPT,
    dataset: str = "arc_easy",
    sampler=None,
    checkpoint_dir=None,
) -> torch.Tensor:
    model.eval()
    acc_largest, acc_middle, acc_smallest = plot_accuracies(
        model, sampler, dataset, checkpoint_dir
    )
    model.train()
    return acc_largest, acc_middle, acc_smallest


@torch.no_grad()
def generate_example(
    fabric: L.Fabric, model: GPT, tokenizer: Tokenizer, eval: EvalArgs, data: DataModule
):
    instruction = (
        "Recommend a movie for me to watch during the weekend and explain the reason."
    )
    fabric.print(instruction)
    prompt = data.prompt_style.apply(instruction)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    model.eval()
    model.reset_super_network()
    max_returned_tokens = len(encoded) + eval.max_new_tokens

    if max_returned_tokens < model.max_seq_length:
        with fabric.init_tensor():
            # do not set `max_seq_length=max_returned_token` because memory is not a concern here
            model.set_kv_cache(batch_size=1)
        output = generate(
            model,
            encoded,
            max_returned_tokens=max_returned_tokens,
            temperature=0.8,
            eos_id=tokenizer.eos_id,
        )
        model.clear_kv_cache()
        model.train()
        output = tokenizer.decode(output)
        fabric.print(output)
    else:
        print(
            f"Length of encoded instruction ({len(encoded)}) and eval.max_new_tokens ({eval.max_new_tokens}) "
            f"exceeds model.max_seq_length ({model.max_seq_length}) used for training. Skipping example generation for efficiency. "
            f"The model's supported context size (post-training) is {model.config.block_size}."
        )


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / warmup_steps
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(max_steps - warmup_steps)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [scheduler1, scheduler2], milestones=[warmup_steps]
    )


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs
) -> Tuple[DataLoader, DataLoader]:
    data.connect(
        tokenizer=tokenizer,
        batch_size=train.micro_batch_size,
        max_seq_length=train.max_seq_length,
    )
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )
    return train_dataloader, val_dataloader


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def save_lora_checkpoint(
    fabric: L.Fabric, model: torch.nn.Module, file_path: Path
) -> None:
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})


def validate_args(train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [
        (train, ["max_tokens", "max_norm", "tie_embeddings", "lr_warmup_fraction"])
    ]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(
                    f"{__file__} doesn't support the {name!r} argument. This is set in {args}"
                )
    required = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(
                    f"{__file__} requires the {name!r} argument. This is set in {args}"
                )
    if not train.epochs and not train.max_steps:
        issues.append(
            f"{__file__} requires either epochs or max_steps to be set. This is set in {train}"
        )
    if issues:
        raise ValueError("\n".join(issues))


if __name__ == "__main__":
    from jsonargparse import CLI

    # setup("EleutherAI/pythia-1b", search_space_gpt)
    CLI(setup)
