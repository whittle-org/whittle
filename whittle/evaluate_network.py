from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings

import torch
from litgpt import Config

from whittle.eval.utils import convert_and_evaluate
from whittle.metrics import compute_latency, compute_parameters
from whittle.models.gpt import GPT
try:
    from whittle.metrics import compute_flops
except ImportError:
    warnings.warn(
        "DeepSpeed not installed. Please install whittle with `pip install whittle[distributed]` "
        "to use DeepSpeed for distributed training and measuring FLOPs.", ImportError
    )


def setup(
    checkpoint_dir: Path,
    out_dir: Path | None = None,
    tasks: str | None = None,
    seed: int = 1337,
    num_fewshot: int | None = None,
    batch_size: int | str = 1,
    latency_batch_size: int = 8,
    device: str | None = None,
    limit: float | None = None,
    tokenizer_name_or_path: str | None = None,
    is_sub_network: bool = False,
    measure_flops: bool = False,
    measure_latency: bool = False,
) -> None:
    """
    Evaluate a model with the LM Evaluation Harness. Compute the latency of a PyTorch model for inference, and FLOPs.

    Arguments:
        checkpoint_dir: The path to the model's checkpoint directory to load for evaluation.
        out_dir: Directory in which to save evaluation results. If not provided, saving to `checkpoint_dir/eval` by default.
        tasks: Task names to evaluate. Example: "hellaswag,mmlu"
        seed: The random seed to use for reproducibility.
        num_fewshot: Number of examples in few-shot context.
        batch_size: Batch size configuration as positive integer value (default: 1),
            "auto", in the format 'auto:N', where 'auto:4' recomputes the batch size 4 times.
        latency_batch_size: Batch size for latency computation.
        device: Device to use for evaluation, for example, "cuda" or "cuda:0".
        limit: Limit on number of examples per task.
        tokenizer_name_or_path: Name or path to the tokenizer file to use for the model. Default is checkpoint_dir.
        is_sub_network: Whether the model is a sub-network config or a whittle model. Default is False.
        measure_flops: Whether to compute FLOPs. Default is False.
        measure_latency: Whether to compute latency. Default is False.
    """
    if out_dir is None:
        out_dir = checkpoint_dir / "eval"

    metrics_path = out_dir / "metrics.json"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # sub-network saved as a config instead of the extracted lit_model.pth (to save memory)
    subnet_config: dict[str, Any] | None = None
    if is_sub_network:
        ckp = torch.load(checkpoint_dir / "sub_network.pkl")
        subnet_config = ckp["config"]
        checkpoint_dir = ckp["parent_dir"]

    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    config.fix_head_size = True
    config.model_type = "gpt"
    config.tie_embeddings = False

    model = GPT(config)
    model.name_or_path = tokenizer_name_or_path  # WhittleLM loads AutoTokenizer inside

    metrics = {}
    metrics["parameters"] = compute_parameters(model)

    if measure_flops:
        metrics["flops"] = compute_latency(model)
    if measure_latency:
        metrics["latency"] = compute_flops(
            model, batch_size=latency_batch_size, previous_device=device
        )

    metrics_path.write_text(json.dumps(metrics, indent=2))

    model.to(device)

    ckp = torch.load(checkpoint_dir / "lit_model.pth", weights_only=False)
    model.load_state_dict(ckp["model"] if "model" in ckp else ckp)
    del ckp

    if is_sub_network:
        assert subnet_config is not None
        model.select_sub_network(subnet_config)

    # import pdb; pdb.set_trace()
    convert_and_evaluate(
        model=model,
        out_dir=out_dir,
        tasks=tasks,
        seed=seed,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
