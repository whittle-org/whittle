from __future__ import annotations

import json
from pathlib import Path

from litgpt import Config

from whittle.eval.utils import convert_and_evaluate
from whittle.metrics import compute_flops, compute_latency, compute_parameters
from whittle.models.gpt import GPT
from whittle.models.gpt.checkpoint import load_checkpoint


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
        measure_flops: Whether to compute FLOPs. Default is False.
        measure_latency: Whether to compute latency. Default is False.
    """
    if out_dir is None:
        out_dir = checkpoint_dir / "eval"

    metrics_path = out_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    config_attr = {
        "fix_head_size": True,
        "model_type": "gpt",
        "tie_embeddings": False,
    }
    model = load_checkpoint(
        checkpoint_dir, model_cls=GPT, config_cls=Config, config_attr=config_attr
    )

    # compute metrics
    metrics = {}
    metrics["parameters"] = compute_parameters(model)
    if measure_latency:
        metrics["latency"] = compute_latency(model)
    if measure_flops:
        metrics["flops"] = compute_flops(
            model, batch_size=latency_batch_size, previous_device=device
        )
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # downstream task evaluation
    model.to(device)
    convert_and_evaluate(
        model,
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
