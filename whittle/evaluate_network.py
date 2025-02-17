from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from litgpt import Config
from litgpt.utils import lazy_load

from whittle.eval.utils import convert_and_evaluate
from whittle.metrics import compute_flops, compute_latency, compute_parameters
from whittle.models.gpt import GPT


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

    # sub-network saved as a config instead of the extracted lit_model.pth (to save memory)
    sub_network_config: dict[str, Any] | None = None
    ckp = lazy_load(checkpoint_dir / "lit_model.pth")

    # sub-network config loading (contains the config and checkpoint path of the parent)
    sub_network_config = ckp.get("sub_network_config", None)
    parent_dir = ckp.get("parent_dir", None)
    # it's either a standalone litgpt model or a sub-network (depending on if there is also a parent_dir)
    if "model" not in ckp:
        # not None: sub-network, None: raw state dict
        if parent_dir is not None:
            checkpoint_dir = Path(parent_dir)
        ckp = lazy_load(checkpoint_dir / "lit_model.pth")

    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    config.fix_head_size = True
    config.model_type = "gpt"
    config.tie_embeddings = False

    model = GPT(config)
    # WhittleLM loads AutoTokenizer inside - either we copied it to checkpoint_dir, or it is referenced in parent_dir
    model.name_or_path = checkpoint_dir if parent_dir is None else parent_dir

    model.load_state_dict(ckp["model"] if "model" in ckp else ckp)
    del ckp

    # if the checkpoint was a sub-network, set it at this point
    if sub_network_config is not None:
        model.select_sub_network(sub_network_config)

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
