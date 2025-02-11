import json
import torch
from pathlib import Path
from typing import Optional, Union
from litgpt import Config
from whittle.models.gpt import GPT
from whittle.metrics import compute_parameters, compute_flops, compute_latency
from whittle.eval.utils import convert_and_evaluate


def setup(
    checkpoint_dir: Path,
    out_dir: Optional[Path] = None,
    tasks: Optional[str] = None,
    seed: int = 1337,
    num_fewshot: Optional[int] = None,
    batch_size: Union[int, str] = 1,
    latency_batch_size: int = 8,
    device: Optional[str] = None,
    limit: Optional[float] = None,
    tokenizer_name_or_path: Optional[str] = None,
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
    """
    if out_dir is None:
        out_dir = checkpoint_dir / "eval"

    metrics_path = out_dir / "metrics.json"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    config.fix_head_size = True
    config.model_type = "gpt"
    config.tie_embeddings = False

    model = GPT(config)
    model.name_or_path = tokenizer_name_or_path  # WhittleLM loads AutoTokenizer inside

    metrics = {}
    metrics["parameters"] = compute_parameters(model)
    metrics["flops"] = compute_flops(model)
    metrics["latency"] = compute_latency(
        model, batch_size=latency_batch_size, device=device
    )

    metrics_path.write_text(json.dumps(metrics, indent=2))

    model.to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "lit_model.pth")["model"])

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
