"""Learning rate schedules for the training scripts."""

from __future__ import annotations

import math


def get_wsd_lr(
    learning_rate: float,
    it: int,
    warmup_iters: int,
    max_iters: int,
    min_lr: float,
    stable_ratio: float = 0.85,
    decay_type: str = "linear",
) -> float:
    """Returns the learning rate of a Warmup-Stable-Decay (WSD) schedule.

    It is a drop-in replacement for the cosine `get_lr` of litgpt. The phases follow
    from `warmup_iters` and `max_iters`:

    - `[0, warmup_iters)`: linear warmup from 0 to `learning_rate`.
    - `[warmup_iters, stable_end)`: constant `learning_rate`.
    - `[stable_end, max_iters)`: decay from `learning_rate` to `min_lr`.
    - `[max_iters, ...)`: constant `min_lr`.

    References: Zhou et al. (2026), "How to Set the Batch Size", arXiv:2601.05034 (a
    1000-step warmup for Qwen3); the Qwen3 Technical Report, arXiv:2505.09388 (WSD with
    a linear decay to 10% of the peak learning rate); Hu et al. (2024), MiniCPM,
    arXiv:2404.06395 (the original WSD schedule).

    Args:
        learning_rate: The peak learning rate (`optimizer.defaults["lr"]`).
        it: The current iteration.
        warmup_iters: The number of warmup iterations.
        max_iters: The total number of iterations (warmup, stable, and decay).
        min_lr: The final learning rate, typically 0.1 times `learning_rate`.
        stable_ratio: The fraction of `max_iters - warmup_iters` at the peak learning
            rate before the decay starts. The default 0.85 means 85% stable and 15%
            decay.
        decay_type: `"linear"` (the Qwen3 default), `"cosine"` (smooth), or
            `"exponential"` (a steep tail).

    Returns:
        The learning rate for iteration `it`.

    Raises:
        ValueError: If `decay_type` is not one of the three types.
    """
    post_warmup = max_iters - warmup_iters
    stable_end = warmup_iters + int(stable_ratio * post_warmup)

    # 1) linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    # 2) past max_iters → floor
    if it >= max_iters:
        return min_lr

    # 3) stable plateau
    if it < stable_end:
        return learning_rate

    # 4) decay  learning_rate → min_lr
    decay_iters = max_iters - stable_end
    progress = (it - stable_end) / decay_iters  # 0.0 → 1.0

    if decay_type == "linear":
        # Qwen3 default: linear anneal
        coeff = 1.0 - progress
    elif decay_type == "cosine":
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif decay_type == "exponential":
        r = min_lr / learning_rate if learning_rate > 0 else 1.0
        return learning_rate * (r**progress)
    else:
        raise ValueError(f"Unknown decay_type '{decay_type}'")

    return min_lr + coeff * (learning_rate - min_lr)
