from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchArgs:
    """search-related arguments"""

    save_interval: int | None = 1000
    """Number of optimizer steps between saving checkpoints"""
    log_interval: int = 1
    """Number of iterations between logging calls"""
    search_strategy: str = "random_search"
    """Multi-objective search strategy"""
    iterations: int = 100
    """Number of iterations for the multi-objective search"""


@dataclass
class DistillArgs:
    """Distillation-related arguments

    Args:
        method: Distillation method to use ('logits' or 'hidden_states') - Only supports 'logits' for now
        temperature: Controls softening of output probabilities. Higher values (>1) produce softer distributions,
                emphasizing less confident predictions. Lower values (<1) make distributions sharper, focusing on
                confident predictions.
        alpha: Weight for the cross-entropy loss. Higher values give more importance to the loss between student logits and ground truth labels.
        beta: Weight for the distillation loss. Higher values give more importance to the loss between student and teacher logits.
        loss: Loss function to use for distillation. Options are 'forward_kld', 'reverse_kld', 'symmetric_kld', 'js_distance', 'simple_cross_entropy', 'cosine_similarity', 'l1_loss', 'l2_loss', 'mmd_loss'.
        weight_scheme: Weight scheme to use for the distillation loss. Options are 'default' (use alpha=1 and beta=hard_target_loss/soft_target_loss).
    """

    method: str = "logits"
    alpha: float = 0.5
    beta: float = 0.5
    temperature: float = 5
    loss: str = "forward_kld"
    weight_scheme: str = "default"


@dataclass
class PruningArgs:
    """pruning-related arguments"""

    pruning_strategy: str = "mag"
    """Structural pruning strategy"""
    prune_n_weights_per_group: int = 2
    """Number of weights to prune per group"""
    weights_per_group: int = 4
    """Total number of weights per group"""
    n_samples: int = 32
    """Number of samples for calibration"""


class ParamBinArgs:
    """parameter bin-related arguments - to limit what networks are sampled"""

    """Number of parameter bins to use"""
    num_bins: int = 20
    """Whether to use log spaced bins"""
    log_bins: bool = False
    """Starting size of the bins (how many configs must be in each bin until the total limit is increased)"""
    start_bin_size: int = 1
    """The total limit will be increased even if K bins are not full yet (some param counts may have only few nets)"""
    empty_bin_tolerance: int = 4
