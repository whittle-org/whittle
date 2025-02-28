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
        alpha: Weight balancing distillation loss vs cross-entropy loss. Values closer to 1 give more importance
                to matching teacher logits, while values closer to 0 prioritize true label prediction.
    """

    method: str = "logits"
    temperature: float = 5
    alpha: float = 0.5


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
