from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchArgs:
    """search-related arguments"""

    save_interval: Optional[int] = 1000
    """Number of optimizer steps between saving checkpoints"""
    log_interval: int = 1
    """Number of iterations between logging calls"""
    search_strategy: str = "random_search"
    """Multi-objective search strategy"""
    iterations: int = 100
    """Number of iterations for the multi-objective search"""


@dataclass
class PruningArgs:
    """pruning-related arguments"""

    pruning_strategy: str = "mag"
    """Structural pruning strategy"""
    n: int = 2
    """Number of weights to prune per group"""
    m: int = 4
    """Total number of weights per group"""
    nsamples: int = 32
    """Number of samples for calibration"""
