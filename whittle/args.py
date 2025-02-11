from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from litgpt.args import TrainArgs


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
class FineTuningArgs(TrainArgs):
    learning_rate: Optional[float] = 2e-5
    temperature: Optional[float] = 10.0
    distillation_weight: Optional[float] = 0.5
