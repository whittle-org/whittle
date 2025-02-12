from __future__ import annotations

from dataclasses import dataclass

from litgpt.args import TrainArgs


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
class FineTuningArgs(TrainArgs):
    learning_rate: float | None = 2e-5
    temperature: float | None = 10.0
    distillation_weight: float | None = 0.5
