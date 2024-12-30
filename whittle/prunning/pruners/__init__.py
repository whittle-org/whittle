from __future__ import annotations

from .magnitude import MagnitudePruner
from .sparsegpt import SparseGptPruner
from .wanda import WandaPruner


__all__ = [
    "MagnitudePruner",
    "SparseGptPruner",
    "WandaPruner",
]
