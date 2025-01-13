from __future__ import annotations

from .magnitude import MagnitudePruner
from .sparsegpt import SparseGPTPruner
from .wanda import WandaPruner

__all__ = [
    "MagnitudePruner",
    "SparseGPTPruner",
    "WandaPruner",
]
