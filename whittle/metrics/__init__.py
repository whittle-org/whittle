from __future__ import annotations

import warnings

try:
    from .flops import compute_flops
except ImportError:
    warnings.warn(
        "DeepSpeed is not installed. If you would like to use DeepSpeed for distributed "
        "training and measuring flops, please install whittle with "
        "`pip install whittle[distributed]`",
        ImportWarning,
    )
from .latency import compute_latency
from .mag import compute_weight_magnitude
from .parameters import compute_all_parameters, compute_parameters

__all__ = [
    "compute_flops",
    "compute_latency",
    "compute_parameters",
    "compute_all_parameters",
    "compute_weight_magnitude",
]
