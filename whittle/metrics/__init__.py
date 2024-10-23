from .flops import compute_flops
from .latency import compute_latency
from .mag import compute_weight_magnitude
from .parameters import compute_parameters, compute_all_parameters


__all__ = [
    "compute_flops",
    "compute_latency",
    "compute_parameters",
    "compute_all_parameters",
    "compute_weight_magnitude",
]
