"""Whittle: compressing large language models by extracting sub-networks.

The central object is `GPT`, which serves as both the super-network and any
sub-network carved out of it. Calling `GPT.set_sub_network` reshapes the forward
pass without re-allocating weights; `extract_current_sub_network` materialises
the currently active sub-network as a stand-alone model.
"""

from __future__ import annotations

from whittle.__version__ import __version__
from whittle.models.gpt import GPT
from whittle.models.gpt.checkpoint import load_checkpoint, save_sub_network
from whittle.models.gpt.extract import extract_current_sub_network, extract_sub_network

__all__ = [
    "GPT",
    "extract_current_sub_network",
    "extract_sub_network",
    "load_checkpoint",
    "save_sub_network",
    "__version__",
]
