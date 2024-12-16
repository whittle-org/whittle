# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from typing import Optional

import pytest
import torch
from lightning.fabric.utilities.testing import _runif_reasons
from lightning_utilities.core.imports import RequirementCache


class MockTokenizer:
    """A dummy tokenizer that encodes each character as its ASCII code."""

    bos_id = 0
    eos_id = 1

    def encode(
        self,
        text: str,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        output = []
        if bos:
            output.append(self.bos_id)
        output.extend([ord(c) for c in text])
        if eos:
            output.append(self.eos_id)
        output = output[:max_length] if max_length > 0 else output
        return torch.tensor(output)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join(chr(int(t)) for t in tokens.tolist())

    def __call__(self, text: str, return_tensors=None, **kwargs):
        class TokenizerOutput:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        encoded = self.encode(text, **kwargs)
        if return_tensors == "pt":
            return TokenizerOutput(encoded.unsqueeze(0))
        return TokenizerOutput(torch.tensor(encoded))


@pytest.fixture()
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture()
def compute_sparsity_ratio():
    def _compute(layer):
        W = layer.weight.data
        total_elements = W.numel()
        zero_elements = torch.sum(W == 0).item()
        sparsity_ratio = zero_elements / total_elements
        return sparsity_ratio

    return _compute


def RunIf(thunder: Optional[bool] = None, **kwargs):
    reasons, marker_kwargs = _runif_reasons(**kwargs)

    if thunder is not None:
        thunder_available = bool(RequirementCache("lightning-thunder", "thunder"))
        if thunder and not thunder_available:
            reasons.append("Thunder")
        elif not thunder and thunder_available:
            reasons.append("not Thunder")

    return pytest.mark.skipif(
        condition=len(reasons) > 0,
        reason=f"Requires: [{' + '.join(reasons)}]",
        **marker_kwargs,
    )
