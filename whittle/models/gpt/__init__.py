# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from __future__ import annotations

import logging
import re

from lightning_utilities.core.imports import RequirementCache
from litgpt import Config

from whittle.models.gpt.blocks import Block
from whittle.models.gpt.model import GPT

_LIGHTNING_AVAILABLE = RequirementCache("lightning>=2.2.0.dev0")
if not bool(_LIGHTNING_AVAILABLE):
    raise ImportError(
        f"whittle requires lightning>=2.2.0. Please run:\n pip install -U lightning\n{_LIGHTNING_AVAILABLE}"
    )

# Suppress excessive warnings, see https://github.com/pytorch/pytorch/issues/111632
_pattern = re.compile(".*Profiler function .* will be ignored")
logging.getLogger("torch._dynamo.variables.torch").addFilter(
    lambda record: not _pattern.search(record.getMessage())
)

# Avoid printing state-dict profiling output at the WARNING level when saving a checkpoint
logging.getLogger("torch.distributed.fsdp._optim_utils").disabled = True
logging.getLogger("torch.distributed.fsdp._debug_utils").disabled = True

__all__ = ["GPT", "Config", "Block"]
