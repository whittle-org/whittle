# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from __future__ import annotations

import warnings

import torch
from jsonargparse import CLI, set_config_read_mode, set_docstring_parse_options
from litgpt.chat.base import main as chat_fn
from litgpt.deploy.serve import run_server as serve_fn
from litgpt.scripts.download import download_from_hub as download_fn

from whittle.eval.utils import convert_and_evaluate as evaluate_fn
from whittle.finetune import setup as finetune_fn
from whittle.pretrain_super_network import setup as pretrain_fn
from whittle.search_sub_networks import setup as search_fn


def main() -> None:
    parser_data = {
        "pretrain": pretrain_fn,
        "search": search_fn,
        "evaluate": evaluate_fn,
        "finetune": finetune_fn,
        "download": download_fn,
        "serve": serve_fn,
        "chat": chat_fn,
    }

    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)

    # PyTorch bug that raises a false-positive warning
    scheduler_warning = r"The epoch parameter in `scheduler.step\(\)` was not necessary and is being deprecated.*"

    warnings.filterwarnings(
        action="ignore",
        message=scheduler_warning,
        category=UserWarning,
        module=r".*torch\.optim\.lr_scheduler.*",
    )

    torch.set_float32_matmul_precision("high")
    CLI(parser_data)


if __name__ == "__main__":
    main()
