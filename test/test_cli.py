from __future__ import annotations

import logging
import sys
from functools import wraps
from unittest.mock import patch

import pytest
from litgpt.chat.base import main as chat_fn
from litgpt.deploy.serve import run_server as serve_fn
from litgpt.scripts.download import download_from_hub as download_fn

from whittle.__main__ import main
from whittle.evaluate_network import setup as evaluate_fn
from whittle.pretrain_super_network import setup as pretrain_fn
from whittle.prune import setup as prune_fn
from whittle.search_sub_networks import setup as search_fn

logging.basicConfig(level=logging.DEBUG)


def create_mock_fn(original_fn):
    """Create a mock function with the same signature and type annotations as the original function."""

    @wraps(
        original_fn
    )  # This preserves the original function's metadata, including annotations
    def mock_fn(*args, **kwargs):
        pass

    return mock_fn


# Create mock functions
mock_pretrain_fn = create_mock_fn(pretrain_fn)
mock_search_fn = create_mock_fn(search_fn)
mock_evaluate_fn = create_mock_fn(evaluate_fn)
mock_prune_fn = create_mock_fn(prune_fn)
mock_download_fn = create_mock_fn(download_fn)
mock_serve_fn = create_mock_fn(serve_fn)
mock_chat_fn = create_mock_fn(chat_fn)


def test_cli_parser_data():
    """Test that the CLI parser data contains all expected commands."""
    with (
        patch("whittle.__main__.pretrain_fn", mock_pretrain_fn),
        patch("whittle.__main__.search_fn", mock_search_fn),
        patch("whittle.__main__.evaluate_fn", mock_evaluate_fn),
        patch("whittle.__main__.prune_fn", mock_prune_fn),
        patch("whittle.__main__.download_fn", mock_download_fn),
        patch("whittle.__main__.serve_fn", mock_serve_fn),
        patch("whittle.__main__.chat_fn", mock_chat_fn),
    ):
        # Call main with --help to trigger CLI initialization
        with pytest.raises(SystemExit) as exc_info:
            sys.argv = ["whittle", "--help"]
            main()

        # Check that exit was clean (0)
        assert exc_info.value.code == 0


@pytest.mark.parametrize(
    "command", ["pretrain", "search", "evaluate", "prune", "download", "serve", "chat"]
)
def test_cli_commands(command):
    """Test that each CLI command is properly registered and can be called with --help."""

    with (
        patch("whittle.__main__.pretrain_fn", mock_pretrain_fn),
        patch("whittle.__main__.search_fn", mock_search_fn),
        patch("whittle.__main__.evaluate_fn", mock_evaluate_fn),
        patch("whittle.__main__.prune_fn", mock_prune_fn),
        patch("whittle.__main__.download_fn", mock_download_fn),
        patch("whittle.__main__.serve_fn", mock_serve_fn),
        patch("whittle.__main__.chat_fn", mock_chat_fn),
    ):
        # Call main with command --help to test command registration
        with pytest.raises(SystemExit) as exc_info:
            sys.argv = ["whittle", command, "--help"]
            main()

        # Check that exit was clean (0)
        assert exc_info.value.code == 0


def test_torch_precision_setting():
    """Test that torch float32 matmul precision is set to high."""
    with patch("torch.set_float32_matmul_precision") as mock_precision:
        with pytest.raises(SystemExit):
            sys.argv = ["whittle", "--help"]
            main()

        mock_precision.assert_called_once_with("high")


def test_warning_filter():
    """Test that the scheduler warning is properly filtered."""
    with patch("warnings.filterwarnings") as mock_filter:
        with pytest.raises(SystemExit):
            sys.argv = ["whittle", "--help"]
            main()

        mock_filter.assert_called_once_with(
            action="ignore",
            message=r"The epoch parameter in `scheduler.step\(\)` was not necessary and is being deprecated.*",
            category=UserWarning,
            module=r".*torch\.optim\.lr_scheduler.*",
        )
