"""Save the arguments of a workflow with its checkpoints.

litgpt's `save_hyperparameters` parses `sys.argv` again to find the arguments. That
fails when a workflow's `setup()` is called from Python, because `sys.argv` then holds
the arguments of the calling script. The functions here use the arguments of the call
itself. The saved `hyperparameters.yaml` has the same format as before, so the workflow's
command line can still read it with `--config`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jsonargparse import ArgumentParser
from litgpt.data import DataModule


def _init_args(value: Any) -> dict[str, Any]:
    if not dataclasses.is_dataclass(value):
        return {}
    return {
        field.name: _to_plain(getattr(value, field.name))
        for field in dataclasses.fields(value)
        if field.init
    }


def _to_plain(value: Any) -> Any:
    """Converts an argument value to the plain data that jsonargparse can dump."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, DataModule):
        # argument dataclasses, such as `TrainArgs` and `Config`
        return _init_args(value)
    # subclass instances, such as data modules and prompt styles
    class_path = f"{type(value).__module__}.{type(value).__qualname__}"
    return {"class_path": class_path, "init_args": _init_args(value)}


def dump_hyperparameters(function: Callable, arguments: dict[str, Any]) -> str:
    """Returns the arguments of a call to `function` as YAML text.

    Args:
        function: The workflow function, for example `setup` of a workflow script.
        arguments: The arguments of the call, by name.

    Returns:
        The YAML text that `function`'s command line reads with `--config`.
    """
    parser = ArgumentParser(exit_on_error=False)
    parser.add_function_arguments(function)
    plain_arguments = {name: _to_plain(value) for name, value in arguments.items()}
    config = parser.parse_object(plain_arguments, defaults=True)
    # keep `None` values, so that they do not turn into the defaults of `function`
    return parser.dump(config, skip_none=False)


def save_hyperparameters(hyperparameters: str, checkpoint_dir: Path) -> None:
    """Writes the text from `dump_hyperparameters` to `checkpoint_dir/hyperparameters.yaml`."""
    (checkpoint_dir / "hyperparameters.yaml").write_text(hyperparameters)
