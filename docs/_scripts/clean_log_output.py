"""The module is a hook which disables warnings and log messages which pollute the
doc build output.

One possible downside is if one of these modules ends up giving an actual
error, such as OpenML failing to retrieve a dataset. I tried to make sure ERROR
log message are still allowed through.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import mkdocs
import mkdocs.plugins
import mkdocs.structure.pages

log = logging.getLogger("mkdocs")


@mkdocs.plugins.event_priority(-50)
def on_startup(**kwargs: Any) -> None:
    # Remove deperecation warnings from the doc build log
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def on_pre_page(
    page: mkdocs.structure.pages.Page,
    config: Any,
    files: Any,
) -> mkdocs.structure.pages.Page | None:
    # If you want to remove certain loggers from doc output, you can do it here

    # logging.getLogger("smac").setLevel(logging.ERROR)
    # logging.getLogger("openml").setLevel(logging.ERROR)
    return page
