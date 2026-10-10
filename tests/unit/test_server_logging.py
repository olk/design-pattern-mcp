"""Process output policy: ``configure_logging`` keeps the SDK's logging-capability notice quiet.

fastmcp 4.x deliberately keeps routing ``Context.info``/``log`` through the MCP logging capability
that the SDK deprecates as of 2026-07-28 (SEP-2577), so every design tool call would otherwise print
that unruled ``UserWarning`` into the server's JSON stderr stream. The pytest surface has its own
message-scoped filter (``pyproject.toml``), which would hide a regression in the runtime one — hence
these tests exercise ``configure_logging`` itself.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator

import pytest
from mcp.shared.exceptions import MCPDeprecationWarning

from src.server import configure_logging

LOGGING_NOTICE = "The logging capability is deprecated as of 2026-07-28 (SEP-2577)."
ROOTS_NOTICE = "The roots capability is deprecated as of 2026-07-28 (SEP-2577)."


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """``configure_logging`` rewrites the root logger; hand the session's state back afterwards."""
    root_logger = logging.getLogger()
    handlers = root_logger.handlers[:]
    level = root_logger.level
    yield
    root_logger.handlers[:] = handlers
    root_logger.setLevel(level)


def _surfaced(message: str) -> list[str]:
    """Warnings surfaced for ``message`` once the runtime filter is installed."""
    with warnings.catch_warnings():
        warnings.resetwarnings()
        configure_logging()
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn(message, MCPDeprecationWarning, stacklevel=2)
        return [str(record.message) for record in caught]


def test_logging_capability_notice_is_silenced() -> None:
    """The deprecated send_log_message path must not leak a warning into the log stream."""
    assert _surfaced(LOGGING_NOTICE) == []


def test_other_mcp_deprecations_still_surface() -> None:
    """The filter is message-scoped: an unrelated SDK deprecation keeps its visibility."""
    assert _surfaced(ROOTS_NOTICE) == [ROOTS_NOTICE]
