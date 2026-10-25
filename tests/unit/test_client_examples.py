"""The demo clients classify failures instead of blaming an unreachable server.

A tool error means the server answered and the pipeline failed (ERR_x in the message) — the
startup block would send the reader down the wrong path. Only transport/protocol failures
(``ClientError``/``McpError``) justify the "is the server running?" guidance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from fastmcp.exceptions import ClientError, McpError, ToolError

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC_CLIENT = REPO_ROOT / "examples" / "design_pattern_client.py"
ASYNC_CLIENT = REPO_ROOT / "examples" / "design_pattern_client_async.py"


def _load(path: Path) -> ModuleType:
    """Import an example module by path (the examples are scripts, not a package)."""
    name = f"example_client_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=[SYNC_CLIENT, ASYNC_CLIENT], ids=["sync", "async"])
def client(request: pytest.FixtureRequest) -> ModuleType:
    return _load(request.param)


def _run_amain(module: ModuleType, monkeypatch: pytest.MonkeyPatch, exc: Exception) -> int:
    """Drive the client's ``amain`` with a stubbed tool call that raises ``exc``."""
    import asyncio

    async def _boom(*_args: object, **_kwargs: object) -> None:
        raise exc

    monkeypatch.setattr(module, "call_select_design_patterns", _boom, raising=False)
    monkeypatch.setattr(module.Client, "call_tool", _boom, raising=False)
    monkeypatch.setattr(module.Client, "__aenter__", _boom, raising=False)
    return int(asyncio.run(module.amain()))


def test_tool_error_never_prints_the_server_startup_block(
    client: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ERR_009/ERR_014 tool failures are reported as tool errors — the server is up."""
    exit_code = _run_amain(client, monkeypatch, ToolError("ERR_014: run exceeded the deadline"))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "MCP tool error" in captured.err
    assert "ERR_014" in captured.err
    assert "Is the MCP server running?" not in captured.err
    assert "Start it with" not in captured.err


def test_transport_error_prints_the_server_startup_block(
    client: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unreachable server is a transport failure and keeps the startup guidance."""
    exit_code = _run_amain(client, monkeypatch, ClientError("connection refused"))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "MCP transport error" in captured.err
    assert "Is the MCP server running?" in captured.err
    assert f"Expected URL:   {client.SERVER_URL}" in captured.err


def test_protocol_error_prints_the_server_startup_block(
    client: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """McpError (protocol-level MCPError) is handled exactly like a transport failure."""
    exit_code = _run_amain(client, monkeypatch, McpError(1, "unexpected protocol frame"))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "MCP transport error" in captured.err
    assert "Is the MCP server running?" in captured.err


def test_unexpected_error_is_labelled_without_a_downtime_claim(
    client: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-MCP client bug is reported as such; the server is not blamed."""
    exit_code = _run_amain(client, monkeypatch, KeyError("job_id"))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Unexpected client error" in captured.err
    assert "Is the MCP server running?" not in captured.err


def test_docstrings_state_the_generator_requirement(client: ModuleType) -> None:
    """Retrieval works without credentials; a complete result still needs a generator."""
    docstring = client.__doc__ or ""

    assert "No LLM credentials" in docstring
    assert "functioning generator" in docstring
    assert "fallbacks" not in docstring


def test_default_url_is_unchanged(client: ModuleType) -> None:
    """The Makefile targets and the documented default port stay as shipped."""
    assert client.SERVER_URL == "http://localhost:8822/mcp"
