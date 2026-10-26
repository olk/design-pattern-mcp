# Copyright (c) 2026 Oliver Kowalke
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Design Pattern MCP — Client Demo

Connects to a running MCP server over HTTP and calls one tool with an
event-driven ingestion service description, then pretty-prints the JSON
result:

- analyze_design (default) — ANALYZE phase only: repaired brief, decision
  parts, per-part selections. Fast; no design generated.
- select_design_patterns (--tool design) — full design artifact: analyze →
  generate → evaluate → refine. Long-running (minutes).

The design call is synchronous from the client's perspective: the tool runs
the whole selection pipeline before answering, so the HTTP request can take
minutes (deterministic retrieval + generator LLM). For short client timeouts
use the job trio instead — see examples/design_pattern_client_async.py.

Usage:
    # Terminal 1: start the server (Docker dev stack)
    $ make docker-up

    # Terminal 2: run the client
    $ make client-analyze    # analyze (fast)
    $ make client-design     # full design

Or pipe the JSON output:
    $ CLIENT_URL=http://localhost:8822/mcp \
        uv run python examples/design_pattern_client.py 2>/dev/null | jq '.selected_patterns'
    $ CLIENT_URL=http://localhost:8822/mcp \
        uv run python examples/design_pattern_client.py --tool design 2>/dev/null | jq '.applications[0]'

Requires:
    - MCP server running on http://localhost:8822/mcp (default; raw server: 8050)
    - No LLM credentials are needed for deterministic retrieval, but a complete
      select_design_patterns result does require a functioning generator: the
      E12 path reports the provider failure instead of fabricating a full design.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import click
from fastmcp import Client  # type: ignore[attr-defined]
from fastmcp.exceptions import ClientError, McpError, ToolError

SERVER_URL = os.environ.get("CLIENT_URL", "http://localhost:8822/mcp")

DESCRIPTION = """
Implement an event-driven IoT ingestion service in Python. The API gateway
component accepts device events over HTTPS, validates each payload against a
versioned JSON schema and returns 422 on violation. The ingest component
publishes accepted events to a Kafka topic; a dispatch component routes each
event by type to one of several worker components that normalize and enrich
it (device metadata lookup, geolocation join). Persisted results land in a
time-series store. Constraints: at least 5,000 events per second, horizontal
scaling of the workers, backpressure towards the gateway, and retries with a
dead-letter topic for poison messages. Worker behavior varies per event type,
so the per-type processing logic must be extensible without touching the
dispatch component.
""".strip()

LANGUAGE = "python"


TOOL_NAMES: dict[str, str] = {
    "analyze": "analyze_design",
    "design": "select_design_patterns",
}


async def call_design_tool(server_url: str, tool: str) -> dict[str, Any]:
    """Connect to the MCP server and call the selected design tool."""
    tool_name = TOOL_NAMES[tool]
    client = Client(server_url)

    async with client:
        tools = await client.list_tools()
        print(f"Connected to {server_url}", file=sys.stderr)
        print(f"Available tools: {[t.name for t in tools]}", file=sys.stderr)

        print(
            f"Calling {tool_name}(language='{LANGUAGE}')...",
            file=sys.stderr,
        )
        result = await client.call_tool(
            tool_name,
            {
                "description": DESCRIPTION,
                "language": LANGUAGE,
            },
        )
        data: dict[str, Any] = result.data
        return data


def print_json(result: dict[str, Any]) -> None:
    """Pretty-print the tool result to stdout as JSON."""
    print(json.dumps(result, indent=2, ensure_ascii=False))


def print_startup_hint() -> None:
    """Print the server-startup guidance — only for transport/protocol failures."""
    print(
        f"\nIs the MCP server running?\n"
        f"  Start it with:  make docker-up   (dev stack, port 8822)\n"
        f"                  python -m src.main   (local, port 8050)\n"
        f"  Expected URL:   {SERVER_URL}",
        file=sys.stderr,
    )


async def amain(tool: str) -> int:
    """Call the tool; classify the failure (tool error / transport error / unexpected)."""
    try:
        result = await call_design_tool(SERVER_URL, tool)
    except ToolError as exc:
        # The server answered and the tool itself failed (ERR_x in the message) —
        # the server is up, so the startup block would be misleading.
        print(f"MCP tool error: {exc}", file=sys.stderr)
        return 1
    except (ClientError, McpError) as exc:
        print(f"MCP transport error: {exc}", file=sys.stderr)
        print_startup_hint()
        return 1
    except Exception as exc:  # noqa: BLE001 - demo client reports everything
        print(f"Unexpected client error: {exc}", file=sys.stderr)
        return 1

    print_json(result)
    return 0


@click.command()
@click.option(
    "--tool",
    type=click.Choice(tuple(TOOL_NAMES)),
    default="analyze",
    show_default=True,
    help="design tool to call (design runs the full pipeline)",
)
def cli(tool: str) -> None:
    """Call analyze_design (default) or select_design_patterns on a running MCP server."""
    exit_code = asyncio.run(amain(tool))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    cli()
