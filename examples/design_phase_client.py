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
Design Pattern MCP — Phase Chain Client (analyze → generate → evaluate)

Connects to a running MCP server over HTTP and drives the three phase tools
in sequence: analyze_design turns a system description into per-decision-part
pattern selections, generate_design materializes the matching design draft,
and evaluate_design scores that draft against the original requirements.

This mirrors examples/design_pattern_client.py (which runs the whole
pipeline in one select_design_patterns call), split into its phases so each
intermediate result is inspectable.

Usage:
    # Terminal 1: start the server (Docker dev stack)
    $ make docker-up

    # Terminal 2: run the client
    $ CLIENT_URL=http://localhost:8822/mcp \
        uv run python examples/design_phase_client.py

Requires:
    - MCP server running on http://localhost:8822/mcp (default)
    - No LLM credentials needed: with an unset generator key the pipeline
      answers from the deterministic retrieval fallbacks.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from fastmcp import Client  # type: ignore[attr-defined]

SERVER_URL = os.environ.get("CLIENT_URL", "http://localhost:8822/mcp")

DESCRIPTION = """
Implement an event-driven IoT ingestion service in TypeScript. The API gateway
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

LANGUAGE = "typescript"


async def call_phase_chain(server_url: str) -> None:
    """Run analyze_design → generate_design → evaluate_design and print each result."""
    client = Client(server_url)

    async with client:
        tools = await client.list_tools()
        print(f"Connected to {server_url}", file=sys.stderr)
        print(f"Available tools: {[t.name for t in tools]}", file=sys.stderr)

        print(f"Calling analyze_design(language='{LANGUAGE}')...", file=sys.stderr)
        analyzed = await client.call_tool(
            "analyze_design",
            {"description": DESCRIPTION, "language": LANGUAGE},
        )
        analysis: dict[str, Any] = analyzed.data
        print(
            json.dumps(
                {
                    "decision_parts": len(analysis["decision_parts"]),
                    "selected_patterns": analysis["selected_patterns"],
                    "effective_language": analysis["effective_language"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        print("Calling generate_design(analysis=...)...", file=sys.stderr)
        generated = await client.call_tool("generate_design", {"analysis": analysis})
        design: dict[str, Any] = generated.data
        print(
            json.dumps(
                {
                    "primary_pattern": design["overview"]["primary_pattern"],
                    "applications": [
                        {
                            "id": application["id"],
                            "pattern": application["pattern"],
                            "language": application["language_binding"]["language"],
                        }
                        for application in design["applications"]
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        print("Calling evaluate_design(design_json=...)...", file=sys.stderr)
        evaluated = await client.call_tool(
            "evaluate_design",
            {"design_json": design, "requirements": DESCRIPTION},
        )
        evaluation: dict[str, Any] = evaluated.data
        print(
            json.dumps(
                {
                    "score": evaluation["score"],
                    "summary": evaluation["summary"],
                    "integrity_findings": evaluation["integrity_findings"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )


async def amain() -> int:
    try:
        await call_phase_chain(SERVER_URL)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            f"\nIs the MCP server running?\n"
            f"  Start it with:  make docker-up   (dev stack)\n"
            f"                  python -m src.main   (local)\n"
            f"  Expected URL:   {SERVER_URL}",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
