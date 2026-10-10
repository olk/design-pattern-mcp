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
Design Pattern MCP — Client Demo (asynchronous job trio)

Connects to a running MCP server over HTTP, starts a background selection
job with submit_design_job, polls get_design_status until it reaches a
terminal state, and prints the DesignResult JSON on success. Useful when
the client cannot hold a long HTTP request open (proxies, short timeouts):
submit returns immediately with a job_id.

Poll cadence and Ctrl+C behavior: the script polls every
DESIGN_PATTERN_CLIENT_POLL_SECONDS (default 15); Ctrl+C during the wait
cancels the job server-side via cancel_design before exiting.

Usage:
    # Terminal 1: start the server (Docker dev stack)
    $ make docker-up

    # Terminal 2: run the client
    $ make client-async

Requires:
    - MCP server running on http://localhost:8060/mcp (default; raw server: 8050)
    - No LLM credentials are needed for deterministic retrieval, but a complete
      job result does require a functioning generator: the E12 path settles the
      job as failed instead of fabricating a full design.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time

from fastmcp import Client  # type: ignore[attr-defined]
from fastmcp.exceptions import ClientError, McpError, ToolError

SERVER_URL = os.environ.get("DESIGN_PATTERN_CLIENT_URL", "http://localhost:8060/mcp")

POLL_EVERY = int(os.environ.get("DESIGN_PATTERN_CLIENT_POLL_SECONDS", "15"))

DESCRIPTION = """
Implement an event-driven IoT ingestion service in Kotlin. The API gateway
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

LANGUAGE = "kotlin"

# Mutable holder set from the signal handler so the poll loop can wind down
# cleanly without a `global` statement.
SHUTDOWN = {"interrupted": False}


def _request_shutdown(signum: int, frame: object) -> None:  # noqa: ARG001
    SHUTDOWN["interrupted"] = True



def print_startup_hint() -> None:
    """Print the server-startup guidance — only for transport/protocol failures."""
    print(
        f"\nIs the MCP server running?\n"
        f"  Start it with:  make docker-up   (dev stack, port 8060)\n"
        f"                  python -m src.main   (local, port 8050)\n"
        f"  Expected URL:   {SERVER_URL}",
        file=sys.stderr,
    )

async def poll_until_terminal(client: Client, job_id: str) -> int:
    """Poll ``get_design_status`` until the job settles; return the process exit code."""
    start_time = time.monotonic()
    while True:
        await asyncio.sleep(POLL_EVERY)
        status_result = await client.call_tool("get_design_status", {"job_id": job_id})
        status_data = status_result.data
        elapsed = time.monotonic() - start_time

        status = status_data["status"]
        print(
            f"[{elapsed:6.0f}s] status={status:10s}  message={status_data['message']}",
            file=sys.stderr,
        )

        if status == "completed":
            print("\n=== Design result ===", file=sys.stderr)
            result_json = json.dumps(
                status_data.get("result") if status_data.get("result") else {}, indent=2
            )
            print(result_json)
            return 0

        if status == "failed":
            print("\n=== Job failed ===", file=sys.stderr)
            print(f"Error: {status_data.get('error', 'unknown')}", file=sys.stderr)
            return 1

        if status == "cancelled":
            print("\n=== Job cancelled ===", file=sys.stderr)
            return 130

        if SHUTDOWN["interrupted"]:
            print("\nCtrl+C received — cancelling job...", file=sys.stderr)
            await client.call_tool("cancel_design", {"job_id": job_id})
            return 130


async def amain() -> int:
    print(f"Connecting to {SERVER_URL}", file=sys.stderr)

    client = Client(SERVER_URL)
    # The try wraps the connection too: a refused/unreachable server fails in
    # __aenter__, before any tool call, and deserves the same startup guidance.
    try:
        async with client:
            # Step 1: start the job
            print("Submitting design job...", file=sys.stderr)
            result = await client.call_tool(
                "submit_design_job",
                {
                    "description": DESCRIPTION,
                    "language": LANGUAGE,
                },
            )
            data = result.data
            job_id = data["job_id"]
            print(f"Job ID: {job_id}", file=sys.stderr)
            print(
                f"Polling every {POLL_EVERY} seconds. Press Ctrl+C to cancel.\n",
                file=sys.stderr,
            )

            # Step 2: poll until terminal state
            return await poll_until_terminal(client, job_id)
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


def main() -> int:
    signal.signal(signal.SIGINT, _request_shutdown)
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
