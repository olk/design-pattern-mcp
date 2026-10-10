# Design Pattern MCP — Client Demo

Standalone Python scripts that connect to a running MCP server over HTTP and
call the selection tools. `design_pattern_client.py` performs one blocking
`select_design_patterns` call; `design_pattern_client_async.py` uses the
background job trio (`submit_design_job` + `get_design_status` +
`cancel_design`) so the client can stay responsive.

## Quick Start

**Terminal 1 — start the server (Docker dev stack):**

```bash
make docker-up
# Server reachable on http://localhost:8060/mcp
```

(or a local server: `python -m src.main` → http://0.0.0.0:8050/mcp)

**Terminal 2 — run a client:**

```bash
make client          # blocking select_design_patterns call
make client-async    # submit + poll + result via the job trio
```

## Output

Both scripts print a single JSON object to stdout (progress goes to stderr)
containing the DesignResult payload — applications (pattern, participants,
implementation steps, bound language variant), supporting patterns,
interactions, quality attributes, test strategy and risks.

Useful views:

```bash
make client 2>/dev/null | jq '.applications[].pattern'
make client 2>/dev/null | jq '.applications[0]'
```

## Customising

Edit these constants at the top of `examples/design_pattern_client.py`
(the async client has the same knobs):

```python
SERVER_URL = os.environ.get("DESIGN_PATTERN_CLIENT_URL", "http://localhost:8060/mcp")
DESCRIPTION = "..."     # your own system description
LANGUAGE = "python"     # generic|c|cplusplus|csharp|go|java|javascript|kotlin|python|ruby|rust|scala|swift|typescript
```

The async poll cadence: `DESIGN_PATTERN_CLIENT_POLL_SECONDS` (default 15).
Ctrl+C during the poll wait cancels the job server-side before exiting.

## Server configuration

No LLM credentials are required for the demos: with an unset generator key
the pipeline answers from its deterministic retrieval fallbacks, and the TEI
sidecars do embedding/reranking in the dev stack. Pass a generator key via
compose environment if you want LLM-backed narrative sections.
