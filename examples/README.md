# Design Pattern MCP — Client Demo

A standalone Python script that connects to a running MCP server over HTTP and
calls one tool with an event-driven ingestion service description, then
pretty-prints the JSON result:

- `analyze_design` (default) — ANALYZE phase only: repaired brief, decision
  parts, per-part selections. Fast; no design generated.
- `select_design_patterns` (`--tool design`) — full design artifact: analyze
  → generate → evaluate → refine. Long-running (minutes).

A second script, `design_pattern_client_async.py`, uses the background job
trio (`submit_design_job` + `get_design_status` + `cancel_design`) so the
client can stay responsive; `design_phase_client.py` drives the three phase
tools (`analyze_design` → `generate_design` → `evaluate_design`) in sequence
so each intermediate result is inspectable.

## Quick Start

**Terminal 1 — start the server (Docker dev stack):**

```bash
make docker-up
# Server reachable on http://localhost:8822/mcp
```

(or a local server: `python -m src.main` → http://0.0.0.0:8050/mcp)

**Terminal 2 — run the client:**

```bash
uv run python examples/design_pattern_client.py                  # analyze (fast)
uv run python examples/design_pattern_client.py --tool design    # full design

# or via make (server on http://localhost:8822/mcp):
make client-analyze
make client-design
make client-async    # submit + poll + result via the job trio
```

## Output

### analyze (default)

The script prints a single JSON object to stdout (progress goes to stderr)
containing:

| Key | Description |
|---|---|
| `design_brief` | Repaired brief: components, flows, constraints |
| `decision_parts` | Decision parts with scope/category hypothesis |
| `selections` | Per-part `PartSelection`: primary, supporting, alternatives, veto_ledger |
| `selected_patterns` | Selected primary patterns in part order (deduplicated) |
| `categories` | Pattern categories represented by the selected primaries |
| `effective_language` | Bound language (`generic` when unrecognized) |

No applications, interactions or quality attributes are generated.

### design (`--tool design`)

The full `DesignResult`:

| Key | Description |
|---|---|
| `overview` | `DesignOverview`: primary pattern, summary, design rationale |
| `applications` | `PatternApplication[]`: pattern, participants, implementation steps, bound language variant |
| `interactions` | Cross-application interactions |
| `quality_attributes` | Per-axis quality assessment |
| `test_strategy` | `TestStrategy` for the applied patterns |
| `risks` | Residual risks of the selection |

Useful views:

```bash
# analyze (default)
uv run python examples/design_pattern_client.py 2>/dev/null | jq '.selected_patterns'
uv run python examples/design_pattern_client.py 2>/dev/null | jq '.selections'

# design
uv run python examples/design_pattern_client.py --tool design 2>/dev/null | jq '.applications[].pattern'
uv run python examples/design_pattern_client.py --tool design 2>/dev/null | jq '.applications[0]'
```

## Customising

Edit these constants at the top of `examples/design_pattern_client.py`
(the async client has the same knobs):

```python
SERVER_URL = os.environ.get("CLIENT_URL", "http://localhost:8822/mcp")
DESCRIPTION = "..."     # your own system description
LANGUAGE = "python"     # generic|c|cplusplus|csharp|go|java|javascript|kotlin|python|ruby|rust|scala|swift|typescript
```

The async poll cadence: `CLIENT_POLL_SECONDS` (default 15).
Ctrl+C during the poll wait cancels the job server-side before exiting.

## Server configuration

No LLM credentials are required for the demos: with an unset generator key
the pipeline answers from its deterministic retrieval fallbacks, and the TEI
sidecars do embedding/reranking in the dev stack. Pass a generator key via
compose environment if you want LLM-backed narrative sections.
