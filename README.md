# design-pattern-mcp

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![M8ven Score](https://m8ven.ai/badge/mcp/olk-design-pattern-mcp-njyc2vv=0bd9bbb0ae6db22b3afee2ea3a58f4fc)](https://m8ven.ai/mcp/olk-design-pattern-mcp-njyc2v?s=readme)

An MCP (Model Context Protocol) server that provides design-pattern selection expertise to AI coding agents. Given a free-text description of a system, it analyses the problem, retrieves matching design patterns (from **486 built-in patterns, each realized in 13 concrete languages plus a generic binding**), generates a concrete design with a primary pattern, supporting patterns, participants, interactions, implementation steps, and language-bound variants, and evaluates it against eight design-level quality axes (testability, extensibility, maintainability, simplicity, performance, reliability, security, scalability). The pipeline (analyze → weights ∥ retrieval → generate → evaluate → bounded refine) and its wire contract are specified in [docs/implementation-guide.md](docs/implementation-guide.md).

---

## Table of Contents

- [⚡ Quickstart](#-quickstart)
- [Install via AI Agent](#install-via-ai-agent)
- [🔌 Connect Your Agent](#-connect-your-agent)
  - [Claude Code](#claude-code)
  - [OpenCode](#opencode)
  - [Codex CLI](#codex-cli)
- [🧑‍🏫 SKILL for AI Agents](#-skill-for-ai-agents)
- [🧪 Use the Tools](#-use-the-tools)
- [🛠️ Tools at a Glance](#️-tools-at-a-glance)
- [💬 Prompts](#-prompts)
- [📖 Pattern Catalog](#-pattern-catalog)
- [Working Together with architecture-pattern-mcp](#-working-together-with-architecture-pattern-mcp)
- [Install Alternatives](#install-alternatives)
  - [Docker (manual)](#docker-manual)
  - [Local Development (uv)](#local-development-uv)
- [Configuration](#configuration)
- [Structured Reasoning](#structured-reasoning)
- [Long-running tools & timeouts](#long-running-tools--timeouts)
- [Benchmarks & Verification](#benchmarks--verification)
- [Troubleshooting](#troubleshooting)
- [Building & Development](#building--development)
- [Publishing](#publishing)
- [systemd Service (Linux)](#systemd-service-linux)
- [License](#license)

---

## ⚡ Quickstart

```bash
# 1. Clone
git clone https://git.odvlh.xyz/olli/design-pattern-mcp.git && cd design-pattern-mcp

# 2. Add your API key
export MINIMAXAI_API_KEY=your_key_here

# 3. Build + start (three images: MCP server + TEI embed + TEI rerank)
make docker-build-all
make docker-up

# 4. Demo
make client           # blocking select_design_patterns call
make client-async     # same run via the background job trio
```

Server starts on **streamable-http** at `http://localhost:8062/mcp` (dev compose maps host `MCP_HOST_PORT:-8062` → container `MCP_CONTAINER_PORT:-8052`). `make client` targets `http://localhost:8062/mcp`; override with `DESIGN_PATTERN_CLIENT_URL`.

---

## Install via AI Agent

To have an AI coding agent install the Docker stack from prebuilt images (no local build), point it at **[INSTALL.md](INSTALL.md)** — the non-interactive agent runbook (copy-paste commands, expected outputs, verification probes).

```text
Clone this repo, then follow INSTALL.md Path A (compose stack, port 8062).
My generator API key is: sk-...
```

Pick one path in `INSTALL.md`:

- **Path A — Compose stack** (`:8062`): evaluating / developing, no sudo needed. Own TEI sidecar copies.
- **Path B — systemd stack** (`:8062`): persistent host service, starts at boot, needs `sudo`. Uses the shared `pattern-tei-infra` TEI stack.

The agent needs a repo clone, Docker ≥ v2.20 with group membership, ~8 GB free, a `MINIMAXAI_API_KEY` (or another provider per `INSTALL.md#switching-the-generator-llm`), and linux/amd64 (on arm64 it builds locally via `make docker-build-all`). Both paths default to host port 8062 — set `MCP_HOST_PORT` in one env file to run both at once. After install, continue with [Connect Your Agent](#-connect-your-agent) below.

---

## 🔌 Connect Your Agent

### Claude Code

```bash
# Install (one-time)
uv pip install -e .

# Run as stdio subprocess — pass the API key via env
claude mcp add design-patterns \
  -e DESIGN_PATTERN_GENERATOR_PROVIDER=minimax \
  -e DESIGN_PATTERN_GENERATOR_MODEL=minimax/MiniMax-M2.7 \
  -e DESIGN_PATTERN_GENERATOR_BASE_URL=https://api.minimax.io/v1 \
  -e DESIGN_PATTERN_GENERATOR_API_KEY=your_key \
  -- design-pattern-mcp --transport stdio
```

Stdio mode needs the TEI endpoints reachable (default `http://design-pattern-tei-embed:8080` and the reranker URL from `config/config.json`) — either run the compose stack's sidecars or set `DESIGN_PATTERN_EMBEDDER_BASE_URL` / `DESIGN_PATTERN_RERANKER_BASE_URL` to your own. For a no-network smoke run, start with `--offline` (deterministic lexical-only retrieval instead of TEI).

### OpenCode

OpenCode uses HTTP transport. Start the server first, then configure opencode:

```bash
# Terminal 1: start the server
make docker-up
# or locally:
uv run design-pattern-mcp --transport streamable-http --port 8050

# Terminal 2: add to ~/.config/opencode/opencode.json
```

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "design-patterns": {
      "type": "remote",
      "url": "http://localhost:8062/mcp"
    }
  }
}
```

> **Note:** the generator API key is read by the server from its config file (`config/config.json`, `{env:...}` placeholders), not from opencode's environment.

### Codex CLI

```bash
# Install (one-time)
uv pip install -e .
```

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.design-patterns]
command = "design-pattern-mcp"
args = ["--transport", "stdio"]

[mcp_servers.design-patterns.env]
DESIGN_PATTERN_GENERATOR_PROVIDER = "minimax"
DESIGN_PATTERN_GENERATOR_MODEL = "minimax/MiniMax-M2.7"
DESIGN_PATTERN_GENERATOR_BASE_URL = "https://api.minimax.io/v1"
DESIGN_PATTERN_GENERATOR_API_KEY = "your_key"
```

Or via CLI:

```bash
codex mcp add design-patterns \
  -e DESIGN_PATTERN_GENERATOR_API_KEY=your_key \
  -- design-pattern-mcp --transport stdio
```

---

## 🧑‍🏫 SKILL for AI Agents

The SKILL in `skills/design-pattern-mcp/` is written for **Oh My Pi (OMP)**, where this server's tools are reached as `xd://mcp__design_pattern_local_*` devices and every MCP request is bounded by a deadline (`OMP_MCP_TIMEOUT_MS` → per-server `timeout` → 30 s). It teaches the agent which entry point fits that deadline, how to read the results, and the full workflow recipes.

```
skills/design-pattern-mcp/
├── SKILL.md                 # OMP constraints, entry-point decision guide, device quick reference
└── references/
    ├── tools.md             # 9 tool signatures, arg shapes, status codes
    └── workflows.md         # workflow recipes, result interpretation, troubleshooting
```

**Install for OMP** — copy the skill directory into OMP's user skills root:

```bash
cp -r skills/design-pattern-mcp ~/.omp/agent/skills/
```

The skill then tells the agent:

- Which entry point fits OMP's 30 s deadline: the async job trio by default; `select_design_patterns` (full pipeline) takes 2–10 min
- That `job_id` is the only durable handle — persist it (SQLite job store, `DESIGN_PATTERN_JOBS_DB`)
- How to phrase `description` (component names + responsibilities + constraints), and `language`, `paradigm`, `focus` as separate structured arguments
- How to interpret `overview.score` (0–100), `findings[]`, and per-part selections

The tool schemas in `references/` are client-agnostic; only the deadline/device guidance is OMP-specific.

---

## 🧪 Use the Tools

All pipeline tools take a free-text `description` as the primary argument. The examples below show the exact tool call shape so you can use them in any MCP client or API consumer.

### Design your first selection

In Claude Code (or any MCP client), paste the natural-language instruction:

```
We're building the ingestion path of an IoT platform: a Kafka consumer
service parses 10k events/sec, an enricher adds geolocation from Redis,
and a writer persists to InfluxDB and S3. Components: consumer, parser,
enricher, sink. Constraints: at-least-once delivery, backpressure, Python.
```

Your agent calls `select_design_patterns` internally. The server returns the `DesignResult`: `overview` (primary pattern, supporting patterns, principles, score, reasoning), `applications[]` (participants, interactions, implementation_steps, code_sketch, language_binding with the bound variant), `interactions[]`, `quality_attributes`, `test_strategy`, and `risks[]`.

**Or call tools directly** from your agent:

```
Call select_design_patterns with:
  description: "IoT ingestion path: Kafka consumer (10k events/sec) → JSON parse
               → Redis geo-enrich → InfluxDB + S3 sink. At-least-once, backpressure."
  language: "python"

Call list_design_patterns()                       # all 486 patterns
Call list_design_patterns(category="workflow-messaging")
Call get_design_pattern(name="process-manager")
```

### Async job pattern: `submit_design_job` + `get_design_status`

For clients with short request timeouts. `submit_design_job` returns a `job_id` immediately; poll `get_design_status` until done:

```
# Step 1: start the job
Call submit_design_job with:
  description: "IoT ingestion path: Kafka consumer → parse → Redis enrich → InfluxDB + S3"

# Step 2: poll every 10-30 seconds
Call get_design_status with:
  job_id: "<job_id from step 1>"

# → status is "pending" | "running" | "completed" | "failed" | "cancelled"
# When status is "completed", the full DesignResult is in result
# When status is "failed", the error is in error
```

In Python (via the MCP client API — see `examples/design_pattern_client_async.py`):

```python
import asyncio
from fastmcp import Client

SERVER = "http://localhost:8062/mcp"
POLL_EVERY = 15  # seconds

async def main():
    client = Client(SERVER)
    async with client:
        # Step 1: start the job
        result = await client.call_tool(
            "submit_design_job",
            {"description": "IoT ingestion: Kafka → parse → enrich → sink"},
        )
        job_id = result.data["job_id"]
        print(f"Job started: {job_id}")

        # Step 2: poll until terminal state
        while True:
            await asyncio.sleep(POLL_EVERY)
            status = await client.call_tool("get_design_status", {"job_id": job_id})
            data = status.data
            print(f"  status={data['status']}")
            if data["status"] in ("completed", "failed", "cancelled"):
                break

        print(data.get("result", data))  # full DesignResult when completed

asyncio.run(main())
```

See `examples/design_pattern_client_async.py` for the complete runnable example. Run it with:

```bash
make docker-up          # Terminal 1
make client-async       # Terminal 2
```

---

## 🛠️ Tools at a Glance

| Tool | Description |
|---|---|
| `select_design_patterns` | Full pipeline (analyze → weights ∥ retrieval → generate → evaluate → refine, `DESIGN_PATTERN_LOOP_MAX_TRIES` attempts). *Long-running (2–10 min). Heartbeat-defended.* |
| `analyze_design` | Requirements analysis only: decision-part decomposition, style/category recommendations. *Long-running (~1 min).* |
| `generate_design` | Generate a design from explicitly named patterns. *Long-running (~1 min).* |
| `evaluate_design` | Score an existing design JSON against the eight quality axes. *Long-running (~30 s).* |
| `submit_design_job` | Start a background selection job and return a `job_id` immediately (SQLite-backed, survives server restart). For short-timeout clients. |
| `get_design_status` | Poll job status; returns the full `DesignResult` when `completed`. |
| `cancel_design` | Cancel a running job (best-effort; takes effect at the next pipeline stage boundary). |
| `list_design_patterns` | List patterns with name/category/description; filter by `category` and/or `language`. |
| `get_design_pattern` | Full pattern JSON for one name (loader-resolvable, alias-aware). |

**Description, language, paradigm, focus are structured parameters** — pass them as separate tool arguments, not embedded in the description text. Variant selection is signal-based: there is no catalog-wide `variant_id` enum (variant ids are per-(pattern, language) and change with catalog edits; valid ids are discoverable via the `design-patterns://variants` resource and previous runs' `bound_variant`).

---

## 💬 Prompts

This server also exposes four user-invoked workflow prompts (slash commands in MCP clients). Unlike tools, the LLM does not autonomously invoke prompts — the user selects one and fills in its arguments. Each prompt encodes a tested tool-orchestration recipe, and none hardcode pattern names (category guidance quotes the live catalog's own structure).

| Prompt | Args | What it does |
|---|---|---|
| `/select_design_patterns_workflow` | description\*, language, architecture_style, paradigm, focus | Guided end-to-end selection workflow |
| `/explore_pattern_catalog` | `category` | Live catalog discovery via the `design-patterns://variants` resource + decision probes |
| `/evaluate_my_design` | description\* | Compare the catalog's selection against an existing design |
| `/compare_pattern_candidates` | decision\*, candidates\* | Two or three named patterns side-by-side for one decision; one selection run per candidate |

\* = required argument

### Tool-only clients

Clients that only support the tools protocol (no native `prompts/list` or `prompts/get`) can access all four workflow prompts via the generated `list_prompts` and `get_prompt` tools, which route through the server's middleware chain exactly as native prompt calls do.

---

## 📖 Pattern Catalog

### Via MCP tools (recommended — works in all clients)

```
list_design_patterns()                                        # all 486 patterns
list_design_patterns(category="concurrency-resilience")       # filter by category
get_design_pattern(name="acceptor-connector")                 # full pattern JSON
```

Valid `category` values (the nine decision groups, catalog sections 1–9): `domain-modeling`, `construction-composition`, `behavior-selection`, `interaction-boundary`, `workflow-messaging`, `persistence-state`, `concurrency-resilience`, `security-verification`, `specialized-components`.

### Via MCP resources

```
mcp_read_resource(uri="design-patterns://variants")
```

Groups every pattern's language realizations by language, then pattern — the realization vocabulary for `language`/`paradigm`/`focus` hints and variant selection.

### What one pattern record contains

The loader (`src/patterns/loader.py`) joins each record from **14 files**: one generic entry plus one per concrete language (13 languages: C, C++, C#, Go, Java, JavaScript, Kotlin, Python, Ruby, Rust, Scala, Swift, TypeScript). Each language binding carries `support` (`native` / `idiomatic` / `discouraged`), `idiom_name`, `implementation_notes`, an optional `code_sketch`, `pitfalls`, and — for `discouraged` support — a required `alternative`. Language bindings can declare `variants[]` (per-language realizations, 4,953 across the catalog) with a `default_variant`. Each record also declares `related_patterns` (loader-resolvable canonical names only) and `related_patterns_notes` (per-link prose distinctions).

Alias resolution is built in: the loader turns **993 raw alias entries → 978 distinct after case-folding → 966 indexed alias mappings**, dropping ambiguous aliases (an alias that is itself another record's name or claimed by two records; the `ambiguous_aliases` health field lists them), so `get_design_pattern(name="circuit-breaker")` and its alias spellings resolve to the same record. Startup fails fast when the catalog's enums drift from `docs/pattern-schema.json` (`assert_enums_in_sync`).

---

## Working Together with architecture-pattern-mcp

This server answers *how each unit is coded* (design patterns); the sibling [architecture-pattern-mcp](https://github.com/olk/architecture-pattern-mcp) answers *which system shape to build* (architecture). An **architecture pattern** structures the whole system — its components, their relationships, and its API/data/event contracts (e.g. `event-driven`, `pipe-and-filter`, `microservices`, chosen from the sibling's 40-pattern catalog). A **design pattern** solves one recurring implementation problem inside a single component — its participants, interactions, and language-bound code (e.g. `observer`, `process-manager`, `circuit-breaker`, chosen from the 486-pattern catalog above with 13 language bindings each).

Recommended order — macro first, micro second:

```
1. Call the sibling's design_architecture with:
     requirements: "ETL pipeline for IoT: Kafka → JSON → Redis geo-enrich → InfluxDB + S3"
     domain: "data-processing"
   # → components (Kafka source, parser filter, enricher, sinks),
   #   relationships, API contracts, data models, event contracts

2. For each component with non-trivial internal logic, call select_design_patterns with:
     description: "<component name + responsibilities + constraints from step 1>"
     language: "python"
   # → primary + supporting patterns, participants, implementation_steps,
   #   code_sketch with the bound language variant
```

Concrete handoff on the IoT example: the sibling emits the `pipe-and-filter` architecture (Kafka source → JSON parser filter → geolocation enricher → InfluxDB/S3 sinks, with at-least-once delivery and backpressure in the contracts). The geolocation enricher's cache-miss path then comes here via `select_design_patterns` ("cache-aside enrichment with Redis fallback, Python") — e.g. a `circuit-breaker` + `retry` design with participants, `implementation_steps`, and a Python `code_sketch` — which the architecture-level component contract references but never implements.

Run both servers side by side (dev host ports `:8060` architecture, `:8062` design-pattern; systemd `:8050` / `:8062`). They are independent — either order works for exploration — but architecture-first keeps component boundaries stable while this server fills in the internals.

## Install Alternatives

### Docker (manual)

```bash
# Build the three images (dense vector cache generated and baked in)
make docker-build-all

# Run with your API key
MINIMAXAI_API_KEY=your_key make docker-up
```

First `docker-build-all` generates the FAISS dense cache (`data/dense-cache/`) via the TEI embed sidecar and bakes it into the server image; `make docker-build-nocache` skips that (first startup then embeds the corpus, ~19 min).

### Local Development (uv)

**Prerequisites:** Python 3.12+, [uv](https://github.com/astral-sh/uv)

```bash
# Install
make install

# Configure — the shipped wire config with {env:...} placeholders
export DESIGN_PATTERN_GENERATOR_PROVIDER=minimax
export DESIGN_PATTERN_GENERATOR_MODEL=minimax/MiniMax-M2.7
export DESIGN_PATTERN_GENERATOR_BASE_URL=https://api.minimax.io/v1
export DESIGN_PATTERN_GENERATOR_API_KEY=your_key

# Run the server
uv run design-pattern-mcp --transport stdio               # for Claude Code / Codex
uv run design-pattern-mcp --transport streamable-http --port 8050   # HTTP
uv run design-pattern-mcp --health                        # offline readiness check (config + catalog + index)
```

Or use the installed console script (after `make install`): `design-pattern-mcp`.

The TEI sidecars are required for full retrieval quality: a dense leg (Qwen3-Embedding-0.6B via the embed sidecar) and a BM25 leg are fused at startup (`service_healthy` ordering guarantees the embedder is up before the server accepts requests — a failed index warm-up refuses server start). The reranker sidecar (`gte-reranker-modernbert-base`) reranks candidate patterns. Without reachable sidecars, start with `--offline` (deterministic lexical-only retrieval, no network).

> **TEI startup:** both sidecar images (`docker/Dockerfile.tei-embed`, `docker/Dockerfile.tei-rerank`) ship a self-built TEI router (v1.9.4 + unmerged [PR #884](https://github.com/huggingface/text-embeddings-inference/pull/884)) with `WARMUP_TOKENS=0` by default — this skips the synthetic warmup pass (measured ~131 s embed / ~78 s rerank) so containers become ready after weight load only. Override per sidecar via `TEI_WARMUP_TOKENS` / `TEI_RERANK_WARMUP_TOKENS` (positive N warms with N tokens instead; values > `MAX_BATCH_TOKENS` are rejected at startup).

**GPU TEI:** the shipped sidecars are CPU images (`cpu-1.9`). On a host with a TEI-CUDA-supported NVIDIA GPU (compute capability ≥ 7.5) plus the NVIDIA container toolkit, `make docker-build-tei-embed-gpu` builds the embed sidecar on the CUDA base (`ghcr.io/huggingface/text-embeddings-inference:cuda-1.9`) and `make docker-up-gpu` starts the dev stack with `docker/docker-compose.gpu.yml` (reserves 1 GPU). The override covers the embed sidecar only — the reranker stays on CPU. On GPU a small positive `WARMUP_TOKENS` warms kernels at startup instead of skipping the pass.

**Already-running embedder/reranker:** when embedding and reranking endpoints already run on other servers, point the MCP server at them via `DESIGN_PATTERN_EMBEDDER_BASE_URL` (OpenAI-compatible route, `http://<host>:8080/v1`) plus `DESIGN_PATTERN_RERANKER_BASE_URL` (`http://<host>:8080`) and skip the sidecars. The embedder forwards `DESIGN_PATTERN_EMBEDDER_MODEL` (default `openai//data/embedding-model`) and `DESIGN_PATTERN_EMBEDDER_API_KEY` through LiteLLM — provider/model syntax is documented in the [LiteLLM embedding docs](https://docs.litellm.ai/docs/embedding/supported_embedding). The reranker is **not** a LiteLLM client — it POSTs `{base_url}/rerank` directly (`src/patterns/safe_tei_rerank.py`), so `DESIGN_PATTERN_RERANKER_MODEL` (default `Alibaba-NLP/gte-reranker-modernbert-base`) is a TEI model name, not a LiteLLM model string. A remote embedder serving a different model invalidates the baked dense cache — regenerate it with `make embed-cache` so vectors match the serving model.

---

## Configuration

### config.json

The server reads `config/config.json` relative to the repo (override with `--config-path` or `DESIGN_PATTERN_CONFIG_PATH`). Every value is a `{env:VAR:-default}` placeholder that expands at load time, so the shipped file doubles as the env-var reference. The schema (abridged):

```json
{
  "logging_level": "{env:DESIGN_PATTERN_LOGGING_LEVEL:-INFO}",
  "transport": "{env:DESIGN_PATTERN_TRANSPORT:-stdio}",
  "host": "{env:DESIGN_PATTERN_HOST:-0.0.0.0}",
  "port": "{env:DESIGN_PATTERN_PORT:-8050}",
  "generator": {
    "provider": "{env:DESIGN_PATTERN_GENERATOR_PROVIDER:-openai}",
    "config": {
      "model": "{env:DESIGN_PATTERN_GENERATOR_MODEL:-gpt-4o-mini}",
      "base_url": "{env:DESIGN_PATTERN_GENERATOR_BASE_URL:-https://api.openai.com/v1}",
      "api_key": "{env:DESIGN_PATTERN_GENERATOR_API_KEY:-}",
      "temperature": "{env:DESIGN_PATTERN_GENERATOR_TEMPERATURE:-0.1}",
      "timeout_seconds": "{env:DESIGN_PATTERN_GENERATOR_TIMEOUT_SECONDS:-120.0}",
      "generation_batch_size": "{env:DESIGN_PATTERN_GENERATOR_GENERATION_BATCH_SIZE:-5}",
      "generation_max_concurrency": "{env:DESIGN_PATTERN_GENERATION_MAX_CONCURRENCY:-2}",
      "circuit_breaker_failure_threshold": "{env:DESIGN_PATTERN_GENERATOR_CIRCUIT_BREAKER_FAILURE_THRESHOLD:-3}",
      "circuit_breaker_cooldown_seconds": "{env:DESIGN_PATTERN_GENERATOR_CIRCUIT_BREAKER_COOLDOWN_SECONDS:-60.0}"
    }
  },
  "embedder":  { "provider": "{env:DESIGN_PATTERN_EMBEDDER_PROVIDER:-tei}", "config": { "base_url": "{env:DESIGN_PATTERN_EMBEDDER_BASE_URL:-http://design-pattern-tei-embed:8080}" } },
  "retrieval": { "bm25_weight": "{env:DESIGN_PATTERN_RETRIEVAL_BM25_WEIGHT:-0.3}", "dense_weight": "{env:DESIGN_PATTERN_RETRIEVAL_DENSE_WEIGHT:-0.7}", "max_concurrent_parts": "{env:DESIGN_PATTERN_RETRIEVAL_MAX_CONCURRENT_PARTS:-1}" },
  "reranker":  { "rerank_text": "{env:DESIGN_PATTERN_RERANK_TEXT:-full}", "rerank_top_n": "{env:DESIGN_PATTERN_RERANK_TOP_N:-10}" },
  "reasoning": { "enabled": "{env:REASONING_ENABLED:-true}", "per_phase": { "analyze": { "enabled": "{env:REASONING_ANALYZE_ENABLED:-true}" }, "weights": { "enabled": "{env:REASONING_WEIGHTS_ENABLED:-true}" }, "generate": { "enabled": "{env:REASONING_GENERATE_ENABLED:-true}" }, "evaluate": { "enabled": "{env:REASONING_EVALUATE_ENABLED:-true}" } }, "max_injected_chars": "{env:REASONING_MAX_INJECTED_CHARS:-}" },
  "validation": { "max_retries": "{env:DESIGN_PATTERN_VALIDATION_MAX_RETRIES:-1}", "retry_on_fail": "{env:DESIGN_PATTERN_VALIDATION_RETRY_ON_FAIL:-true}", "truncation_rerun": "{env:DESIGN_PATTERN_VALIDATION_TRUNCATION_RERUN:-repair}", "truncation_boost_max_completion_tokens": "{env:DESIGN_PATTERN_VALIDATION_TRUNCATION_BOOST_TOKENS:-65536}" },
  "prompts": { "stable_prefix_first": "{env:DESIGN_PATTERN_PROMPT_STABLE_PREFIX:-false}" },
  "loop":  { "max_tries": "{env:DESIGN_PATTERN_LOOP_MAX_TRIES:-2}", "min_quality_score": "{env:DESIGN_PATTERN_LOOP_MIN_QUALITY_SCORE:-50}" },
  "tasks": { "deadline_seconds": "{env:DESIGN_PATTERN_TASKS_DEADLINE_SECONDS:-1800.0}", "heartbeat_enabled": "{env:DESIGN_PATTERN_TASKS_HEARTBEAT_ENABLED:-true}", "heartbeat_interval_seconds": "{env:DESIGN_PATTERN_TASKS_HEARTBEAT_INTERVAL_SECONDS:-30}" },
  "jobs":  { "db_path": "{env:DESIGN_PATTERN_JOBS_DB:-~/.config/design-pattern-mcp/jobs.db}" }
}
```

### Generator LLM (LlamaIndex LiteLLM)

The generator LLM is accessed through the **LlamaIndex LiteLLM integration**. All provider settings follow **LiteLLM's model syntax**: `<provider>/<model>`. The dev stack and benchmark harness default to MiniMax:

| Config / env | Dev-stack default |
|---|---|
| `DESIGN_PATTERN_GENERATOR_PROVIDER` | `minimax` |
| `DESIGN_PATTERN_GENERATOR_MODEL` | `minimax/MiniMax-M2.7` |
| `DESIGN_PATTERN_GENERATOR_BASE_URL` | `https://api.minimax.io/v1` |
| `DESIGN_PATTERN_GENERATOR_API_KEY` | falls back to `MINIMAXAI_API_KEY` (host env or `.env`) |

Provider list and syntax: [LiteLLM Providers documentation](https://docs.litellm.ai/docs/providers). `temperature`, `top_p`, `top_k`, `timeout_seconds`, and the circuit-breaker knobs map to the corresponding LiteLLM/provider parameters.

### Key environment variables

| Variable | Default | Description |
|---|---|---|
| `DESIGN_PATTERN_GENERATOR_API_KEY` | *(required)* | Generator API key (dev stack falls back to `MINIMAXAI_API_KEY`) |
| `DESIGN_PATTERN_GENERATOR_PROVIDER` | `openai` | LiteLLM provider prefix (`minimax`, `openai`, `anthropic`, `openrouter`, …) |
| `DESIGN_PATTERN_GENERATOR_MODEL` | `gpt-4o-mini` | Model name; `minimax/MiniMax-M2.7` is the measured dev-stack default |
| `DESIGN_PATTERN_GENERATOR_BASE_URL` | `https://api.openai.com/v1` | API base URL (LiteLLM `api_base`) |
| `DESIGN_PATTERN_GENERATOR_TEMPERATURE` | `0.1` | Sampling temperature |
| `DESIGN_PATTERN_GENERATOR_TIMEOUT_SECONDS` | `120.0` | Bounds ONE HTTP request to the model |
| `DESIGN_PATTERN_GENERATOR_GENERATION_BATCH_SIZE` | `5` | Patterns per GENERATE batch |
| `DESIGN_PATTERN_GENERATION_MAX_CONCURRENCY` | `2` | GENERATE batches in flight at once |
| `DESIGN_PATTERN_GENERATOR_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `3` | Consecutive qualifying provider failures (timeout, connection, 408/429/5xx, empty response) opening the circuit; auth/configuration 4xx never count |
| `DESIGN_PATTERN_GENERATOR_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `60.0` | Open-state window before one half-open probe |
| `DESIGN_PATTERN_EMBEDDER_BASE_URL` | `http://design-pattern-tei-embed:8080` | TEI embedder endpoint |
| `DESIGN_PATTERN_EMBEDDER_BATCH_SIZE` | `16` | Embedding batch size |
| `DESIGN_PATTERN_RETRIEVAL_DENSE_WEIGHT` | `0.7` | Dense-leg fusion weight (must sum with `..._BM25_WEIGHT` to 1.0 ± 1e-3 — config-validated; a leg below 0.05 logs a warning) |
| `DESIGN_PATTERN_RETRIEVAL_BM25_WEIGHT` | `0.3` | BM25-leg fusion weight |
| `DESIGN_PATTERN_RETRIEVAL_MAX_CONCURRENT_PARTS` | `1` | Per-part rerank concurrency. Cap-2 saturates the CPU TEI rerank sidecar's 16,384-token buffer (HTTP 429) — leave at 1 unless you measured your sidecar |
| `DESIGN_PATTERN_RERANK_TEXT` | `full` | Rerank input text: `full` (rankings as documented) or `capsule` (3.6–4.3× faster, diverging rankings — opt-in) |
| `DESIGN_PATTERN_RERANK_TOP_N` | `10` | Rerank top N |
| `REASONING_ENABLED` | `true` | Master switch for the pre-LLM reasoning layer |
| `REASONING_{ANALYZE,WEIGHTS,GENERATE,EVALUATE}_ENABLED` | `true` | Per-phase reasoning switches; a disabled phase contributes an empty reasoning context (no generator call, no subprocess, no cache entry), while an *outage* in an enabled phase degrades silently to the in-prompt scaffold |
| `REASONING_{...}_THOUGHTS` | `2` / phase | Reasoning steps before each phase's main call |
| `REASONING_STEP_TIMEOUT_SECONDS` | `20` | Per-thought tool-call timeout |
| `REASONING_MAX_TOTAL_STEPS` | `8` | Hard cap on reasoning steps per phase |
| `REASONING_MAX_INJECTED_CHARS` | uncapped | Cap on the rendered reasoning trace injected into downstream prompts; overflow is cut and marked `(reasoning trace truncated at N chars)` (measured latency lever) |
| `REASONING_FAIL_FAST` | `false` | Fail server startup when a reasoning tool is unreachable (default: degrade, log ERROR) |
| `REASONING_QUIET_STDERR` | `true` | Silence reasoning-subprocess stderr; set `false` to debug spawn failures |
| `DESIGN_PATTERN_VALIDATION_MAX_RETRIES` | `1` | Structured-output repair budget (model-output failures only — provider errors are re-raised, never repair-prompted) |
| `DESIGN_PATTERN_VALIDATION_TRUNCATION_RERUN` | `repair` | Truncation handling: `repair` (default), `boost` (truncation-first: one rerun with the boost cap, no repair into the same ceiling), `off` |
| `DESIGN_PATTERN_VALIDATION_TRUNCATION_BOOST_TOKENS` | `65536` | `max_completion_tokens` of the boost rerun (MiniMax M2.x recommended cap) |
| `DESIGN_PATTERN_PROMPT_STABLE_PREFIX` | `false` | Reorder each operation's user prompt so the static response-contract block leads and per-run payload trails (MiniMax passive prompt-cache optimization; content-identical either way) |
| `DESIGN_PATTERN_LOOP_MAX_TRIES` | `2` | Design-loop GENERATE→EVALUATE attempts |
| `DESIGN_PATTERN_LOOP_MIN_QUALITY_SCORE` | `50.0` | Early-stop score threshold |
| `DESIGN_PATTERN_TASKS_DEADLINE_SECONDS` | `1800.0` | Whole-run budget for one tool call / job; on expiry the run ends with `ERR_014` (the pipeline's internal llama-index Workflow timeout is 0.9× this) |
| `DESIGN_PATTERN_TASKS_HEARTBEAT_INTERVAL_SECONDS` | `30` | Progress-notification interval (keep below the client's idle timeout) |
| `DESIGN_PATTERN_JOBS_DB` | `~/.config/design-pattern-mcp/jobs.db` | SQLite path for the async job trio |
| `DESIGN_PATTERN_PATTERN_DIRECTORY` | `pattern/` | Pattern files directory |
| `DESIGN_PATTERN_LOGGING_LEVEL` | `INFO` | `DEBUG` captures every reasoning thought + per-step tool response |
| `DESIGN_PATTERN_TRANSPORT` | `stdio` | `stdio` or `streamable-http` |
| `DESIGN_PATTERN_PORT` | `8050` | HTTP bind port (dev compose maps host `8062` → container) |
| `DESIGN_PATTERN_CONFIG_PATH` | `config/config.json` | Config file path |

### CLI flags

| Flag | Description |
|---|---|
| `--config-path` | Path to the config file (default: `config/config.json`) |
| `--transport {stdio,streamable-http}` | Override transport mode |
| `--host` | Override HTTP bind host (default: `0.0.0.0`) |
| `--port` | Override HTTP port (default: `8050`) |
| `--health` | Run the offline readiness check (config + catalog + index) and exit |
| `--offline` | Serve with the deterministic offline embedder/reranker (no network; lexical-only retrieval) |

---

## Structured Reasoning

Before each prompted LLM phase call (analyze / weights / generate / evaluate), the server runs a bounded **ThoughtGenerator loop**: each step is authored with the generator LLM (one completion per step) and submitted to the [shannonthinking](https://github.com/olaservo/shannon-thinking) and/or [code-reasoning](https://github.com/mettamatt/code-reasoning) MCP servers — structured thinking scratchpads that validate, number, and record each step. The resulting trace is injected into the phase prompt as a `<reasoning_context>` block. Contract: **each thought = 1 LLM completion + 1 MCP tool call**, capped by `REASONING_MAX_TOTAL_STEPS` (default 8). The tools list per phase defaults to both scratchpads; startup health-checks exactly the tools the enabled phases use.

Key properties:

- **Embedded in Docker** — the image bakes both npm packages into the image; the runtime invokes them directly via `node` (no network, no npx).
- **Auto-fallback to npx** outside Docker — missing embedded entry points fall back to `npx -y <pkg>` (first call downloads).
- **Silent per-call degradation** — any spawn/timeout/tool failure in an *enabled* phase logs a WARNING and the phase proceeds with a degraded in-prompt thinking scaffold; a reasoning outage never fails a tool call. A phase disabled via `REASONING_<PHASE>_ENABLED=false` instead contributes an empty reasoning context — no generator call, no subprocess, no cache entry.
- **Loud startup, soft failure** — the lifespan health-check reports reasoning health; set `REASONING_FAIL_FAST=true` to make a broken tool fail startup instead.
- **Trace caching** — analyze and generate traces are computed once per design request and reused across design-loop attempts; weights and evaluate traces are per-run.
- **`REASONING_MAX_INJECTED_CHARS`** — the rendered trace injected downstream can be capped (default uncapped) to cut input tokens on every downstream call.

Latency note: a full run issues 8 reasoning tool calls (2 per phase) as serial subprocesses; reasoning is a material share of end-to-end latency. Measured on the Stage-0 benchmark (MiniMax-M2.7 generator, 18 scenarios, `NO_WARMUP=1`): shipping config (all phases on) p50 **288.4 s** / p95 439.6 s; with `REASONING_WEIGHTS_ENABLED=false REASONING_EVALUATE_ENABLED=false` p50 **253.0 s** / p95 413.9 s — both arms passed the quality gate (overall hit ≥ baseline Wilson low). The per-phase toggles are the documented way to opt in.

---

## Long-running tools & timeouts

`select_design_patterns` runs a multi-stage pipeline that **takes minutes per call**. This is inherent to the workload: the generator LLM processes the fused retrieval candidates, per-part quality weights, and the full output of every previous stage, and produces a large, strictly structured JSON document one token at a time. The design loop repeats generate → evaluate up to `DESIGN_PATTERN_LOOP_MAX_TRIES` times.

### Four bounds govern a run — do not confuse them

| Bound | Default | Scope |
|---|---|---|
| `DESIGN_PATTERN_GENERATOR_TIMEOUT_SECONDS` | 120 s | ONE HTTP request to the model |
| `DESIGN_PATTERN_VALIDATION_MAX_RETRIES` | 1 | Repair re-issues exactly one call, and only for model-output failures |
| `DESIGN_PATTERN_LOOP_MAX_TRIES` | 2 | GENERATE→EVALUATE design-loop attempts |
| `DESIGN_PATTERN_TASKS_DEADLINE_SECONDS` | 1800 s | The whole run; expiry cancels in-flight work and ends with `ERR_014` |

### The heartbeat defence (applied by default)

Every long-running tool emits `progress` notifications from a parallel coroutine every 30 seconds (`DESIGN_PATTERN_TASKS_HEARTBEAT_INTERVAL_SECONDS`). Clients whose idle timers reset on any received data stay alive for the full pipeline duration.

> **TS-SDK clients (Claude Desktop, Cursor) do not reset their timeout on progress notifications.**

### The async job trio (for timeout-constrained clients)

```
submit_design_job(description, language?, ...)  → job_id
get_design_status(job_id)                       → {status, result | error}
cancel_design(job_id)                           → best-effort, next stage boundary
```

`submit_design_job` returns a `job_id` in milliseconds; the pipeline runs in a background task and the store is SQLite (`DESIGN_PATTERN_JOBS_DB`). Poll every 10–30 seconds. **This is the fix that works for TS-SDK clients.**

### Bypassing client timeouts entirely: `make client`

`examples/design_pattern_client.py` is a direct Python HTTP client with **no client-side idle timeout** — one blocking request, full response regardless of duration:

```bash
make docker-up          # Terminal 1
make client             # Terminal 2
```

---

## Benchmarks & Verification

This repo treats selection quality and pipeline latency as measured properties, not claims.

**Benchmark harness** (`tests/benchmark/`, `make benchmark-*`): an 18-scenario corpus with holdout split; `make benchmark-live NO_WARMUP=1 OUT=benchmarks/<arm> RUN_ID=<id>` runs the full pipeline in-process with per-stage attribution (reasoning.*, llm.*, rerank, embed — calls, latency, chars/s, repair counters). `make benchmark-selfcheck` hand-verifies every metric formula. `make benchmark-compare A=<baseline> B=<candidate>` applies the preregistered rule — *candidate wins iff holdout p95 e2e improves ≥ 15 % AND candidate overall acceptable-primary hit rate ≥ the baseline's overall Wilson lower bound* — and **exits 1 on candidate-loss**. The e2e mode measures the deployed compose stack and is not comparable with the in-process mode.

**Reference arms** (committed under `benchmarks/`, all MiniMax-M2.7, 18 scenarios):

| Arm | Configuration | e2e p50 | e2e p95 |
|---|---|---|---|
| `base-fresh` | shipped defaults (reasoning all phases on) | 288.4 s | 439.6 s |
| `cfg-reasoning` | + weights/evaluate reasoning off (opt-in env) | 253.0 s | 413.9 s |
| `code-levers` | + L5 truncation-boost, L6 trace cap, L7 stable-prefix (all default-off, env-driven) | 223.6 s | 379.2 s |

The code levers are shipped as opt-in env configuration, not defaults: the cumulative win claim passed the preregistered gate against `base-fresh` (holdout p95 410.9 → 297.5 s = 27.6 % ≥ 15 %, overall 13/18 ≥ floor), but the attribution comparison against `cfg-reasoning` recorded a candidate-loss (+12.5 % < 15 %), so defaults stay behavior-neutral. To opt in:

```bash
export DESIGN_PATTERN_VALIDATION_TRUNCATION_RERUN=boost
export REASONING_MAX_INJECTED_CHARS=4000
export DESIGN_PATTERN_PROMPT_STABLE_PREFIX=true
```

**Verification layers** (the repo's differentiator):

| Layer | What | Where |
|---|---|---|
| L1 | Unit tests (388) | `tests/unit/` |
| L2 | Hypothesis property oracles (110) | `tests/verification/` |
| L3 | Spec gardens (FG-* mutants) | `verify/`, `make verify-fizz-garden` |
| L4 | Exhaustive FizzBee models (7 specs: jobs runner/protocol, pipeline control, design loop, reasoning retry, TEI fallback, retrieval fusion) | `verify/fizz/`, `make verify-fizz` |
| — | Mutation ratchet (`mutmut` baseline; new survivors fail the gate) | `verify/mutmut-baseline.json`, `make test-mutations` |
| — | Ledger + cross-consistency (fizz ↔ tests ↔ docs) | `make verify-ledger`, `make verify-cross-consistency` |

Every `.fizz` assertion must kill at least one spec-garden mutant or carry a `# spec-explains:` justification; a change to a decision module updates the matching `.fizz` model in the same PR. The normative pipeline spec is [docs/implementation-guide.md](docs/implementation-guide.md) (§-numbered, mirrored in German as `docs/selektions-process-de.md`); the wire contract of the tool result is `docs/result-schema.json`, the catalog record contract is `docs/pattern-schema.json`.

---

## Troubleshooting

### Server starts but tools are not visible

1. Check the agent's MCP connection: Claude Code `/mcp`, OpenCode `opencode mcp list`, Codex `codex mcp list`
2. Verify the server process started and the catalog loaded: `uv run design-pattern-mcp --health` prints `OK` plus a JSON readiness report (records, language files, aliases)
3. Confirm the TEI embedder is healthy: `docker compose -f docker/docker-compose.yml ps` (healthchecks with 120 s embed-model start period)

### "Connection refused" or startup failure

Index warm-up refuses server start when the embedder is unreachable (fail-fast rather than breaking your first request):

```bash
docker compose -f docker/docker-compose.yml logs design-pattern-tei-embed
```

Local non-Docker runs must point `DESIGN_PATTERN_EMBEDDER_BASE_URL` / `DESIGN_PATTERN_RERANKER_BASE_URL` at reachable endpoints — or start with `--offline`.

### LLM provider errors (502 / 401)

- Confirm `DESIGN_PATTERN_GENERATOR_API_KEY` (or `MINIMAXAI_API_KEY` on the dev stack) is set and not expired
- Verify `DESIGN_PATTERN_GENERATOR_BASE_URL` matches your provider's endpoint
- After 3 consecutive transport failures the circuit opens for 60 s (`..._CIRCUIT_BREAKER_*` knobs); auth/configuration 4xx never count toward it

### `ERR_012` / `ERR_014`

- `ERR_012` = cancelled (via `cancel_design` or a client disconnect at the next stage boundary)
- `ERR_014` = whole-run deadline exceeded (`DESIGN_PATTERN_TASKS_DEADLINE_SECONDS`); raise it explicitly for legitimately longer runs — the in-flight provider coroutine is cancelled, partial results are discarded

### Reranker 429s

One full-text rerank request already saturates the CPU TEI sidecar's 16,384-token inference buffer (`MAX_BATCH_TOKENS=8192` interacts with request packing; measured 2026-09-25): a cap-2 `retrieve_many` leg fails outright with HTTP 429 "Model is overloaded". Keep `DESIGN_PATTERN_RETRIEVAL_MAX_CONCURRENT_PARTS=1` (shipped default); the server-side invariant also keeps `dense_top_k + bm25_top_k` ≤ the sidecar's `MAX_CLIENT_BATCH_SIZE` (48).

### Pattern JSON files not loading

- Records are joined from 14 files per pattern (`pattern/<name>.generic.json` + 13 `<language>.json`); the generic file is required
- Validate a record against `docs/pattern-schema.json` (`scripts/validate_patterns.py`)
- Startup fails on enum drift from the schema (`assert_enums_in_sync`) — that is intentional; fix the data, not the guard

---

## Building & Development

**Common make targets:**

| Target | Description |
|---|---|
| `make install` | Sync dependencies into `.venv` |
| `make install-mcps` | Install the reasoning MCP servers globally (local dev; Docker embeds them) |
| `make check-lint` | ruff |
| `make check-static-typing` | mypy `--strict` on `src/` (60 files) |
| `make check-deadcode` / `make check-depcheck` | vulture / deptry |
| `make check-all` | All four check gates |
| `make test-unit` | Unit tests (`tests/unit/`) |
| `make test-oracles` | Executable oracles: L1 canary, L2 property oracles, trace replay |
| `make test-mutations` | mutmut ratchet against `verify/mutmut-baseline.json` |
| `make verify-fizz` | Exhaustive FizzBee model checks (L4) |
| `make verify-ledger` / `make verify-cross-consistency` | Ledger + NL-doc consistency gates |
| `make benchmark-live` / `make benchmark-compare` / `make benchmark-selfcheck` | Benchmark harness (see above) |
| `make embed-cache` | Force-regenerate `data/dense-cache/` |
| `make docker-build-all` | Build all three images (dense cache baked in) |
| `make docker-up` / `make docker-down` | Start / stop the dev stack |
| `make client` / `make client-async` | Demo clients against `http://localhost:8062/mcp` |
| `make docker-publish` | Push the server image to Docker Hub + GHCR (+ git tag) |

**Development workflow:**

```bash
make install            # first-time setup
make check-all          # before pushing
make test-unit          # behavioral oracle
uv run mypy --strict    # src/ is strict; must exit 0
make docker-up          # after image-affecting changes
```

Ground rules (from `AGENTS.md`): `src/` is mypy-strict; schema changes change `docs/implementation-guide.md` + the schema + the code in the same commit; pipeline control-flow changes update the matching FizzBee model in the same PR.

---

## Publishing

All three images are published to two registries simultaneously:

| Image | Docker Hub | GHCR | Tags |
|---|---|---|---|
| MCP server | `olkowa/design-pattern-mcp` | `ghcr.io/olk/design-pattern-mcp` | `$(DOCKER_TAG)`, `latest` |
| TEI embedder | `olkowa/pattern-tei-embed` | `ghcr.io/olk/pattern-tei-embed` | `$(DOCKER_TAG)`, `latest` |
| TEI reranker | `olkowa/pattern-tei-rerank` | `ghcr.io/olk/pattern-tei-rerank` | `$(DOCKER_TAG)`, `latest` |

All three images share the same `$(DOCKER_TAG)` (the version from `pyproject.toml`). The TEI images are shared with the sibling `architecture-pattern-mcp` server and change rarely — blob deduplication keeps re-tagging unchanged TEI images cheap, and TEI pushes can be skipped when the sidecars are unchanged.

> **Bandwidth note:** the TEI embedder image is ~5 GB (ONNX fp32 weights baked in). First push to each registry is ~5 GB upload. Subsequent pushes are incremental — only changed layers are transferred.

### Prerequisites

**Docker Hub** — already authenticated locally (`docker login`).

**GitHub Container Registry** — requires a [classic PAT](https://github.com/settings/tokens/new?scopes=write:packages) with `write:packages` scope. 2FA is not an issue — PATs bypass it. After login the token is discarded; the credential persists in `~/.docker/config.json` until you log out.

### Publish (one-time setup + per-session)

```bash
# 1. Login to GHCR (interactive — paste token at the password prompt)
docker login ghcr.io -u olk

# 2. Build and push all three images (MCP + TEI embedder + TEI reranker)
#    The umbrella target builds the MCP image, tags it, pushes it, creates the git tag,
#    then builds and pushes each TEI image in sequence.
make docker-publish-all

# 3. Logout from GHCR immediately after publishing
#    (removes the ghcr.io credential from ~/.docker/config.json)
docker logout ghcr.io
```

On subsequent publishes repeat steps 1–3. If your PAT has expired, generate a new one at the link above. `make docker-publish` pushes only the MCP server image (plus the `v$(DOCKER_TAG)` git tag and the MCP Registry entry via `mcp-publisher`); `make docker-publish-tei` pushes only the two TEI sidecars.

### First push — set packages public (GHCR only)

GHCR packages default to **private**. After the first `make docker-publish-all`, flip all three packages to public:

| Package | Settings URL |
|---|---|
| MCP server | `https://github.com/users/olk/packages/container/design-pattern-mcp/settings` |
| TEI embedder | `https://github.com/users/olk/packages/container/pattern-tei-embed/settings` |
| TEI reranker | `https://github.com/users/olk/packages/container/pattern-tei-rerank/settings` |

Set each to **Public** and save.

### Partial failure recovery

If the push fails mid-way (e.g., GHCR auth was not configured), Docker Hub layers are already uploaded. After fixing auth, re-running `make docker-publish-all` is safe — each registry reports a cache hit for already-uploaded layers and completes the remaining push. For targeted retries, the TEI sidecars can be pushed with `make docker-publish-tei`.

---

## systemd Service (Linux)

The server can run as a systemd service on any systemd-based Linux host. It starts the Docker Compose stack automatically at boot. Full agent-runbook install: [INSTALL.md](INSTALL.md#path-b--systemd-stack-persistent-sudo-required) (Path B). `systemd/` contains:

| File | Purpose |
|---|---|
| `systemd/design-pattern-mcp.service` | The systemd unit |
| `systemd/docker-compose.yml` | Production compose variant (no `build:`, absolute paths, container port 8062) |
| `systemd/README.md` | Full runbook: install, verify, troubleshooting |

The production compose is the deployment variant of the dev file: pre-built images only, absolute paths, deployable to `/etc/design-pattern-mcp/`. The TEI sidecars live in the shared [pattern-tei-infra](https://github.com/olk/pattern-tei-infra) stack; this stack joins its Docker network and reaches them by DNS name. Start the service only after the shared TEI infra is `active (running)`.

```bash
# Fast path (see INSTALL.md Path B for the full verified sequence):
VERSION=$(grep -m 1 '^version' pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/')
docker pull "olkowa/design-pattern-mcp:${VERSION}"
docker tag "olkowa/design-pattern-mcp:${VERSION}" design-pattern-mcp:latest
sudo install -d /etc/design-pattern-mcp/config
sudo install -m 644 systemd/docker-compose.yml /etc/design-pattern-mcp/
sudo install -m 644 config/config.json /etc/design-pattern-mcp/config/
sudo install -m 644 systemd/design-pattern-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now design-pattern-mcp
```

Day-to-day: `systemctl start|stop|restart design-pattern-mcp`, logs via `journalctl -u design-pattern-mcp -f`. Full details in `systemd/README.md`.

---

## License

MIT License. Copyright (c) 2026 Oliver Kowalke. See the license headers in the source files.
