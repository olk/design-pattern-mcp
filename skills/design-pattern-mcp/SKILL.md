---
name: design-pattern-mcp
description: Selects design patterns for a described system via the design-pattern-mcp MCP server. Covers 486 patterns across 9 categories (domain-modeling, construction-composition, behavior-selection, interaction-boundary, workflow-messaging, persistence-state, concurrency-resilience, security-verification, specialized-components). Per-decision-part retrieval instead of domain-slug matching. Triggers on: select design patterns for X, which patterns handle concurrency/event-driven/retry/deadlock, improve testability of my code, which patterns are best for Python microservices.
---

# design-pattern-mcp

Design-pattern selection pipeline exposed by the `design-pattern-mcp` MCP server.

## OMP constraints

### 1. OMP is a short-timeout client → the async job trio is the default entry point

OMP applies one MCP request deadline, resolved as: `OMP_MCP_TIMEOUT_MS` env → `~/.omp/agent/mcp.json` `timeout` field (ms) → **30 000 ms default**.

`design-pattern-mcp` has no `timeout` entry in `~/.omp/agent/mcp.json`, so the 30 s deadline applies. Measured against `http://localhost:8822/mcp`:

| Call | Latency |
|------|---------|
| `submit_design_job` | <1 s |
| `get_design_status` | <1 s |
| `cancel_design` | <1 s |
| `list_design_patterns` / `get_design_pattern` | <1 s |
| `select_design_patterns` (full pipeline) | 2–10 min |
| `analyze_design` (analysis only) | ~1 min |

Use the job trio for the full pipeline. `analyze_design` also exceeds the 30 s deadline and has no job counterpart — raise the server timeout before calling it. See the device quick reference.

### 2. Tools are `xd://` devices

```
write  xd://mcp__design_pattern_local_<tool>   JSON args → tool result
read   xd://mcp__design_pattern_local_<tool>   device doc + input schema
```

Device prefix = server name lowercased with `-` → `_` (`design-pattern-mcp` → `mcp__design_pattern_local`). Invalid args echo the schema.

### 3. Job ids are the only durable handle

`submit_design_job` returns a `job_id` persisted in SQLite (`~/.config/design-pattern-mcp/jobs.db`, override `JOBS_DB`). Keep the `job_id` — write it to a scratch file.

## Entry-point decision guide

```
Existing code/design to score?
├── YES → evaluate_design (device mcp__design_pattern_local_evaluate_design)
└── NO
    ├── Explore the catalog first   → list_design_patterns (category?) → get_design_pattern(name)
    ├── Agent writes the code       → analyze_design → generate_design(analysis): selections,
    │   itself                        then bound applications (participants, steps, sketch,
    │                                 variant) — implement from the applications
    ├── Caller wants the design     → job trio: submit_design_job → get_design_status
    │                                 (full pipeline: analyze → generate → evaluate → refine)
    └── Known patterns              → generate_design (explicit pattern names)
```

## Device quick reference

| Device suffix | Purpose | Latency |
|---|---|---|
| `select_design_patterns` | Design path: full pipeline, one call | 2–10 min |
| `analyze_design` | Analysis-only: brief + per-part selections; follow with `generate_design` (same payload) for the implementation material (participants, steps, bindings) — the path when the agent writes the code itself | ~1 min + ~1 min |
| `generate_design` | Applications from a prior analyze_design payload or explicit pattern names | ~1 min |
| `evaluate_design` | Score a design JSON | ~30 s |
| `submit_design_job` | Start pipeline, returns `job_id` | <1 s |
| `get_design_status` | Poll job; `result` when `completed` | <1 s |
| `cancel_design` | Best-effort cancel | <1 s |
| `list_design_patterns` | Pattern list (name, category, description) | <1 s |
| `get_design_pattern` | Full pattern JSON | <1 s |

## Job trio recipe

1. **Submit** — `write xd://mcp__design_pattern_local_submit_design_job` with `{"description": "..."}`
2. **Persist the `job_id`**
3. **Poll** — `write xd://mcp__design_pattern_local_get_design_status` with `{"job_id": "..."}` every 10–30 s

| Status | Action |
|--------|--------|
| `pending` | poll again |
| `running` | poll again |
| `completed` | read `result` |
| `failed` | read `error` |
| `cancelled` | report |

4. **Cancel** — `write xd://mcp__design_pattern_local_cancel_design` with `{"job_id": "..."}`

## Result interpretation (`DesignResult` — see `docs/result-schema.json`)

- `overview.score` (0–1, blended selection quality): dominant pattern, supporting
  patterns, principles, reasoning in `overview`
- `applications[]`: one entry per decision part (participants, interactions,
  implementation steps, code sketch, language binding with `bound_variant`)
- `quality_attributes`, `test_strategy`, `risks[]`: evaluation axes and follow-ups
