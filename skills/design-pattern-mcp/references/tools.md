# Design Pattern tools (9 tools)

## Pipeline tools (blocking, heartbeat-defended)

### `select_design_patterns`
Design path for callers that want the generated design: full pipeline (analyze →
weights ∥ retrieval → generate → evaluate → bounded refine), returning
applications with participants, implementation steps, code sketches and bound
variants, plus quality attributes, a test strategy and risks. Long-running
(2–10 min) — prefer the job trio under short timeouts.

**Args:** `{"description": "...", "architecture_style"?: str, "language"?: str, "paradigm"?: str, ...}`

**Returns:** `DesignResult` (`overview`, `applications[]`, ... — see `docs/result-schema.json`)

### `analyze_design`
Analysis-only path for coding agents that write the code themselves: the same
ANALYZE phase (§1–§8) as `select_design_patterns`, stopping before generation
and evaluation. Returns the repaired brief, the decision parts and one selection
per part (primary, supporting patterns with roles, ranked alternatives, veto
ledger, weights); no `applications[]`.

**Args:** `{"description": "...", "architecture_style"?: str, "language"?: str, "paradigm"?: str, "focus"?: str}`

**Returns:** `AnalysisResult` (`design_brief`, `decision_parts[]`, `selections{}`)
plus `selected_patterns` (deduplicated primaries) and `categories`. The selection carries no
implementation material: call `generate_design` with this payload to materialize the
applications (participants, steps, code sketches, bound variants) before implementing;
`get_design_pattern` returns the catalog record when generation is not wanted.

### `generate_design`
Materialize applications from a prior `analyze_design` payload or explicit
pattern names.

**Args:** `{"analysis"?: {...}}` or `{"description": "...", "patterns": [...], "language"?: str}`

### `evaluate_design`
Evaluate an existing design.

**Args:** `{"design": {...}}`

## Async job trio (OMP default, each <1 s)

### `submit_design_job`
Start the full pipeline in the background.

**Args:** `{"description": "...", ...}`

**Returns:** `{"job_id": "..."}`

### `get_design_status`
Poll job status or retrieve result.

**Args:** `{"job_id": "..."}`

**Returns:** `{"status": "pending|running|completed|failed|cancelled", "result"?: {...}}`

### `cancel_design`
Best-effort cancel of a pending/running job.

**Args:** `{"job_id": "..."}`

**Returns:** job record with terminal `cancelled` status (or prior terminal state).

## Catalog tools

### `list_design_patterns`
List patterns, optionally filtered by category and/or language.

**Args:** `{"category"?: str, "language"?: str}` — category must be a
`PatternCategory` value (e.g. `concurrency-resilience`); unknown values return `[]`.

**Returns:** `[{name, category, description}]`

### `get_design_pattern`
Get full pattern JSON by name.

**Args:** `{"name": "..."}`

## Resources

- `design-patterns://variants` — per-(pattern, language) `variant_id` bindings
  bound as `bound_variant` on `applications[]` language bindings.

## Error codes

- `PatternNotFound`: unknown pattern name
- `JobNotFound`: unknown job_id
- `InvalidCategory`: unknown category name
