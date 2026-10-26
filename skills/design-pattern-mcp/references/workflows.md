# Design Pattern workflows

## Analysis-only (the agent writes the code)

```
analyze_design → brief + decision parts + per-part selections
```

1. `analyze_design({"description": "...", "language": "python"})` — costs the ANALYZE phase only,
   no generation or evaluation
2. Implement from `decision_parts[]` + `selections[]` (primary, supporting, alternatives, vetoes,
   weights); `selected_patterns` lists the deduplicated primaries
3. Want the generated applications after all? `generate_design({"analysis": <payload>})`

## Job trio (OMP default)

```
submit_design_job → poll get_design_status → read result
```

1. `submit_design_job({"description": "..."})` → `{"job_id": "<uuid>"}`
2. Poll `get_design_status({"job_id": "..."})` every 10–30 s
3. When `status: "completed"`, read `result`

## Raised-timeout one-shot

Set `timeout: 0` in `~/.omp/agent/mcp.json` for `design-pattern-mcp`, then use `select_design_patterns` directly.

## Catalog → generate

1. `list_design_patterns({"category": "concurrency-resilience"})`
2. `get_design_pattern({"name": "producer-consumer"})`
3. `generate_design({"description": "...", "patterns": ["producer-consumer"]})`
