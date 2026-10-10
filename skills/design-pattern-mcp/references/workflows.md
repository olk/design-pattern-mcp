# Design Pattern workflows

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

1. `list_design_patterns({"category": "concurrency"})`
2. `get_design_pattern({"name": "producer-consumer"})`
3. `generate_design({"description": "...", "selected_patterns": ["producer-consumer"]})`
