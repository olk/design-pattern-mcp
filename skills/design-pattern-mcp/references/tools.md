# Design Pattern tools

## Core tools

### `select_design_patterns`
Start a full design-pattern selection job.

**Args:** `{"description": "..."}`

**Returns:** `{"job_id": "..."}`

### `get_design_status`
Poll job status or retrieve result.

**Args:** `{"job_id": "..."}`

**Returns:** `{"status": "pending|running|completed|failed|cancelled", "result": {...}}`

### `cancel_design`
Cancel a running job.

**Args:** `{"job_id": "..."}`

### `analyze_design`
Analyze requirements and recommend patterns.

**Args:** `{"description": "..."}`

### `generate_design`
Generate design from selected patterns.

**Args:** `{"description": "...", "selected_patterns": [...]}`

### `evaluate_design`
Evaluate an existing design.

**Args:** `{"design": {...}}`

## Catalog tools

### `list_design_patterns`
List patterns, optionally filtered by category.

**Args:** `{"category"?: str}`

**Returns:** `[{name, category, description}]`

### `get_design_pattern`
Get full pattern JSON by name.

**Args:** `{"name": "..."}`

## Error codes

- `PatternNotFound`: unknown pattern name
- `JobNotFound`: unknown job_id
- `InvalidCategory`: unknown category name
