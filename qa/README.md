# QA pairs for decision-model finetuning

`qa.jsonl` — 169 question–answer pairs exported 2026-11-05 from
`tests/benchmark/scenarios/full.json` (see its `_meta` for authoring
provenance: LLM-drafted, agent-curated, **human expert review pending**).

One JSON object per line:

| field | content |
|---|---|
| `id` | scenario id (stable, matches the corpus) |
| `question` | `description` + `"\n\nDecision: "` + `decision_statement` (the model input) |
| `answer` | `expected.primary` (the ground-truth label) |
| `labels` | full label sets: `category`, `family`, `primary`, `acceptable_primary`, `acceptable_supporting` |
| `language` | requested language or null |
| `split` | `train` / `holdout` (family-disjoint; 139/30) |
| `notes` | review rationale for the labels |
| `source` | corpus path + sha12 + scenario id (regeneration provenance) |

Finetuning notes:

- Train on `split == "train"` rows only; hold out `holdout` for evaluation
  (families never cross the split — see the corpus `_meta`).
- `answer` is the primary label; `labels.acceptable_primary` holds the
  near-miss siblings a reviewer would also accept (soft-label source).
- Corpus contract: family-disjoint split, sibling near-miss labels, no
  pattern-name leakage in the question text.
