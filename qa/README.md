# QA pairs for decision-model finetuning

`qa.jsonl` — 100 question–answer pairs exported 2026-10-27 from
`tests/benchmark/scenarios/full.json` (see its `_meta` for authoring
provenance: LLM-drafted, agent-curated, **human expert review pending**).

One JSON object per line:

| field | content |
|---|---|
| `id` | scenario id (stable, matches the corpus) |
| `question` | `description` + `decision_statement` (the model input) |
| `answer` | `expected.primary` (the ground-truth label) |
| `labels` | full label sets: `category`, `family`, `primary`, `acceptable_primary`, `acceptable_supporting` |
| `language` | requested language or null |
| `split` | `train` / `holdout` (family-disjoint; 70/30) |
| `notes` | review rationale for the labels |
| `source` | corpus path + sha12 + scenario id (regeneration provenance) |

Finetuning notes:

- Train on `split == "train"` rows only; hold out `holdout` for evaluation
  (families never cross the split — see the corpus `_meta`).
- `answer` is the primary label; `labels.acceptable_primary` holds the
  near-miss siblings a reviewer would also accept (soft-label source).
- Label budget gate (`docs/recommendation_design-pattern-mcp_decision-models.md`
  §7.5): do not fine-tune before the label budget and a measured gap exist —
  <200 labels means prompting, 200–1,000 is possible with review, and the
  AnyJev/CLM/Eikos/Clef/Kev recipes want 500–2,000 labeled decisions. These
  100 pairs (still awaiting human expert review) are seed data to extend
  toward that budget, not a training set for a 4B–9B model.
- Per `docs/recommendation_design-pattern-mcp_decision-models.md` §1.5, decision
  models belong only in bounded judgments (category, per-candidate relevance,
  weights, defer) — never as the final pattern selector.
- Corpus contract: family-disjoint split, sibling near-miss labels, no
  pattern-name leakage in the question text.
