# Typed labels for decision-model finetuning

2,000 typed labels derived 2026-10-27 from `tests/benchmark/scenarios/full.json`
(no new LLM drafting — every label is script-derived from the 100 reviewed
scenarios, so no new review burden). Corpus labels are agent-curated,
**human expert review pending**.

## Files

| file | n | shape | seam |
|---|---|---|---|
| `choice.jsonl` | 100 | `state` + 9 `criteria` + `label` (category) | J2 category |
| `noul.jsonl` | 1900 | `state` + `proposition` + boolean `label` | J1 per-candidate relevance |

- Choice: 1 per scenario, criteria = the 9 catalog categories with a one-line
  gloss each (same `_STYLE_GLOSSES` idea as the style seam), label = the
  scenario's category.
- Noul: 19 per scenario — every labeled pattern is a positive
  (`label_tier`: `primary` / `acceptable_primary` / `supporting`), padded
  with cross-category negatives (`label_tier: negative`, deterministic
  per-scenario seed) for a fixed 19. Verdict: 307 true / 1593 false.
  `candidate_text` is the compact catalog evidence (`name: intent` + problem
  excerpt) — the CLM action text / J1 compact evidence; `proposition` embeds
  it so single-text trainers also see description alignment, not just names.

## Split

Train = 1,400, holdout = 600. Families never cross the split. Train on
`source.split == "train"`; fit thresholds / temperature on held-out
**families**, not training rows (recommendation §6.6).

## Recipe fit

Per `docs/recommendation_design-pattern-mcp_decision-models.md` §4/§7.5:

- **AnyJev Tacit**: needs zero labels (self-distilled) — use these as
  *evaluation* data; only a later QLoRA of the Qwen3-8B base (500–2,000
  labeled decisions) would train on them.
- **CLM-8B**: heads-only `finetune.py --task clm --init-ckpt`, wants 1,000+
  state/candidate pairs — the 1,900 noul rows (1,330 train) are that shape.
- **Eikos-4B** (LoRA r64), **Clef-flash-9B** (QLoRA), **Kev-4B**
  (`kev.train --init_from`) each want 500–2,000 labeled decisions — this
  export (1,400 train) is in budget. Kev's data shape is the API shape plus
  `label`, exactly these records.

Label tiers double as sample weights: `primary` > `acceptable_primary` >
`supporting` > `negative`.
