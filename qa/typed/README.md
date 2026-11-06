# Typed labels for decision-model finetuning

2,338 typed labels derived from `tests/benchmark/scenarios/full.json`
(the 2,169 script-derived rows plus 169 reviewed style rows): the choice/noul
labels are script-derived from the 169 reviewed scenarios (no new review burden);
the style labels are per-scenario reviewed judgments with a quoted rationale each
(`scripts/decision_models/derive_style_labels.py`).
Corpus labels are agent-curated, **human expert review pending**.

## Files

| file | n | shape | seam |
|---|---|---|---|
| `choice.jsonl` | 169 | `state` + 9 `criteria` + `label` (category) | J2 category |
| `style.jsonl` | 169 | `state` + 36 `criteria` + `label` (style) | Ch2 style |
| `noul.jsonl` | 2000 | `state` + `proposition` + boolean `label` (+ `candidate_text`) | J1 per-candidate relevance |
| `rerank_pairs.jsonl` | 2000 | `statement` + `document` (+ `gain`, `split`, `name`) | J1 serving bytes (train + eval) |
| `review.json` | — | per-scenario human review ledger | Ch2 enable-path gate |

- Choice: 1 per scenario, criteria = the 9 catalog categories with a one-line
  gloss each, label = the scenario's category.
- Style: 1 per scenario, criteria/instructions = the served Ch2 vocabulary verbatim
  (`src/decisions/seam_prompts.py` shared vocabulary), `state` = the raw scenario
  description bytes the seam receives, label = the reviewed dominant
  architecture style with a quoted rationale. Train split: 139 rows over all 36
  criterion keys — every served style has at least one train positive (the 69
  train-only style-exemplar scenarios added 2026-11 closed the last 23 gaps;
  largest classes are monolithic 20, none 11, layered-monolith 10); the 30
  holdout rows are labeled for eval only and never reach training weights.
  Review state is machine-readable: `review.json` (currently `unreviewed`)
  records per-scenario human certification, and `make tacit-calibrate` refuses to
  recommend a confidence floor until the ledger certifies the label source
  byte-for-byte — no calibrated floor exists for unreviewed labels.
- Noul: 20 per scenario over the 100 scenarios authored 2026-10-27 (the 2026-11
  style-exemplar scenarios added no noul rows) — every labeled pattern is a
  positive (`label_tier`: `primary` / `acceptable_primary` / `supporting`),
  padded with cross-category negatives (`label_tier: negative`, deterministic
  per-scenario seed) for a fixed 20. Verdict: 307 true / 1693 false.
  `candidate_text` is the compact catalog evidence (`name: intent` + problem
  excerpt); `proposition` embeds it so single-text trainers also see description
  alignment, not just names. It is **not** the served rerank text, which is each
  record's `document_text` (`src/patterns/loader.py`).
- Rerank pairs: one row per (scenario, candidate) over **both** splits — the
  `decision_statement` the pipeline serves plus the candidate record's
  `document_text`, i.e. exactly one stage-2 `yes_no` item
  (`seam_prompts.rerank_state`). `train` rows (1,400) feed
  `tacit_self_distill.py --rerank-pairs`; `holdout` rows (600) are eval material
  only. `gain` is the reviewed `label_tier` as a TREC graded-relevance step
  (primary 3 / acceptable_primary 2 / supporting 1 / negative 0) for nDCG.
  Derived by `scripts/decision_models/derive_rerank_pairs.py` (`make tacit-payloads`)
  because the Tacit image has no pattern catalog; regenerate after any corpus,
  noul or catalog change (`make tacit-payloads-check` byte-compares, and the Tacit
  cache marker pins it).

## Split

Train = 1,678, holdout = 660 (choice + style: 139/30 each; noul: 1,400/600).
Families never cross the split. Train on `source.split == "train"`; fit thresholds
/ temperature on held-out **families**, not training rows (recommendation §6.6).

The rerank pairs keep the same rule: `tacit_self_distill.load_rerank_pairs` reads
`split == "train"` rows only, so the holdout pairs can never reach the weights.

## Recipe fit

Per `docs/recommendation_design-pattern-mcp_decision-models.md` §4/§7.5:

- **AnyJev Tacit-4B** serves every seam and is trained by self-distillation
  (`scripts/decision_models/tacit_self_distill.py`), which needs no human labels:
  the released checkpoint writes states, answers its own questions with thinking,
  and a QLoRA student reproduces the label distribution in one forward. The
  training problems are the served questions — category, weights and gate from
  `build_problems`, the Ch2 style shards from `build_style_problems`, and the
  J1 `yes_no` items from `rerank_pairs.jsonl`. These typed labels are therefore
  **evaluation and calibration** data: `make tacit-calibrate` replays the holdout
  style rows through the served sharded path to produce the seam's floor, and the
  holdout rerank pairs score the leg (nDCG) without touching the weights.
- **Eikos-4B** (LoRA r64), **Clef-flash-9B** (QLoRA), **Kev-4B**
  (`kev.train --init_from`) each want 500–2,000 labeled decisions — this
  export (1,678 train) is in budget. Kev's data shape is the API shape plus
  `label`, exactly these records.

Label tiers double as sample weights: `primary` > `acceptable_primary` >
`supporting` > `negative`.
