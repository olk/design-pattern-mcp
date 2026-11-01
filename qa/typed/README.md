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
| `rank_payloads.jsonl` | 30 | `context` + `question` + 20 served `answers` (+ `answer_names`) | J1 rank-parity gate |

- Choice: 1 per scenario, criteria = the 9 catalog categories with a one-line
  gloss each (same `STYLE_GLOSSES` idea as the style seam), label = the
  scenario's category.
- Style: 1 per scenario, criteria/instructions = the served Ch2 prompt verbatim
  (`src/decisions/seam_prompts.py` shared vocabulary), `state` = the raw scenario
  description bytes the seam receives, label = the reviewed dominant
  architecture style with a quoted rationale. Train split: 139 rows over all 36
  criterion keys — every served style has at least one train positive (the 69
  train-only style-exemplar scenarios added 2026-11 closed the last 23 gaps;
  largest classes are monolithic 20, none 11, layered-monolith 10); the 30
  holdout rows are labeled for eval only and never reach training weights.
  Review state is machine-readable: `review.json` (currently `unreviewed`)
  records per-scenario human certification, and `make clm-calibrate` refuses to
  recommend a confidence floor until the ledger certifies the label source
  byte-for-byte — no calibrated floor exists for unreviewed labels.
- Noul: 20 per scenario over the 100 scenarios authored 2026-10-27 (the 2026-11
  style-exemplar scenarios added no noul rows) — every labeled pattern is a
  positive (`label_tier`: `primary` / `acceptable_primary` / `supporting`),
  padded with cross-category negatives (`label_tier: negative`, deterministic
  per-scenario seed) for a fixed 20. Verdict: 307 true / 1693 false.
  `candidate_text` is the compact catalog evidence (`name: intent` + problem
  excerpt) — the `--task clm` action text / J1 compact evidence; `proposition`
  embeds it so single-text trainers also see description alignment, not just names.
  It is **not** the served `/v1/rank` answer text, which is each record's
  `document_text` (`src/patterns/loader.py:210`).
- Rank payloads: one per holdout scenario, in the exact `CLMReranker._request`
  shape — `context` empty, `question` = the scenario's `decision_statement`,
  `answers` = the 20 candidate records' `document_text` in noul file order.
  Derived by `scripts/decision_models/derive_rank_payloads.py` (`make clm-payloads`)
  because the CLM image has no pattern catalog; regenerate it after any corpus or
  catalog change (the CLM cache marker pins it).

## Split

Train = 1,678, holdout = 660 (choice + style: 139/30 each; noul: 1,400/600).
Families never cross the split. Train on `source.split == "train"`; fit thresholds
/ temperature on held-out **families**, not training rows (recommendation §6.6).

That is also how the CLM export is sharded: `export_clm_dataset.py` writes the
`train` rows to the trainer's train shard and the `holdout` rows to its **test**
shard (30 rows per question), so the build gate scores the head on families it
never trained on. `finetune.py` reads the test shard only after checkpoint
selection, so those rows cannot reach the weights.

## Recipe fit

Per `docs/recommendation_design-pattern-mcp_decision-models.md` §4/§7.5:

- **AnyJev Tacit**: needs zero labels (self-distilled) — use these as
  *evaluation* data; only a later QLoRA of the Qwen3-8B base (500–2,000
  labeled decisions) would train on them.
- **CLM-8B**: heads-only `finetune.py --task choice --init-ckpt` over the 278
  train rows (139 category + 139 style — `scripts/decision_models/export_clm_dataset.py`
  to the typed-decisions parquet layout). The style set is the exact question the
  Ch2 seam serves; the 2,000 noul rows (1,400 train) remain the `--task clm`
  rank-follow-up shape: `state` is the CLM state text, `candidate_text` the action text.
  The build gates the result on the holdout shard: `clm_eval_gate.py` (test ≥ init per
  served question, within one row of granularity, above the label prior, every
  test label supported by a train positive — R6 — and every train-uncovered
  criterion explicitly acknowledged in `review.json` — R7) plus `clm_rank_parity.py`
  (released-vs-trained top-1 within budget AND mean nDCG@10 no worse than the
  released head, scored on the `relevance` grades carried in `rank_payloads.jsonl`;
  the 0.25 flip budget is asserted policy, not data-derived — reset it from the
  first head's parity report + a rank A/B). Category training stays unreviewed
  (R8 advisory — no served CLM consumer asks it). The shipped head is
  **uncalibrated** until `make clm-calibrate` produces a floor (override-stratified
  acted accuracy with 95% Wilson bands and a minimum acted count, gated on
  `review.json` certification); the compose overlay keeps `DECISIONS_STYLE_ENABLED=false`
  until then.
- **Eikos-4B** (LoRA r64), **Clef-flash-9B** (QLoRA), **Kev-4B**
  (`kev.train --init_from`) each want 500–2,000 labeled decisions — this
  export (1,678 train) is in budget. Kev's data shape is the API shape plus
  `label`, exactly these records.

Label tiers double as sample weights: `primary` > `acceptable_primary` >
`supporting` > `negative`.
