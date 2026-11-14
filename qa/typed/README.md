# Typed labels for decision-model finetuning

3,914 typed labels derived from `tests/benchmark/scenarios/full.json`
(the 2,169 script-derived rows over the original 169 scenarios, plus 857 reviewed
style rows and 857 mirrored choice rows): the noul labels are script-derived from
the 169 original scenarios (no new review burden); the style labels are per-scenario
reviewed judgments with a quoted rationale each
(`scripts/decision_models/derive_style_labels.py`).
Corpus labels are agent-curated, **human expert review pending**.

The 2026-11 style-label gap work (§4.7.1 of `docs/tacit-fine-tune.md`) took the
corpus from 169 to 857 scenarios: 688 candidates were labelled by **two blind
model annotators** under `qa/typed/style-rubric.md` with opaque row ids, Cohen's
kappa recorded per batch (0.978-0.993 train, 0.981 and 1.000 holdout), and every
disagreement adjudicated in writing (each adjudicated row carries the adjudicator's reason in
`review.json`). Candidate selection was **stratified by served class with a per-class floor**, not
uncertainty-sampled: the plan's least-confidence ranking needs a served base checkpoint, and no
gateway was reachable from the workspace that built this corpus, so class coverage is guaranteed
by construction instead (`style_label_gap.py`'s pool report records the method). `review.json` certifies all 857 labels at the
label source's current bytes, and `human_review` now records the project owner's acceptance of
a 47-row audit sheet spanning all 36 labels (sampled, not a row-by-row expert review — the block
says so explicitly).

## Files

| file | n | shape | seam |
|---|---|---|---|
| `choice.jsonl` | 857 | `state` + 9 `criteria` + `label` (category) | J2 category |
| `style.jsonl` | 857 | `state` + 36 `criteria` + `label` (style) | Ch2 style |
| `noul.jsonl` | 2000 | `state` + `proposition` + boolean `label` (+ `candidate_text`) | J1 per-candidate relevance |
| `rerank_pairs.jsonl` | 2000 | `statement` + `document` (+ `gain`, `split`, `name`) | J1 serving bytes (train + eval) |
| `review.json` | — | per-scenario human review ledger | Ch2 enable-path gate |

- Choice: 1 per scenario, criteria = the 9 catalog categories with a one-line
  gloss each, label = the scenario's category.
- Style: 1 per scenario, criteria/instructions = the served Ch2 vocabulary verbatim
  (`src/decisions/seam_prompts.py` shared vocabulary), `state` = the raw scenario
  description bytes the seam receives, label = the reviewed dominant
  architecture style with a quoted rationale. Train split: 719 rows over all 36
  criterion keys — every served style has at least 13 train positives, so no
  class is starved and the 2026-11 audit exception is closed: `api-gateway` lost
  its only two train labels when that review corrected them to monolithic, and
  the gap work added 25 reviewed gateway train rows (Spring Cloud Gateway and
  Kong routes, an MCP gateway endpoint, and deliberately negative rows that are
  NOT gateways). `tests/unit/test_benchmark_draft_style.py` fails if the
  uncovered-styles set, the 500-label train floor, the per-class floor or the
  100-row holdout floor regresses. The 138 holdout rows are labeled for eval only
  and never reach training weights.
  Review state is machine-readable: `review.json` records per-scenario review
  (reviewer identity, label, `labels_sha256`, `rationale_ok`) plus a
  `human_review` block — a model review pass is recorded as such, never as
  expert sign-off — and `make style-head-calibrate` refuses to recommend a confidence
  floor until the ledger certifies the label source byte-for-byte — no calibrated
  floor exists for unreviewed labels.
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

Train = 2,258, holdout = 768 (choice + style: 719/138 each; noul: 1,400/600).
Families never cross the split. Train on `source.split == "train"`; fit thresholds
/ temperature on held-out **families**, not training rows (recommendation §6.6).

The style split is the §4.7.1 instrument: 719 train labels (>= the 500 floor, every
class >= 13 rows) and 138 family-disjoint holdout rows (>= the 100 floor, every class
>= 2 rows) so a confidence floor is measured over the whole served vocabulary
instead of the 11 classes the 30-row holdout happened to contain.

The rerank pairs keep the same rule: `tacit_self_distill.load_rerank_pairs` reads
`split == "train"` rows only, so the holdout pairs can never reach the weights.

## Recipe fit

Per `docs/recommendation_design-pattern-mcp_decision-models.md` §4/§7.5:

- **AnyJev Tacit-4B** serves every seam and is trained by self-distillation
  (`scripts/decision_models/tacit_self_distill.py`), which needs no human labels:
  the released checkpoint writes states, answers its own questions with thinking,
  and a QLoRA student reproduces the label distribution in one forward. The
  training problems are the served questions — category, weights and gate from
  `build_problems`, and the J1 `yes_no` items from `rerank_pairs.jsonl`. These typed
  labels are therefore **evaluation and calibration** data for the Tacit seams, and
  **training** data for the linear Ch2 style head: `make style-head-calibrate` replays
  the holdout style rows through the served `/v1/systemone` path to produce the seam's
  floor, and the holdout rerank pairs score the leg (nDCG) without touching the weights.
- **Eikos-4B** (LoRA r64), **Clef-flash-9B** (QLoRA), **Kev-4B**
  (`kev.train --init_from`) each want 500–2,000 labeled decisions — this
  export (2,258 train) is in budget. Kev's data shape is the API shape plus
  `label`, exactly these records.

Label tiers double as sample weights: `primary` > `acceptable_primary` >
`supporting` > `negative`.
