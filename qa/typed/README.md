# Typed labels for the Ch2 style seam

857 reviewed style rows: one per scenario of `tests/benchmark/scenarios/full.json` (the corpus grew from 169 to 857 scenarios in the 2026-11 style-label gap work). Each row is a per-scenario reviewed judgment with a quoted rationale (`scripts/decision_models/derive_style_labels.py`). The label source is agent-curated; `review.json` records the review state (two-blind-annotator pass plus adjudication, and the project owner's sampled audit sheet).

The 2026-11 style-label gap work (the 2026-11 gap plan, since deleted — see git history) took the
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

| file | n | shape | use |
|---|---|---|---|
| `style.jsonl` | 857 | `state` + 36 `criteria` + `label` (style) | Ch2 style: head training and floor calibration |
| `review.json` | — | per-scenario review ledger | Ch2 enable-path gate |

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

## Split

Train = 719, holdout = 138. Families never cross the split. Train on `source.split == "train"`; fit thresholds / temperature on held-out **families**, not training rows. The style split is the enable-path instrument: 719 train labels (>= the 500 floor, every class >= 13 rows) and 138 family-disjoint holdout rows (>= the 100 floor, every class >= 2 rows), so a confidence floor is measured over the whole served vocabulary.

## Ledger re-stamp (2026-11-26)

`scripts/decision_models/derive_style_labels.py` changed in a comment-only edit (three stale references removed: two to the deleted `qa/typed/choice.jsonl`, one to a script that no longer exists). `review.json` was re-stamped to the new sha256 with `style_label_review.py certify --carry-forward`: the tool's documented semantics — a carried entry is a claim about the label, not about the file bytes — so every one of the 857 verdicts was carried unchanged. This is a re-stamp, **not a re-review**; `models/style-head-calibration.json` keeps the digest it recorded at calibration time (`8b1bc3313211`).
