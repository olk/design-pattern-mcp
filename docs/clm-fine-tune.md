# CLM-8B fine-tune for the Ch2 architecture-style seam

The Ch2 seam samples `brief.architecture_style` from the A+B LLM call (entropy channel 2,
`docs/SELECTION_VARIABILITY_ANALYSIS.md` §2.2). Two decision models have been proposed to
replace that sample. This document is the CLM lane: **heads-only fine-tune of CLM-8B as a
36-way `choice`** (35 styles + `none`), served over `POST /v1/systemone`.

The Tacit lane and its verdict are in `docs/tacit-fine-tune.md`. That arm is measured AND
closed: gold supervision bought fitting, not generalization (§4.7.1). This document is not a
re-run of that — it is a different instrument on the same seam, with its own gate numbers.

## 1. Why CLM is a different bet than Tacit

`STYLE_VOCABULARY` has 35 styles. AnyJev's `choice` wire caps options at 26, so the Tacit
lane answers the seam as TWO questions (category → ≤26-style shortlist) and keeps the most
confident named style across the shards. That shape has two costs the CLM lane does not pay:

- **Cross-shard comparison.** A softmax over 18 options concentrates more mass than one over
  17, so the two answers are not on one scale and the floor is fitted through that seam.
- **Two chances to abstain.** `none` rides in exactly one shard, so a confident wrong
  category question can lose the row before the shortlist is even asked.

CLM embeds each criterion independently against the state, so the seam is **one unsharded
36-criterion question**: one softmax, one abstention candidate, no shard arithmetic. The
measured Tacit failure (0.442 holdout accuracy, `none` sub-skill 0.293) is not evidence about
this instrument.

## 2. Corpus

`qa/typed/style.jsonl`, 857 rows, audited and certified (`qa/typed/review.json`,
`status: reviewed`).

| split | rows | classes | purpose |
|---|---|---|---|
| `train` | 719 | 36, min 13 positives (`event-sourcing`), max 53 (`monolithic`) | head fine-tune |
| `holdout` | 138 | 36, min 2 (`edge-computing`, `enterprise-service-bus`, `modular-monolith`) | generalization instrument |

Family-disjoint, and now MEASURED rather than asserted. The corpus carries a `family` field;
the split is 35 train-only families and 17 holdout-only families with **zero straddling**, so
the holdout scenarios belong to families the head never sees. Zero shared
`source.scenario_id` across the splits, and no paraphrase leakage either: the highest 5-gram
Jaccard between any of the 99 222 train×holdout description pairs is **0.148** (none reaches
0.60, let alone the 0.80 that would flag a paraphrase), and there are no exact duplicate
descriptions. The exporter loads exactly one split and refuses the other. The val slice that
drives checkpoint selection is carved from train ids only (10 %, `random.Random(1234)`), so
the holdout families never reach a weight or a checkpoint choice.

The category question (`qa/typed/choice.jsonl`, 719/138) is exported alongside and trained,
but it is **advisory, never gating** — no CLM client asks it on this branch; J2 stays on
Tacit. Its coverage is deliberately reported and *not* asserted (`specialized-components` and
`workflow-messaging` have zero train positives).

## 3. Recipe

Pinned from `train/finetune.py` at CLM pin `d5f9ef0`, heads only; encoder
`Qwen/Qwen3-8B` frozen (`--max-len 2048`, last-token pooled, no chat template — CLM scores,
it never generates).

```
--task choice --workflow all --targets hard --loss infonce \
--init-ckpt CLM_v0.1-8B.pt --max-len 2048 --batch 32
```

Everything else is the upstream default this build inherits: lr `5e-4` (the `choice` branch's
`args.lr or 5e-4`), epochs `20` with early stop `patience 5` on val accuracy, `val-frac 0.1`,
seed `1234`, weight-decay `0.0`, AdamW, `OneCycleLR(pct_start=0.1, anneal_strategy="cos")`,
grad-norm clip `1.0`. Head geometry (width 1536 / depth 3 / proj 512) comes from the released
checkpoint. No LoRA, no quantization, no gradient accumulation: `state_head` + `action_head` +
`logit_scale` are ~40 M trainable parameters and the embeddings are precomputed behind a
sha1-keyed `TextCache`, so there is nothing for accumulation to buy.

**Why `--targets hard` is passed explicitly.** Both `--targets soft` and `--loss infonce` are
the upstream defaults, and on this corpus the soft path is a no-op: gold rows carry `label`
only (no `probabilities`), so `train/adapters.py::typed_decision_examples` computes
`z = sum([]) = 0` and falls to `[float(k == label)]`. Verified by reading the exported shard,
not by assumption. Stating `hard` makes the intent auditable — a default is not a statement
about our data. If the exporter ever emits graded targets, this flag is the thing that has to
change deliberately.

## 4. Serving contract

One `POST {base_url}/v1/systemone`:

```json
{"model": "clm-latest",
 "state": "<description>",
 "questions": {"architecture_style": {"type": "choice",
                                      "instructions": STYLE_INSTRUCTIONS,
                                      "criteria": {"<style>": "<gloss>", ..., "none": "<none gloss>"}}}}
```

`criteria` is keyed and ORDERED by `STYLE_VOCABULARY` with `none` last
(`src/decisions/client.py::build_style_criteria`). It is derived from the vocabulary, not from
`src/style_affinity.py::STYLE_AFFINITY` — that table is the style→category PRIOR, and reading
its order here coupled the served bytes to a table that exists for another purpose.

Byte parity is enforced, not assumed: `test_style_corpus_matches_the_served_prompt_bytes`
compares `list(row["criteria"]) == list(build_style_criteria())` against **every** row of
`qa/typed/style.jsonl`. Ordered comparison matters — `==` on dicts ignores insertion order,
so a re-ordering would pass dict equality while serving the head a serialisation it never
trained on.

`confidence` is TypeSafe-style (top probability − mean of the rest, `src/clm/schema.py` at the
pin), NOT the top probability. `--metric` replays the floor table under either definition.

## 5. Gates

Every gate runs inside the image build, so a candidate that cannot pass never becomes an image.

| gate | where | rule |
|---|---|---|
| R0–R8 eval gate | `clm_eval_gate.py` (build `[4/6]`) | no regression vs released heads on `architecture_style`; one flipped row tolerated (1/n granularity); R5 majority-prior beat; R6 every test label needs a train positive; R7 `uncovered` criteria must be acknowledged |
| rank parity | `clm_rank_parity.py` (build `[5/6]`) | released-vs-trained top-1 flip ≤ `0.25` over the 30 holdout payloads, plus mean top-5 overlap |
| coverage | `export_clm_dataset.py` manifest | per-question counts and label histogram; measured `architecture_style` train 719 / test 138 |
| pin | `clm_preflight.py` | exporter's declared pin == `--expect-pin` == the Dockerfile's `CLM_REPO_PIN` |

The rank gate exists because there is no separate rank head to leave untouched: `/v1/rank` is
served by the SAME `state_head`/`action_head`/`logit_scale` triple that `--task choice` trains
(`src/clm/server.py::rank` → `engine.rank` builds a synthetic choice question and answers it
with the loaded `HeadPair`). Training moves the rank surface by construction; the gate bounds
it.

Both gates' INPUTS were rehearsed on the real corpus on the workstation, so a FAIL at `[4/6]`
or `[5/6]` on the host attributes to the trained head, not to the data:

- `clm_eval_gate.py` against the real `manifest.json` (138 rows per question) reports
  `R6-support: all 36 test labels have train positives` and `R7-coverage: every criterion has
  a train positive` — no acknowledgment is needed, and the R5 majority share is `0.0942`
  (13/138 `none`), so the 0.65 accuracy target clears the prior bar by a wide margin.
  `R8-choice-review` reports the category training as uncertified — advisory, non-blocking,
  and correctly so: no served CLM consumer asks that question.
- `clm_rank_parity.load_payloads` accepts all 30 committed payloads (20 answers each,
  relevance aligned, at least one positive gain, empty context).

## 6. Selection-consistency probes

A floor fitted on a head that does not answer the same question the same way twice is a
threshold on noise. `clm_calibrate.py` runs two probes beside the floor table:

- `--repeats N` replays the same bytes N times. Serving is one temperature-0 forward per
  question, so every row's choice MUST be identical across replays; the recommendation is
  refused when agreement drops below `MIN_REPEAT_AGREEMENT = 0.99`. That number is the
  seam's own `C_repeat` target, not 1.0, deliberately: with 138 holdout rows 0.99 tolerates
  one flipped row, and a script stricter than the gate it feeds would report FAIL on an arm
  the objective calls PASS — two contradicting verdicts on one measurement. On a
  temperature-0 head the expected flip count is 0, so any flip is still listed in
  `consistency.unstable_rows`. Reported: `style_choice_repeat_agreement`, the per-floor
  acted-set flips, and the max confidence spread.
- `--permutations K` replays each row K times under a seeded random ordering of its same
  36-criterion map. The criteria are an unordered answer set, so the argmax must not depend on
  the order they were written in. Reported: `style_choice_order_agreement` (the gated number)
  and `canonical_agreement` (drift from the served order, i.e. a head that is order-stable but
  systematically wrong). NOT gated in the script — order invariance is a property of the head
  measured for the arm comparison, whereas a repeat flip is a determinism defect.

The report also pins **which artifact produced these numbers** (`served`): container name,
image ID, requested tag, container start time and the heads repo. A docker TAG is not a
served artifact — a long-running container survives the tag being moved onto a newer image,
which is exactly how an earlier Tacit calibration measured a stale checkpoint — so the image
ID and start time travel with the measurement instead of being re-derived afterwards.

Each report also carries **every row's decision** (`rows`: scenario id, gold label, chosen
criterion, confidence). The E1/E2 comparison is paired per row — which rows flipped, per-class
deltas — and that cannot be reconstructed from aggregate accuracies, so the decisions travel
with the report rather than being re-derived from a container that may since have been
replaced.

## 7. Rollout and rollback

The seam is default-off (`decisions.style.enabled = false`) and stays off until the paired
style A/B clears the pre-registered rule. Two providers, one switch:

```
make docker-up-decisions STYLE_PROVIDER=clm
make benchmark-style-ab STYLE_PROVIDER=clm
```

`STYLE_PROVIDER` moves provider, endpoint and model together. A mismatched pair is
unexpressible rather than merely discouraged: `DecisionSeamConfig` rejects a `provider` and
`base_url` shape that disagree in EITHER direction (clm without the `/v1` root, or tacit on a
`/v1` root), and `DecisionsConfig` rejects `clm` on any seam but `style`. The A/B reads its
floor from the calibration artifact of the provider under test — never hand-copied.

Rollback is a config flip, no schema or data migration. Move the provider, endpoint and model
TOGETHER — use the same knob as the forward path, because flipping `provider` alone would
leave a clm-serve URL on a Tacit client (the config validator rejects that pairing, so it
fails at startup rather than serving a dead seam):

```
make docker-up-decisions STYLE_PROVIDER=tacit      # or DECISIONS_STYLE_ENABLED=false
```

Then `docker compose … down design-pattern-clm` to free the card. Keep `models/clm-heads/` +
both calibration reports as evidence; `rm -rf models/clm-heads` only if the cached-heads path
is itself implicated (that forces a retrain instead of re-baking).

## 8. State of this branch

Landed and verified on the workstation:

- provider switch (`DecisionSeamConfig.provider`, env `DECISIONS_STYLE_PROVIDER`), both
  clients wired per provider, `DecisionModelClient` restored from `tacit-clm`;
- `build_style_criteria()` re-keyed to `STYLE_VOCABULARY` + ordered byte-parity test;
- the CLM scripts and their two Dockerfile targets (`runtime-released` and
  `runtime-finetuned` exist and pass `docker build --check` — statically only, neither image
  has been BUILT yet), the compose sidecar, the `clm-*` Makefile targets, the consistency
  probes;
- `make clm-calibrate` proved end to end against an oracle stub on this workstation (138
  rows, 7 floors, both probes populated, `served` provenance recorded, exit 0). That is a
  PLUMBING check, not a measurement: the stub knows the gold labels, so its 1.0 accuracy says
  nothing about the head.
- every in-build script invocation is argv-checked against its own parser
  (`test_clm_scripts_accept_the_argv_the_image_build_passes`), because a mistyped option
  otherwise only surfaces hours into a GPU build — `docker build --check` cannot see inside
  a `RUN`, and `make -n` only prints the recipe;
- corpus re-adopted at 857 rows (719/138, min 13 per class, zero shared ids), export smoke
  clean (`architecture_style` train 719 / test 138, `uncovered: []`);
- `make check-all` + `make test-unit` green; `docker build --check` clean on both targets.

Every deterministic link in the build→serve path was rehearsed here, so what remains on the
host is GPU compute and nothing else:

| Link | Status |
|---|---|
| `[2/6]` export → manifest | 719 train / 138 test, `uncovered: []` |
| `[2/6]` preflight pin vs manifest | PASS on the real manifest; fails closed on a wrong pin |
| `[4/6]` eval gate on the real manifest | PASS (R6 all 36 test labels supported, R7 nothing to acknowledge) |
| `[5/6]` rank-parity payload loader | accepts all 30 committed payloads |
| extract chain (trainer → COPY → `docker cp` → marker) | filenames and paths match at the pin |
| `clm-entrypoint.sh` ↔ training ↔ serving | 2048 token budget, `qwen3-8b` served name, `/v1/embeddings` upstream and `/clm/heads/best_head.pt` all agree |

The `:base` build was deliberately NOT run locally while the host is down. This branch's only
`Dockerfile.clm` changes are header comments plus `--targets hard` / `--loss infonce` inside
the `finetune` stage, so `runtime-released` is byte-identical to the branch that already built
and served the sidecar; building it here would spend ~25 GB plus a 16 GB encoder download on a
69 GB-free disk to produce an image this box cannot serve (Pascal, CC 6.1).

The extract chain was verified statically, link by link, because a mismatch anywhere in it
costs a finished 10-hour build: the trainer writes `best_head.pt` and `finetune_summary.json`
into `--out-dir /clm/heads-trained` (`train/finetune.py:323,743` at the pin); the two gates
write `clm-eval-gate.json` and `clm-rank-parity.json` beside them; `runtime-finetuned` is
`COPY --from=finetune /clm/heads-trained /clm/heads`, so `docker cp "$cid:/clm/heads/."`
captures the checkpoint AND both reports, which the Makefile then hashes into
`.extract-complete`; `runtime-released` places `CLM_v0.1-8B.pt` at exactly `/clm/heads/best_head.pt`,
the path `CLM_CKPT` serves.

MERGED AND RUN, on `tacit-clm-ft` (4× RTX 3090) from this branch. Every step executed:

1. `docker-build-clm-base` — succeeded once `docker/clm-entrypoint.sh` and the four CLM
   scripts were whitelisted in `.dockerignore` (they were not; the build failed at that COPY
   in 3 s, and `docker build --check` does NOT validate COPY sources against the context);
   verified `/clm/heads/best_head.pt` = 75 557 149 B in the `:base` image;
2. `docker-build-clm` — run 1 stopped at the rank-parity gate (8/30 vs a 0.250 budget, with
   nDCG slightly BETTER trained); run 2, the plan's single retry at `lr 2.5e-4`, passed both
   gates and produced `design-pattern-clm:1.0.1` (image `7e186c051de9`);
3. heads + both gate reports extracted to `models/clm-heads/`, marker verified
   (`sha256sum -c .extract-complete` clean on the workstation);
4. E1 baseline replay against `:base` → `models/clm-calibration-base.json`;
5. E2 tuned replay against `:1.0.1` → `models/clm-calibration.json`.

The host needs `uv` for `clm-payloads-check` / `check-dockerfile-shell`; it did not have it,
installed from astral.sh. `rsync` is also absent — use `tar -c host | tar -x` over ssh.
Free GPU 0 first: the in-build pooling encoder defaults to `cuda:0`, and the Tacit dev
container holds it. Restart `design-pattern-tacit-dev` afterwards.

Acceptance (absolute AND delta, so the baseline cannot be gamed):
holdout top-1 ≥ 0.65 and ≥ `init_test + 0.10`; no regression vs the released heads (one
flipped row tolerated); majority-prior beat; `C_order ≥ 0.95`; `C_repeat ≥ 0.99`; tuned
consistency ≥ baseline − 0.02; a qualifying floor at acted-accuracy ≥ 0.8 / acted-rate ≥ 0.3
on ≥ 10 acted rows. One controlled retry only, with a stated hypothesis — used.

**Both runs miss the absolute bar and neither E1 nor E2 yields a qualifying floor, so the
seam stays off. The numbers and the verdict are in §9.**

## 9. Measured results

**Verdict: REJECT.** The seam stays off. Measured on `tacit-clm-ft` (4× RTX 3090) from this
branch; every number below is from a committed artifact.

Same instrument for every arm: the 138-row family-disjoint holdout, 36 classes, chance
`1/36 ≈ 0.028`.

| arm | holdout top-1 | source |
|---|---|---|
| released CLM-v0.1-8B heads | **0.0362** | `models/clm-calibration-base.json` |
| AnyJev Tacit-4B (committed report) | **0.4203** | `models/tacit-calibration.json`, n=138 |
| CLM-8B heads-only, run 1 (`lr 5e-4`) | **0.5072** | in-build eval gate (build stopped at rank parity, heads not extracted) |
| CLM-8B heads-only, run 2 (`lr 2.5e-4`) | **0.4565** | `models/clm-calibration.json` (served replay) |
| TF-IDF(1-2gram) + calibrated LinearSVC | **0.833** | `models/style-head-calibration.json`, n=138 (`docs/linear-style-head.md` §2) |

### The two builds

| | run 1 (`lr 5e-4`) | run 2 (`lr 2.5e-4`) |
|---|---|---|
| best epoch / early stop | 9 / 14 | 14 / 19 |
| val accuracy | 0.6319 | 0.5833 |
| `architecture_style` test | **0.5072** | **0.4493** |
| rank parity (budget 0.25) | **FAIL** 8/30 = 0.267 | **PASS** 7/30 = 0.233 |
| mean nDCG@10 trained vs released | 0.8108 vs 0.8052 | 0.8305 vs 0.8049 |
| mean top-5 overlap | 0.827 | 0.847 |

Run 1 passed every eval-gate rule (R0–R7) and was stopped only by rank parity, one payload
over budget. That is what triggered the plan's single pre-registered retry (train loss still
falling while val plateaued → overfit → `lr 2.5e-4`). The retry bought rank parity and paid
for it in accuracy: **0.5072 → 0.4493**. Neither run reaches 0.65.

### The two replays

`make clm-calibrate` with the plan's protocol (`--floors 0.1…0.7`, `--metric confidence`,
`--repeats 5`, `--permutations 5`), each against a live `clm-serve` container:

| | E1 (`:base`, released heads) | E2 (`:1.0.1`, trained) |
|---|---|---|
| holdout accuracy | 0.0362 | **0.4565** |
| ECE / Brier | 0.7132 / 0.5621 | — |
| repeat agreement (5 replays) | 1.0 | 1.0 |
| order agreement (5 permutations) | 1.0 | 1.0 |
| qualifying floor (≥0.8 acted at ≥0.3 rate) | none | **none** |

Two cross-checks worth noting. E1's served accuracy (0.0362) equals the in-build
`init_test.per_question.architecture_style` (0.0362) exactly — the calibration harness and the
trainer's eval agree to four decimals on the same rows. And E2's served 0.4565 sits within
0.007 of the trainer's in-process 0.4493, so the served wire reproduces the training-time
measurement.

E2's acted accuracy never reaches 0.8 at any floor: 0.443 @0.1, 0.450 @0.2, 0.466 @0.3,
0.500 @0.4, 0.519 @0.5, 0.548 @0.6. The head carries real signal — 16× the released heads,
and it beats Tacit-4B — but it has no confident region worth handing the seam.

### Why it fails, and what that means

The gate arithmetic against the plan's objectives:

- **FAIL** absolute holdout accuracy ≥ 0.65 (0.4565)
- PASS delta over the released baseline ≥ +0.10 (+0.4203)
- PASS repeat agreement ≥ 0.99 (1.0 both arms)
- PASS order agreement ≥ 0.95 (1.0 both arms)
- PASS consistency not worse by > 0.02
- **FAIL** a qualifying floor exists (none)

So the seam would have to act on every row and still be wrong a third of the time. Per the
plan's stop rule the effort ends here; this document is the verdict.

The shape of the result is the same one `docs/tacit-fine-tune.md` §4.7.1 reached from the other
direction: a fine-tuned sidecar over a frozen encoder scores 0.46–0.51 on this 36-way task
while a TF-IDF(1-2gram) + max-margin linear model over the *same 719 labels* scores 0.833,
and the released heads score 0.036. Twenty times the parameters, ten times the compute, less
than two thirds of the accuracy. This is not a case for another retrain — the plan allows one
retry and it made things worse. It is evidence that the task is a lexical read the encoder
adapters are the wrong instrument for. (`docs/linear-style-head.md` §2 measures that baseline
per classifier: the *logistic-regression* form of it scores 0.4348 on the same bytes, so §4.7.1's
"0.833" is the max-margin arm, not the softmax one. The conclusion is unchanged — a linear
model reads this task and the 8B heads do not — but the recipe that reaches 0.833 is a
calibrated `LinearSVC`.)

What a future attempt would have to change is the *instrument*, not the hyperparameters
options already measured out: a shortlist-then-discriminate head, a linear style head served
directly, or a genuinely different fine-tune objective. Those are design decisions for the
owner of this seam, not continuations of this experiment.
