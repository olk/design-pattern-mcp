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

NOT yet measured — the GPU host (`192.168.1.153`, 4× RTX 3090) was unreachable (`No route to
host`, and its Proxmox parent with it) when this branch was cut. The remaining steps are
gated on that host. The branch travels as `~/clm-style-seam.bundle` (complete history;
`git bundle verify` names the tip it carries, so this paragraph cannot go stale):

1. `scp ~/clm-style-seam.bundle <host>:/tmp/`, then on the host
   `git fetch /tmp/clm-style-seam.bundle clm-style-seam:clm-style-seam && git checkout clm-style-seam`,
   then `make docker-build-clm-base` (no GPU; verify `/clm/heads/best_head.pt` exists in the
   image). Both build targets run `clm-payloads-check` first, so the host needs `uv` — the
   same prerequisite the Tacit lane already satisfies;
2. `make docker-build-clm` (GPU: free card 0 first — the in-build pooling encoder lands
   there, so `docker stop design-pattern-tacit-dev` and never build Tacit and CLM
   concurrently); the build writes `clm-eval-gate: PASS` and the rank-parity report;
3. `rsync` `models/clm-heads/` back, then `sha256sum -c models/clm-heads/.extract-complete`;
4. baseline replay E1 against the `:base` image → `models/clm-calibration-base.json`;
5. tuned replay E2 against the trained image → `models/clm-calibration.json`.

The `tacit-clm-ft` ssh alias does not exist on this workstation (no `~/.ssh/config` stanza);
use the IP, or add the stanza before running the plan's commands verbatim.

Acceptance (absolute AND delta, so the unmeasured baseline cannot be gamed):
holdout top-1 ≥ 0.65 and ≥ `init_test + 0.10`; no regression vs the released heads (one
flipped row tolerated); majority-prior beat; `C_order ≥ 0.95`; `C_repeat ≥ 0.99`; tuned
consistency ≥ baseline − 0.02; a qualifying floor at acted-accuracy ≥ 0.8 / acted-rate ≥ 0.3
on ≥ 10 acted rows. One controlled retry only, with a stated hypothesis (val flat while train
falls → overfit: lr 2.5e-4 or 10 epochs; both poor → underfit: 40 epochs). A second failure
stops the effort and is recorded here — the `docs/tacit-fine-tune.md` §4.7.1 lesson is that an
unfalsifiable fourth attempt is not a plan.
