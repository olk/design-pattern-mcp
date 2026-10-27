# Stage-0 selection benchmark (`tests/benchmark/`)

Instrument for the Jev decision-model A/B work (`docs/JEV_DECISION_MODEL_ANALYSIS_PLAN.md` §6
Stage 0): it attributes wall-clock to pipeline stages and measures selection quality against a
labelled scenario corpus, so a Jev cutover at a seam can be judged as a **paired
baseline-vs-candidate** comparison on identical scenarios.

Nothing here is a pytest test: no module matches `test_*.py` and there is no `conftest.py`, so
`make test-unit` / `make test-oracles` never collect it. `src/` is untouched — the pipeline is
constructor-injected (`src/pipeline.py:1294`), so the harness only wraps the instances it passes
to `RetrievalIndex.build` and `DesignPipeline.__init__`.

## Commands

| Command | What it does |
|---|---|
| `make benchmark-selfcheck` | hand-computed anchors for every metric formula (no network, no LLM) |
| `make benchmark-offline` | scripted deterministic run over `scenarios/seed.json` (<60 s) |
| `make benchmark-offline FLIP=<scenario_id>` | same, with one scenario scripted to miss (compare dry run) |
| `make benchmark-live LIMIT=2` | real in-process pipeline with full stage attribution |
| `make benchmark-analyze LIMIT=2` | same live seams, ANALYZE phase only (selection hit/recall, no generation) |
| `make benchmark-e2e LIMIT=2` | black-box run against the running compose stack |
| `make benchmark-compare A=<dir> B=<dir>` | two-arm comparison + verdict |
| `make benchmark-draft` | draft `full.json` candidates from the catalog (`DRY_RUN=1` prints the prompt) |

Shared knobs: `LIMIT=<n>`, `SPLIT=train|holdout`, `REPEAT=<n>`, `NO_WARMUP=1`, `OUT=<dir>`,
`RUN_ID=<id>`, `BENCH_SCENARIOS=<path>`; `FLIP=<scenario_id>` (offline only) and
`ALLOW_MISMATCH=1` (compare) are documented above. Each command also exposes the same options via
`--help`.

Raw entry points (same options via `--help`):

```bash
uv run python -m tests.benchmark.harness.main --mode offline --limit 3 --out data/benchmark-runs/offline-smoke
uv run python -m tests.benchmark.harness.main --mode analyze --out data/benchmark-runs/analyze-all
uv run python -m tests.benchmark.harness.e2e --limit 2 --out data/benchmark-runs/e2e-smoke
uv run python -m tests.benchmark.harness.compare data/benchmark-runs/base data/benchmark-runs/cand
uv run python -m tests.benchmark.harness.draft --dry-run          # corpus drafting prompt
```

`make benchmark-live` mirrors the dev stack's provider defaults (MiniMax via
`MINIMAXAI_API_KEY`) and points `REASONING_SHANNONTHINKING_CMD` / `REASONING_CODE_REASONING_CMD`
at the globally installed reasoning MCP packages (`make install-mcps`) when they are present —
the shipped config default names the Docker-embedded entry points, which do not exist on a
workstation. Without those packages the pipeline still runs: each reasoning phase degrades
silently and the manifest's `reasoning_health` records exactly which tool was unavailable (the
`reasoning.<phase>` stage timing then measures the failed attach, not real reasoning).

## Modes: what each one proves

| | offline | live (in-process) | analyze (in-process) | e2e (black-box) |
|---|---|---|---|---|
| retrieval | scripted label pool (`retrieval.scripted`) | real dense + BM25 + cross-encoder (`embed.query`, `rerank`) | same as live | server-side, not observable |
| LLM / reasoning | scripted doubles | real LiteLLM generator + reasoning MCP client (`llm.*`, `reasoning.<phase>`) | same as live, ANALYZE phases only (`llm.analyze`, `llm.weights`) | server-side |
| stage attribution | llm/reasoning only | every seam (incl. `build.embed_documents`) | every ANALYZE seam (`embed.query`, `rerank`, `llm.analyze`, `llm.weights`, `reasoning.*`) | none (total latency only) |
| quality observables | all metrics | all metrics | hit + recall + score/calibration/risk-coverage; pipeline block null | hit + recall + `overview.score` only |
| network / keys | none | TEI sidecars + generator key | TEI sidecars + generator key | full stack |
| runtime | ~15 s | 300–800 s/request (CPU tier) | ~60 % of a live request's p50 (no generation) | 300–800 s/request |

**Offline mode is not a quality instrument.** Its Stage-C hits are derived from the scenario's own
label pool (acceptable primaries first, then acceptable supporting patterns) and the request
language vetoes are applied when the pool is built, so its hit rate is a fixture property. What it
does prove is that the whole pipeline executes against the real 486-record catalog, that the
metric code consumes real `AnalysisResult`/`PipelineResult` objects, and that the two-arm compare
workflow runs on two complete run directories. The runner *asserts* fixture integrity after every
offline run: every non-flipped scenario must hit and every flipped scenario must miss — a
deviation fails loudly instead of being averaged away.

Offline also pins `selection.primary_score_threshold` and `supporting_score_threshold` to `0` so
the scripted pool always clears the §8.1 dual gate; the effective `ServerConfig` is recorded in
every manifest, and the override is visible in `summary.json` → `per_scenario[].effective_config`.

**Analyze mode is the analysis prefix of live.** It runs the same shipped configuration and the
same seams, then stops after `DesignPipeline.analyze`: no GENERATE, no EVALUATE, no design loop.
Hit, recall, supporting F1, acceptable-primary F1, veto conflicts and the calibration score all
come from the `AnalysisResult`, so they stay observable; the dominant pattern is the deterministic
§9 composition (`compose_selection`) replayed over the analyze selections — exactly the value the
full arm's `final_pattern` carries when its loop does not reselect. `pipeline` (attempts,
evaluation score, matched categories) is `null`, never a zero, and `e2e_ms` is a phase subset, not
a full-run latency. Use it to price the generation work and to check that a generation-skipping
default keeps the full arm's selection quality.

**Measured analyze vs live, 2026-10-25** — baseline `benchmarks/live-20261025`, candidate
`benchmarks/analyze-20261025` (same seed corpus, `NO_WARMUP=1`, MiniMax-M2.7, identical effective
config/environment/worktree; comparison committed under `benchmarks/analyze-20261025/comparison/`):

| split | hit analyze | hit live | Δ hit (analyze − live) [95 % Newcombe CI] | recall@3 A/L | recall@5 A/L |
|---|---|---|---|---|---|
| overall | 12/18 (66.7 %) | 12/18 (66.7 %) | +0.000 [−0.286, +0.286] | 0.267 / 0.300 | 0.378 / 0.367 |
| train | 8/12 (66.7 %) | 7/12 (58.3 %) | +0.083 [−0.272, +0.412] | 0.267 / 0.283 | 0.400 / 0.350 |
| holdout | 4/6 (66.7 %) | 5/6 (83.3 %) | −0.167 [−0.558, +0.295] | 0.267 / 0.333 | 0.333 / 0.400 |

Four scenarios flip (2 hits lost, 2 gained), so the equal overall totals are not the same
scenarios. Latency: analyze p50 161.5 s / p95 378.1 s vs live p50 273.0 s / p95 474.6 s — the
candidate is 93.2 s faster per scenario at the median (sign test p = 0.0075, 15/18 faster), by
construction because generation and evaluation are skipped. At n=18 the hit-rate intervals cover
zero on every split; this arm does not establish a quality regression, and it does not justify
changing a default either.

## Stage attribution

`TimingProxy` classes wrap the four injected seams and append one `StageRecord`
(`stage`, `op`, `started_ns`, `duration_ms`, `ok`, `meta`) per call into a shared
`StageRecorder`. Buckets (`tests/benchmark/harness/probes.py`):

- `build.embed_documents` — index build; reported, excluded from per-request sums;
- per request: `embed.query`, `rerank`, `llm.analyze` (`extract_brief_and_parts`), `llm.weights`
  (`extract_weights_batch`), `llm.generate` (one record per GENERATE batch), `llm.evaluate`,
  `retrieval.scripted` (offline only) and `reasoning.<phase>` (analyze/weights/generate/evaluate/…);
- `e2e_ms` — wall clock around `DesignPipeline.analyze` + `design_loop` (the two phases
  `run_design` itself calls, `src/pipeline.py:1358`); the analyze arm times `analyze` only;
- `residual_ms = e2e_ms − Σ(stage totals)` — a diagnostic of the deterministic core (scoring,
  planning, assembly) and harness overhead. It can go **negative** in live runs: GENERATE
  dispatches up to `generation_max_concurrency` batches in flight
  (`src/pipeline.py:1473`), so parallel per-call durations are counted twice in the sum. The
  record then carries a `residual_note` and the summary a `negative_residual_scenarios` count —
  read the per-stage table, not the residual, for concurrency-heavy runs.

Call counts are recorded alongside durations, so a latency delta can be attributed to *fewer
calls* (e.g. a Jev reasoning gate skipping phases) versus *faster calls*. p50/p95 come from
`statistics.quantiles(n=20, method="inclusive")` (p50 = cut 10/20, p95 = cut 19/20); fewer than
four samples report `null` plus a note — never NaN.

**LLM data seam (Step-0b, 2026-10-07 latency-reduction round).** `log_capture.py` reads the
production log path — one INFO line per transport call (`Provider call succeeded`, carrying
`operation`, `prompt_chars`, `response_chars`) and the `src.validation` WARNING line that marks
a structured-output repair re-call — and the `llm.*` proxies attach the values to the stage
record after the call returns. Stage tables expose `prompt_chars_mean`, `response_chars_mean`,
`chars_per_s_mean` and a `repair_calls` counter. The handler raises `src.agent`'s log level to
INFO (verbosity only); values come from the record extras, never the response body — sparse
data (scripted doubles) stays absent. `cached_tokens` is null at this seam: the llama-index
LiteLLM wrapper exposes `prompt_tokens/completion_tokens` but not
`prompt_tokens_details.cached_tokens` to the protocol caller. Concurrent GENERATE batches of
the same operation can attach a sibling batch's chars to a record (the log line carries no
per-call id); reads are scoped per operation, so cross-operation attribution is exact.

## Metrics (definitions)

Implemented and hand-pinned in `tests/benchmark/harness/scoring.py` + `selfcheck.py`:

- **acceptable-primary hit** — `final_pattern` ∈ `acceptable_primary`, and neither the run nor any
  part is a fallback; a fallback is an automatic miss, never a dropped scenario.
- **recall@K (K = 3, 5)** — the ordered selected set (primaries → per-part supporting → per-part
  alternatives, deduplicated) capped at K, scored against
  `acceptable_primary ∪ acceptable_supporting`.
- **supporting F1** — micro-averaged over decision parts against `acceptable_supporting`.
- **acceptable-primary F1** — a hit scores P = 1.0, R = 1/|A|; a miss scores 0/0.
- **veto-conflict violations** (expected 0) — a veto-ledger pattern elected/supported, or a part's
  supporting set containing its own primary.
- **Brier / ECE (10 equal-width bins)** — monitoring only, train split only. Score source: the
  elected primary's `fit_score` of the first decision part, clipped to [0, 1] (`PartSelection` has
  no score field); in e2e mode the only score on the wire is `overview.score` (the evaluation
  score), so calibration is not comparable across modes.
- **risk-coverage** — scenarios ordered by their own score descending; risk (miss rate) at
  80/90/100 % coverage plus the trapezoidal AUC of the risk-coverage curve anchored at (0, 0).

Confidence machinery is hand-rolled (no new dependencies): Wilson score intervals, Newcombe's
Wilson-based difference interval, and the exact two-sided binomial sign test (`math.comb`).
**Tie policy:** the sign test's sample space is the signs of the *non-zero* paired deltas —
zero deltas count neither as wins nor losses, `n` is the non-zero count, and the payload's
`n_dropped` records how many tied pairs were excluded.

## A/B protocol and the pre-registered decision rule

Both arms run the **same scenario file**; every manifest pins the corpus sha256, the git HEAD +
worktree hash, the masked deployment environment (`REASONING_*` plus every config
`{env:...}` name) and the effective
`ServerConfig`. `compare` refuses two runs whose corpus hashes differ (`--allow-mismatch`
overrides, with a warning) and refuses cross-mode pairs outright — except `analyze` vs `live`,
which is accepted as a phase-subset pair: hit/recall and the selection metrics compare directly,
the pipeline block is null on the analyze arm, and the pre-registered p95 rule is reported
`inconclusive` because the e2e delta is the omitted generate/evaluate work by construction.
Config/env drift is warned on.

### Live-run prerequisite (sidecar loopback ports)

`make benchmark-sidecars-up` merges `docker/docker-compose.benchmark.yml` on top of the base file
and publishes the TEI sidecars on `127.0.0.1:18081`/`18082`. Any later **base-only**
`docker compose -f docker/docker-compose.yml up …` recreates those containers *without* the
override's ports, and the next live run fails with `Connection refused` on the embedder — re-run
`make benchmark-sidecars-up` after any such stack operation. The merged form is always safe:

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml up -d …
```


It then reports per-scenario e2e deltas, a sign test over those deltas, hit flips, Newcombe
intervals on hit-rate differences (overall/train/holdout) and per-stage call/total deltas.

**Pre-registered rule:** the candidate (Jev) arm wins iff

1. holdout p95 e2e improves by ≥ 15 %, **and**
2. overall acceptable-primary hit rate ≥ the baseline's **overall** Wilson lower bound
   (same estimator both sides: no quality regression beyond the baseline's own statistical
   floor).

With n≈60 single-sample scenarios the Wilson floor is ≈±13 %, and at seed-corpus size (6 holdout
scenarios) the floor is ≈±39 %; the rule therefore resolves to `inconclusive` whenever fewer than
four holdout e2e samples per arm exist. That is reported, never hidden. The quality gate uses the
overall split on both sides, never a tiny split: at n=6 the holdout's own Wilson lower bound
collapses to ≈0.10, and gating a candidate's n=6 holdout rate against a baseline-wide floor
handed a `candidate-win` to an arm whose overall hit rate collapsed 66.7 % → 33.3 % (observed on
`base-noreasoning`, 2026-10-02).

## Scenario corpus

`scenarios/seed.json` — 18 hand-authored scenarios (2 per category × 9), family-tagged by catalog
`###` subgroup, split 12 train / 6 holdout (family-disjoint, so no subgroup appears in both
splits). Labels: `acceptable_primary` always carries ≥ 2 ids including a same-family sibling of
the expected primary (near-miss discrimination), and no description names or paraphrases the
pattern. `scenarios/scenario.schema.json` is the record contract; `load_scenarios` validates the
schema, resolves every label id against `pattern/<id>.json`, requires the family to equal the
primary's subgroup and enforces the family-disjoint split.

Seed labels are author-drafted (catalog-grounded) and **await expert review**; the seed run is the
Stage-0 dry run.

`scenarios/full.json` — 100 reviewed scenarios across all 52 sibling-capable families
(whole-part excluded: a single record, so no sibling is possible), split 70 train / 30
holdout by `draft.py`'s family rule (last third of the sorted family order → holdout,
family-disjoint). Produced 2026-10-27 via `make benchmark-draft` (MiniMax-M2.7,
`--per-family 3`, 156 candidates) + agent curation to 100; descriptions never name the
pattern or its family's aliases, and each scenario carries a concrete language chosen
under the no-discouraged-label rule. Drafts live under `data/benchmark-runs/drafts/`
(gitignored, never committed). Full labels are agent-curated and **await human expert
review**.

## Run directory

```
data/benchmark-runs/<run-id>/
  manifest.json               provenance (git, env, config, corpus hash, executor seams, warmup)
  warmup.json                 untimed warmup record (a repeat of scenario[0], excluded from metrics)
  scenarios/<id>-r<repeat>.json  per-scenario record: stages, residual, quality, pipeline, selection
  summary.json                aggregates: quality per split, latency, calibration, risk-coverage
  report.md                   human-readable tables
```

`compare` writes `comparison.json` + `comparison.md` (default: `<candidate>/comparison/`) and exits
non-zero for a `candidate-loss` verdict, so a CI step can gate on it. `compare` also **refuses**
cross-mode pairs (e.g. live vs e2e) and different scenario files; the analyze/live phase-subset
pair is the documented exception, and `--allow-mismatch` overrides the rest with a recorded
warning.

A failed arm leaves an `aborted.json` marker next to its partial `scenarios/` records (the runner
propagates the failure — live attribution must not average over it) and **no**
`manifest.json`/`summary.json`; `Arm.load` refuses such directories and names the marker.
**Discard an aborted run directory** and re-run the arm.

`make benchmark-e2e CALL_TIMEOUT=<seconds>` (or `--call-timeout-seconds`) caps each black-box
request; the default is no client-side timeout because live requests take 300–800 s on the CPU tier.

## Deviations from the Stage-0 plan text

- The compose override lives at `docker/docker-compose.benchmark.yml` (next to the base file the
  Makefile already uses) instead of the repository root.
- Offline mode scripts the Stage-C retrieval hits (via the existing `tests.unit.fakes.StubIndex`
  double) instead of building a real index over `OfflineHashEmbedder`/`LexicalReranker`: the
  scripted pool is what makes the compare dry run's single hit flip deterministic. The real offline
  doubles remain the production test seams (`make health`, `tests/unit/`), and live mode exercises
  the real embed/rerank stages.
- The scenario record carries one extra required field, `decision_statement`, so the offline
  script and the reviewer have an explicit statement of the decision under test.
