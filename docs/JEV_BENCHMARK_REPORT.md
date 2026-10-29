# JEV Decision-Model Benchmark: A/B Report

Date: 2026-10-20/21 · Mode: `benchmark-live` (in-process, Stage-0 harness) ·
Corpus: `tests/benchmark/scenarios/seed.json`, holdout split (6 scenarios × REPEAT=3 = 18 records/arm) ·
Generator: `minimax/MiniMax-M2.7` · Decision model: `jev-1.13-free` via OpenCode SystemOne ·
Verdict: **`candidate-loss` — keep `DECISIONS_ENABLED=false`**

> **Renamed per-seam 2026-10-27** (`DECISIONS_ENABLED` → `decisions.{style,weights,category,reasoning_gate}`,
> all default-off; hosted Jev is no longer a default). The record below is unedited: it measures the
> *style* seam against hosted `jev-1.13-free`, and its verdict still holds for that seam.

## 1. What was tested

Whether routing the Ch2 architecture-style choice through the hosted JEV decision model
(`POST {base}/systemone`, one `choice` question over 35 `STYLE_AFFINITY` keys + `none`,
override only on confident non-`none` answer per `src/pipeline.py::_apply_decision_style`)
improves **latency** (holdout p95 e2e) and **selection quality/consistency**
(acceptable-primary hit rate, within-arm repeat stability) versus the LLM-sampled style.

Arms (same corpus, split, repeats; `benchmark-sidecars-up` first):

| Arm | Dir | Flag | Run |
|---|---|---|---|
| Baseline (no JEV) | `benchmarks/live-nojev2` | `DECISIONS_ENABLED=false` | `live-20261020-173400` |
| Candidate (JEV) | `benchmarks/live-jev2` | `=true` | `live-20261020-191306` |

Follow-up: `benchmarks/live-sv-jev` (`live-20261021-051648`, LIMIT=6 whole holdout,
`LOG_LEVEL=info`) to attribute the `sv-order-double-tests` flip with per-call style lines.

## 2. How it was executed

- `make benchmark-live SPLIT=holdout REPEAT=3 OUT=<dir>` per arm (sequential, ~3h each;
  concurrent arms would contaminate the p95 comparison via shared TEI/CPU), then
  `make benchmark-compare A=… B=…` (pre-registered rule: win iff holdout p95 ≥15% better
  **and** candidate overall hit rate ≥ baseline overall Wilson lower bound).
- Proof before full arms: `LIMIT=1` JEV run logged
  `Decision-model style choice` + `Architecture style resolved by decision model`;
  ruff / mypy-strict / vulture / selfcheck green throughout.
- Validity of the final pair: candidate records carry `style_affinity+5` on every
  candidate whose category intersects the JEV style (baseline: none); zero
  `Decision-model style choice degraded` lines (no silent per-scenario fallback to LLM style).

## 3. Single results

### 3.1 Latency (e2e wall clock, 18 samples/arm)

| Metric | Baseline | Candidate | Δ |
|---|---|---|---|
| Holdout p95 | 559 745.8 ms | 508 471.8 ms | **+9.2% (< +15% → leg fails)** |
| Holdout p50 | 269 386.7 ms | 242 716.9 ms | +9.9% |
| Paired deltas | — | mean −13.7 s · median −51.8 s · 11/18 faster · sign test p=0.48 (n=18) | indistinguishable |

Stage deltas (candidate − baseline): `llm.generate` +455 s / +5 calls, `rerank` +162 s,
all other LLM/reasoning stages negative. JEV's own cost is ~0.6 s/call — the swing is
downstream LLM variance, and the arms ran sequentially hours apart (time drift confounds p95).

### 3.2 Quality (acceptable-primary hit rate)

| Split | Baseline | Candidate | Difference [95% Newcombe CI] |
|---|---|---|---|
| Overall | 72.2% (13/18) | 55.6% (10/18) | −0.167 [−0.433, +0.138] |
| Holdout | 72.2% | 55.6% | same (all-holdout run) |

Quality leg passes (55.6% ≥ 49.1% baseline Wilson floor) but drops 16.7 pts.
Hit flips (9 cross-arm pairs): the full −3 net swing is `sv-order-double-tests`
(baseline `stub` 3/3 hit → candidate `configurable-test-double` 3/3 miss),
partly offset by `cr-structured-fanout` +1 (candidate stabilised to `structured-concurrency` 3/3).

### 3.3 Consistency (post-hoc repeat grouping; `compare` does not compute this)

| Arm | Unstable scenarios (distinct finals across 3 repeats) |
|---|---|
| Baseline | 4/6 (`cc` 3 distinct, `cr`/`ib`/`sc` 2 each; `ps`, `sv` stable) |
| Candidate | 3/6 (`cc`/`ib`/`sc` 2 each; `ps`, `cr`, `sv` stable) |

Marginal; repeats 2–3 reuse repeat 1's LRU-cached JEV style, so this measures
cache+LLM stability, not JEV repeatability. n=18 is 6 scenarios × 3 — `compare`'s
Wilson/sign-test/p95 intervals treat them as independent and are overconfident.

### 3.4 `sv-order-double-tests` attribution (follow-up run)

- JEV's replayed answer for the sv description: **`hexagonal`, conf 0.74**
  (`{domain-modeling, behavior-selection, interaction-boundary}`,
  `src/style_affinity.py:29`) — a fresh sample, not the A/B's answer (A/B ran at default
  `warning`, per-call style never logged; hosted JEV flips labels on identical bytes).
- The deciding part's candidates are all `security-verification`: zero affinity overlap,
  zero `style_affinity` lines in the A/B record (all `category_mismatch-10`).
- **The `sv` flip is LLM variance between run windows, not a JEV effect** — for or against.
  The lone `style_affinity+5` in the sv record sits on `interpreter-as-data` in the
  non-deciding part.

## 4. Summary

Real A/B (wired client, key gate, no degrade lines): JEV fires (~0.6 s/call) and its style
reaches scoring wherever the affinity set intersects candidate categories. It produced no
demonstrated gain on any leg — p95 +9.2% misses the +15% rule at p=0.48, hit rate fell
16.7 pts on LLM-window variance, repeat stability is a wash under a shared style cache.
No evidence of a JEV-caused regression either: the systematic-looking `sv` flip carries
no style-affinity fingerprint.

## 5. Final verdict

**`candidate-loss` per the pre-registered rule (p95 leg). Keep JEV disabled.**

Actionable notes:

- The dev stack is JEV-**on** by default: `docker/docker-compose.yml:100` defaults
  `DECISIONS_ENABLED` to `true` (only `config/config.json` defaults false,
  which compose overrides). Set `DECISIONS_ENABLED=false` in the compose
  env, or flip that default.
- A future A/B needs a corpus where JEV styles intersect candidate categories (so the
  override actually bites), per-call style logging (`LOG_LEVEL=info`), and more distinct
  scenarios rather than repeats (`full.json`, not REPEAT=3) for honest intervals.
