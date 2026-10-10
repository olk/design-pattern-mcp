# Benchmark comparison

- baseline (baseline): `live-base-fresh` mode `live` (18 records) in `benchmarks/base-fresh`
- candidate (candidate): `live-cfg-fast` mode `live` (18 records) in `benchmarks/cfg-fast`
- scenario file match: True · paired scenarios: 18

## Verdict: `candidate-loss`

_candidate wins iff holdout p95 e2e improves >= 15% AND candidate overall acceptable-primary hit rate >= baseline overall Wilson lower bound_

- p95 e2e improved 14.2% (< 15%)
- holdout p95 e2e: baseline 410921.9 ms → candidate 352384.1 ms (14.2%)

## Hit rate (Wilson / Newcombe)

| split | baseline | candidate | difference [95% CI] |
|---|---|---|---|
| overall | 61.1% | 55.6% | -0.056 [-0.342, +0.245] |
| train | 83.3% | 58.3% | -0.250 [-0.540, +0.109] |
| holdout | 16.7% | 50.0% | +0.333 [-0.172, +0.674] |

## Paired deltas (candidate minus baseline)

- e2e: mean +10690.9037 ms · median -9456.0335 ms · faster 9 / slower 9 · sign test p=1.0 (n=18)

| metric | compared | mean Δ | median Δ | improved | worsened | unchanged |
|---|---|---|---|---|---|---|
| recall_at_3 | 18 | +0.0222 | +0.0000 | 4 | 3 | 11 |
| recall_at_5 | 18 | +0.0333 | +0.0000 | 6 | 4 | 8 |
| supporting_f1 | 18 | +0.0164 | +0.0000 | 2 | 2 | 14 |
| acceptable_primary_f1 | 18 | -0.0278 | +0.0000 | 3 | 4 | 11 |
| veto_violations | 18 | +0.0000 | +0.0000 | 0 | 0 | 18 |
| score | 18 | +0.0026 | +0.0012 | 10 | 8 | 0 |

## Hit flips

| key | baseline | candidate |
|---|---|---|
| bs-ast-visitor#1 | hit `visitor` | miss `collecting-parameter` |
| cc-module-wiring#1 | miss `interface-class` | hit `dependency-injection` |
| cc-report-builder#1 | hit `builder` | miss `macro-method` |
| dm-order-lifecycle#1 | hit `entity` | miss `make-illegal-states-unrepresentable` |
| ib-admin-controller#1 | miss `service-to-worker` | hit `page-controller` |
| ps-catalog-repository#1 | hit `repository` | miss `data-mapper` |
| sv-order-double-tests#1 | miss `configurable-test-double` | hit `stub` |

## Stage deltas

| stage | calls Δ | total ms baseline | total ms candidate | Δ ms |
|---|---|---|---|---|
| `embed.query` | +0 | 12956.1 | 14019.1 | +1063.0 |
| `llm.analyze` | +0 | 297201.1 | 345591.0 | +48389.9 |
| `llm.evaluate` | +0 | 543191.6 | 543050.7 | -140.9 |
| `llm.generate` | -1 | 1964090.8 | 1980763.7 | +16672.9 |
| `llm.weights` | +1 | 309180.9 | 584056.9 | +274876.0 |
| `reasoning.analyze` | +0 | 451002.0 | 528078.1 | +77076.1 |
| `reasoning.evaluate` | +0 | 532586.4 | 601504.3 | +68917.9 |
| `reasoning.generate` | +0 | 570787.6 | 665260.0 | +94472.4 |
| `reasoning.weights` | +0 | 538530.7 | 611162.5 | +72631.8 |
| `rerank` | -4 | 969259.4 | 273546.8 | -695712.7 |

## Warnings

- effective config differs in ['reranker']
- environment differs in ['DESIGN_PATTERN_RERANK_TEXT', 'DESIGN_PATTERN_RETRIEVAL_MAX_CONCURRENT_PARTS']
