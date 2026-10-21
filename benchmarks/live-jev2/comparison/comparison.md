# Benchmark comparison

- baseline (baseline): `live-20261020-173400` mode `live` (18 records) in `benchmarks/live-nojev2`
- candidate (candidate): `live-20261020-191306` mode `live` (18 records) in `benchmarks/live-jev2`
- scenario file match: True · paired scenarios: 18

## Verdict: `candidate-loss`

_candidate wins iff holdout p95 e2e improves >= 15% AND candidate overall acceptable-primary hit rate >= baseline overall Wilson lower bound_

- p95 e2e improved 9.2% (< 15%)
- holdout p95 e2e: baseline 559745.8 ms → candidate 508471.8 ms (9.2%)

## Hit rate (Wilson / Newcombe)

| split | baseline | candidate | difference [95% CI] |
|---|---|---|---|
| overall | 72.2% | 55.6% | -0.167 [-0.433, +0.138] |
| train | null | null | null |
| holdout | 72.2% | 55.6% | -0.167 [-0.433, +0.138] |

## Paired deltas (candidate minus baseline)

- e2e: mean -13660.1821 ms · median -51771.3045 ms · faster 11 / slower 7 · sign test p=0.480682 (n=18)

| metric | compared | mean Δ | median Δ | improved | worsened | unchanged |
|---|---|---|---|---|---|---|
| recall_at_3 | 18 | -0.0222 | +0.0000 | 3 | 5 | 10 |
| recall_at_5 | 18 | -0.0444 | +0.0000 | 1 | 5 | 12 |
| supporting_f1 | 18 | -0.0965 | +0.0000 | 0 | 4 | 14 |
| acceptable_primary_f1 | 18 | -0.0833 | +0.0000 | 3 | 6 | 9 |
| veto_violations | 18 | +0.0000 | +0.0000 | 0 | 0 | 18 |
| score | 18 | +0.0157 | +0.0148 | 11 | 7 | 0 |

## Hit flips

| key | baseline | candidate |
|---|---|---|
| cr-structured-fanout#1 | miss `deadline-propagation` | hit `structured-concurrency` |
| ib-admin-controller#1 | hit `page-controller` | miss `service-to-worker` |
| ib-admin-controller#2 | miss `service-to-worker` | hit `page-controller` |
| sc-dashboard-mvvm#1 | hit `model-view-viewmodel` | miss `flux-redux` |
| sc-dashboard-mvvm#2 | miss `model-view-viewmodel` | hit `presentation-model` |
| sc-dashboard-mvvm#3 | hit `presentation-model` | miss `flux-redux` |
| sv-order-double-tests#1 | hit `stub` | miss `configurable-test-double` |
| sv-order-double-tests#2 | hit `stub` | miss `configurable-test-double` |
| sv-order-double-tests#3 | hit `stub` | miss `configurable-test-double` |

## Stage deltas

| stage | calls Δ | total ms baseline | total ms candidate | Δ ms |
|---|---|---|---|---|
| `embed.query` | +0 | 12596.1 | 15673.5 | +3077.4 |
| `llm.analyze` | +0 | 365063.2 | 335639.0 | -29424.2 |
| `llm.evaluate` | +0 | 583993.7 | 510710.7 | -73283.0 |
| `llm.generate` | +5 | 2126301.9 | 2581736.4 | +455434.5 |
| `llm.weights` | +0 | 300300.3 | 274670.8 | -25629.5 |
| `reasoning.analyze` | +0 | 153668.2 | 116881.1 | -36787.1 |
| `reasoning.evaluate` | +0 | 608240.3 | 532255.8 | -75984.5 |
| `reasoning.generate` | -1 | 634834.7 | 539901.9 | -94932.8 |
| `reasoning.weights` | +0 | 608582.6 | 547062.0 | -61520.6 |
| `rerank` | +5 | 734155.8 | 896363.0 | +162207.2 |

## Warnings

- effective config differs in ['decisions']
- environment differs in ['DESIGN_PATTERN_DECISIONS_ENABLED']
