# Benchmark comparison

- baseline (baseline): `live-20261020-120933` mode `live` (18 records) in `benchmarks/live-nojev`
- candidate (candidate): `live-20261020-152201` mode `live` (18 records) in `benchmarks/live-jev`
- scenario file match: True · paired scenarios: 18

## Verdict: `candidate-loss`

_candidate wins iff holdout p95 e2e improves >= 15% AND candidate overall acceptable-primary hit rate >= baseline overall Wilson lower bound_

- p95 e2e improved -1.6% (< 15%)
- holdout p95 e2e: baseline 429722.1 ms → candidate 436587.9 ms (-1.6%)

## Hit rate (Wilson / Newcombe)

| split | baseline | candidate | difference [95% CI] |
|---|---|---|---|
| overall | 77.8% | 61.1% | -0.167 [-0.428, +0.129] |
| train | null | null | null |
| holdout | 77.8% | 61.1% | -0.167 [-0.428, +0.129] |

## Paired deltas (candidate minus baseline)

- e2e: mean +10606.6045 ms · median -139.4280 ms · faster 9 / slower 9 · sign test p=1.0 (n=18)

| metric | compared | mean Δ | median Δ | improved | worsened | unchanged |
|---|---|---|---|---|---|---|
| recall_at_3 | 18 | -0.0333 | +0.0000 | 2 | 4 | 12 |
| recall_at_5 | 18 | -0.0889 | +0.0000 | 0 | 5 | 13 |
| supporting_f1 | 18 | -0.0624 | +0.0000 | 0 | 2 | 16 |
| acceptable_primary_f1 | 18 | -0.0833 | +0.0000 | 3 | 6 | 9 |
| veto_violations | 18 | +0.0000 | +0.0000 | 0 | 0 | 18 |
| score | 18 | -0.0014 | -0.0051 | 8 | 10 | 0 |

## Hit flips

| key | baseline | candidate |
|---|---|---|
| cc-legacy-adapter#2 | miss `gateway` | hit `adapter` |
| cc-legacy-adapter#3 | hit `adapter` | miss `mapper` |
| cr-structured-fanout#1 | hit `structured-concurrency` | miss `thread-pool` |
| cr-structured-fanout#3 | miss `deadline-propagation` | hit `structured-concurrency` |
| ib-admin-controller#1 | hit `page-controller` | miss `service-to-worker` |
| ib-admin-controller#3 | miss `chain-of-responsibility` | hit `page-controller` |
| sv-order-double-tests#1 | hit `stub` | miss `configurable-test-double` |
| sv-order-double-tests#2 | hit `stub` | miss `configurable-test-double` |
| sv-order-double-tests#3 | hit `stub` | miss `configurable-test-double` |

## Stage deltas

| stage | calls Δ | total ms baseline | total ms candidate | Δ ms |
|---|---|---|---|---|
| `embed.query` | +0 | 12971.2 | 11128.2 | -1842.9 |
| `llm.analyze` | +0 | 352000.7 | 368564.2 | +16563.5 |
| `llm.evaluate` | -1 | 538991.2 | 614720.8 | +75729.7 |
| `llm.generate` | +2 | 1798142.1 | 1936227.0 | +138084.9 |
| `llm.weights` | +0 | 298217.1 | 383199.9 | +84982.8 |
| `reasoning.analyze` | +0 | 136109.4 | 157591.7 | +21482.3 |
| `reasoning.evaluate` | -1 | 563409.3 | 531554.4 | -31854.9 |
| `reasoning.generate` | -1 | 541765.8 | 539951.7 | -1814.2 |
| `reasoning.weights` | +0 | 614935.9 | 648367.7 | +33431.8 |
| `rerank` | +2 | 755749.6 | 856183.7 | +100434.1 |

## Warnings

- effective config differs in ['decisions']
- environment differs in ['DESIGN_PATTERN_DECISIONS_ENABLED']
