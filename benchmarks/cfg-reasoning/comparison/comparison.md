# Benchmark comparison

- baseline (baseline): `live-base-fresh` mode `live` (18 records) in `benchmarks/base-fresh`
- candidate (candidate): `live-cfg-reasoning` mode `live` (18 records) in `benchmarks/cfg-reasoning`
- scenario file match: True · paired scenarios: 18

## Verdict: `candidate-win`

_candidate wins iff holdout p95 e2e improves >= 15% AND candidate overall acceptable-primary hit rate >= baseline overall Wilson lower bound_

- every pre-registered condition held
- holdout p95 e2e: baseline 410921.9 ms → candidate 339921.3 ms (17.3%)

## Hit rate (Wilson / Newcombe)

| split | baseline | candidate | difference [95% CI] |
|---|---|---|---|
| overall | 61.1% | 66.7% | +0.056 [-0.239, +0.338] |
| train | 83.3% | 66.7% | -0.167 [-0.468, +0.176] |
| holdout | 16.7% | 66.7% | +0.500 [-0.040, +0.773] |

## Paired deltas (candidate minus baseline)

- e2e: mean -15232.6028 ms · median -18604.7490 ms · faster 10 / slower 8 · sign test p=0.814529 (n=18)

| metric | compared | mean Δ | median Δ | improved | worsened | unchanged |
|---|---|---|---|---|---|---|
| recall_at_3 | 18 | -0.0000 | +0.0000 | 4 | 5 | 9 |
| recall_at_5 | 18 | +0.0333 | +0.0000 | 6 | 4 | 8 |
| supporting_f1 | 18 | +0.0139 | +0.0000 | 2 | 1 | 15 |
| acceptable_primary_f1 | 18 | +0.0278 | +0.0000 | 5 | 4 | 9 |
| veto_violations | 18 | +0.0000 | +0.0000 | 0 | 0 | 18 |
| score | 18 | -0.0022 | -0.0011 | 9 | 9 | 0 |

## Hit flips

| key | baseline | candidate |
|---|---|---|
| cc-legacy-adapter#1 | miss `mapper` | hit `adapter` |
| cc-module-wiring#1 | miss `interface-class` | hit `dependency-injection` |
| cc-report-builder#1 | hit `builder` | miss `functional-options` |
| cr-billing-circuit-breaker#1 | hit `circuit-breaker` | miss `fail-fast` |
| cr-structured-fanout#1 | miss `thread-pool` | hit `structured-concurrency` |
| sc-dashboard-mvvm#1 | miss `presentation-model` | hit `model-view-viewmodel` |
| sc-expression-parser#1 | miss `interpreter` | hit `recursive-descent-parser` |
| sv-api-authorization#1 | hit `authorization-enforcer` | miss `policy-guard` |
| wm-order-router#1 | hit `content-based-router` | miss `content-based-serializer` |

## Stage deltas

| stage | calls Δ | total ms baseline | total ms candidate | Δ ms |
|---|---|---|---|---|
| `embed.query` | +0 | 12956.1 | 17216.8 | +4260.8 |
| `llm.analyze` | +0 | 297201.1 | 346421.0 | +49219.9 |
| `llm.evaluate` | +0 | 543191.6 | 706632.1 | +163440.5 |
| `llm.generate` | -1 | 1964090.8 | 2103241.5 | +139150.7 |
| `llm.weights` | +0 | 309180.9 | 264940.4 | -44240.5 |
| `reasoning.analyze` | +0 | 451002.0 | 510356.8 | +59354.8 |
| `reasoning.evaluate` | +0 | 532586.4 | 0.6 | -532585.8 |
| `reasoning.generate` | +0 | 570787.6 | 534933.0 | -35854.6 |
| `reasoning.weights` | +0 | 538530.7 | 0.4 | -538530.3 |
| `rerank` | +2 | 969259.4 | 1230239.2 | +260979.7 |

## Warnings

- effective config differs in ['reasoning']
- environment differs in ['REASONING_EVALUATE_ENABLED', 'REASONING_WEIGHTS_ENABLED']
