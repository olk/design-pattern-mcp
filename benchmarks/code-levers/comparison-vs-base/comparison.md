# Benchmark comparison

- baseline (baseline): `live-base-fresh` mode `live` (18 records) in `benchmarks/base-fresh`
- candidate (candidate): `live-code-levers` mode `live` (18 records) in `benchmarks/code-levers`
- scenario file match: True · paired scenarios: 18

## Verdict: `candidate-win`

_candidate wins iff holdout p95 e2e improves >= 15% AND candidate overall acceptable-primary hit rate >= baseline overall Wilson lower bound_

- every pre-registered condition held
- holdout p95 e2e: baseline 410921.9 ms → candidate 297476.7 ms (27.6%)

## Hit rate (Wilson / Newcombe)

| split | baseline | candidate | difference [95% CI] |
|---|---|---|---|
| overall | 61.1% | 72.2% | +0.111 [-0.185, +0.383] |
| train | 83.3% | 75.0% | -0.083 [-0.390, +0.241] |
| holdout | 16.7% | 66.7% | +0.500 [-0.040, +0.773] |

## Paired deltas (candidate minus baseline)

- e2e: mean -51504.8522 ms · median -72780.9680 ms · faster 13 / slower 5 · sign test p=0.096252 (n=18)

| metric | compared | mean Δ | median Δ | improved | worsened | unchanged |
|---|---|---|---|---|---|---|
| recall_at_3 | 18 | -0.0444 | +0.0000 | 1 | 5 | 12 |
| recall_at_5 | 18 | -0.0444 | +0.0000 | 2 | 6 | 10 |
| supporting_f1 | 18 | -0.0444 | +0.0000 | 1 | 2 | 15 |
| acceptable_primary_f1 | 18 | +0.0556 | +0.0000 | 5 | 3 | 10 |
| veto_violations | 18 | +0.0000 | +0.0000 | 0 | 0 | 18 |
| score | 18 | +0.0051 | +0.0024 | 9 | 9 | 0 |

## Hit flips

| key | baseline | candidate |
|---|---|---|
| cc-module-wiring#1 | miss `interface-class` | hit `dependency-injection` |
| cr-structured-fanout#1 | miss `thread-pool` | hit `structured-concurrency` |
| ib-admin-controller#1 | miss `service-to-worker` | hit `page-controller` |
| ib-billing-gateway#1 | hit `gateway` | miss `messaging-gateway` |
| sc-expression-parser#1 | miss `interpreter` | hit `recursive-descent-parser` |
| sv-api-authorization#1 | hit `authorization-enforcer` | miss `front-controller` |
| sv-order-double-tests#1 | miss `configurable-test-double` | hit `stub` |
| wm-order-router#1 | hit `content-based-router` | miss `content-based-serializer` |

## Stage deltas

| stage | calls Δ | total ms baseline | total ms candidate | Δ ms |
|---|---|---|---|---|
| `embed.query` | +0 | 12956.1 | 18744.7 | +5788.6 |
| `llm.analyze` | +0 | 297201.1 | 331995.3 | +34794.3 |
| `llm.evaluate` | -2 | 543191.6 | 565384.0 | +22192.4 |
| `llm.generate` | -3 | 1964090.8 | 1646422.6 | -317668.2 |
| `llm.weights` | +0 | 309180.9 | 230973.0 | -78207.9 |
| `reasoning.analyze` | +0 | 451002.0 | 465422.7 | +14420.7 |
| `reasoning.evaluate` | -2 | 532586.4 | 0.5 | -532585.9 |
| `reasoning.generate` | -2 | 570787.6 | 513315.3 | -57472.3 |
| `reasoning.weights` | +0 | 538530.7 | 0.4 | -538530.4 |
| `rerank` | +2 | 969259.4 | 1278365.6 | +309106.1 |

## Warnings

- effective config differs in ['prompts', 'reasoning', 'validation']
- environment differs in ['DESIGN_PATTERN_PROMPT_STABLE_PREFIX', 'DESIGN_PATTERN_VALIDATION_TRUNCATION_RERUN', 'REASONING_EVALUATE_ENABLED', 'REASONING_MAX_INJECTED_CHARS', 'REASONING_WEIGHTS_ENABLED']
