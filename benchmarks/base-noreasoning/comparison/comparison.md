# Benchmark comparison

- baseline (baseline): `live-20261002-070756` mode `live` (18 records) in `benchmarks/base`
- candidate (candidate): `live-20261002-101554` mode `live` (18 records) in `benchmarks/base-noreasoning`
- scenario file match: True · paired scenarios: 18

## Verdict: `candidate-loss`

_candidate wins iff holdout p95 e2e improves >= 15% AND candidate overall acceptable-primary hit rate >= baseline overall Wilson lower bound_

- candidate overall hit rate 0.333 is below the baseline overall Wilson lower bound 0.437
- holdout p95 e2e: baseline 516069.9 ms → candidate 265090.7 ms (48.6%)

## Hit rate (Wilson / Newcombe)

| split | baseline | candidate | difference [95% CI] |
|---|---|---|---|
| overall | 66.7% | 33.3% | -0.333 [-0.575, -0.009] |
| train | 83.3% | 25.0% | -0.583 [-0.784, -0.185] |
| holdout | 33.3% | 50.0% | +0.167 [-0.315, +0.559] |

## Paired deltas (candidate minus baseline)

- e2e: mean -118927.3677 ms · median -103741.3215 ms · faster 16 / slower 2 · sign test p=0.001312 (n=18)

| metric | compared | mean Δ | median Δ | improved | worsened | unchanged |
|---|---|---|---|---|---|---|
| recall_at_3 | 18 | -0.0000 | +0.0000 | 3 | 3 | 12 |
| recall_at_5 | 18 | -0.0667 | +0.0000 | 2 | 7 | 9 |
| supporting_f1 | 18 | -0.0830 | +0.0000 | 1 | 7 | 10 |
| acceptable_primary_f1 | 18 | -0.1667 | -0.2500 | 3 | 9 | 6 |
| veto_violations | 18 | +0.0000 | +0.0000 | 0 | 0 | 18 |
| score | 18 | -0.0342 | -0.0273 | 5 | 12 | 1 |

## Hit flips

| key | baseline | candidate |
|---|---|---|
| bs-pricing-strategy#1 | hit `strategy` | miss `interface-class` |
| cc-legacy-adapter#1 | hit `adapter` | miss `gateway` |
| cc-module-wiring#1 | hit `dependency-injection` | miss `dao-factory` |
| cr-billing-circuit-breaker#1 | hit `circuit-breaker` | miss `health-check-endpoint` |
| cr-structured-fanout#1 | miss `async-method-invocation` | hit `structured-concurrency` |
| dm-currency-ledger#1 | hit `money` | miss `quantity` |
| dm-order-lifecycle#1 | miss `domain-event` | hit `aggregate` |
| ps-catalog-repository#1 | hit `repository` | miss `data-mapper` |
| sc-expression-parser#1 | hit `recursive-descent-parser` | miss `irregular-heterogeneous-ast` |
| sv-api-authorization#1 | hit `authorization-enforcer` | miss `policy-guard` |
| sv-order-double-tests#1 | miss `configurable-test-double` | hit `stub` |
| wm-order-router#1 | hit `content-based-router` | miss `content-based-serializer` |

## Stage deltas

| stage | calls Δ | total ms baseline | total ms candidate | Δ ms |
|---|---|---|---|---|
| `embed.query` | +0 | 13295.1 | 10243.0 | -3052.1 |
| `llm.analyze` | +0 | 371048.2 | 335839.9 | -35208.3 |
| `llm.evaluate` | +3 | 720230.6 | 778526.5 | +58295.9 |
| `llm.generate` | +3 | 2005135.7 | 2261095.9 | +255960.2 |
| `llm.weights` | +0 | 374028.4 | 310583.1 | -63445.3 |
| `reasoning.analyze` | -18 | 467428.2 | 0.0 | -467428.2 |
| `reasoning.evaluate` | -20 | 570670.5 | 0.0 | -570670.5 |
| `reasoning.generate` | -20 | 597880.2 | 0.0 | -597880.2 |
| `reasoning.weights` | -18 | 647984.5 | 0.0 | -647984.5 |
| `rerank` | +5 | 1033518.0 | 1164241.1 | +130723.1 |

## Warnings

- effective config differs in ['reasoning']
- environment differs in ['REASONING_ENABLED']
