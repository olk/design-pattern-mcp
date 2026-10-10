# Benchmark comparison

- baseline (baseline): `live-cfg-reasoning` mode `live` (18 records) in `benchmarks/cfg-reasoning`
- candidate (candidate): `live-code-levers` mode `live` (18 records) in `benchmarks/code-levers`
- scenario file match: True · paired scenarios: 18

## Verdict: `candidate-loss`

_candidate wins iff holdout p95 e2e improves >= 15% AND candidate overall acceptable-primary hit rate >= baseline overall Wilson lower bound_

- p95 e2e improved 12.5% (< 15%)
- holdout p95 e2e: baseline 339921.3 ms → candidate 297476.7 ms (12.5%)

## Hit rate (Wilson / Newcombe)

| split | baseline | candidate | difference [95% CI] |
|---|---|---|---|
| overall | 66.7% | 72.2% | +0.056 [-0.232, +0.331] |
| train | 66.7% | 75.0% | +0.083 [-0.260, +0.403] |
| holdout | 66.7% | 66.7% | +0.000 [-0.436, +0.436] |

## Paired deltas (candidate minus baseline)

- e2e: mean -36272.2494 ms · median -10610.3110 ms · faster 10 / slower 8 · sign test p=0.814529 (n=18)

| metric | compared | mean Δ | median Δ | improved | worsened | unchanged |
|---|---|---|---|---|---|---|
| recall_at_3 | 18 | -0.0444 | +0.0000 | 2 | 5 | 11 |
| recall_at_5 | 18 | -0.0778 | +0.0000 | 1 | 7 | 10 |
| supporting_f1 | 18 | -0.0583 | +0.0000 | 1 | 3 | 14 |
| acceptable_primary_f1 | 18 | +0.0278 | +0.0000 | 4 | 3 | 11 |
| veto_violations | 18 | +0.0000 | +0.0000 | 0 | 0 | 18 |
| score | 18 | +0.0073 | +0.0029 | 10 | 8 | 0 |

## Hit flips

| key | baseline | candidate |
|---|---|---|
| cc-legacy-adapter#1 | hit `adapter` | miss `mapper` |
| cc-report-builder#1 | miss `functional-options` | hit `builder` |
| cr-billing-circuit-breaker#1 | miss `fail-fast` | hit `circuit-breaker` |
| ib-admin-controller#1 | miss `service-to-worker` | hit `page-controller` |
| ib-billing-gateway#1 | hit `gateway` | miss `messaging-gateway` |
| sc-dashboard-mvvm#1 | hit `model-view-viewmodel` | miss `presentation-model` |
| sv-order-double-tests#1 | miss `recording-fake` | hit `stub` |

## Stage deltas

| stage | calls Δ | total ms baseline | total ms candidate | Δ ms |
|---|---|---|---|---|
| `embed.query` | +0 | 17216.8 | 18744.7 | +1527.8 |
| `llm.analyze` | +0 | 346421.0 | 331995.3 | -14425.7 |
| `llm.evaluate` | -2 | 706632.1 | 565384.0 | -141248.1 |
| `llm.generate` | -2 | 2103241.5 | 1646422.6 | -456818.9 |
| `llm.weights` | +0 | 264940.4 | 230973.0 | -33967.4 |
| `reasoning.analyze` | +0 | 510356.8 | 465422.7 | -44934.1 |
| `reasoning.evaluate` | -2 | 0.6 | 0.5 | -0.1 |
| `reasoning.generate` | -2 | 534933.0 | 513315.3 | -21617.7 |
| `reasoning.weights` | +0 | 0.4 | 0.4 | -0.1 |
| `rerank` | +0 | 1230239.2 | 1278365.6 | +48126.4 |

## Warnings

- effective config differs in ['prompts', 'reasoning', 'validation']
- environment differs in ['DESIGN_PATTERN_PROMPT_STABLE_PREFIX', 'DESIGN_PATTERN_VALIDATION_TRUNCATION_RERUN', 'REASONING_MAX_INJECTED_CHARS']
