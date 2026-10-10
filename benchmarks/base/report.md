# Benchmark run live-20261002-070756

- mode: `live` · split `all` · limit `None` · repeats `1`
- scenarios: 18 records (train 12 / holdout 6)
- corpus: `tests/benchmark/scenarios/seed.json` sha256 `ebfa882dce2b82e7…`
- git: `29cee564fabd` dirty=True
- catalog: 438 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 18 | 66.7% | [0.437, 0.837] | 1 | 0.244 | 0.356 | 0.155 | 0.333 | 0 |
| train | 12 | 83.3% | [0.552, 0.953] | 0 | 0.267 | 0.383 | 0.186 | 0.417 | 0 |
| holdout | 6 | 33.3% | [0.097, 0.700] | 1 | 0.200 | 0.300 | 0.092 | 0.167 | 0 |

## Latency

- e2e: n=18 p50=310448.2 p95=536899.5 (warmup 405362.2)
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 18 | 0 | 13295.1 | 587.3 | 1856.5 | embed_queries x18 |
| `rerank` | 40 | 0 | 1033518.0 | 25675.6 | 31535.5 | rerank x40 |
| `llm.analyze` | 18 | 0 | 371048.2 | 19646.9 | 26834.8 | extract_brief_and_parts x18 |
| `llm.weights` | 18 | 0 | 374028.4 | 19914.3 | 41957.4 | extract_weights_batch x18 |
| `llm.generate` | 26 | 0 | 2005135.7 | 80401.5 | 108771.5 | generate x26 |
| `llm.evaluate` | 20 | 0 | 720230.6 | 28296.9 | 56528.4 | evaluate x20 |
| `reasoning.weights` | 18 | 0 | 647984.5 | 35097.7 | 45663.1 | run_pre_llm x18 |
| `reasoning.generate` | 20 | 0 | 597880.2 | 29861.3 | 33469.6 | run_pre_llm x20 |
| `reasoning.evaluate` | 20 | 0 | 570670.5 | 28774.9 | 36205.9 | run_pre_llm x20 |
| `reasoning.analyze` | 18 | 0 | 467428.2 | 27288.8 | 34156.0 | run_pre_llm x18 |

## Calibration (train split, monitoring only)

- Brier: 0.126 · ECE (10 bins): 0.079

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|
| 7 | [0.7, 0.8) | 4 | 0.778 | 0.750 | 0.028 |
| 8 | [0.8, 0.9) | 3 | 0.852 | 0.667 | 0.185 |
| 9 | [0.9, 1.0) | 5 | 0.945 | 1.000 | 0.055 |

## Risk-coverage

- holdout: n=6 risk@80=0.600 risk@90=0.667 risk@100=0.667 AURC=0.294
- train: n=12 risk@80=0.200 risk@90=0.182 risk@100=0.167 AURC=0.058

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `dm-order-lifecycle` | train | domain-modeling | ✘ | `domain-event` | `entity` | 0.786 | 434842.1 | -77782.5 | 0 |
| `dm-currency-ledger` | train | domain-modeling | ✔ | `money` | `money` | 0.937 | 247389.5 | -18829.4 | 0 |
| `cc-report-builder` | train | construction-composition | ✔ | `builder` | `builder` | 0.840 | 363334.3 | -22176.1 | 0 |
| `bs-pricing-strategy` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.885 | 539822.7 | -119122.8 | 0 |
| `bs-ast-visitor` | train | behavior-selection | ✔ | `visitor` | `visitor` | 0.793 | 248786.1 | -10644.8 | 0 |
| `ib-billing-gateway` | train | interaction-boundary | ✘ | `adapter` | `gateway` | 0.831 | 232560.7 | -12822.1 | 0 |
| `wm-order-router` | train | workflow-messaging | ✔ | `content-based-router` | `content-based-router` | 0.760 | 374968.0 | -14480.6 | 0 |
| `ps-catalog-repository` | train | persistence-state | ✔ | `repository` | `repository` | 0.918 | 284891.0 | -20282.1 | 0 |
| `cr-billing-circuit-breaker` | train | concurrency-resilience | ✔ | `circuit-breaker` | `circuit-breaker` | 0.952 | 318515.9 | -12740.4 | 0 |
| `sv-api-authorization` | train | security-verification | ✔ | `authorization-enforcer` | `authorization-enforcer` | 0.927 | 278417.2 | -13336.7 | 0 |
| `sc-expression-parser` | train | specialized-components | ✔ | `recursive-descent-parser` | `recursive-descent-parser` | 0.774 | 302380.5 | -22702.1 | 0 |
| `cc-module-wiring` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.990 | 335789.4 | -95835.8 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✔ | `adapter` | `adapter` | 0.845 | 261294.6 | -8456.6 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✘ | `service-to-worker` | `application-controller` | 0.822 | 291456.3 | -19442.0 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.895 | 265784.8 | -10958.1 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✘ | `async-method-invocation` | `structured-concurrency` | 0.822 | 455128.6 | -15536.6 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✘ | `configurable-test-double` | `mock` | 0.777 | 400215.1 | -92795.9 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✘ | `flux-redux` | `model-view-viewmodel` | 0.000 | 536383.6 | -41314.3 | 0 |
