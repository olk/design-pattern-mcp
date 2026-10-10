# Benchmark run live-20261002-101554

- mode: `live` · split `all` · limit `None` · repeats `1`
- scenarios: 18 records (train 12 / holdout 6)
- corpus: `tests/benchmark/scenarios/seed.json` sha256 `ebfa882dce2b82e7…`
- git: `29cee564fabd` dirty=True
- catalog: 438 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 18 | 33.3% | [0.163, 0.563] | 3 | 0.244 | 0.289 | 0.072 | 0.167 | 0 |
| train | 12 | 25.0% | [0.089, 0.532] | 1 | 0.250 | 0.283 | 0.044 | 0.125 | 0 |
| holdout | 6 | 50.0% | [0.188, 0.812] | 2 | 0.233 | 0.300 | 0.127 | 0.250 | 0 |

## Latency

- e2e: n=18 p50=220641.8 p95=376227.5 (warmup 255356.7)
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 18 | 0 | 10243.0 | 566.7 | 966.5 | embed_queries x18 |
| `rerank` | 45 | 0 | 1164241.1 | 26224.2 | 33317.6 | rerank x45 |
| `llm.analyze` | 18 | 0 | 335839.9 | 18653.8 | 25733.7 | extract_brief_and_parts x18 |
| `llm.weights` | 18 | 0 | 310583.1 | 14474.2 | 33774.5 | extract_weights_batch x18 |
| `llm.generate` | 29 | 0 | 2261095.9 | 77290.4 | 112578.8 | generate x29 |
| `llm.evaluate` | 23 | 0 | 778526.5 | 29239.0 | 68112.1 | evaluate x23 |

## Calibration (train split, monitoring only)

- Brier: 0.542 · ECE (10 bins): 0.568

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|
| 6 | [0.6, 0.7) | 2 | 0.673 | 0.000 | 0.673 |
| 7 | [0.7, 0.8) | 3 | 0.725 | 0.667 | 0.058 |
| 8 | [0.8, 0.9) | 5 | 0.876 | 0.200 | 0.676 |
| 9 | [0.9, 1.0) | 2 | 0.955 | 0.000 | 0.955 |

## Risk-coverage

- holdout: n=6 risk@80=0.400 risk@90=0.500 risk@100=0.500 AURC=0.247
- train: n=12 risk@80=0.700 risk@90=0.727 risk@100=0.750 AURC=0.840

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `dm-order-lifecycle` | train | domain-modeling | ✔ | `aggregate` | `entity` | 0.722 | 449611.9 | -217101.3 | 0 |
| `dm-currency-ledger` | train | domain-modeling | ✘ | `quantity` | `money` | 0.898 | 138353.8 | -10933.9 | 0 |
| `cc-report-builder` | train | construction-composition | ✔ | `builder` | `builder` | 0.854 | 113156.8 | -14038.2 | 0 |
| `bs-pricing-strategy` | train | behavior-selection | ✘ | `interface-class` | `strategy` | 0.877 | 254646.7 | -22165.3 | 0 |
| `bs-ast-visitor` | train | behavior-selection | ✔ | `visitor` | `visitor` | 0.745 | 225781.6 | -9968.8 | 0 |
| `ib-billing-gateway` | train | interaction-boundary | ✘ | `adapter` | `gateway` | 0.690 | 215270.0 | -34499.3 | 0 |
| `wm-order-router` | train | workflow-messaging | ✘ | `content-based-serializer` | `content-based-router` | 0.869 | 242990.1 | -83426.1 | 0 |
| `ps-catalog-repository` | train | persistence-state | ✘ | `data-mapper` | `repository` | 0.883 | 363277.4 | -14791.1 | 0 |
| `cr-billing-circuit-breaker` | train | concurrency-resilience | ✘ | `health-check-endpoint` | `circuit-breaker` | 0.960 | 254001.3 | -91213.6 | 0 |
| `sv-api-authorization` | train | security-verification | ✘ | `policy-guard` | `authorization-enforcer` | 0.950 | 180251.6 | -23765.6 | 0 |
| `sc-expression-parser` | train | specialized-components | ✘ | `irregular-heterogeneous-ast` | `recursive-descent-parser` | 0.656 | 121364.6 | -12735.0 | 0 |
| `cc-module-wiring` | train | construction-composition | ✘ | `dao-factory` | `dependency-injection` | 0.707 | 237342.5 | -16761.2 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `gateway` | `adapter` | 0.810 | 242252.1 | -8578.9 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✘ | `service-to-worker` | `application-controller` | 0.802 | 127962.7 | -23418.8 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.884 | 210335.2 | -115020.6 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✔ | `structured-concurrency` | `structured-concurrency` | 0.905 | 272703.5 | -110361.2 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✔ | `stub` | `mock` | 0.722 | 166464.2 | -7922.3 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✘ | `transaction-script` | `model-view-viewmodel` | 0.000 | 215501.9 | -12560.4 | 0 |
