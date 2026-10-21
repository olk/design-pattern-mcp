# Benchmark run live-20261020-120933

- mode: `live` · split `holdout` · limit `None` · repeats `3`
- scenarios: 18 records (train 0 / holdout 18)
- corpus: `tests/benchmark/scenarios/seed.json` sha256 `ebfa882dce2b82e7…`
- git: `c643c2be0d30` dirty=True
- catalog: 486 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 18 | 77.8% | [0.548, 0.910] | 1 | 0.278 | 0.344 | 0.095 | 0.389 | 0 |
| train | 0 | null | null | null | null | null | null | null | null |
| holdout | 18 | 77.8% | [0.548, 0.910] | 1 | 0.278 | 0.344 | 0.095 | 0.389 | 0 |

## Latency

- e2e: n=18 p50=265227.9 p95=429722.1 (warmup 260174.1)
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 18 | 0 | 12971.2 | 497.9 | 1596.7 | embed_queries x18 |
| `rerank` | 33 | 0 | 755749.6 | 22320.5 | 32952.9 | rerank x33 |
| `llm.analyze` | 18 | 0 | 352000.7 | 19294.9 | 29744.0 | extract_brief_and_parts x18 |
| `llm.weights` | 18 | 0 | 298217.1 | 14020.2 | 31272.9 | extract_weights_batch x18 |
| `llm.generate` | 20 | 0 | 1798142.1 | 93137.2 | 132847.9 | generate x20 |
| `llm.evaluate` | 19 | 0 | 538991.2 | 23489.1 | 51377.3 | evaluate x19 |
| `reasoning.weights` | 18 | 0 | 614935.9 | 32980.2 | 46313.3 | run_pre_llm x18 |
| `reasoning.evaluate` | 19 | 0 | 563409.3 | 28594.7 | 42179.8 | run_pre_llm x19 |
| `reasoning.generate` | 19 | 0 | 541765.8 | 29112.8 | 37531.2 | run_pre_llm x19 |
| `reasoning.analyze` | 18 | 0 | 136109.4 | 0.0 | 28284.0 | run_pre_llm x18 |

## Calibration (train split, monitoring only)

- Brier: null · ECE (10 bins): null

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|

## Risk-coverage

- holdout: n=18 risk@80=0.200 risk@90=0.235 risk@100=0.222 AURC=0.161
- train: n=0 risk@80=null risk@90=null risk@100=null AURC=null

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `gateway` | `adapter` | 0.821 | 262662.1 | -13224.0 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✔ | `page-controller` | `application-controller` | 0.823 | 196253.7 | -10134.6 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.899 | 267793.7 | -10772.0 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✔ | `structured-concurrency` | `structured-concurrency` | 0.921 | 331114.6 | -22900.4 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✔ | `stub` | `mock` | 0.725 | 374190.1 | -16523.4 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✔ | `presentation-model` | `model-view-viewmodel` | 0.745 | 300496.3 | -10382.0 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `gateway` | `adapter` | 0.796 | 256053.1 | -18963.3 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✔ | `page-controller` | `application-controller` | 0.648 | 170995.0 | -15108.9 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.886 | 216466.4 | -8966.9 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✔ | `structured-concurrency` | `structured-concurrency` | 0.909 | 213268.6 | -11120.7 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✔ | `stub` | `mock` | 0.768 | 580158.1 | -26552.9 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✔ | `presentation-model` | `model-view-viewmodel` | 0.741 | 345907.2 | -30848.2 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✔ | `adapter` | `adapter` | 0.694 | 214478.9 | -10609.1 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✘ | `chain-of-responsibility` | `application-controller` | 0.677 | 214792.6 | -11319.9 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.896 | 319224.4 | -14717.8 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✘ | `deadline-propagation` | `structured-concurrency` | 0.837 | 202855.7 | -12315.7 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✔ | `stub` | `mock` | 0.730 | 403174.6 | -33197.2 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✔ | `presentation-model` | `model-view-viewmodel` | 0.736 | 349125.6 | -115624.7 | 0 |

## Notes

- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
