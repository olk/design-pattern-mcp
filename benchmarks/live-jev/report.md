# Benchmark run live-20261020-152201

- mode: `live` · split `holdout` · limit `None` · repeats `3`
- scenarios: 18 records (train 0 / holdout 18)
- corpus: `tests/benchmark/scenarios/seed.json` sha256 `ebfa882dce2b82e7…`
- git: `c643c2be0d30` dirty=True
- catalog: 486 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 18 | 61.1% | [0.386, 0.797] | 0 | 0.244 | 0.256 | 0.033 | 0.306 | 0 |
| train | 0 | null | null | null | null | null | null | null | null |
| holdout | 18 | 61.1% | [0.386, 0.797] | 0 | 0.244 | 0.256 | 0.033 | 0.306 | 0 |

## Latency

- e2e: n=18 p50=282062.4 p95=436587.9 (warmup 308049.4)
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 18 | 0 | 11128.2 | 400.4 | 1477.5 | embed_queries x18 |
| `rerank` | 35 | 0 | 856183.7 | 22326.4 | 34955.2 | rerank x35 |
| `llm.analyze` | 18 | 0 | 368564.2 | 20570.0 | 25480.7 | extract_brief_and_parts x18 |
| `llm.weights` | 18 | 0 | 383199.9 | 16517.0 | 42621.0 | extract_weights_batch x18 |
| `llm.generate` | 22 | 0 | 1936227.0 | 86681.7 | 129938.7 | generate x22 |
| `llm.evaluate` | 18 | 0 | 614720.8 | 30265.8 | 58096.1 | evaluate x18 |
| `reasoning.weights` | 18 | 0 | 648367.7 | 33462.6 | 47613.9 | run_pre_llm x18 |
| `reasoning.generate` | 18 | 0 | 539951.7 | 31640.0 | 36062.8 | run_pre_llm x18 |
| `reasoning.evaluate` | 18 | 0 | 531554.4 | 27632.5 | 36841.7 | run_pre_llm x18 |
| `reasoning.analyze` | 18 | 0 | 157591.7 | 0.0 | 31726.2 | run_pre_llm x18 |

## Calibration (train split, monitoring only)

- Brier: null · ECE (10 bins): null

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|

## Risk-coverage

- holdout: n=18 risk@80=0.467 risk@90=0.412 risk@100=0.389 AURC=0.365
- train: n=0 risk@80=null risk@90=null risk@100=null AURC=null

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `mapper` | `adapter` | 0.856 | 261271.4 | -25096.5 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✘ | `service-to-worker` | `application-controller` | 0.852 | 200544.0 | -12371.7 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.739 | 258409.1 | -14685.0 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✘ | `thread-pool` | `structured-concurrency` | 0.816 | 403897.0 | -70100.6 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✘ | `configurable-test-double` | `mock` | 0.758 | 343759.9 | -15994.7 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✔ | `presentation-model` | `model-view-viewmodel` | 0.736 | 301608.1 | -16566.2 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✔ | `adapter` | `adapter` | 0.679 | 216960.3 | -16169.3 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✔ | `page-controller` | `application-controller` | 0.785 | 202748.5 | -16383.4 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.885 | 321993.7 | -15896.3 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✔ | `structured-concurrency` | `structured-concurrency` | 0.935 | 529774.3 | -172452.5 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✘ | `configurable-test-double` | `mock` | 0.752 | 295563.3 | -31751.4 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✔ | `model-view-viewmodel` | `model-view-viewmodel` | 0.700 | 268561.6 | -21018.4 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `mapper` | `adapter` | 0.847 | 224304.1 | -10956.5 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✔ | `page-controller` | `application-controller` | 0.806 | 203399.0 | -13045.7 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.874 | 238576.2 | -8903.9 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✔ | `structured-concurrency` | `structured-concurrency` | 0.776 | 384468.6 | -114029.4 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✘ | `configurable-test-double` | `mock` | 0.750 | 333947.1 | -44893.0 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✔ | `model-view-viewmodel` | `model-view-viewmodel` | 0.682 | 420143.2 | -17245.5 | 0 |

## Notes

- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
