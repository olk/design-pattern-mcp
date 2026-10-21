# Benchmark run live-20261020-191306

- mode: `live` · split `holdout` · limit `None` · repeats `3`
- scenarios: 18 records (train 0 / holdout 18)
- corpus: `tests/benchmark/scenarios/seed.json` sha256 `ebfa882dce2b82e7…`
- git: `c643c2be0d30` dirty=True
- catalog: 486 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 18 | 55.6% | [0.337, 0.754] | 0 | 0.289 | 0.322 | 0.042 | 0.278 | 0 |
| train | 0 | null | null | null | null | null | null | null | null |
| holdout | 18 | 55.6% | [0.337, 0.754] | 0 | 0.289 | 0.322 | 0.042 | 0.278 | 0 |

## Latency

- e2e: n=18 p50=242716.9 p95=508471.8 (warmup 241351.9)
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 18 | 0 | 15673.5 | 501.1 | 1981.9 | embed_queries x18 |
| `rerank` | 33 | 0 | 896363.0 | 25540.1 | 40782.3 | rerank x33 |
| `llm.analyze` | 18 | 0 | 335639.0 | 15993.8 | 33910.0 | extract_brief_and_parts x18 |
| `llm.weights` | 18 | 0 | 274670.8 | 13558.5 | 27374.5 | extract_weights_batch x18 |
| `llm.generate` | 30 | 0 | 2581736.4 | 77811.4 | 210357.5 | generate x30 |
| `llm.evaluate` | 21 | 0 | 510710.7 | 22005.6 | 43102.0 | evaluate x21 |
| `reasoning.weights` | 18 | 0 | 547062.0 | 29894.4 | 40189.4 | run_pre_llm x18 |
| `reasoning.generate` | 21 | 0 | 539901.9 | 24922.2 | 36646.9 | run_pre_llm x21 |
| `reasoning.evaluate` | 21 | 0 | 532255.8 | 24717.7 | 31558.4 | run_pre_llm x21 |
| `reasoning.analyze` | 18 | 0 | 116881.1 | 0.0 | 26028.9 | run_pre_llm x18 |

## Calibration (train split, monitoring only)

- Brier: null · ECE (10 bins): null

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|

## Risk-coverage

- holdout: n=18 risk@80=0.400 risk@90=0.412 risk@100=0.444 AURC=0.277
- train: n=0 risk@80=null risk@90=null risk@100=null AURC=null

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `mapper` | `adapter` | 0.893 | 214140.8 | -7712.2 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✘ | `service-to-worker` | `application-controller` | 0.846 | 296028.7 | -12713.3 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.721 | 402639.3 | -14148.2 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✔ | `structured-concurrency` | `structured-concurrency` | 0.963 | 487482.7 | -182391.0 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✘ | `configurable-test-double` | `mock` | 0.756 | 188455.9 | -13494.2 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✘ | `flux-redux` | `model-view-viewmodel` | 0.715 | 494527.3 | -105534.0 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `mapper` | `adapter` | 0.861 | 211475.9 | -12052.5 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✔ | `page-controller` | `application-controller` | 0.819 | 152373.6 | -9490.1 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.760 | 261070.8 | -22113.7 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✔ | `structured-concurrency` | `structured-concurrency` | 0.932 | 587490.8 | -187964.0 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✘ | `configurable-test-double` | `mock` | 0.774 | 180829.7 | -11190.2 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✔ | `presentation-model` | `model-view-viewmodel` | 0.768 | 455895.5 | -101716.0 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✔ | `adapter` | `adapter` | 0.840 | 179837.4 | -8892.5 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✔ | `page-controller` | `application-controller` | 0.800 | 172533.7 | -11868.1 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.865 | 224363.0 | -11654.4 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✔ | `structured-concurrency` | `structured-concurrency` | 0.993 | 375405.3 | -108156.2 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✘ | `configurable-test-double` | `mock` | 0.747 | 174652.7 | -13025.0 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✘ | `flux-redux` | `model-view-viewmodel` | 0.690 | 330104.3 | -127471.2 | 0 |

## Notes

- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
