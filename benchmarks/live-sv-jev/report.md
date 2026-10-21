# Benchmark run live-20261021-051648

- mode: `live` · split `holdout` · limit `6` · repeats `1`
- scenarios: 6 records (train 0 / holdout 6)
- corpus: `tests/benchmark/scenarios/seed.json` sha256 `ebfa882dce2b82e7…`
- git: `c643c2be0d30` dirty=True
- catalog: 486 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 6 | 50.0% | [0.188, 0.812] | 0 | 0.267 | 0.267 | 0.083 | 0.250 | 0 |
| train | 0 | null | null | null | null | null | null | null | null |
| holdout | 6 | 50.0% | [0.188, 0.812] | 0 | 0.267 | 0.267 | 0.083 | 0.250 | 0 |

## Latency

- e2e: n=6 p50=353626.9 p95=546569.9 (warmup 302839.6)
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 6 | 0 | 4973.6 | 572.5 | 1596.2 | embed_queries x6 |
| `rerank` | 9 | 0 | 253795.1 | 26955.8 | 35714.7 | rerank x9 |
| `llm.analyze` | 6 | 0 | 135657.7 | 20730.7 | 31106.0 | extract_brief_and_parts x6 |
| `llm.weights` | 6 | 0 | 132067.1 | 14834.1 | 44508.0 | extract_weights_batch x6 |
| `llm.generate` | 8 | 0 | 858199.4 | 116504.0 | 173512.7 | generate x8 |
| `llm.evaluate` | 7 | 0 | 235186.8 | 31188.5 | 54624.2 | evaluate x7 |
| `reasoning.weights` | 6 | 0 | 225871.6 | 36998.8 | 48502.1 | run_pre_llm x6 |
| `reasoning.evaluate` | 7 | 0 | 217208.4 | 30038.5 | 42075.3 | run_pre_llm x7 |
| `reasoning.generate` | 7 | 0 | 201657.6 | 33570.1 | 37029.6 | run_pre_llm x7 |
| `reasoning.analyze` | 6 | 0 | 170542.8 | 28452.8 | 46035.8 | run_pre_llm x6 |

## Calibration (train split, monitoring only)

- Brier: null · ECE (10 bins): null

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|

## Risk-coverage

- holdout: n=6 risk@80=0.400 risk@90=0.500 risk@100=0.500 AURC=0.247
- train: n=0 risk@80=null risk@90=null risk@100=null AURC=null

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `mapper` | `adapter` | 0.822 | 238911.5 | -27394.4 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✘ | `chain-of-responsibility` | `application-controller` | 0.715 | 239086.8 | -13685.5 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.893 | 429816.2 | -10995.2 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✔ | `structured-concurrency` | `structured-concurrency` | 0.984 | 585487.8 | -14747.5 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✘ | `configurable-test-double` | `mock` | 0.766 | 284465.4 | -47804.3 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✔ | `presentation-model` | `model-view-viewmodel` | 0.743 | 422788.5 | -119977.0 | 0 |

## Notes

- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
