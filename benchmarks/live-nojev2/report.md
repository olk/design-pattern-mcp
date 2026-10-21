# Benchmark run live-20261020-173400

- mode: `live` · split `holdout` · limit `None` · repeats `3`
- scenarios: 18 records (train 0 / holdout 18)
- corpus: `tests/benchmark/scenarios/seed.json` sha256 `ebfa882dce2b82e7…`
- git: `c643c2be0d30` dirty=True
- catalog: 486 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 18 | 72.2% | [0.491, 0.875] | 1 | 0.311 | 0.367 | 0.139 | 0.361 | 0 |
| train | 0 | null | null | null | null | null | null | null | null |
| holdout | 18 | 72.2% | [0.491, 0.875] | 1 | 0.311 | 0.367 | 0.139 | 0.361 | 0 |

## Latency

- e2e: n=18 p50=269386.7 p95=559745.8 (warmup 295973.9)
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 18 | 0 | 12596.1 | 539.8 | 1599.8 | embed_queries x18 |
| `rerank` | 28 | 0 | 734155.8 | 25145.2 | 36192.9 | rerank x28 |
| `llm.analyze` | 18 | 0 | 365063.2 | 19489.0 | 27597.6 | extract_brief_and_parts x18 |
| `llm.weights` | 18 | 0 | 300300.3 | 14909.1 | 25430.7 | extract_weights_batch x18 |
| `llm.generate` | 25 | 1 | 2126301.9 | 81251.0 | 134854.4 | generate x25 |
| `llm.evaluate` | 21 | 0 | 583993.7 | 25631.8 | 46438.4 | evaluate x21 |
| `reasoning.generate` | 22 | 0 | 634834.7 | 29896.9 | 36394.0 | run_pre_llm x22 |
| `reasoning.weights` | 18 | 0 | 608582.6 | 32042.4 | 44133.3 | run_pre_llm x18 |
| `reasoning.evaluate` | 21 | 0 | 608240.3 | 28096.1 | 37671.4 | run_pre_llm x21 |
| `reasoning.analyze` | 18 | 0 | 153668.2 | 0.0 | 33097.9 | run_pre_llm x18 |

## Calibration (train split, monitoring only)

- Brier: null · ECE (10 bins): null

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|

## Risk-coverage

- holdout: n=18 risk@80=0.267 risk@90=0.235 risk@100=0.278 AURC=0.165
- train: n=0 risk@80=null risk@90=null risk@100=null AURC=null

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `mapper` | `adapter` | 0.842 | 272714.7 | -14559.8 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✔ | `page-controller` | `application-controller` | 0.809 | 209861.5 | -17179.2 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.890 | 474610.5 | -22119.8 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✘ | `deadline-propagation` | `structured-concurrency` | 0.685 | 393463.7 | -10552.0 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✔ | `stub` | `mock` | 0.708 | 237415.7 | -14607.0 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✔ | `model-view-viewmodel` | `model-view-viewmodel` | 0.726 | 372739.1 | -118614.6 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `gateway` | `adapter` | 0.755 | 266058.7 | -13479.9 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✘ | `service-to-worker` | `application-controller` | 0.853 | 212391.4 | -18112.7 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.861 | 220385.3 | -10611.6 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✔ | `structured-concurrency` | `structured-concurrency` | 0.916 | 247897.8 | -10917.3 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✔ | `stub` | `mock` | 0.722 | 367108.7 | -20772.7 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✘ | `model-view-viewmodel` | `model-view-viewmodel` | 0.731 | 557682.9 | -80695.1 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✔ | `adapter` | `adapter` | 0.827 | 300444.0 | -7701.0 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✔ | `page-controller` | `application-controller` | 0.806 | 174177.4 | -15125.7 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.899 | 571435.8 | -13708.7 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✔ | `structured-concurrency` | `structured-concurrency` | 0.920 | 239375.7 | -19493.8 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✔ | `stub` | `mock` | 0.735 | 234709.4 | -11447.7 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✔ | `presentation-model` | `model-view-viewmodel` | 0.776 | 282718.6 | -72847.4 | 0 |

## Notes

- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
