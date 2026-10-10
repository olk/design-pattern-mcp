# Benchmark run live-code-levers

- mode: `live` · split `all` · limit `None` · repeats `1`
- scenarios: 18 records (train 12 / holdout 6)
- corpus: `tests/benchmark/scenarios/seed.json` sha256 `ebfa882dce2b82e7…`
- git: `d3b0d4626308` dirty=True
- catalog: 438 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 18 | 72.2% | [0.491, 0.875] | 1 | 0.233 | 0.289 | 0.067 | 0.361 | 0 |
| train | 12 | 75.0% | [0.468, 0.911] | 0 | 0.217 | 0.283 | 0.100 | 0.375 | 0 |
| holdout | 6 | 66.7% | [0.300, 0.903] | 1 | 0.267 | 0.300 | 0.000 | 0.333 | 0 |

## Latency

- e2e: n=18 p50=223647.0 p95=379190.6
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 18 | 0 | 18744.7 | 827.1 | 2053.7 | embed_queries x18 |
| `rerank` | 37 | 0 | 1278365.6 | 34537.5 | 43575.2 | rerank x37 |
| `llm.analyze` | 18 | 0 | 331995.3 | 16424.4 | 30158.9 | extract_brief_and_parts x18 |
| `llm.weights` | 18 | 0 | 230973.0 | 12228.0 | 18610.9 | extract_weights_batch x18 |
| `llm.generate` | 23 | 0 | 1646422.6 | 75620.5 | 118456.0 | generate x23 |
| `llm.evaluate` | 18 | 0 | 565384.0 | 25369.6 | 54750.5 | evaluate x18 |
| `reasoning.generate` | 18 | 0 | 513315.3 | 27670.8 | 40009.0 | run_pre_llm x18 |
| `reasoning.analyze` | 18 | 0 | 465422.7 | 24835.4 | 32710.4 | run_pre_llm x18 |
| `reasoning.evaluate` | 18 | 0 | 0.5 | 0.0 | 0.0 | run_pre_llm x18 |
| `reasoning.weights` | 18 | 0 | 0.4 | 0.0 | 0.0 | run_pre_llm x18 |

## Calibration (train split, monitoring only)

- Brier: 0.211 · ECE (10 bins): 0.222

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|
| 6 | [0.6, 0.7) | 1 | 0.688 | 1.000 | 0.312 |
| 7 | [0.7, 0.8) | 2 | 0.782 | 1.000 | 0.218 |
| 8 | [0.8, 0.9) | 5 | 0.843 | 0.600 | 0.243 |
| 9 | [0.9, 1.0) | 4 | 0.927 | 0.750 | 0.177 |

## Risk-coverage

- holdout: n=6 risk@80=0.400 risk@90=0.333 risk@100=0.333 AURC=0.192
- train: n=12 risk@80=0.300 risk@90=0.273 risk@100=0.250 AURC=0.251

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `dm-order-lifecycle` | train | domain-modeling | ✔ | `entity` | `entity` | 0.864 | 370037.8 | -90129.3 | 0 |
| `dm-currency-ledger` | train | domain-modeling | ✔ | `money` | `money` | 0.780 | 247923.9 | -17520.1 | 0 |
| `cc-report-builder` | train | construction-composition | ✔ | `builder` | `builder` | 0.848 | 431056.3 | -132376.8 | 0 |
| `bs-pricing-strategy` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.914 | 311508.8 | -61026.7 | 0 |
| `bs-ast-visitor` | train | behavior-selection | ✔ | `visitor` | `visitor` | 0.784 | 304404.4 | -12133.4 | 0 |
| `ib-billing-gateway` | train | interaction-boundary | ✘ | `messaging-gateway` | `gateway` | 0.818 | 318851.6 | -12178.0 | 0 |
| `wm-order-router` | train | workflow-messaging | ✘ | `content-based-serializer` | `content-based-router` | 0.929 | 176393.3 | -8300.5 | 0 |
| `ps-catalog-repository` | train | persistence-state | ✔ | `repository` | `repository` | 0.903 | 178302.5 | -6734.1 | 0 |
| `cr-billing-circuit-breaker` | train | concurrency-resilience | ✔ | `circuit-breaker` | `circuit-breaker` | 0.961 | 178241.1 | -13533.2 | 0 |
| `sv-api-authorization` | train | security-verification | ✘ | `front-controller` | `authorization-enforcer` | 0.848 | 363648.4 | -18105.5 | 0 |
| `sc-expression-parser` | train | specialized-components | ✔ | `recursive-descent-parser` | `recursive-descent-parser` | 0.688 | 215579.7 | -8889.9 | 0 |
| `cc-module-wiring` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.835 | 196450.2 | -10159.0 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `mapper` | `adapter` | 0.857 | 213845.3 | -9551.1 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✔ | `page-controller` | `application-controller` | 0.813 | 132624.2 | -15299.6 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.876 | 231714.3 | -8954.9 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✔ | `structured-concurrency` | `structured-concurrency` | 0.914 | 165547.4 | -13098.0 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✔ | `stub` | `mock` | 0.748 | 142631.4 | -11731.9 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✘ | `presentation-model` | `model-view-viewmodel` | 0.751 | 319397.5 | -102743.9 | 0 |

## Notes

- NO_WARMUP protocol deviation (2026-10-07 latency-reduction round): the warmup scenario is also a measured scenario, so it stays in the sample set; the reasoning LRU would otherwise hand it a ~0 ms reasoning.analyze cache hit (warmup-cache artifact understating the run mean). Warmup-on arms (e.g. the committed benchmarks/base, live-20261002-070756) are NOT mutually comparable with NO_WARMUP arms; base-fresh is the baseline for every comparison this round.
- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
