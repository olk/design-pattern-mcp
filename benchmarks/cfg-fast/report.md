# Benchmark run live-cfg-fast

- mode: `live` · split `all` · limit `None` · repeats `1`
- scenarios: 18 records (train 12 / holdout 6)
- corpus: `tests/benchmark/scenarios/seed.json` sha256 `ebfa882dce2b82e7…`
- git: `d3b0d4626308` dirty=True
- catalog: 438 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 18 | 55.6% | [0.337, 0.754] | 0 | 0.300 | 0.367 | 0.127 | 0.278 | 0 |
| train | 12 | 58.3% | [0.320, 0.807] | 0 | 0.300 | 0.367 | 0.136 | 0.292 | 0 |
| holdout | 6 | 50.0% | [0.188, 0.812] | 0 | 0.300 | 0.367 | 0.111 | 0.250 | 0 |

## Latency

- e2e: n=18 p50=287556.6 p95=499132.7
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 18 | 0 | 14019.1 | 719.5 | 1369.8 | embed_queries x18 |
| `rerank` | 31 | 0 | 273546.8 | 8855.8 | 10682.3 | rerank x31 |
| `llm.analyze` | 18 | 0 | 345591.0 | 18808.5 | 25819.2 | extract_brief_and_parts x18 |
| `llm.weights` | 19 | 1 | 584056.9 | 15135.6 | 57646.6 | extract_weights_batch x19 |
| `llm.generate` | 25 | 0 | 1980763.7 | 74359.3 | 122372.0 | generate x25 |
| `llm.evaluate` | 20 | 0 | 543050.7 | 24110.4 | 45452.4 | evaluate x20 |
| `reasoning.generate` | 20 | 0 | 665260.0 | 29827.7 | 51969.2 | run_pre_llm x20 |
| `reasoning.weights` | 18 | 0 | 611162.5 | 33409.4 | 40423.6 | run_pre_llm x18 |
| `reasoning.evaluate` | 20 | 0 | 601504.3 | 30128.9 | 35826.4 | run_pre_llm x20 |
| `reasoning.analyze` | 18 | 0 | 528078.1 | 28051.7 | 36263.4 | run_pre_llm x18 |

## Calibration (train split, monitoring only)

- Brier: 0.293 · ECE (10 bins): 0.280

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|
| 6 | [0.6, 0.7) | 1 | 0.668 | 0.000 | 0.668 |
| 7 | [0.7, 0.8) | 2 | 0.785 | 0.500 | 0.285 |
| 8 | [0.8, 0.9) | 3 | 0.857 | 0.333 | 0.524 |
| 9 | [0.9, 1.0) | 6 | 0.926 | 0.833 | 0.093 |

## Risk-coverage

- holdout: n=6 risk@80=0.400 risk@90=0.500 risk@100=0.500 AURC=0.289
- train: n=12 risk@80=0.400 risk@90=0.364 risk@100=0.417 AURC=0.167

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `dm-order-lifecycle` | train | domain-modeling | ✘ | `make-illegal-states-unrepresentable` | `entity` | 0.895 | 332120.8 | -93580.6 | 0 |
| `dm-currency-ledger` | train | domain-modeling | ✔ | `money` | `money` | 0.915 | 231211.5 | -8520.4 | 0 |
| `cc-report-builder` | train | construction-composition | ✘ | `macro-method` | `builder` | 0.833 | 318519.5 | -22466.1 | 0 |
| `bs-pricing-strategy` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.921 | 225934.4 | -11447.1 | 0 |
| `bs-ast-visitor` | train | behavior-selection | ✘ | `collecting-parameter` | `visitor` | 0.799 | 698078.1 | -18040.1 | 0 |
| `ib-billing-gateway` | train | interaction-boundary | ✔ | `mapper` | `gateway` | 0.843 | 234591.6 | -11115.6 | 0 |
| `wm-order-router` | train | workflow-messaging | ✔ | `content-based-router` | `content-based-router` | 0.770 | 339577.9 | -55228.6 | 0 |
| `ps-catalog-repository` | train | persistence-state | ✘ | `data-mapper` | `repository` | 0.911 | 464024.7 | -109509.6 | 0 |
| `cr-billing-circuit-breaker` | train | concurrency-resilience | ✔ | `circuit-breaker` | `circuit-breaker` | 0.936 | 246829.5 | -8498.0 | 0 |
| `sv-api-authorization` | train | security-verification | ✔ | `authorization-enforcer` | `authorization-enforcer` | 0.939 | 212892.4 | -8640.1 | 0 |
| `sc-expression-parser` | train | specialized-components | ✘ | `irregular-heterogeneous-ast` | `recursive-descent-parser` | 0.668 | 329845.4 | -64456.0 | 0 |
| `cc-module-wiring` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.934 | 254186.0 | -9235.8 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `mapper` | `adapter` | 0.700 | 242499.4 | -7818.6 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✔ | `page-controller` | `application-controller` | 0.836 | 279359.0 | -8642.8 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.885 | 359287.0 | -7985.3 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✘ | `future-promise` | `structured-concurrency` | 0.837 | 295754.2 | -9598.0 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✔ | `stub` | `mock` | 0.733 | 221294.8 | -8467.5 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✘ | `model-view-intent` | `model-view-viewmodel` | 0.729 | 331675.5 | -66101.1 | 0 |

## Notes

- NO_WARMUP protocol deviation (2026-10-07 latency-reduction round): the warmup scenario is also a measured scenario, so it stays in the sample set; the reasoning LRU would otherwise hand it a ~0 ms reasoning.analyze cache hit (warmup-cache artifact understating the run mean). Warmup-on arms (e.g. the committed benchmarks/base, live-20261002-070756) are NOT mutually comparable with NO_WARMUP arms; base-fresh is the baseline for every comparison this round.
- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
