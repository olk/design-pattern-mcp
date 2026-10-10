# Benchmark run live-base-fresh

- mode: `live` · split `all` · limit `None` · repeats `1`
- scenarios: 18 records (train 12 / holdout 6)
- corpus: `tests/benchmark/scenarios/seed.json` sha256 `ebfa882dce2b82e7…`
- git: `d3b0d4626308` dirty=True
- catalog: 438 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 18 | 61.1% | [0.386, 0.797] | 1 | 0.278 | 0.333 | 0.111 | 0.306 | 0 |
| train | 12 | 83.3% | [0.552, 0.953] | 0 | 0.267 | 0.333 | 0.125 | 0.417 | 0 |
| holdout | 6 | 16.7% | [0.030, 0.564] | 1 | 0.300 | 0.333 | 0.083 | 0.083 | 0 |

## Latency

- e2e: n=18 p50=288386.5 p95=439610.3
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 18 | 0 | 12956.1 | 569.5 | 1396.9 | embed_queries x18 |
| `rerank` | 35 | 0 | 969259.4 | 26463.0 | 40326.1 | rerank x35 |
| `llm.analyze` | 18 | 0 | 297201.1 | 15014.0 | 25317.4 | extract_brief_and_parts x18 |
| `llm.weights` | 18 | 0 | 309180.9 | 11702.0 | 42785.0 | extract_weights_batch x18 |
| `llm.generate` | 26 | 0 | 1964090.8 | 67919.5 | 138430.0 | generate x26 |
| `llm.evaluate` | 20 | 0 | 543191.6 | 24833.6 | 46053.2 | evaluate x20 |
| `reasoning.generate` | 20 | 0 | 570787.6 | 27542.2 | 36974.0 | run_pre_llm x20 |
| `reasoning.weights` | 18 | 0 | 538530.7 | 29603.3 | 37016.7 | run_pre_llm x18 |
| `reasoning.evaluate` | 20 | 0 | 532586.4 | 25135.4 | 41939.9 | run_pre_llm x20 |
| `reasoning.analyze` | 18 | 0 | 451002.0 | 25599.3 | 30800.2 | run_pre_llm x18 |

## Calibration (train split, monitoring only)

- Brier: 0.086 · ECE (10 bins): 0.213

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|
| 6 | [0.6, 0.7) | 2 | 0.649 | 0.000 | 0.649 |
| 7 | [0.7, 0.8) | 2 | 0.781 | 1.000 | 0.219 |
| 8 | [0.8, 0.9) | 3 | 0.855 | 1.000 | 0.145 |
| 9 | [0.9, 1.0) | 5 | 0.924 | 1.000 | 0.076 |

## Risk-coverage

- holdout: n=6 risk@80=0.800 risk@90=0.833 risk@100=0.833 AURC=0.689
- train: n=12 risk@80=0.000 risk@90=0.091 risk@100=0.167 AURC=0.015

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `dm-order-lifecycle` | train | domain-modeling | ✔ | `entity` | `entity` | 0.865 | 236867.5 | -10543.5 | 0 |
| `dm-currency-ledger` | train | domain-modeling | ✔ | `money` | `money` | 0.926 | 358540.8 | -98761.1 | 0 |
| `cc-report-builder` | train | construction-composition | ✔ | `builder` | `builder` | 0.811 | 172823.8 | -11182.1 | 0 |
| `bs-pricing-strategy` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.929 | 245415.1 | -9191.2 | 0 |
| `bs-ast-visitor` | train | behavior-selection | ✔ | `visitor` | `visitor` | 0.799 | 387377.4 | -66659.5 | 0 |
| `ib-billing-gateway` | train | interaction-boundary | ✔ | `gateway` | `gateway` | 0.906 | 217335.0 | -12943.8 | 0 |
| `wm-order-router` | train | workflow-messaging | ✔ | `content-based-router` | `content-based-router` | 0.763 | 296539.3 | -10691.4 | 0 |
| `ps-catalog-repository` | train | persistence-state | ✔ | `repository` | `repository` | 0.889 | 257476.1 | -10872.7 | 0 |
| `cr-billing-circuit-breaker` | train | concurrency-resilience | ✔ | `circuit-breaker` | `circuit-breaker` | 0.934 | 206578.4 | -12774.3 | 0 |
| `sv-api-authorization` | train | security-verification | ✔ | `authorization-enforcer` | `authorization-enforcer` | 0.927 | 234180.5 | -8105.6 | 0 |
| `sc-expression-parser` | train | specialized-components | ✘ | `interpreter` | `recursive-descent-parser` | 0.615 | 451416.2 | -170587.6 | 0 |
| `cc-module-wiring` | train | construction-composition | ✘ | `interface-class` | `dependency-injection` | 0.682 | 405486.5 | -97167.4 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `mapper` | `adapter` | 0.822 | 280233.6 | -11781.2 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✘ | `service-to-worker` | `application-controller` | 0.860 | 330868.9 | -9288.1 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.866 | 246454.7 | -11526.9 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✘ | `thread-pool` | `structured-concurrency` | 0.915 | 437527.0 | -8424.9 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✘ | `configurable-test-double` | `mock` | 0.742 | 329017.8 | -88685.2 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✘ | `presentation-model` | `model-view-viewmodel` | 0.788 | 331106.9 | -114354.7 | 0 |

## Notes

- NO_WARMUP protocol deviation (2026-10-07 latency-reduction round): the warmup scenario is also a measured scenario, so it stays in the sample set; the reasoning LRU would otherwise hand it a ~0 ms reasoning.analyze cache hit (warmup-cache artifact understating the run mean). Warmup-on arms (e.g. the committed benchmarks/base, live-20261002-070756) are NOT mutually comparable with NO_WARMUP arms; base-fresh is the baseline for every comparison this round.
- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
