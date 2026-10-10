# Benchmark run live-cfg-reasoning

- mode: `live` · split `all` · limit `None` · repeats `1`
- scenarios: 18 records (train 12 / holdout 6)
- corpus: `tests/benchmark/scenarios/seed.json` sha256 `ebfa882dce2b82e7…`
- git: `d3b0d4626308` dirty=True
- catalog: 438 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 18 | 66.7% | [0.437, 0.837] | 0 | 0.278 | 0.367 | 0.125 | 0.333 | 0 |
| train | 12 | 66.7% | [0.391, 0.862] | 0 | 0.283 | 0.367 | 0.104 | 0.333 | 0 |
| holdout | 6 | 66.7% | [0.300, 0.903] | 0 | 0.267 | 0.367 | 0.167 | 0.333 | 0 |

## Latency

- e2e: n=18 p50=252955.2 p95=413884.8
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 18 | 0 | 17216.8 | 878.7 | 1946.5 | embed_queries x18 |
| `rerank` | 37 | 0 | 1230239.2 | 32398.7 | 42041.8 | rerank x37 |
| `llm.analyze` | 18 | 0 | 346421.0 | 18042.9 | 27751.9 | extract_brief_and_parts x18 |
| `llm.weights` | 18 | 0 | 264940.4 | 14093.9 | 25233.7 | extract_weights_batch x18 |
| `llm.generate` | 25 | 0 | 2103241.5 | 84060.8 | 131776.2 | generate x25 |
| `llm.evaluate` | 20 | 0 | 706632.1 | 35346.9 | 57109.9 | evaluate x20 |
| `reasoning.generate` | 20 | 0 | 534933.0 | 27154.1 | 34809.2 | run_pre_llm x20 |
| `reasoning.analyze` | 18 | 0 | 510356.8 | 28506.6 | 31659.3 | run_pre_llm x18 |
| `reasoning.evaluate` | 20 | 0 | 0.6 | 0.0 | 0.0 | run_pre_llm x20 |
| `reasoning.weights` | 18 | 0 | 0.4 | 0.0 | 0.0 | run_pre_llm x18 |

## Calibration (train split, monitoring only)

- Brier: 0.292 · ECE (10 bins): 0.321

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|
| 6 | [0.6, 0.7) | 1 | 0.687 | 1.000 | 0.313 |
| 7 | [0.7, 0.8) | 2 | 0.748 | 1.000 | 0.252 |
| 8 | [0.8, 0.9) | 6 | 0.873 | 0.500 | 0.373 |
| 9 | [0.9, 1.0) | 3 | 0.933 | 0.667 | 0.266 |

## Risk-coverage

- holdout: n=6 risk@80=0.200 risk@90=0.333 risk@100=0.333 AURC=0.061
- train: n=12 risk@80=0.400 risk@90=0.364 risk@100=0.333 AURC=0.425

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `dm-order-lifecycle` | train | domain-modeling | ✔ | `entity` | `entity` | 0.873 | 352204.9 | -63999.7 | 0 |
| `dm-currency-ledger` | train | domain-modeling | ✔ | `money` | `money` | 0.915 | 202509.6 | -11937.6 | 0 |
| `cc-report-builder` | train | construction-composition | ✘ | `functional-options` | `builder` | 0.859 | 339016.7 | -18712.6 | 0 |
| `bs-pricing-strategy` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.921 | 188535.9 | -6927.7 | 0 |
| `bs-ast-visitor` | train | behavior-selection | ✔ | `extensible-visitor` | `visitor` | 0.758 | 209262.1 | -7885.0 | 0 |
| `ib-billing-gateway` | train | interaction-boundary | ✔ | `gateway` | `gateway` | 0.883 | 331989.6 | -111887.6 | 0 |
| `wm-order-router` | train | workflow-messaging | ✘ | `content-based-serializer` | `content-based-router` | 0.862 | 580381.5 | -51610.9 | 0 |
| `ps-catalog-repository` | train | persistence-state | ✔ | `repository` | `repository` | 0.864 | 384503.0 | -8094.9 | 0 |
| `cr-billing-circuit-breaker` | train | concurrency-resilience | ✘ | `fail-fast` | `circuit-breaker` | 0.962 | 186323.8 | -14344.3 | 0 |
| `sv-api-authorization` | train | security-verification | ✘ | `policy-guard` | `authorization-enforcer` | 0.900 | 217225.6 | -15411.8 | 0 |
| `sc-expression-parser` | train | specialized-components | ✔ | `recursive-descent-parser` | `recursive-descent-parser` | 0.687 | 184664.3 | -11579.1 | 0 |
| `cc-module-wiring` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.738 | 366353.2 | -15400.9 | 0 |
| `cc-legacy-adapter` | holdout | construction-composition | ✔ | `adapter` | `adapter` | 0.841 | 209787.2 | -10178.9 | 0 |
| `ib-admin-controller` | holdout | interaction-boundary | ✘ | `service-to-worker` | `application-controller` | 0.708 | 219696.1 | -15108.7 | 0 |
| `ps-mapping-layer` | holdout | persistence-state | ✔ | `data-mapper` | `data-mapper` | 0.871 | 286137.1 | -50713.5 | 0 |
| `cr-structured-fanout` | holdout | concurrency-resilience | ✔ | `structured-concurrency` | `structured-concurrency` | 0.923 | 219773.4 | -8516.3 | 0 |
| `sv-order-double-tests` | holdout | security-verification | ✘ | `recording-fake` | `mock` | 0.705 | 329199.6 | -24562.9 | 0 |
| `sc-dashboard-mvvm` | holdout | specialized-components | ✔ | `model-view-viewmodel` | `model-view-viewmodel` | 0.729 | 343495.2 | -116051.0 | 0 |

## Notes

- NO_WARMUP protocol deviation (2026-10-07 latency-reduction round): the warmup scenario is also a measured scenario, so it stays in the sample set; the reasoning LRU would otherwise hand it a ~0 ms reasoning.analyze cache hit (warmup-cache artifact understating the run mean). Warmup-on arms (e.g. the committed benchmarks/base, live-20261002-070756) are NOT mutually comparable with NO_WARMUP arms; base-fresh is the baseline for every comparison this round.
- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
