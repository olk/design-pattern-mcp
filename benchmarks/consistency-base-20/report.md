# Benchmark run consistency-base-20-spark

- mode: `analyze` · split `train` · limit `20` · repeats `3`
- scenarios: 60 records (train 60 / holdout 0)
- corpus: `/app/tests/benchmark/scenarios/full.json` sha256 `92a8558a78ec9e9d…`
- git: `unknown` dirty=False
- catalog: 486 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 60 | 60.0% | [0.474, 0.714] | 2 | 0.398 | 0.474 | 0.003 | 0.384 | 0 |
| train | 60 | 60.0% | [0.474, 0.714] | 2 | 0.398 | 0.474 | 0.003 | 0.384 | 0 |
| holdout | 0 | null | null | null | null | null | null | null | null |

## Latency

- e2e: n=60 p50=209722.3 p95=264651.9
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 60 | 0 | 78034.0 | 1232.1 | 2055.4 | embed_queries x60 |
| `rerank` | 182 | 0 | 4170512.5 | 22604.3 | 28989.6 | rerank x182 |
| `llm.analyze` | 60 | 0 | 3464140.1 | 55344.7 | 85711.0 | extract_brief_and_parts x60 |
| `llm.weights` | 60 | 0 | 2367067.8 | 40648.7 | 55429.8 | extract_weights_batch x60 |
| `reasoning.weights` | 60 | 0 | 3617418.0 | 58093.5 | 74868.0 | run_pre_llm x60 |
| `reasoning.analyze` | 60 | 0 | 1174576.4 | 0.0 | 65155.2 | run_pre_llm x60 |

## Calibration (train split, monitoring only)

- Brier: 0.259 · ECE (10 bins): 0.216

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|
| 6 | [0.6, 0.7) | 7 | 0.666 | 0.143 | 0.523 |
| 7 | [0.7, 0.8) | 17 | 0.763 | 0.471 | 0.292 |
| 8 | [0.8, 0.9) | 27 | 0.853 | 0.704 | 0.149 |
| 9 | [0.9, 1.0) | 9 | 0.923 | 0.889 | 0.034 |

## Risk-coverage

- holdout: n=0 risk@80=null risk@90=null risk@100=null AURC=null
- train: n=60 risk@80=0.333 risk@90=0.352 risk@100=0.400 AURC=0.233

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `algorithms-traversal-variation-pricing-rules` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.855 | 225824.8 | -34496.3 | 0 |
| `dispatch-pair-interactions` | train | behavior-selection | ✔ | `double-dispatch` | `double-dispatch` | 0.833 | 286785.4 | -47747.3 | 0 |
| `extension-compile-time-matrix` | train | behavior-selection | ✘ | `newtype` | `policy-based-design` | 0.718 | 245565.9 | -22251.1 | 0 |
| `functional-composition-request-validation` | train | behavior-selection | ✘ | `contravariant-validators` | `applicative` | 0.952 | 276253.1 | -26743.0 | 0 |
| `concurrency-resilience-order-mailbox` | train | concurrency-resilience | ✘ | `structured-concurrency` | `guarded-suspension` | 0.775 | 317128.5 | -47170.8 | 0 |
| `coordination-primitives-bounded-payment-handoff` | train | concurrency-resilience | ✘ | `channels` | `producer-consumer` | 0.790 | 234200.0 | -44216.4 | 0 |
| `concurrency-resilience-fanout-deadline` | train | concurrency-resilience | ✔ | `timeout` | `deadline-propagation` | 0.881 | 263874.4 | -43191.4 | 0 |
| `concurrency-resilience-hot-config` | train | concurrency-resilience | ✔ | `thread-safe-copy-on-write` | `thread-safe-copy-on-write` | 0.904 | 216323.0 | -29361.2 | 0 |
| `queue-drain-shutdown` | train | concurrency-resilience | ✘ | `producer-consumer` | `poison-pill` | 0.646 | 249814.5 | -30820.8 | 0 |
| `composition-encapsulation-checkout-entry` | train | construction-composition | ✘ | `application-controller` | `facade` | 0.699 | 228075.2 | -30317.7 | 0 |
| `creation-config-growing-client-options` | train | construction-composition | ✔ | `functional-options` | `functional-options` | 0.920 | 250960.3 | -49300.7 | 0 |
| `dependency-wiring-api-handler-collaborators` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.761 | 264041.3 | -48138.0 | 0 |
| `functional-composition-inbound-normalization` | train | construction-composition | ✔ | `function-composition` | `function-composition` | 0.877 | 247609.5 | -55395.2 | 0 |
| `method-structure-buffer-assignment` | train | construction-composition | ✔ | `copy-and-swap` | `copy-and-swap` | 0.840 | 252095.3 | -40496.3 | 0 |
| `domain-logic-shipment-import` | train | domain-modeling | ✔ | `table-module` | `table-module` | 0.783 | 237915.7 | -53338.4 | 0 |
| `functional-data-modeling-expression-evaluation` | train | domain-modeling | ✘ | `sealed-hierarchy-exhaustive-match` | `pattern-matching` | 0.822 | 217355.8 | -39562.5 | 0 |
| `model-clarity-ledger-contract-checks` | train | domain-modeling | ✘ | `guard-clause` | `assertion` | 0.717 | 239274.0 | -35481.0 | 0 |
| `tactical-customer-identity` | train | domain-modeling | ✘ | `value-object` | `entity` | 0.891 | 218706.6 | -41026.1 | 0 |
| `time-booking-intervals` | train | domain-modeling | ✔ | `range` | `range` | 0.798 | 260421.8 | -41349.5 | 0 |
| `values-correctness-distinct-identifiers` | train | domain-modeling | ✔ | `newtype` | `newtype` | 0.899 | 238717.3 | -30993.5 | 0 |
| `algorithms-traversal-variation-pricing-rules` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.842 | 148846.0 | -31258.3 | 0 |
| `dispatch-pair-interactions` | train | behavior-selection | ✔ | `double-dispatch` | `double-dispatch` | 0.832 | 237485.2 | -55564.6 | 0 |
| `extension-compile-time-matrix` | train | behavior-selection | ✘ | `type-safe-builder` | `policy-based-design` | 0.771 | 139666.0 | -22593.1 | 0 |
| `functional-composition-request-validation` | train | behavior-selection | ✘ | `contravariant-validators` | `applicative` | 0.793 | 188313.4 | -22144.1 | 0 |
| `concurrency-resilience-order-mailbox` | train | concurrency-resilience | ✘ | `poison-pill` | `guarded-suspension` | 0.675 | 238874.4 | -41007.1 | 0 |
| `coordination-primitives-bounded-payment-handoff` | train | concurrency-resilience | ✘ | `careful-buffering` | `producer-consumer` | 0.827 | 213558.2 | -33291.7 | 0 |
| `concurrency-resilience-fanout-deadline` | train | concurrency-resilience | ✔ | `timeout` | `deadline-propagation` | 0.886 | 247828.8 | -33648.4 | 0 |
| `concurrency-resilience-hot-config` | train | concurrency-resilience | ✔ | `thread-safe-copy-on-write` | `thread-safe-copy-on-write` | 0.727 | 179776.1 | -31737.8 | 0 |
| `queue-drain-shutdown` | train | concurrency-resilience | ✘ | `producer-consumer` | `poison-pill` | 0.852 | 205007.0 | -45641.7 | 0 |
| `composition-encapsulation-checkout-entry` | train | construction-composition | ✘ | `unit-of-work` | `facade` | 0.630 | 199308.3 | -32507.7 | 0 |
| `creation-config-growing-client-options` | train | construction-composition | ✔ | `functional-options` | `functional-options` | 0.909 | 147467.5 | -21322.9 | 0 |
| `dependency-wiring-api-handler-collaborators` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.968 | 138630.3 | -43293.6 | 0 |
| `functional-composition-inbound-normalization` | train | construction-composition | ✔ | `function-composition` | `function-composition` | 0.894 | 190713.6 | -42554.7 | 0 |
| `method-structure-buffer-assignment` | train | construction-composition | ✘ | `smart-pointer` | `copy-and-swap` | 0.841 | 184312.7 | -50074.2 | 0 |
| `domain-logic-shipment-import` | train | domain-modeling | ✔ | `table-module` | `table-module` | 0.769 | 193211.2 | -37310.9 | 0 |
| `functional-data-modeling-expression-evaluation` | train | domain-modeling | ✘ | `sealed-hierarchy-exhaustive-match` | `pattern-matching` | 0.807 | 167604.7 | -45636.7 | 0 |
| `model-clarity-ledger-contract-checks` | train | domain-modeling | ✘ | `fail-fast` | `assertion` | 0.760 | 205886.3 | -42647.4 | 0 |
| `tactical-customer-identity` | train | domain-modeling | ✔ | `entity` | `entity` | 0.879 | 213749.6 | -41889.5 | 0 |
| `time-booking-intervals` | train | domain-modeling | ✔ | `range` | `range` | 0.795 | 189006.2 | -63482.0 | 0 |
| `values-correctness-distinct-identifiers` | train | domain-modeling | ✔ | `newtype` | `newtype` | 0.887 | 190507.3 | -48519.7 | 0 |
| `algorithms-traversal-variation-pricing-rules` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.840 | 175212.8 | -39430.9 | 0 |
| `dispatch-pair-interactions` | train | behavior-selection | ✔ | `double-dispatch` | `double-dispatch` | 0.832 | 193798.4 | -40744.3 | 0 |
| `extension-compile-time-matrix` | train | behavior-selection | ✔ | `policy-based-design` | `policy-based-design` | 0.730 | 230258.4 | -22197.2 | 0 |
| `functional-composition-request-validation` | train | behavior-selection | ✔ | `applicative` | `applicative` | 0.804 | 155559.7 | -24134.5 | 0 |
| `concurrency-resilience-order-mailbox` | train | concurrency-resilience | ✘ | `poison-pill` | `guarded-suspension` | 0.817 | 199795.0 | -43931.9 | 0 |
| `coordination-primitives-bounded-payment-handoff` | train | concurrency-resilience | ✘ | `careful-buffering` | `producer-consumer` | 0.778 | 204015.0 | -39871.5 | 0 |
| `concurrency-resilience-fanout-deadline` | train | concurrency-resilience | ✔ | `timeout` | `deadline-propagation` | 0.902 | 217471.4 | -40831.7 | 0 |
| `concurrency-resilience-hot-config` | train | concurrency-resilience | ✔ | `thread-safe-copy-on-write` | `thread-safe-copy-on-write` | 0.887 | 166797.9 | -34811.4 | 0 |
| `queue-drain-shutdown` | train | concurrency-resilience | ✔ | `poison-pill` | `poison-pill` | 0.659 | 219446.9 | -26672.2 | 0 |
| `composition-encapsulation-checkout-entry` | train | construction-composition | ✘ | `application-controller` | `facade` | 0.683 | 241941.6 | -56317.8 | 0 |
| `creation-config-growing-client-options` | train | construction-composition | ✔ | `functional-options` | `functional-options` | 0.910 | 128222.3 | -19737.8 | 0 |
| `dependency-wiring-api-handler-collaborators` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.924 | 160932.4 | -25558.2 | 0 |
| `functional-composition-inbound-normalization` | train | construction-composition | ✔ | `function-composition` | `function-composition` | 0.892 | 178302.3 | -35973.8 | 0 |
| `method-structure-buffer-assignment` | train | construction-composition | ✔ | `copy-and-swap` | `copy-and-swap` | 0.835 | 158588.4 | -37939.0 | 0 |
| `domain-logic-shipment-import` | train | domain-modeling | ✘ | `table-data-gateway` | `table-module` | 0.672 | 231232.5 | -29874.8 | 0 |
| `functional-data-modeling-expression-evaluation` | train | domain-modeling | ✘ | `extensible-visitor` | `pattern-matching` | 0.784 | 189816.9 | -52230.0 | 0 |
| `model-clarity-ledger-contract-checks` | train | domain-modeling | ✔ | `assertion` | `assertion` | 0.717 | 183109.9 | -32532.4 | 0 |
| `tactical-customer-identity` | train | domain-modeling | ✘ | `value-object` | `entity` | 0.876 | 178943.2 | -38976.9 | 0 |
| `time-booking-intervals` | train | domain-modeling | ✔ | `range` | `range` | 0.806 | 190395.1 | -43410.6 | 0 |
| `values-correctness-distinct-identifiers` | train | domain-modeling | ✔ | `newtype` | `newtype` | 0.914 | 170614.1 | -43899.0 | 0 |

## Notes

- analyze-only arm: hit = the §9 dominant election replayed over the ANALYZE selections (the same definition as the full arm's final_pattern before loop reselection), recall = the analysis selection; generate/evaluate are skipped, so the pipeline block is null and e2e_ms is a phase subset, not a full-run latency
- NO_WARMUP protocol deviation (2026-10-07 latency-reduction round): the warmup scenario is also a measured scenario, so it stays in the sample set; the reasoning LRU would otherwise hand it a ~0 ms reasoning.analyze cache hit (warmup-cache artifact understating the run mean). Warmup-on arms (e.g. the committed benchmarks/base, live-20261002-070756) are NOT mutually comparable with NO_WARMUP arms; base-fresh is the baseline for every comparison this round.
- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
