# Benchmark run consistency-jev-20

- mode: `analyze` · split `train` · limit `20` · repeats `3`
- scenarios: 60 records (train 60 / holdout 0)
- corpus: `/app/tests/benchmark/scenarios/full.json` sha256 `92a8558a78ec9e9d…`
- git: `unknown` dirty=False
- catalog: 486 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 60 | 66.7% | [0.541, 0.773] | 2 | 0.401 | 0.477 | 0.038 | 0.432 | 0 |
| train | 60 | 66.7% | [0.541, 0.773] | 2 | 0.401 | 0.477 | 0.038 | 0.432 | 0 |
| holdout | 0 | null | null | null | null | null | null | null | null |

## Latency

- e2e: n=60 p50=198936.5 p95=262239.4
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 60 | 0 | 62644.4 | 1112.0 | 1557.9 | embed_queries x60 |
| `rerank` | 149 | 0 | 3276293.6 | 21503.3 | 28863.3 | rerank x149 |
| `llm.analyze` | 60 | 0 | 3356788.5 | 54496.7 | 85415.3 | extract_brief_and_parts x60 |
| `llm.weights` | 60 | 0 | 2317243.5 | 38137.0 | 55359.9 | extract_weights_batch x60 |
| `reasoning.weights` | 60 | 0 | 3868829.0 | 63077.7 | 109057.0 | run_pre_llm x60 |
| `reasoning.analyze` | 60 | 0 | 1117369.2 | 0.0 | 63899.9 | run_pre_llm x60 |

## Calibration (train split, monitoring only)

- Brier: 0.231 · ECE (10 bins): 0.185

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|
| 6 | [0.6, 0.7) | 1 | 0.697 | 0.000 | 0.697 |
| 7 | [0.7, 0.8) | 24 | 0.755 | 0.542 | 0.214 |
| 8 | [0.8, 0.9) | 23 | 0.843 | 0.652 | 0.191 |
| 9 | [0.9, 1.0) | 12 | 0.923 | 1.000 | 0.077 |

## Risk-coverage

- holdout: n=0 risk@80=null risk@90=null risk@100=null AURC=null
- train: n=60 risk@80=0.312 risk@90=0.333 risk@100=0.333 AURC=0.183

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `algorithms-traversal-variation-pricing-rules` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.891 | 222184.4 | -31635.4 | 0 |
| `dispatch-pair-interactions` | train | behavior-selection | ✔ | `double-dispatch` | `double-dispatch` | 0.827 | 259410.3 | -42014.4 | 0 |
| `extension-compile-time-matrix` | train | behavior-selection | ✘ | `named-constructor` | `policy-based-design` | 0.758 | 212922.4 | -18147.3 | 0 |
| `functional-composition-request-validation` | train | behavior-selection | ✘ | `contravariant-validators` | `applicative` | 0.765 | 184823.7 | -18349.1 | 0 |
| `concurrency-resilience-order-mailbox` | train | concurrency-resilience | ✘ | `poison-pill` | `guarded-suspension` | 0.731 | 289018.1 | -41103.8 | 0 |
| `coordination-primitives-bounded-payment-handoff` | train | concurrency-resilience | ✘ | `channels` | `producer-consumer` | 0.844 | 244269.8 | -45490.9 | 0 |
| `concurrency-resilience-fanout-deadline` | train | concurrency-resilience | ✔ | `timeout` | `deadline-propagation` | 0.948 | 244347.8 | -34657.9 | 0 |
| `concurrency-resilience-hot-config` | train | concurrency-resilience | ✔ | `thread-safe-copy-on-write` | `thread-safe-copy-on-write` | 0.730 | 238962.2 | -33250.6 | 0 |
| `queue-drain-shutdown` | train | concurrency-resilience | ✘ | `careful-buffering` | `poison-pill` | 0.846 | 234753.0 | -21738.7 | 0 |
| `composition-encapsulation-checkout-entry` | train | construction-composition | ✘ | `application-controller` | `facade` | 0.743 | 170151.3 | -18544.3 | 0 |
| `creation-config-growing-client-options` | train | construction-composition | ✔ | `functional-options` | `functional-options` | 0.908 | 164675.0 | -19488.1 | 0 |
| `dependency-wiring-api-handler-collaborators` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.763 | 208682.3 | -29730.7 | 0 |
| `functional-composition-inbound-normalization` | train | construction-composition | ✔ | `function-composition` | `function-composition` | 0.976 | 249000.8 | -45759.9 | 0 |
| `method-structure-buffer-assignment` | train | construction-composition | ✔ | `copy-and-swap` | `copy-and-swap` | 0.849 | 261142.8 | -38134.1 | 0 |
| `domain-logic-shipment-import` | train | domain-modeling | ✔ | `table-module` | `table-module` | 0.748 | 198054.6 | -35949.5 | 0 |
| `functional-data-modeling-expression-evaluation` | train | domain-modeling | ✘ | `type-safe-enum` | `pattern-matching` | 0.879 | 248660.8 | -39820.6 | 0 |
| `model-clarity-ledger-contract-checks` | train | domain-modeling | ✘ | `guard-clause` | `assertion` | 0.742 | 269677.8 | -33348.5 | 0 |
| `tactical-customer-identity` | train | domain-modeling | ✔ | `entity` | `entity` | 0.865 | 279042.5 | -38479.6 | 0 |
| `time-booking-intervals` | train | domain-modeling | ✔ | `effectivity` | `range` | 0.771 | 223470.7 | -50764.4 | 0 |
| `values-correctness-distinct-identifiers` | train | domain-modeling | ✔ | `newtype` | `newtype` | 0.905 | 207829.8 | -42697.2 | 0 |
| `algorithms-traversal-variation-pricing-rules` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.906 | 171736.6 | -27317.4 | 0 |
| `dispatch-pair-interactions` | train | behavior-selection | ✔ | `double-dispatch` | `double-dispatch` | 0.800 | 223180.6 | -38765.0 | 0 |
| `extension-compile-time-matrix` | train | behavior-selection | ✔ | `policy-based-design` | `policy-based-design` | 0.709 | 104757.8 | -17153.9 | 0 |
| `functional-composition-request-validation` | train | behavior-selection | ✘ | `contravariant-validators` | `applicative` | 0.800 | 114669.7 | -21036.4 | 0 |
| `concurrency-resilience-order-mailbox` | train | concurrency-resilience | ✘ | `poison-pill` | `guarded-suspension` | 0.829 | 206533.0 | -35106.6 | 0 |
| `coordination-primitives-bounded-payment-handoff` | train | concurrency-resilience | ✘ | `careful-buffering` | `producer-consumer` | 0.813 | 204097.4 | -36041.5 | 0 |
| `concurrency-resilience-fanout-deadline` | train | concurrency-resilience | ✔ | `timeout` | `deadline-propagation` | 0.949 | 179232.1 | -34771.5 | 0 |
| `concurrency-resilience-hot-config` | train | concurrency-resilience | ✔ | `thread-safe-copy-on-write` | `thread-safe-copy-on-write` | 0.732 | 199818.5 | -40410.8 | 0 |
| `queue-drain-shutdown` | train | concurrency-resilience | ✔ | `poison-pill` | `poison-pill` | 0.855 | 184382.6 | -31108.8 | 0 |
| `composition-encapsulation-checkout-entry` | train | construction-composition | ✔ | `facade` | `facade` | 0.700 | 246650.6 | -18044.0 | 0 |
| `creation-config-growing-client-options` | train | construction-composition | ✔ | `functional-options` | `functional-options` | 0.906 | 111727.7 | -18664.5 | 0 |
| `dependency-wiring-api-handler-collaborators` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.841 | 178920.4 | -36564.2 | 0 |
| `functional-composition-inbound-normalization` | train | construction-composition | ✔ | `function-composition` | `function-composition` | 0.828 | 191176.3 | -37012.8 | 0 |
| `method-structure-buffer-assignment` | train | construction-composition | ✔ | `non-throwing-swap` | `copy-and-swap` | 0.850 | 172946.3 | -31962.6 | 0 |
| `domain-logic-shipment-import` | train | domain-modeling | ✔ | `table-module` | `table-module` | 0.766 | 246609.7 | -40110.7 | 0 |
| `functional-data-modeling-expression-evaluation` | train | domain-modeling | ✘ | `type-safe-enum` | `pattern-matching` | 0.858 | 203792.0 | -39256.1 | 0 |
| `model-clarity-ledger-contract-checks` | train | domain-modeling | ✘ | `guard-clause` | `assertion` | 0.852 | 165769.3 | -23114.8 | 0 |
| `tactical-customer-identity` | train | domain-modeling | ✔ | `entity` | `entity` | 0.870 | 178235.7 | -38233.6 | 0 |
| `time-booking-intervals` | train | domain-modeling | ✔ | `range` | `range` | 0.789 | 151267.5 | -25513.5 | 0 |
| `values-correctness-distinct-identifiers` | train | domain-modeling | ✔ | `newtype` | `newtype` | 0.914 | 208261.5 | -55118.5 | 0 |
| `algorithms-traversal-variation-pricing-rules` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.903 | 166432.1 | -47430.7 | 0 |
| `dispatch-pair-interactions` | train | behavior-selection | ✔ | `double-dispatch` | `double-dispatch` | 0.810 | 157450.5 | -49364.1 | 0 |
| `extension-compile-time-matrix` | train | behavior-selection | ✘ | `named-constructor` | `policy-based-design` | 0.753 | 135962.5 | -17747.6 | 0 |
| `functional-composition-request-validation` | train | behavior-selection | ✘ | `contravariant-validators` | `applicative` | 0.753 | 212601.3 | -24771.5 | 0 |
| `concurrency-resilience-order-mailbox` | train | concurrency-resilience | ✘ | `lock-free-queue` | `guarded-suspension` | 0.800 | 204032.1 | -33764.2 | 0 |
| `coordination-primitives-bounded-payment-handoff` | train | concurrency-resilience | ✘ | `load-leveller` | `producer-consumer` | 0.798 | 225462.3 | -41077.6 | 0 |
| `concurrency-resilience-fanout-deadline` | train | concurrency-resilience | ✔ | `timeout` | `deadline-propagation` | 0.946 | 196389.4 | -36292.4 | 0 |
| `concurrency-resilience-hot-config` | train | concurrency-resilience | ✔ | `thread-safe-copy-on-write` | `thread-safe-copy-on-write` | 0.734 | 261847.9 | -43503.3 | 0 |
| `queue-drain-shutdown` | train | concurrency-resilience | ✔ | `poison-pill` | `poison-pill` | 0.707 | 184259.6 | -36785.8 | 0 |
| `composition-encapsulation-checkout-entry` | train | construction-composition | ✘ | `execute-around` | `facade` | 0.770 | 140748.9 | -19867.5 | 0 |
| `creation-config-growing-client-options` | train | construction-composition | ✔ | `functional-options` | `functional-options` | 0.905 | 149742.0 | -18395.3 | 0 |
| `dependency-wiring-api-handler-collaborators` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.824 | 176479.2 | -30267.4 | 0 |
| `functional-composition-inbound-normalization` | train | construction-composition | ✔ | `function-composition` | `function-composition` | 0.832 | 218953.3 | -40493.2 | 0 |
| `method-structure-buffer-assignment` | train | construction-composition | ✔ | `copy-and-swap` | `copy-and-swap` | 0.844 | 202502.7 | -56263.6 | 0 |
| `domain-logic-shipment-import` | train | domain-modeling | ✔ | `table-module` | `table-module` | 0.780 | 190401.2 | -39834.7 | 0 |
| `functional-data-modeling-expression-evaluation` | train | domain-modeling | ✘ | `type-safe-enum` | `pattern-matching` | 0.697 | 156624.9 | -27394.1 | 0 |
| `model-clarity-ledger-contract-checks` | train | domain-modeling | ✘ | `guard-clause` | `assertion` | 0.808 | 182714.4 | -39434.0 | 0 |
| `tactical-customer-identity` | train | domain-modeling | ✔ | `entity` | `entity` | 0.869 | 174269.2 | -52259.3 | 0 |
| `time-booking-intervals` | train | domain-modeling | ✔ | `range` | `range` | 0.784 | 159934.5 | -33342.4 | 0 |
| `values-correctness-distinct-identifiers` | train | domain-modeling | ✔ | `newtype` | `newtype` | 0.913 | 186635.5 | -48178.1 | 0 |

## Notes

- analyze-only arm: hit = the §9 dominant election replayed over the ANALYZE selections (the same definition as the full arm's final_pattern before loop reselection), recall = the analysis selection; generate/evaluate are skipped, so the pipeline block is null and e2e_ms is a phase subset, not a full-run latency
- NO_WARMUP protocol deviation (2026-10-07 latency-reduction round): the warmup scenario is also a measured scenario, so it stays in the sample set; the reasoning LRU would otherwise hand it a ~0 ms reasoning.analyze cache hit (warmup-cache artifact understating the run mean). Warmup-on arms (e.g. the committed benchmarks/base, live-20261002-070756) are NOT mutually comparable with NO_WARMUP arms; base-fresh is the baseline for every comparison this round.
- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
