# Benchmark run consistency-base-20

- mode: `analyze` · split `train` · limit `20` · repeats `3`
- scenarios: 60 records (train 60 / holdout 0)
- corpus: `/app/tests/benchmark/scenarios/full.json` sha256 `92a8558a78ec9e9d…`
- git: `unknown` dirty=False
- catalog: 486 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 60 | 55.0% | [0.425, 0.669] | 3 | 0.343 | 0.411 | 0.019 | 0.353 | 0 |
| train | 60 | 55.0% | [0.425, 0.669] | 3 | 0.343 | 0.411 | 0.019 | 0.353 | 0 |
| holdout | 0 | null | null | null | null | null | null | null | null |

## Latency

- e2e: n=60 p50=165723.4 p95=257155.6
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 60 | 0 | 133300.9 | 2102.0 | 3928.0 | embed_queries x60 |
| `rerank` | 285 | 0 | 6678056.8 | 22833.4 | 30422.5 | rerank x285 |
| `llm.analyze` | 60 | 3 | 1542538.4 | 24509.7 | 48853.8 | extract_brief_and_parts x60 |
| `llm.weights` | 63 | 6 | 727840.1 | 10346.0 | 26315.6 | extract_weights_batch x63 |
| `reasoning.weights` | 60 | 0 | 1449704.7 | 23974.5 | 40031.7 | run_pre_llm x60 |
| `reasoning.analyze` | 60 | 0 | 423535.7 | 0.0 | 29099.3 | run_pre_llm x60 |

## Calibration (train split, monitoring only)

- Brier: 0.265 · ECE (10 bins): 0.218

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|
| 0 | [0.0, 0.1) | 1 | 0.000 | 0.000 | 0.000 |
| 6 | [0.6, 0.7) | 13 | 0.655 | 0.231 | 0.424 |
| 7 | [0.7, 0.8) | 22 | 0.752 | 0.545 | 0.207 |
| 8 | [0.8, 0.9) | 16 | 0.858 | 0.750 | 0.108 |
| 9 | [0.9, 1.0) | 8 | 0.913 | 0.750 | 0.163 |

## Risk-coverage

- holdout: n=0 risk@80=null risk@90=null risk@100=null AURC=null
- train: n=60 risk@80=0.375 risk@90=0.389 risk@100=0.450 AURC=0.316

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `algorithms-traversal-variation-pricing-rules` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.700 | 174489.4 | -10970.8 | 0 |
| `dispatch-pair-interactions` | train | behavior-selection | ✔ | `double-dispatch` | `double-dispatch` | 0.798 | 232205.6 | -8953.7 | 0 |
| `extension-compile-time-matrix` | train | behavior-selection | ✘ | `sealed-hierarchy-exhaustive-match` | `policy-based-design` | 0.674 | 213945.2 | -10909.4 | 0 |
| `functional-composition-request-validation` | train | behavior-selection | ✘ | `contravariant-validators` | `applicative` | 0.910 | 139614.0 | -8054.5 | 0 |
| `concurrency-resilience-order-mailbox` | train | concurrency-resilience | ✘ | `asynchronous-completion-token` | `guarded-suspension` | 0.626 | 288996.7 | -4540.3 | 0 |
| `coordination-primitives-bounded-payment-handoff` | train | concurrency-resilience | ✘ | `channels` | `producer-consumer` | 0.831 | 185839.6 | -9053.0 | 0 |
| `concurrency-resilience-fanout-deadline` | train | concurrency-resilience | ✘ | `deadline-propagation` | `deadline-propagation` | 0.000 | 194106.6 | -7299.9 | 0 |
| `concurrency-resilience-hot-config` | train | concurrency-resilience | ✔ | `thread-safe-copy-on-write` | `thread-safe-copy-on-write` | 0.876 | 220926.8 | -15974.1 | 0 |
| `queue-drain-shutdown` | train | concurrency-resilience | ✔ | `poison-pill` | `poison-pill` | 0.701 | 247397.2 | -13014.0 | 0 |
| `composition-encapsulation-checkout-entry` | train | construction-composition | ✘ | `gateway` | `facade` | 0.639 | 275917.6 | -15502.7 | 0 |
| `creation-config-growing-client-options` | train | construction-composition | ✔ | `functional-options` | `functional-options` | 0.886 | 163489.1 | -8338.1 | 0 |
| `dependency-wiring-api-handler-collaborators` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.721 | 151592.8 | -12496.2 | 0 |
| `functional-composition-inbound-normalization` | train | construction-composition | ✔ | `function-composition` | `function-composition` | 0.914 | 195732.5 | -13763.4 | 0 |
| `method-structure-buffer-assignment` | train | construction-composition | ✔ | `non-throwing-swap` | `copy-and-swap` | 0.673 | 176772.6 | -10736.5 | 0 |
| `domain-logic-shipment-import` | train | domain-modeling | ✘ | `deferred-validation` | `table-module` | 0.628 | 241164.1 | -10279.0 | 0 |
| `functional-data-modeling-expression-evaluation` | train | domain-modeling | ✘ | `interpreter` | `pattern-matching` | 0.718 | 164832.1 | -7079.8 | 0 |
| `model-clarity-ledger-contract-checks` | train | domain-modeling | ✔ | `assertion` | `assertion` | 0.754 | 264026.1 | -12057.2 | 0 |
| `tactical-customer-identity` | train | domain-modeling | ✘ | `value-object` | `entity` | 0.884 | 161255.5 | -21400.3 | 0 |
| `time-booking-intervals` | train | domain-modeling | ✔ | `range` | `range` | 0.798 | 185503.2 | -3702.3 | 0 |
| `values-correctness-distinct-identifiers` | train | domain-modeling | ✔ | `newtype` | `newtype` | 0.893 | 115381.7 | -7542.4 | 0 |
| `algorithms-traversal-variation-pricing-rules` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.821 | 115416.3 | -10843.9 | 0 |
| `dispatch-pair-interactions` | train | behavior-selection | ✔ | `double-dispatch` | `double-dispatch` | 0.827 | 179650.6 | -27829.5 | 0 |
| `extension-compile-time-matrix` | train | behavior-selection | ✘ | `sealed-hierarchy-exhaustive-match` | `policy-based-design` | 0.664 | 171340.0 | -17735.1 | 0 |
| `functional-composition-request-validation` | train | behavior-selection | ✘ | `contravariant-validators` | `applicative` | 0.936 | 108245.8 | -11892.4 | 0 |
| `concurrency-resilience-order-mailbox` | train | concurrency-resilience | ✘ | `timeout` | `guarded-suspension` | 0.798 | 234259.0 | -16738.5 | 0 |
| `coordination-primitives-bounded-payment-handoff` | train | concurrency-resilience | ✘ | `channels` | `producer-consumer` | 0.782 | 154970.7 | -8801.0 | 0 |
| `concurrency-resilience-fanout-deadline` | train | concurrency-resilience | ✔ | `timeout` | `deadline-propagation` | 0.907 | 242810.2 | -10493.8 | 0 |
| `concurrency-resilience-hot-config` | train | concurrency-resilience | ✘ | `immutability` | `thread-safe-copy-on-write` | 0.770 | 155334.6 | -13047.8 | 0 |
| `queue-drain-shutdown` | train | concurrency-resilience | ✘ | `guaranteed-delivery` | `poison-pill` | 0.641 | 178799.4 | -9385.7 | 0 |
| `composition-encapsulation-checkout-entry` | train | construction-composition | ✘ | `application-service` | `facade` | 0.678 | 203179.5 | -7473.4 | 0 |
| `creation-config-growing-client-options` | train | construction-composition | ✔ | `functional-options` | `functional-options` | 0.903 | 139406.2 | -8030.9 | 0 |
| `dependency-wiring-api-handler-collaborators` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.729 | 147644.1 | -4507.9 | 0 |
| `functional-composition-inbound-normalization` | train | construction-composition | ✔ | `function-composition` | `function-composition` | 0.892 | 116692.0 | -12003.4 | 0 |
| `method-structure-buffer-assignment` | train | construction-composition | ✔ | `non-throwing-swap` | `copy-and-swap` | 0.672 | 159658.4 | -11140.4 | 0 |
| `domain-logic-shipment-import` | train | domain-modeling | ✘ | `table-data-gateway` | `table-module` | 0.712 | 197064.0 | -29942.8 | 0 |
| `functional-data-modeling-expression-evaluation` | train | domain-modeling | ✔ | `pattern-matching` | `pattern-matching` | 0.710 | 126432.0 | -3978.2 | 0 |
| `model-clarity-ledger-contract-checks` | train | domain-modeling | ✔ | `assertion` | `assertion` | 0.859 | 166614.7 | -25018.4 | 0 |
| `tactical-customer-identity` | train | domain-modeling | ✔ | `entity` | `entity` | 0.874 | 129487.2 | -8720.5 | 0 |
| `time-booking-intervals` | train | domain-modeling | ✔ | `range` | `range` | 0.785 | 158680.1 | -14586.5 | 0 |
| `values-correctness-distinct-identifiers` | train | domain-modeling | ✔ | `newtype` | `newtype` | 0.911 | 113795.2 | -22139.5 | 0 |
| `algorithms-traversal-variation-pricing-rules` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.859 | 116678.7 | -4202.0 | 0 |
| `dispatch-pair-interactions` | train | behavior-selection | ✘ | `capability-query` | `double-dispatch` | 0.820 | 170583.4 | -26414.8 | 0 |
| `extension-compile-time-matrix` | train | behavior-selection | ✘ | `non-throwing-swap` | `policy-based-design` | 0.851 | 159548.3 | -13066.3 | 0 |
| `functional-composition-request-validation` | train | behavior-selection | ✘ | `contravariant-validators` | `applicative` | 0.760 | 128216.9 | -14562.9 | 0 |
| `concurrency-resilience-order-mailbox` | train | concurrency-resilience | ✘ | `async-method-invocation` | `guarded-suspension` | 0.628 | 231771.5 | -10208.9 | 0 |
| `coordination-primitives-bounded-payment-handoff` | train | concurrency-resilience | ✘ | `channels` | `producer-consumer` | 0.790 | 194047.8 | -21035.2 | 0 |
| `concurrency-resilience-fanout-deadline` | train | concurrency-resilience | ✔ | `timeout` | `deadline-propagation` | 0.894 | 242920.4 | -6688.2 | 0 |
| `concurrency-resilience-hot-config` | train | concurrency-resilience | ✘ | `thread-local-storage` | `thread-safe-copy-on-write` | 0.746 | 192574.9 | -35429.3 | 0 |
| `queue-drain-shutdown` | train | concurrency-resilience | ✔ | `poison-pill` | `poison-pill` | 0.674 | 217879.9 | -4879.2 | 0 |
| `composition-encapsulation-checkout-entry` | train | construction-composition | ✘ | `gateway` | `facade` | 0.723 | 256794.0 | -24536.2 | 0 |
| `creation-config-growing-client-options` | train | construction-composition | ✔ | `functional-options` | `functional-options` | 0.915 | 130155.9 | -9269.4 | 0 |
| `dependency-wiring-api-handler-collaborators` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.765 | 150117.3 | -6799.4 | 0 |
| `functional-composition-inbound-normalization` | train | construction-composition | ✔ | `function-composition` | `function-composition` | 0.913 | 152490.1 | -7793.0 | 0 |
| `method-structure-buffer-assignment` | train | construction-composition | ✔ | `copy-and-swap` | `copy-and-swap` | 0.827 | 134256.8 | -21103.0 | 0 |
| `domain-logic-shipment-import` | train | domain-modeling | ✘ | `deferred-validation` | `table-module` | 0.640 | 149986.6 | -13138.4 | 0 |
| `functional-data-modeling-expression-evaluation` | train | domain-modeling | ✘ | `interpreter` | `pattern-matching` | 0.679 | 122049.5 | -10160.2 | 0 |
| `model-clarity-ledger-contract-checks` | train | domain-modeling | ✘ | `fail-fast` | `assertion` | 0.767 | 214129.8 | -7830.1 | 0 |
| `tactical-customer-identity` | train | domain-modeling | ✔ | `entity` | `entity` | 0.796 | 21403.1 | -826.2 | 0 |
| `time-booking-intervals` | train | domain-modeling | ✔ | `range` | `range` | 0.721 | 26561.9 | -661.4 | 0 |
| `values-correctness-distinct-identifiers` | train | domain-modeling | ✔ | `newtype` | `newtype` | 0.840 | 24705.2 | -854.9 | 0 |

## Notes

- analyze-only arm: hit = the §9 dominant election replayed over the ANALYZE selections (the same definition as the full arm's final_pattern before loop reselection), recall = the analysis selection; generate/evaluate are skipped, so the pipeline block is null and e2e_ms is a phase subset, not a full-run latency
- NO_WARMUP protocol deviation (2026-10-07 latency-reduction round): the warmup scenario is also a measured scenario, so it stays in the sample set; the reasoning LRU would otherwise hand it a ~0 ms reasoning.analyze cache hit (warmup-cache artifact understating the run mean). Warmup-on arms (e.g. the committed benchmarks/base, live-20261002-070756) are NOT mutually comparable with NO_WARMUP arms; base-fresh is the baseline for every comparison this round.
- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
