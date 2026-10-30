# Benchmark run base-analyze-100

- mode: `analyze` · split `all` · limit `100` · repeats `1`
- scenarios: 100 records (train 70 / holdout 30)
- corpus: `/app/tests/benchmark/scenarios/full.json` sha256 `92a8558a78ec9e9d…`
- git: `unknown` dirty=False
- catalog: 486 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 100 | 45.0% | [0.356, 0.548] | 8 | 0.301 | 0.413 | 0.037 | 0.283 | 0 |
| train | 70 | 44.3% | [0.332, 0.559] | 8 | 0.287 | 0.390 | 0.023 | 0.276 | 0 |
| holdout | 30 | 46.7% | [0.302, 0.639] | 0 | 0.333 | 0.467 | 0.069 | 0.300 | 0 |

## Latency

- e2e: n=100 p50=189166.4 p95=279142.7
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 100 | 0 | 257243.7 | 2536.0 | 3818.2 | embed_queries x100 |
| `rerank` | 531 | 0 | 13508203.2 | 25097.9 | 31838.1 | rerank x531 |
| `llm.analyze` | 100 | 0 | 2180076.7 | 19656.1 | 42256.7 | extract_brief_and_parts x100 |
| `llm.weights` | 100 | 0 | 811196.2 | 6872.8 | 15768.9 | extract_weights_batch x100 |
| `reasoning.analyze` | 100 | 0 | 1835168.2 | 17321.6 | 30245.7 | run_pre_llm x100 |
| `reasoning.weights` | 100 | 0 | 1623201.5 | 15740.6 | 23730.7 | run_pre_llm x100 |

## Calibration (train split, monitoring only)

- Brier: 0.369 · ECE (10 bins): 0.329

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|
| 0 | [0.0, 0.1) | 1 | 0.000 | 0.000 | 0.000 |
| 5 | [0.5, 0.6) | 1 | 0.593 | 0.000 | 0.593 |
| 6 | [0.6, 0.7) | 15 | 0.661 | 0.533 | 0.128 |
| 7 | [0.7, 0.8) | 21 | 0.753 | 0.429 | 0.324 |
| 8 | [0.8, 0.9) | 25 | 0.852 | 0.440 | 0.412 |
| 9 | [0.9, 1.0) | 7 | 0.917 | 0.429 | 0.488 |

## Risk-coverage

- holdout: n=30 risk@80=0.500 risk@90=0.481 risk@100=0.533 AURC=0.500
- train: n=70 risk@80=0.571 risk@90=0.571 risk@100=0.557 AURC=0.553

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `algorithms-traversal-variation-pricing-rules` | train | behavior-selection | ✔ | `strategy` | `strategy` | 0.706 | 267399.0 | -10176.5 | 0 |
| `dispatch-pair-interactions` | train | behavior-selection | ✔ | `double-dispatch` | `double-dispatch` | 0.807 | 169165.4 | -2850.8 | 0 |
| `extension-compile-time-matrix` | train | behavior-selection | ✘ | `type-safe-builder` | `policy-based-design` | 0.776 | 215178.0 | -9049.2 | 0 |
| `functional-composition-request-validation` | train | behavior-selection | ✘ | `contravariant-validators` | `applicative` | 0.794 | 123220.0 | -9213.3 | 0 |
| `concurrency-resilience-order-mailbox` | train | concurrency-resilience | ✘ | `two-phase-termination` | `guarded-suspension` | 0.683 | 270179.1 | -7784.3 | 0 |
| `coordination-primitives-bounded-payment-handoff` | train | concurrency-resilience | ✘ | `channels` | `producer-consumer` | 0.793 | 230140.1 | -9301.2 | 0 |
| `concurrency-resilience-fanout-deadline` | train | concurrency-resilience | ✘ | `deadline-propagation` | `deadline-propagation` | 0.866 | 253617.9 | -4733.8 | 0 |
| `concurrency-resilience-hot-config` | train | concurrency-resilience | ✘ | `singleflight` | `thread-safe-copy-on-write` | 0.775 | 185277.8 | -16478.6 | 0 |
| `queue-drain-shutdown` | train | concurrency-resilience | ✔ | `poison-pill` | `poison-pill` | 0.878 | 224658.9 | -9335.4 | 0 |
| `composition-encapsulation-checkout-entry` | train | construction-composition | ✘ | `unit-of-work` | `facade` | 0.711 | 208363.0 | -4826.8 | 0 |
| `creation-config-growing-client-options` | train | construction-composition | ✔ | `functional-options` | `functional-options` | 0.700 | 175652.0 | -3578.6 | 0 |
| `dependency-wiring-api-handler-collaborators` | train | construction-composition | ✔ | `dependency-injection` | `dependency-injection` | 0.706 | 195139.1 | -12949.0 | 0 |
| `functional-composition-inbound-normalization` | train | construction-composition | ✔ | `function-composition` | `function-composition` | 0.888 | 192301.4 | -7619.1 | 0 |
| `method-structure-buffer-assignment` | train | construction-composition | ✔ | `non-throwing-swap` | `copy-and-swap` | 0.605 | 197055.2 | -15682.0 | 0 |
| `domain-logic-shipment-import` | train | domain-modeling | ✘ | `deferred-validation` | `table-module` | 0.671 | 195778.4 | -8952.2 | 0 |
| `functional-data-modeling-expression-evaluation` | train | domain-modeling | ✘ | `sealed-hierarchy-exhaustive-match` | `pattern-matching` | 0.732 | 136660.1 | -6179.7 | 0 |
| `model-clarity-ledger-contract-checks` | train | domain-modeling | ✔ | `assertion` | `assertion` | 0.755 | 250201.5 | -10186.1 | 0 |
| `tactical-customer-identity` | train | domain-modeling | ✘ | `value-object` | `entity` | 0.915 | 161935.5 | -9764.4 | 0 |
| `time-booking-intervals` | train | domain-modeling | ✔ | `range` | `range` | 0.797 | 189784.5 | -10731.7 | 0 |
| `values-correctness-distinct-identifiers` | train | domain-modeling | ✔ | `newtype` | `newtype` | 0.897 | 138371.9 | -5357.5 | 0 |
| `message-contracts-order-facts` | train | interaction-boundary | ✘ | `publish-subscribe-channel` | `event-message` | 0.876 | 275019.9 | -7818.1 | 0 |
| `message-endpoints-durable-order-projection` | train | interaction-boundary | ✘ | `idempotent-client` | `durable-subscriber` | 0.734 | 160291.6 | -4130.4 | 0 |
| `remote-contracts-audit-traversal` | train | interaction-boundary | ✔ | `cursor-pagination` | `cursor-pagination` | 0.926 | 217641.0 | -8043.3 | 0 |
| `server-presentation-multichannel-invoice` | train | interaction-boundary | ✘ | `money` | `two-step-view` | 0.879 | 180470.5 | -4148.7 | 0 |
| `translation-checkout-mapping` | train | interaction-boundary | ✔ | `converter` | `converter` | 0.854 | 197679.6 | -14253.9 | 0 |
| `change-tracking-claim-editor` | train | persistence-state | ✔ | `version-number` | `version-number` | 0.844 | 174868.1 | -3625.2 | 0 |
| `data-access-gateways-customer-rows` | train | persistence-state | ✘ | `data-mapper` | `row-data-gateway` | 0.782 | 135763.9 | -3655.0 | 0 |
| `domain-to-store-catalog-record` | train | persistence-state | ✔ | `active-record` | `active-record` | 0.644 | 175461.3 | -5253.9 | 0 |
| `persistence-state-central-request-enforcement` | train | persistence-state | ✘ | `authorization-enforcer` | `implicit-lock` | 0.883 | 155426.9 | -12121.1 | 0 |
| `queries-metadata-external-field-declarations` | train | persistence-state | ✘ | `data-mapper` | `metadata-mapping` | 0.860 | 149262.6 | -2896.1 | 0 |
| `relational-state-hierarchy` | train | persistence-state | ✔ | `class-table-inheritance` | `single-table-inheritance` | 0.614 | 263775.8 | -6600.8 | 0 |
| `lifecycle-archive-channel` | train | persistence-state | ✘ | `singleflight` | `lazy-acquisition` | 0.684 | 209073.3 | -8970.3 | 0 |
| `audit-sensitive-admin-evidence` | train | security-verification | ✔ | `secure-logger` | `secure-logger` | 0.835 | 164120.6 | -5020.9 | 0 |
| `capabilities-isolation-plugin-control` | train | security-verification | ✔ | `revocable-forwarder` | `revocable-forwarder` | 0.632 | 149400.8 | -6063.2 | 0 |
| `compile-time-access-factory-construction` | train | security-verification | ✔ | `passkey` | `passkey` | 0.822 | 154995.5 | -46093.3 | 0 |
| `enforcement-api-policy` | holdout | security-verification | ✘ | `authorization-enforcer` | `policy-guard` | 0.917 | 199255.2 | -4088.8 | 0 |
| `identity-multi-channel-normalization` | holdout | security-verification | ✘ | `authenticator` | `subject-descriptor` | 0.839 | 154161.1 | -7594.9 | 0 |
| `regression-boundary-independent-mobile-api` | holdout | security-verification | ✔ | `consumer-driven-contract-test` | `consumer-driven-contract-test` | 0.907 | 152829.5 | -2679.4 | 0 |
| `security-pipeline-api-gateway` | holdout | security-verification | ✔ | `authentication-enforcer` | `authentication-enforcer` | 0.936 | 135569.4 | -3218.0 | 0 |
| `security-verification-payment-release` | holdout | security-verification | ✘ | `rbac` | `separation-of-duties` | 0.737 | 219541.1 | -6825.8 | 0 |
| `test-doubles-device-protocol` | holdout | security-verification | ✔ | `mock` | `mock` | 0.664 | 143186.4 | -3289.1 | 0 |
| `security-verification-admin-ui` | holdout | security-verification | ✔ | `page-object` | `page-object` | 0.711 | 76753.3 | -2918.6 | 0 |
| `testability-seams-certificate-clock` | holdout | security-verification | ✘ | `timeout` | `test-specific-subclass` | 0.650 | 129371.1 | -3789.0 | 0 |
| `validation-error-contracts-payment-authorization` | holdout | security-verification | ✔ | `result` | `either` | 0.762 | 261119.8 | -10220.9 | 0 |
| `client-composition-tabs` | holdout | specialized-components | ✔ | `compound-components` | `compound-components` | 0.678 | 217832.3 | -6875.5 | 0 |
| `language-implementation-expression-operators` | holdout | specialized-components | ✘ | `barton-nackman` | `pratt-parser` | 0.653 | 241677.4 | -8456.5 | 0 |
| `crowd-proximity-lookup` | holdout | specialized-components | ✘ | `priority-queue` | `spatial-partition` | 0.855 | 204330.5 | -5362.1 | 0 |
| `checkout-state-transitions` | holdout | specialized-components | ✘ | `side-effect-free-function` | `model-view-intent` | 0.687 | 231560.3 | -9930.4 | 0 |
| `workflow-messaging-module-event-wiring` | holdout | workflow-messaging | ✔ | `publisher-subscriber` | `event-aggregator` | 0.694 | 168400.1 | -5466.3 | 0 |
| `workflow-messaging-silent-inventory-flow` | holdout | workflow-messaging | ✔ | `watchman` | `watchman` | 0.795 | 193485.2 | -14913.3 | 0 |
| `messaging-delivery-order-commands` | holdout | workflow-messaging | ✘ | `load-leveller` | `point-to-point-channel` | 0.834 | 219217.2 | -12108.7 | 0 |
| `workflow-messaging-operational-route-changes` | holdout | workflow-messaging | ✔ | `dynamic-router` | `dynamic-router` | 0.703 | 296326.7 | -4781.3 | 0 |
| `algorithms-traversal-variation-route-traversal` | train | behavior-selection | ✔ | `iterator` | `iterator` | 0.752 | 187623.2 | -4549.9 | 0 |
| `dispatch-plugin-extensions` | train | behavior-selection | ✘ | `sealed-hierarchy-exhaustive-match` | `extensible-visitor` | 0.593 | 206649.2 | -9372.1 | 0 |
| `extension-module-host` | train | behavior-selection | ✘ | `fail-fast` | `plugin` | 0.734 | 230958.1 | -5407.9 | 0 |
| `functional-composition-order-pipeline` | train | behavior-selection | ✘ | `process-manager` | `monad` | 0.749 | 200026.3 | -6210.7 | 0 |
| `concurrency-resilience-registry-capacity` | train | concurrency-resilience | ✘ | `balking` | `monitor-object` | 0.838 | 172901.2 | -5743.6 | 0 |
| `coordination-primitives-lazy-schema-cache` | train | concurrency-resilience | ✘ | `caching` | `double-checked-locking` | 0.707 | 175823.3 | -16572.6 | 0 |
| `concurrency-resilience-multi-dependency-pools` | train | concurrency-resilience | ✘ | `rate-limiting` | `bulkhead` | 0.857 | 158478.5 | -3631.2 | 0 |
| `concurrency-resilience-payload-handles` | train | concurrency-resilience | ✘ | `message` | `copy-on-write` | 0.644 | 170292.4 | -7103.9 | 0 |
| `request-cancel-propagation` | train | concurrency-resilience | ✔ | `cancellation-token` | `cancellation-token` | 0.771 | 252183.5 | -11669.0 | 0 |
| `composition-encapsulation-provider-boundary` | train | construction-composition | ✘ | `gateway` | `adapter` | 0.793 | 175000.2 | -3484.1 | 0 |
| `creation-config-matched-database-products` | train | construction-composition | ✘ | `separated-interface` | `abstract-factory` | 0.825 | 188767.5 | -16689.7 | 0 |
| `dependency-wiring-job-runner-lifecycle` | train | construction-composition | ✘ | `process-manager` | `inversion-of-control` | 0.819 | 176064.3 | -12850.2 | 0 |
| `functional-composition-request-stages` | train | construction-composition | ✔ | `pipeline` | `pipeline` | 0.687 | 243447.3 | -7555.1 | 0 |
| `method-structure-quote-decomposition` | train | construction-composition | ✔ | `macro-method` | `composed-method` | 0.907 | 283016.0 | -8595.9 | 0 |
| `domain-logic-supplier-update` | train | domain-modeling | ✔ | `transaction-script` | `transaction-script` | 0.661 | 71940.8 | -2676.8 | 0 |
| `functional-data-modeling-order-lifecycle` | train | domain-modeling | ✔ | `make-illegal-states-unrepresentable` | `make-illegal-states-unrepresentable` | 0.763 | 139347.1 | -5491.5 | 0 |
| `model-clarity-reproducible-pricing` | train | domain-modeling | ✘ | `money` | `side-effect-free-function` | 0.911 | 157380.4 | -4505.3 | 0 |
| `tactical-inventory-consistency` | train | domain-modeling | ✔ | `aggregate` | `aggregate` | 0.831 | 226935.7 | -7543.7 | 0 |
| `time-measurement-normalization` | train | domain-modeling | ✔ | `quantity` | `quantity` | 0.917 | 357565.3 | -6563.3 | 0 |
| `values-correctness-upload-lifecycle` | train | domain-modeling | ✘ | `immutability` | `typestate` | 0.650 | 180230.8 | -6626.0 | 0 |
| `message-contracts-async-quotes` | train | interaction-boundary | ✘ | `scatter-gather` | `request-reply` | 0.840 | 179647.2 | -29860.1 | 0 |
| `message-endpoints-inventory-request` | train | interaction-boundary | ✘ | `messaging-gateway` | `messaging-gateway` | 0.000 | 330632.9 | -11718.4 | 0 |
| `remote-contracts-order-submission` | train | interaction-boundary | ✔ | `data-transfer-object` | `remote-facade` | 0.700 | 180122.8 | -5219.7 | 0 |
| `server-presentation-page-ownership` | train | interaction-boundary | ✘ | `action-domain-responder` | `page-controller` | 0.924 | 150314.3 | -5248.3 | 0 |
| `translation-erp-procurement` | train | interaction-boundary | ✔ | `anti-corruption-layer` | `anti-corruption-layer` | 0.866 | 71453.8 | -4576.0 | 0 |
| `change-tracking-customer-session` | train | persistence-state | ✘ | `identity-field` | `identity-map` | 0.692 | 186497.7 | -15651.8 | 0 |
| `data-access-gateways-order-lines` | train | persistence-state | ✔ | `row-data-gateway` | `table-data-gateway` | 0.785 | 222234.5 | -13868.8 | 0 |
| `domain-to-store-order-bundle` | train | persistence-state | ✘ | `unit-of-work` | `composite-entity` | 0.857 | 160632.5 | -6871.4 | 0 |
| `persistence-state-order-workspace` | train | persistence-state | ✔ | `optimistic-offline-lock` | `coarse-grained-lock` | 0.830 | 199910.6 | -12451.7 | 0 |
| `queries-metadata-invoice-collection-access` | train | persistence-state | ✘ | `data-access-object` | `repository` | 0.865 | 153785.2 | -9787.0 | 0 |
| `relational-state-identity` | train | persistence-state | ✘ | `version-number` | `identity-field` | 0.688 | 186874.3 | -4973.5 | 0 |
| `lifecycle-catalog-cache` | train | persistence-state | ✘ | `cache-aside` | `cache-aside` | 0.877 | 272360.9 | -6791.1 | 0 |
| `audit-sensitive-payment-boundary` | train | security-verification | ✘ | `authorization-enforcer` | `audit-interceptor` | 0.916 | 240749.2 | -10041.1 | 0 |
| `capabilities-isolation-session-logout` | train | security-verification | ✘ | `revocable-forwarder` | `revocable-forwarder` | 0.809 | 189565.2 | -6811.1 | 0 |
| `compile-time-access-iterator-core` | train | security-verification | ✔ | `attorney-client` | `attorney-client` | 0.666 | 155377.1 | -2831.4 | 0 |
| `enforcement-owned-secret-vault` | holdout | security-verification | ✔ | `protected-system` | `protected-system` | 0.780 | 208303.4 | -5306.1 | 0 |
| `identity-pluggable-credential-verification` | holdout | security-verification | ✔ | `authenticator` | `authenticator` | 0.768 | 235031.1 | -7574.8 | 0 |
| `regression-boundary-legacy-reporting` | holdout | security-verification | ✘ | `composed-method` | `characterization-test` | 0.720 | 173464.1 | -3137.0 | 0 |
| `security-pipeline-event-ingress` | holdout | security-verification | ✘ | `content-based-serializer` | `intercepting-validator` | 0.671 | 195213.1 | -8977.5 | 0 |
| `security-verification-unmatched-api-access` | holdout | security-verification | ✔ | `fail-safe-defaults` | `fail-safe-defaults` | 0.888 | 222270.0 | -7135.7 | 0 |
| `test-doubles-policy-outcomes` | holdout | security-verification | ✘ | `policy-guard` | `stub` | 0.792 | 104512.7 | -4762.2 | 0 |
| `security-verification-focused-user-setup` | holdout | security-verification | ✘ | `make-illegal-states-unrepresentable` | `test-data-builder` | 0.716 | 173033.2 | -5290.9 | 0 |
| `testability-seams-checkout-screen` | holdout | security-verification | ✔ | `humble-object` | `humble-object` | 0.745 | 130910.3 | -8287.5 | 0 |
| `validation-error-contracts-registry-lookup` | holdout | security-verification | ✔ | `option-maybe` | `option-maybe` | 0.772 | 211678.0 | -5124.3 | 0 |
| `client-composition-permissions` | holdout | specialized-components | ✘ | `policy-guard` | `higher-order-component` | 0.932 | 217371.7 | -6377.6 | 0 |
| `language-implementation-handwritten-grammar` | holdout | specialized-components | ✘ | `combinator-library` | `recursive-descent-parser` | 0.661 | 295412.7 | -10523.4 | 0 |
| `runtime-actor-advance` | holdout | specialized-components | ✘ | `subclass-sandbox` | `update-method` | 0.853 | 278938.9 | -4685.4 | 0 |
| `local-search-field-state` | holdout | specialized-components | ✘ | `partial-response` | `state-colocation` | 0.729 | 139870.8 | -3654.4 | 0 |

## Notes

- analyze-only arm: hit = the §9 dominant election replayed over the ANALYZE selections (the same definition as the full arm's final_pattern before loop reselection), recall = the analysis selection; generate/evaluate are skipped, so the pipeline block is null and e2e_ms is a phase subset, not a full-run latency
- NO_WARMUP protocol deviation (2026-10-07 latency-reduction round): the warmup scenario is also a measured scenario, so it stays in the sample set; the reasoning LRU would otherwise hand it a ~0 ms reasoning.analyze cache hit (warmup-cache artifact understating the run mean). Warmup-on arms (e.g. the committed benchmarks/base, live-20261002-070756) are NOT mutually comparable with NO_WARMUP arms; base-fresh is the baseline for every comparison this round.
- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
