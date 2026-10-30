# Benchmark comparison

- baseline (baseline): `consistency-base-20-spark` mode `analyze` (60 records) in `/mnt/a`
- candidate (candidate): `consistency-jev-20` mode `analyze` (60 records) in `/mnt/b`
- scenario file match: True · paired scenarios: 60

## Verdict: `inconclusive`

_candidate wins iff holdout p95 e2e improves >= 15% AND candidate overall acceptable-primary hit rate >= baseline overall Wilson lower bound_

- fewer than 4 holdout e2e samples per arm (baseline 0, candidate 0): p95 is undefined

## Hit rate (Wilson / Newcombe)

| split | baseline | candidate | difference [95% CI] |
|---|---|---|---|
| overall | 60.0% | 66.7% | +0.067 [-0.104, +0.232] |
| train | 60.0% | 66.7% | +0.067 [-0.104, +0.232] |
| holdout | null | null | null |

## Selection metrics (mean per arm)

| split | metric | baseline | candidate | Δ |
|---|---|---|---|---|
| overall | recall_at_3 | 0.3975 | 0.4008 | +0.0033 |
| overall | recall_at_5 | 0.4736 | 0.4775 | +0.0039 |
| overall | supporting_f1 | 0.0033 | 0.0381 | +0.0348 |
| overall | acceptable_primary_f1 | 0.3844 | 0.4317 | +0.0472 |
| overall | veto_violations | 0.0000 | 0.0000 | +0.0000 |
| overall | score | 0.8161 | 0.8214 | +0.0053 |
| holdout | recall_at_3 | null | null | null |
| holdout | recall_at_5 | null | null | null |
| holdout | supporting_f1 | null | null | null |
| holdout | acceptable_primary_f1 | null | null | null |
| holdout | veto_violations | null | null | null |
| holdout | score | null | null | null |

## Paired deltas (candidate minus baseline)

- e2e: mean -10447.6788 ms · median -12835.5135 ms · faster 36 / slower 24 · sign test p=0.155002 (n=60)

| metric | compared | mean Δ | median Δ | improved | worsened | unchanged |
|---|---|---|---|---|---|---|
| recall_at_3 | 60 | +0.0033 | +0.0000 | 8 | 8 | 44 |
| recall_at_5 | 60 | +0.0039 | +0.0000 | 6 | 5 | 49 |
| supporting_f1 | 60 | +0.0348 | +0.0000 | 4 | 1 | 55 |
| acceptable_primary_f1 | 60 | +0.0472 | +0.0000 | 7 | 3 | 50 |
| veto_violations | 60 | +0.0000 | +0.0000 | 0 | 0 | 60 |
| score | 60 | +0.0053 | +0.0042 | 32 | 28 | 0 |

## Hit flips

| key | baseline | candidate |
|---|---|---|
| composition-encapsulation-checkout-entry#2 | miss `unit-of-work` | hit `facade` |
| domain-logic-shipment-import#3 | miss `table-data-gateway` | hit `table-module` |
| extension-compile-time-matrix#2 | miss `type-safe-builder` | hit `policy-based-design` |
| extension-compile-time-matrix#3 | hit `policy-based-design` | miss `named-constructor` |
| functional-composition-request-validation#3 | hit `applicative` | miss `contravariant-validators` |
| method-structure-buffer-assignment#2 | miss `smart-pointer` | hit `non-throwing-swap` |
| model-clarity-ledger-contract-checks#3 | hit `assertion` | miss `guard-clause` |
| queue-drain-shutdown#2 | miss `producer-consumer` | hit `poison-pill` |
| tactical-customer-identity#1 | miss `value-object` | hit `entity` |
| tactical-customer-identity#3 | miss `value-object` | hit `entity` |

## Stage deltas

| stage | calls Δ | total ms baseline | total ms candidate | Δ ms |
|---|---|---|---|---|
| `embed.query` | +0 | 78034.0 | 62644.4 | -15389.6 |
| `llm.analyze` | +0 | 3464140.1 | 3356788.5 | -107351.6 |
| `llm.weights` | +0 | 2367067.8 | 2317243.5 | -49824.3 |
| `reasoning.analyze` | +0 | 1174576.4 | 1117369.2 | -57207.2 |
| `reasoning.weights` | +0 | 3617418.0 | 3868829.0 | +251411.0 |
| `rerank` | -33 | 4170512.5 | 3276293.6 | -894218.8 |

## Warnings

- effective config differs in ['decisions']
- environment differs in ['DECISIONS_ENABLED']
