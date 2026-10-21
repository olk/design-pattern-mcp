# Benchmark run live-20261020-171914

- mode: `live` · split `holdout` · limit `1` · repeats `1`
- scenarios: 1 records (train 0 / holdout 1)
- corpus: `tests/benchmark/scenarios/seed.json` sha256 `ebfa882dce2b82e7…`
- git: `c643c2be0d30` dirty=True
- catalog: 486 records

## Quality (per split)

| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |
|---|---|---|---|---|---|---|---|---|---|
| overall | 1 | 0.0% | [0.000, 0.793] | 0 | 0.200 | 0.200 | 0.000 | 0.000 | 0 |
| train | 0 | null | null | null | null | null | null | null | null |
| holdout | 1 | 0.0% | [0.000, 0.793] | 0 | 0.200 | 0.200 | 0.000 | 0.000 | 0 |

## Latency

- e2e: n=1 p50=null p95=null (warmup 272326.3) — 1 sample(s): p50/p95 need at least 4
- residual: stage sums double-count wall clock when GENERATE batches run concurrently (generation_max_concurrency); a negative residual means overlap, not missing time

| stage | calls | failures | total ms | p50 ms | p95 ms | ops |
|---|---|---|---|---|---|---|
| `embed.query` | 1 | 0 | 406.2 | null | null | embed_queries x1 |
| `rerank` | 1 | 0 | 24916.2 | null | null | rerank x1 |
| `llm.analyze` | 1 | 0 | 18899.9 | null | null | extract_brief_and_parts x1 |
| `llm.weights` | 1 | 0 | 7852.2 | null | null | extract_weights_batch x1 |
| `llm.generate` | 1 | 0 | 78977.6 | null | null | generate x1 |
| `llm.evaluate` | 1 | 0 | 40261.7 | null | null | evaluate x1 |
| `reasoning.evaluate` | 1 | 0 | 41135.1 | null | null | run_pre_llm x1 |
| `reasoning.weights` | 1 | 0 | 35492.8 | null | null | run_pre_llm x1 |
| `reasoning.generate` | 1 | 0 | 25199.5 | null | null | run_pre_llm x1 |
| `reasoning.analyze` | 1 | 0 | 0.0 | null | null | run_pre_llm x1 |

## Calibration (train split, monitoring only)

- Brier: null · ECE (10 bins): null

| bin | range | n | avg score | accuracy | gap |
|---|---|---|---|---|---|

## Risk-coverage

- holdout: n=1 risk@80=1.000 risk@90=1.000 risk@100=1.000 AURC=0.500
- train: n=0 risk@80=null risk@90=null risk@100=null AURC=null

## Per scenario

| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |
|---|---|---|---|---|---|---|---|---|---|
| `cc-legacy-adapter` | holdout | construction-composition | ✘ | `gateway` | `adapter` | 0.826 | 265345.6 | -7795.6 | 0 |

## Notes

- Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the src.* log record extras; a repair re-call after an 'output failed validation' WARNING is flagged repair=True) attached to llm.* stage records and aggregated per stage as prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose prompt_tokens_details to the protocol caller)
