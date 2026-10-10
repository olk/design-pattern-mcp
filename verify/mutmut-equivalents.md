# mutmut Equivalents Ledger — Governed Survivors

> Companion to `verify/mutmut-scope.md` and `verify/mutmut-baseline.json`.
> Doctrine: survivors are hints, never verdicts —
> this ledger is where a hint becomes a *decision*. A survivor may only stop
> failing the gate through the deliberate ratchet refresh
> (`make regen-mutmut-baseline`), never by ignoring the gate.

## Rules

1. Every survivor triage ends in exactly one of two states:
   - **killed**: write the missing test (preferred — do it in the same PR
     that regenerates the baseline);
   - **equivalent**: record the mutant key cluster here with a reason, then
     refresh the baseline.
2. An equivalence claim must state the *observable behaviour* that is
   unchanged, not just "the test can't see it". "The test can't see it" is a
   test gap, not an equivalence.
3. Rows are per survivor-key cluster (mutmut keys share the mutated token),
   never per individual mutant index — indices shift on every source edit.

## Known survivor clusters (re-derived 2026-09-26)

> The 2026-09-11 table was ported from the sibling architecture-pattern server
> and named functions that do not exist here (`_category_code`,
> `evaluate_printable_text`, `StripWindow`/`TextVerdict`) and test modules that
> were never ported (`test_text_validation_internals.py`,
> `test_design_normalization_internals.py`). The rows below name the real
> functions in this repo's mutation scope. Counts are refreshed from
> `verify/mutmut-baseline.json` at every regen; most are expected to be test
> gaps (kill them), not equivalents.
>
> Measured at the 2026-09-26 regen — 6576 mutants, 3827 killed, 2724 survived
> (25 unclassified), module kill ratios 0.52–1.00. These are the first real
> numbers for this repo's functions; the ported 2026-09-11 counts are gone.
>
| Function | Survivors | First-glance classification | Disposition |
|---|---|---|---|
| `src/validation.py::validate_with_retries` | 35/75 | **suspected test gap** — the repair loop's `range(max_retries)` bound, the `last_error is None` guard and the `attempt + 1` report offset are only driven at one budget. Kill by parametrizing the retry budget and asserting the exact call count. | pending |
| `src/tools/jobs.py::_init` | 21/48 | likely gap — the `CREATE TABLE` DDL string mutants are invisible to behaviour tests. Candidates: a `PRAGMA table_info` shape assertion (pragma semantics on 3.7 still unverified). | pending |
| `src/tools/jobs.py::create_job` | 12/22 | gap — debug-log f-string mutants plus the `uuid4()`/timestamp interleave. The log mutants become observational once a caplog test exists; until then they are candidate equivalents. | pending |
| `src/validation.py::format_validation_errors` | 9/19 | gap — message-shape mutants (the `repr(...)[:80]` truncation, the join separator) survive. Kill by pinning the exact rendered string. | pending |
| `src/text_validation.py::ensure_printable_text` | 18/44 | gap — the strip-window and length-clamp arithmetic mutants. Kill by pinning the exact `(lo, hi)` window and the `max_length` boundary for the all-whitespace and over-max corners. | pending |
| `src/tools/jobs.py::_guarded_update` | 8/18 | gap — the guarded-UPDATE parameter tuples and the `rowcount == 0` re-read. Kill by pinning the exact SQL and params sent per transition. | pending |
| `src/config_expansion.py::expand_env_in_obj` | 6/12 | gap — the dict/list recursion branches. Kill with a nested `{env:…}` structure assertion (the CE-6 shadow-walker oracle does this for the flat case only). | pending |
| `src/patterns/retriever.py::_min_max` | 6/16 | gap — the `span <= 0.0` degenerate branch and the `(score - low) / span` arithmetic. Kill by pinning the all-equal-range → 1.0 law and a two-point normalization. | pending |
| `src/text_validation.py::_pattern_name_validator` / `_domain_validator` / `_freetext_validator` | 1–3 each | gap — residual mutants are the `field=` literal and `allow_line_breaks` flag in each wrapper call. The per-field error-name assertions added in `tests/unit/test_input_validation.py` (2026-09-26) kill most; the rest are the documented length-bound literals. | confirm at next regen |
| `src/tools/jobs.py::_conn` | 4/5 | the `RuntimeError` guard text + condition — guard-presence is tested via a not-initialised store? if not, that is the gap. | pending |
| `src/pipeline.py` prompt builders (`_build_analyze_user_prompt`, `_build_generate_system_prompt`, …) | part of 1285 | **equivalent** — prompt text is a byte string passed to a scripted LLM double that echoes it; a mutated token changes wording no oracle asserts and nothing downstream parses. | equivalent |
| `src/patterns/retriever.py::_dense_leg`, `::_build_bm25_sync` (per-leg `min(..., n)` clamps) | see below | **equivalent** — `faiss` pads past `ntotal` with id `-1` and the `int(idx) != -1` filter drops them, and `BM25Retriever` refuses an oversized `k` itself. Verified 2026-09-24 by hand; the engine now covers these functions. | equivalent |

## Superseded hand-applied sample — per-leg recall caps (2026-09-24, closed 2026-09-26)

> **Historical.** The rows below are kept as the record of how the cap arithmetic
> was verified on 2026-09-24, when `make test-mutations` could not run: the
> ported `pytest_add_cli_args_test_selection` named
> `tests/verification/test_*_properties.py` files that do not exist in this
> repo, so the stats pass aborted at collection. That is fixed — the selection
> now names the two test DIRECTORIES — and both cap functions
> (`src/patterns/retriever.py::_dense_leg`, `::_build_bm25_sync`) are inside the
> live mutation scope, so the engine now decides. The two clamp survivors
> recorded here remain the expected equivalents; re-confirm them against the
> current `verify/mutmut-baseline.json` before treating the rows as still open.

| Function | Mutant-key token cluster | Observable behaviour unchanged | Disposition | Date |
|---|---|---|---|---|
| `src/patterns/retriever.py::_dense_leg` | `min(self._retrieval.dense_top_k or len(self._names), len(self._names))` — the `min(..., len(self._names))` clamp | `faiss.IndexFlatIP.search` pads every neighbour beyond `ntotal` with id `-1`; `_dense_leg`'s `int(idx) != -1` filter drops exactly those, so an unclamped `k` yields an identical `{name: score}` map (probe: 30-vector index, `k=10**6` → 30 kept neighbours, last raw id `-1`, pairs identical to `k=30`) | equivalent — verified 2026-09-24 | 2026-09-24 |
| `src/patterns/retriever.py::_build_bm25_sync` | `min(self._retrieval.bm25_top_k or corpus_n, corpus_n)` — the `min(..., corpus_n)` clamp | `BM25Retriever` refuses `k` above its node count itself ("Overriding the value of similarity_top_k to number of nodes added"), so the unclamped cap returns the same 30 results with identical scores (probe: 30 nodes, `similarity_top_k=10**6` vs `30` → identical maps) | equivalent — verified 2026-09-24 | 2026-09-24 |

## Entry format (for future rows)

```
| <module>::<function> | <mutant-key token cluster> | <observable behaviour unchanged> | killed-by <test id> / equivalent <reason> | <date> |
```

## DesignPipeline Workflow conversion (2026-10-01)

Triage of the fresh post-conversion scratch (`mutants/src/pipeline.py.meta`, 1015 survivors)
against the committed 2026-09-26 floor: the conversion-new survivors are the rows below plus
`_generate_in_batches` 69–72; every other fresh survivor belongs to a pre-existing floor
function/cluster (helper functions, prompt builders) and keeps its 2026-09-26 classification.
`_finish` 9/10 were conversion-new survivors at scratch time and are killed by
`tests/unit/test_pipeline_workflow.py::test_reselection_header_reports_yes` (asserts the exact
`reselection=yes` header token); they are therefore not ledgered as survivors.

| Function | Mutant-key token cluster | Observable behaviour unchanged | Disposition | Date |
|---|---|---|---|---|
| `src/pipeline.py::DesignPipeline.run_design/analyze/generate/evaluate/design_loop` | `__mutmut_20/20/12/13/12` — `cast(PipelineDone, await handler)` → `cast(None, await handler)` | `typing.cast` is a runtime no-op; the awaited handler result is unchanged and `PipelineDone.payload_as` re-asserts the payload type | equivalent | 2026-10-01 |
| `src/pipeline.py::DesignPipeline.design_loop` | `__mutmut_9/10` — `mode="design_loop"` → `"XXdesign_loopXX"`/`"DESIGN_LOOP"` | `route`'s step dispatch ends in an unconditional `SelectionReady` fallback (src/pipeline.py:1568-69); the mutated mode string is never re-read for control flow | equivalent | 2026-10-01 |
| `src/pipeline.py::DesignPipeline.validate` | `__mutmut_1/2` — defaults `validate_resource_configs=True→False`, `validate_resources=False→True`; `__mutmut_21/22` — passthrough kwarg `=None`; `__mutmut_23/24` — kwarg dropped | DesignPipeline registers no workflow resources, so the base `Workflow.validate` branches (`if validate_resource_configs:` / `if validate_resources:`) run over empty step-resource configs in every variant and produce the same result; the return value (human-in-the-loop flag) is untouched | equivalent | 2026-10-01 |
| `src/pipeline.py::DesignPipeline._finish` | `__mutmut_3` — `assert best is not None and best_evaluation is not None` → `or` | `best` and `best_evaluation` are assigned together in `_reselect`; the assert outcome is identical for every reachable state | equivalent | 2026-10-01 |
| `src/pipeline.py::DesignPipeline._generate_in_batches` | `__mutmut_69/71/72` — cleanup `await asyncio.gather(*tasks, return_exceptions=True)` → `return_exceptions=None` / positional-only / `=False`; `__mutmut_70` — drops `*tasks` (awaits nothing) | With a falsy `return_exceptions` the cleanup gather re-raises the same first-failed-child exception instance that the original bare `raise` re-raises, so the outward exception and already-cancelled sibling tasks are identical (mutant 70 skips only the no-op await of cancelled tasks) | equivalent | 2026-10-01 |
| `src/pipeline.py::DesignPipeline._generate_in_batches` | `__mutmut_73/85` — log message `"GENERATE batches complete"` → `None`; log extra key `returned_applications` → `RETURNED_APPLICATIONS` | Structured log text/extras are never parsed by any test or downstream consumer (2026-09-26 log-text precedent) | equivalent | 2026-10-01 |
