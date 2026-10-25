# mutmut Scope Ledger — this repo's decision modules

> Owned by the L3 (mutation) section of the verification program described in
> [`AGENTS.md`](../AGENTS.md) (drift countermeasure, mirroring
> `verify/fizz/README.md`: a PR that changes `[tool.mutmut]` in
> `pyproject.toml` must change a row here in the same PR).
> Run: `make test-mutations` · Enforcement: `scripts/mutmut_baseline.py --gate`
> · Baseline: `verify/mutmut-baseline.json` · Parameters:
> `verify/mutmut-thresholds.toml` · Equivalents: `verify/mutmut-equivalents.md`
>
> (`docs/verification.md` and `docs/testing-strategies.md` are cited by earlier
> revisions of this file; neither exists in this repo — the doctrine they
> carried, "survivors are hints, never verdicts", is restated in
> `verify/mutmut-equivalents.md`.)

## Why a narrow scope

The mutation surface is the decision logic — the code where a silent bug is a
kill-class bug — never the whole tree. A module enters scope only when all four
checklist rows hold; it leaves scope only through a row edit here.

> The file previously carried a "Tier A/B/C + R5" scheme ported from the
> sibling architecture-pattern server, and every row cited modules or
> functions this repo does not have. The scheme is dropped rather than
> re-derived: the tiering existed to stage that repo's scope growth, and here
> the scope is settled (see the module ledger for the current, measured
> surface). A future scope addition re-introduces whatever staging scheme that
> change actually needs.

## Module ledger

> Re-derived 2026-09-26 from the modules mutmut actually mutates in THIS repo
> (`mutants/src/**/*.meta` after a full `make test-mutations` run), which is
> the exact complement of `do_not_mutate` in `pyproject.toml`. The previous
> table was ported from the sibling architecture-pattern server and named
> modules that do not exist here (`src/reasoning/tools.py`,
> `src/tools/submit_architecture_design.py`,
> `src/tools/cancel_architecture_design.py`, `src/patterns/embedder.py` as a
> scope row) and function names that do not exist in the files it cited.

| Module | Mutated surface | Ownership (kill authority) | Why in scope | Expected survivors |
|---|---|---|---|---|
| `src/pipeline.py` | 54 functions / 2695 mutants — the whole orchestration + analyze/generate/evaluate seam (`run_design`, `design_loop`, `analyze`, `generate`, `evaluate`, `score_candidates`, `select_for_part`, `apply_vetoes`, `_fallback_selection`, `_reasoning`, `validate`, the prompt builders) | L1 `test_analyze.py`, `test_generate.py`, `test_evaluate.py`, `test_scoring.py`, `test_selection.py`, `test_fallbacks.py`, `test_composition.py`, `test_input_validation.py`, `test_style_precedence.py`, `test_pipeline_validate.py` + L2 `tests/verification/test_bug_garden.py` | the decision core of the whole server: intake, decomposition, per-part scoring, vetoes, fallback, DL attempt loop | **yes — 1285/2695 survive (ratio 0.52); the largest surface, and prompt-string plus log-format mutants dominate. Triage pending** |
| `src/patterns/loader.py` | 22 functions / 1262 mutants — `load_catalog`, `assert_enums_in_sync`, `parse_catalog_order`, `parse_out_of_scope_terms`, `parse_locally_implementable_names`, `_check_variants`, `_build_record`, `_build_alias_index`, `_load_one_record` | L1 `test_catalog.py` | catalog discovery, schema-enum sync and alias closure decide which patterns a query can ever see; a dropped membership check silently ships a broken catalog | yes — 519/1262 (ratio 0.59); JSON/DDL construction mutants. Triage pending |
| `src/patterns/dense_cache.py` | 21 functions / 818 mutants *(pre-2026-10-25 snapshot — the logging/write-back edit shifted per-function indices, regen pending)* — `resolve_dense_vectors`, `load_manifest`, `load_vectors`, `store_cache`, `build_manifest`, `corpus_digest`, `_metadata_fingerprint_match`, `_corpus_fingerprint_match`, `probe_cosine`, `_probe_embed`, `_write_back` | L1 `test_dense_cache.py` + `test_config.py` + `test_retrieval.py::test_warm_start_embeds_only_the_probe_text` + L4 `dense_cache.fizz` (EMB-1..4, FG-37..40) | the two-layer cache fingerprint + semantic probe decide "reuse the cache or re-embed"; a stale-hit is a silent-wrong-answer class. Since 2026-10-25 the module also owns restart durability: a dropped write-back guard makes every restart re-embed (~19 min), and a write-back that raises or discards the freshly embedded vectors turns a read-only cache directory into a startup failure — `_write_back` is best-effort by contract (EMB-4). The decision log lines (`Dense cache hit/miss/written/disabled`, each carrying `path`) are pinned by `test_hit_and_write_lines_name_the_cache_directory` / `test_miss_and_disabled_lines_name_the_cache_directory` (message **and** extras). Those insertions shift mutmut's per-function indices, so `scripts/mutmut_baseline.py --gate` will report pre-existing survivors under new keys until `make regen-mutmut-baseline` runs deliberately — **the ratchet was not re-run for this edit**; the log-mutant kill is unit-pinned, the ratchet status is unverified | yes — 336/818 (ratio 0.59, **pre-change snapshot**); manifest/npy IO string mutants. Regen + triage pending |
| `src/patterns/retriever.py` | 16 functions / 420 mutants — `retrieve`, `retrieve_many`, `build`, `_fuse`, `_blend`, `_min_max`, `reciprocal_rank_score`, `_rerank`, `_leg_scores_many`, `_blend_and_cap` | L1 `test_retrieval.py` | fusion weights, the per-leg min–max normalization, the RRF arithmetic and the per-leg recall caps decide whether a query sees real patterns or the fallback | yes — 100/420 (ratio 0.76, the strongest large module); async/concurrency mutants need the fakes. Triage pending |
| `src/variants.py` | 14 functions / 317 mutants — `select_variant`, `bind_entry`, `variants_for_pattern`, `_by_explicit_id`, `_by_paradigm`, `_by_capability_floor`, `_satisfies`, `_floor_sort_key`, `variants_payload` | L1 `test_variants.py` + `test_catalog.py` | the §8.4 four-layer variant precedence: a wrong layer wins and a design silently binds a non-default realization | yes — 100/317 (ratio 0.68); payload-shaping mutants. Triage pending |
| `src/design_normalization.py` | 12 functions / 353 mutants — `build_brief`, `repair_parts`, `_normalize_priorities`, `_merge_oversized`, `synthesize_construction_part`, `truncate`, `first_sentence`, `kebab`, the four `as_*` coercers | L1 `test_fallbacks.py`, `test_input_validation.py` (indirect) | §4.11/§5 denormalization precedence, the priority invariant ("exactly one part at 1.0") and the bounded merge repair | yes — 157/353 (ratio 0.56); the coercers and the merge search are search-heavy. Triage pending |
| `src/tools/jobs.py` | 16 functions / 182 mutants — the whole `JobsStore` lifecycle (`create_job`, `set_running`, `set_completed`, `set_failed`, `set_cancelled`, `is_cancelled`, `get_job`, `close`) + `_guarded_update`, `_init` | L1 `test_jobs.py` (transition matrix) + L2 `tests/verification/test_bug_garden.py` (J-1/J-2) | W0-1 guarded transitions: the `AND status = ?` clause is the whole check-and-set; the single most safety-critical automaton in the server | yes — 69/182 (ratio 0.62); mostly SQL-string f-string mutants, killable only through behaviour. Triage pending |
| `src/validation.py` | 3 functions / 96 mutants — `validate_with_retries`, `format_validation_errors`, `_format_failure` | L1 `test_validation.py` | the self-healing repair loop at the LLM boundary: an unbounded retry is a hang class, a swallowed `LLMError` is a silent-degradation class | **yes — 44/96 (ratio 0.54); `validate_with_retries` alone 35/75 survive, the weakest audited function. Triage pending** |
| `src/text_validation.py` | 6 functions / 84 mutants — `ensure_printable_text` (strip window + length clamp), the three `AfterValidator` wrappers, `_is_control_or_format`, `_is_letter_or_digit` | L1 `test_input_validation.py` | the E1/E2 intake gate: the strip window and the `hi - lo > max_length` clamp decide what text ever reaches the LLM | yes — 23/84 (ratio 0.73); `ensure_printable_text` 18/44, the wrapper `field=` literals are the known cluster |
| `src/provider_health.py` | 6 functions / 155 mutants — `before_call`, `record_failure`, `record_success`, `is_circuit_qualifying_status`, `extract_status_code` | L1 `test_provider_health.py` | the circuit breaker: which failures count as qualifying decides whether a dead provider burns the full call budget | yes — 34/155 (ratio 0.78); timestamp/threshold arithmetic. Triage pending |
| `src/anti_scope.py` | 6 functions / 156 mutants — `is_architecture_only`, `matched_terms`, `catalog_phrases`, `hypothesis_for`, `_normalize` | L1 `test_anti_scope.py` + `test_pipeline_validate.py` | §5 rule 5: an architecture-only part that survives drives selection on a name the catalog deliberately excludes | yes — 49/156 (ratio 0.69); token-count threshold mutants. Triage pending |
| `src/conflicts.py` | 1 function / 3 mutants — `conflicting` | L1 `test_selection.py` | §8.3 rule 3: a missing conflict pair lets two incompatible patterns share a decision part | none (3/3 killed) |
| `src/config_expansion.py` | 2 functions / 30 mutants — `expand_env`, `expand_env_in_obj` | L1 `test_config.py` + L2 `tests/verification/test_config_properties.py` (CE-1..CE-6) | `{env:VAR}` expansion feeds every config secret and URL — silent mis-expansion is a config-injection class | yes — 6/30 (ratio 0.80); the nested dict/list recursion in `expand_env_in_obj` is the open cluster. Triage pending |
| `src/style_affinity.py` | 1 function / 5 mutants — `affinity_for` | L1 `test_style_precedence.py` | §10.1 soft prior: a wrong category set silently disables or misapplies the style bonus | 2/5 (ratio 0.60) — the affinity-set membership mutants; triage pending |
| `src/__main__.py`, `src/patterns/__init__.py` | 0 mutants (re-export shims) | — | nothing to mutate; listed so the ledger accounts for every `do_not_mutate`-excluded module that still appears in the scratch tree | n/a |
>
> Survivor counts above are from the 2026-09-26 regen — the first run against
> this repo's own source. Earlier revisions of this file quoted the sibling
> server's ratios (`src/validation.py` 0.375, `src/tools/jobs.py` 0.67) for
> functions that do not exist here.

## Deferred (excluded from mutmut scope)

> Drift countermeasure mirroring `verify/fizz/README.md`: a PR that changes
> `[tool.mutmut]` in `pyproject.toml` must change a row in this file in the
> same PR. Two exclusion rationales live here: modules that may move to scope
> (awaiting sign-off) and modules whose decisions other verification layers
> already own (model-checked).

### Awaiting scope sign-off — do NOT add without a ledger row

| Module | Candidate functions | Why deferred |
|---|---|---|
| `src/config.py` | the four `_check_*` model validators (weight-sum `1e-3`/`0.05` tolerances, floor ≤ `2/60`) | boundary logic is a good mutmut target, but the file is 564 lines of Pydantic models — mutating field defaults and `Field` wiring would flood the run with equivalents; needs `# pragma: no mutate` support verification first (3.7 pragma semantics unverified) and dedicated boundary tests. Boundary tolerance IS covered meanwhile by the garden's `boundary-tolerance` class (M-27..M-31). |
| `src/tools/analyze.py` / `src/tools/design.py` / `src/tools/evaluate.py` / `src/tools/generate.py` | tool handlers | dedicated handler tests exist (`test_analyze.py` 3, `test_generate.py` 10, `test_evaluate.py` 4; no `test_design.py`) and handlers carry real formatting/error-wrapping decisions, but the tests mock the pipeline — expect survivor noise; ~1 600 LOC would grow the mutant surface by ~4x. Add as a dedicated PR AFTER the first nightly run reports wall-clock for the current scope (job cap: 300 min per `verification.yml`). |
| `src/tools/get_design_status.py` | status mapping handler | covered via `tests/unit/test_protocol_smoke.py` + `test_server_logging.py` (no dedicated `test_server.py` exists); pulling a full server test module into the slice would drag server wiring into every slice. Needs a dedicated handler test file first. |
| `src/tools/patterns.py` | list/get pattern handlers | thin loader delegation; `src/patterns/loader.py` itself is already in scope. |
| `src/reasoning/config.py` | `get_strategy`, command validation | mostly declarative pydantic; the tested decisions (defaults, override parsing) are pinned via `test_reasoning_client.py` but the model-default bulk would be equivalent noise. Revisit together with the `src/config.py` pragma work. |
| `src/agent.py`, `src/reasoning/client.py`, `src/reasoning/prompts.py`, `src/reasoning/tools.py`, `src/errors.py`, `src/tools/_adapters.py`, `src/design_validation.py`, `src/patterns/embedder.py`, `src/patterns/safe_tei_rerank.py` | see the *Model-checked* table below | these modules are in `do_not_mutate` because another verification layer already owns their decisions; they are **not** awaiting sign-off. A row here means "excluded pending scope sign-off", which is a different claim — the *Model-checked* table is the truthful one for them. |

### Model-checked (L4) — decisions owned by the fizz/L1/L2 layers

> Excluded from mutmut because the decision surface is already pinned by
> layers whose authority does not degrade with survivor noise: FizzBee checks
> the state machine exhaustively up to bounds, and the L1/L2 oracles call the
> REAL pipeline methods. Mutating individual statements would re-test
> decisions these layers already enforce.

| Module | Decision functions | Why excluded | Coverage authority |
|---|---|---|---|
| `src/agent.py` | `_generate_structured_once`, `generate_structured` retry wiring, the raw-exception → `LLMError` mapping | boundary/transport: an unmapped exception escaping into the pipeline is caught by the pipeline's own error handling, and the LLM seam is already exercised at every call site. Mutating the mapping would re-test what L1 `test_agent.py` + the seam fakes already pin | L1 `test_agent.py` + L1 `test_provider_health.py` |
| `src/reasoning/client.py` | `_run_cached` (LRU + single-flight), `_generate_trace` (step bound), `_validate_draft` | silent-degradation failure mode by design — a reasoning outage must never fail a tool call, so the module is written defensively and its mutants are dominated by `logger.`/formatting lines. `logger\.` lines are OBSERVATIONAL here (caplog tests), so a global logger exclusion would be wrong | L1 `test_reasoning_client.py` (`TestCacheLRUEviction`, `TestSingleFlight`, `TestToolContracts`) + L4 `reasoning_retry.fizz` (FG-20..22) |
| `src/reasoning/prompts.py` | `build_step_user_prompt`, scaffold rendering, truncation caps | prompt construction is string-mutant noise — the decisions (which scaffold, which cap) are declarative data, not branches | L1 `test_reasoning_client.py` + L0 |
| `src/errors.py` | `JobStateError.__init__`, `MalformedArchitectureOverviewError.__init__` | error attribute contracts (`job_id`, `current_status`) are asserted by every caller; the classes are pure data holders with no decision branches to mutate | L1 `test_jobs.py` (attribute asserts) + L1 `test_adapters.py` (locator/message) |
| `src/tools/_adapters.py` | sub-record parsers (`_parse_relationship`, `_parse_event_contract`, `_parse_api_contract`), `design_integrity_findings`, the overview→Pydantic conversion | boundary/serialization: the parsers turn an untrusted LLM dict into the schema, and the schema validation that follows is the authority. Mutating a `.get(...)` default duplicates `test_adapters.py`'s both-directions coverage | L1 `test_adapters.py` + schema enforcement at the tool boundary |
| `src/design_validation.py` | `evaluate_design_integrity`, `_duplicate_id_violations`, `_relationship_violations`, `_event_violations`, `_api_contract_violations` | the INV-2..5 cross-reference gate runs on the LLM output *inside* `assemble_result`, which is itself mutated (`src/pipeline.py` is in scope); a second mutation layer over the same rule mostly re-tests the schema validators | L1 `test_adapters.py::TestDesignIntegrityFindings` + schema enforcement at the tool boundary |
| `src/patterns/embedder.py` | `_normalize` (row normalisation, zero-vector guard) | the zero-vector → NaN guard is the decision and is pinned directly; the wrapper classes around it have no branches. NaN embeddings are also caught downstream by the dense-cache cosine probe (mutated via `src/patterns/dense_cache.py`) | L1 `test_retrieval.py` / `test_dense_cache.py` + L0 |
| `src/patterns/safe_tei_rerank.py` | `_safe_tei_rerank_call`, `SafeTEIReranker._call_api` | error verdicts must stay pure signals — a partial result masquerading as REAL is the RET-1 kill class, and the fallback decision is a single `except` whose branches are pinned by the generated status/body matrix | L1 `test_retrieval.py` HTTP-error tests + L4 `tei_fallback.fizz` (FG-23/24) |

> **`src/pipeline.py` is IN the mutmut scope** (54 functions / 2695 mutants — the
> largest mutated module; see the module ledger). The four `src/pipeline.py`
> rows the table used to carry claimed an exclusion that does not exist in
> `do_not_mutate`; they were removed rather than restated. FizzBee
> (`design_loop.fizz` DL-2..5, `pipeline_control.fizz` P-1/FP-2..7) and L4
> `jobs_runner.fizz` (RUN-1/RUN-3/C-1) *cross-cover* the loop's mutants —
> a layer that reduces survivor noise, not one that removes the surface.

## Permanent (excluded by design)

> Excluded for structural reasons (wiring / declarative / prompt text), not
> pending triage. A module leaves this section only by growing real decision
> density AND a dedicated kill oracle — same-PR ledger row required
> (AGENTS.md L3 rule). Cross-ref `pyproject.toml [tool.mutmut].do_not_mutate`
> for the canonical list.

| Module | Decision-density zones | Why permanently excluded | Coverage authority |
|---|---|---|---|
| `src/main.py` | `cli()` health-check branch; option dispatch | thin click CLI wrapping `server_main` — mutation surface is dispatcher noise | L0 (ruff/mypy) + server-lifecycle integration tests (outside the mutation selection) |
| `src/__main__.py` | — | one-line `python -m src` shim delegating to `src.main.cli`; mutmut generates 0 mutants for it | L0 + import smoke at startup |
| `src/server.py` | `_with_prompts_as_tools_annotations`; transport validation (`stdio`/`streamable-http`); job-task cancel gather; JSON log-formatter branch | FastMCP wiring + lifespan management; the decision zones are integration-tested but survivors would be framework noise (string literals, gather patterns, formatter construction) | L1 `test_server_logging.py` + L0 |
| `src/mcp_prompts/*` | `register_prompts` dispatch | user-invoked workflow templates; prompt text = string-mutant noise bombs; registration is a dispatch table | L0 + `test_protocol_smoke.py` (outside the mutation selection) |
| `src/patterns/nodes.py` | `tokenize`, `dense_query`, `bm25_query`, `build_nodes` | thin LlamaIndex node factories over the shared domain-slug set; the text-shaping decisions have direct pins in the retrieval tests | L1 `test_retrieval.py` + L0 |
| `src/prompts/*` | `get_style_guidance` lookup | internal LLM prompt content + few-shot examples; string-mutant noise bombs; schema drift is caught at import time (examples instantiate against the Pydantic models) | L0 import-time schema validation |
| `src/reasoning/schemas.py` | `ReasoningTrace.render_lines` formatting | declarative Pydantic schemas for the reasoning integration; `render_lines` is pure formatting observed via caplog asserts | L1 `test_reasoning_client.py` + L0 |
| `src/reasoning/__init__.py` | `__getattr__` PEP 562 lazy import | re-exports + one lazy import breaking the `client → src.agent → src.config` cycle; pure wiring | L0 + import smoke |
| `src/resources/*` | resource-handler JSON dumps + slug→variant lookup | FastMCP resource registration wiring; handlers are format-and-lookup | L1 `test_variants.py` (loader-backed payloads) + L0 |
| `src/schemas/*` | — | declarative Pydantic models + enums; unasserted defaults would flood equivalents (same rationale as `src/config.py`'s model bulk, without the boundary logic) | L0 (`mypy --strict` schema gate) + Pydantic validation at tool boundaries |
| `src/tools/__init__.py` | `create_all_tools` dispatch table | tool factory: maps tool keys to constructors; handler decisions live in the per-tool modules | L0 + `test_protocol_smoke.py` |
| `src/tools/_execution.py`, `src/tools/_heartbeat.py` | the deadline wrapper and the heartbeat loop | transport plumbing around the job runner; the deadline semantics are model-checked by L4 `pipeline_control.fizz` (DeadlineExpire) and the heartbeat is timing-only | L4 `pipeline_control.fizz` + L1 `test_jobs.py` |
| `src/tools/analyze.py`, `src/tools/design.py`, `src/tools/evaluate.py`, `src/tools/generate.py`, `src/tools/patterns.py`, `src/tools/get_design_status.py`, `src/tools/cancel_design.py`, `src/tools/submit_design_job.py` | the eight tool handlers | see the *Awaiting scope sign-off* table — these are the handler candidates deferred for survivor-noise reasons, not permanent | L1 `test_analyze.py` / `test_generate.py` / `test_evaluate.py` / `test_fallbacks.py` |

## Gate mechanics (what `make test-mutations` now enforces)

1. mutmut is pinned `==3.7.*` in lockstep with `verify/mutmut/mutmut_compat.py`
   (source-string guards fail loudly on an upstream change — bump pin + shim
   together).
2. `also_copy` carries the shim, the scripts package the `tests/verification`
   meta-tests import, the real pattern catalogue, the docs the loader
   validates against, the repo config, and the client examples + compose files
   the unit tests read by path: a missing copy aborts the run at stats
   collection (observed 2026-09-11 for `scripts/`, 2026-09-12 for `pattern/`,
   2026-09-26 for `examples/` — `tests/unit/test_client_examples.py` imports
   them by path and `tests/unit/test_config.py` pins the compose files).
3. `pytest_add_cli_args_test_selection` names the two test DIRECTORIES
   (`tests/unit`, `tests/verification`), not a hand-listed set of files: a new
   test file joins the kill authority automatically instead of silently
   dropping out. The single exclusion is
   `--ignore=tests/unit/test_protocol_smoke.py` — a RUNTIME decision, not a
   scope one: its 9 tests cost 137 s of the 179 s clean suite and were the
   "cover" for 133 of the 174 mutated functions, so every mutant paid that
   bill (sweep ~4.5 h with it, ~40 min without). The three functions it alone
   covered (`_freetext_validator`, `_pattern_name_validator`,
   `DesignPipeline.validate`) are pinned directly by
   `tests/unit/test_input_validation.py` and
   `tests/unit/test_pipeline_validate.py`. It stays in `make test-unit`.
4. Per-mutant budget: `timeout_constant = 5.0`, `timeout_multiplier = 6.0`
   (verified present in the 3.7.0 `Config` dataclass).
5. The recipe runs in one shell with an `EXIT` trap — the per-run Hypothesis
   database is cleaned up even on failure.
6. `scripts/mutmut_baseline.py --gate` distills the run and fails on: any new
   survivor key, any per-function kill-ratio drop beyond `ratio_tolerance`
   (0.02), any function that lost all its classified mutants, any module that
   lost ALL its data (one line per module, not per function), and any garden
   class without engine-side classified mutants (R7 bridge).
7. `do_not_mutate_patterns` is deliberately unset: logger lines became
   observational when `retriever.py` entered scope (caplog tests), so a
   global `logger\.` exclusion would delete live mutation targets
   (review finding P-02).
8. Scratch reset duty: a materially changed `[tool.mutmut]` (scope or
   selection) plus an interrupted run can leave `mutants/` with a stale
   config fingerprint and half-invalidated results — the documented reset is
   `rm -rf mutants/` (gitignored, fully regenerable; the committed baseline
   is unaffected). Done 2026-09-11 after the round-2 scope expansion,
   2026-09-12 after adding `pattern/` to `also_copy`, 2026-09-12 after
   the consecutive-run cache warning ("pyproject.toml changed … cached
   results were kept").
9. Dependency-change policy: `on_dependency_change = "rerun"` +
   `cache_invalidation_files = ["docs/design-patterns.md",
   "docs/pattern-schema.json", "pattern/*.json"]` (both keys verified
   present in the 3.7.0 `Config` dataclass). The default "warn" KEEPS cached
   mutant results when a watched non-Python file changes — for a manual
   nightly ratchet that means `scripts/mutmut_baseline.py --gate` could
   distill stale verdicts (observed 2026-09-12). "rerun" discards all
   results instead; the catalogue globs additionally re-test the loader
   mutants when the `pattern/` data the loader oracles read changes.

## mutmut 4 migration note (R11)

The compat shim guards two upstream source strings verbatim
(`_TRAMPOLINE_GUARD`, `_NAMING_GUARD` in `verify/mutmut/mutmut_compat.py`) and
the Makefile pins the version. A mutmut 4 bump must, in one PR: update the
pin, re-derive both guarded strings from the new source, re-verify the
trampoline-hit module-name convention (the whole reason the shim exists is the
`src`-package layout), re-check `Config` field names for the timeout keys, run
one full `make test-mutations`, and `make regen-mutmut-baseline` (indices and
keys will shift wholesale).

## Survivor triage duty

Every NEW-SURVIVOR gate failure is either (a) a test-suite gap — write the
missing kill, or (b) a true equivalent — record it in
`verify/mutmut-equivalents.md` with its mutant key and a reason, then
`make regen-mutmut-baseline`. The baseline only ever moves deliberately.

Current per-module ratios are read from the committed
`verify/mutmut-baseline.json` after each regen. That baseline (regenerated at commit
`923eec5`, per `run.git_commit` in the JSON) records
6328 mutants / 3862 killed / 2440 survived — overall kill ratio 0.610. The weakest
ratios in scope are `src/validation.py` (0.54), `src/design_normalization.py`
(0.57), `src/pipeline.py` (0.58) and `src/patterns/dense_cache.py` (0.59).
Cluster-by-cluster dispositions live in `verify/mutmut-equivalents.md`.

**Post-baseline edits not yet ratcheted (2026-10-25):** `src/patterns/dense_cache.py`
gained the best-effort `_write_back` and the four cache-path log lines. Statements were
inserted inside mutated functions, so per-function mutant indices moved; the next
`make test-mutations` will surface pre-existing survivors under new keys and needs a
deliberate `make regen-mutmut-baseline` (the numbers above still describe the
`923eec5` snapshot, not this tree).
