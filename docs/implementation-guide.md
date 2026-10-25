# Design-Pattern Selection Process — Implementation Guide

**Scope of this document.** This guide specifies *only* how the design-pattern MCP server selects design patterns when it receives a description from an AI coding agent: the input contract, the stage-by-stage selection algorithm (deterministic vs LLM steps), the retrieval and scoring formulas, ranking and tie-breaking, selection thresholds, supporting-pattern rules, interaction with the architecture-level context, error and ambiguity handling, and a worked example. The design of the surrounding server (tools, transports, job store, packaging) is out of scope.

**Position in the agent workflow.** The AI coding agent first calls the *architecture-pattern* MCP server, which returns an `ArchitectureDesign` result (components, relationships, API/event contracts, quality attributes — see `../architecture-pattern-mcp/docs/result-schema.json`). The agent then derives a **description** from that result and sends it to *this* server. This server answers one question: *which design patterns (from `docs/design-patterns.md`, as `DesignPattern` records per `docs/pattern-schema.json`) should be applied to implement the described parts, and how do they compose into a `DesignResult` (`docs/result-schema.json`)?*

The selection pipeline deliberately mirrors the architecture server's proven structure — ANALYZE (hybrid retrieval + requirements-weighted scoring) → GENERATE → EVALUATE → bounded REFINE loop with best-attempt retention — but operates over design-pattern data: nine decision categories instead of architecture styles, per-decision-part retrieval instead of domain-slug retrieval, and eight design-level quality axes instead of six architecture-level ones.

---

## 1. Input Contract

### 1.1 What the agent sends

The primary tool accepts a **description** (free text, 1–100 000 printable characters; control/format characters rejected at validation) plus optional structured hints. The description is expected to be derived from the architecture server's `ArchitectureDesign` result. The following table defines which parts of that result the agent is expected to carry across, and what this server does with them:

| Architecture result field (`ArchitectureDesign`) | Expected content in the description / hints | Consumed by stage |
|---|---|---|
| `overview.style` | Named verbatim (e.g. "event-driven") or passed as the optional `architecture_style` hint (same enum) | Style→category affinity prior (§11.1) |
| `overview.principles`, `overview.constraints` | Quoted constraints ("at-least-once delivery", "p99 < 50 ms") | Part constraints, weight-extraction evidence (§6, §8.2) |
| `components[]` (`id`, `name`, `type`, `description`, `responsibilities`, `technology_stack`) | Each implementable component named with its responsibilities | Decomposition into decision parts (§6) |
| `relationships[]` (`source`, `target`, `type`, `description`) | Cross-component flows (sync / async / event-stream / data-flow) | Part-dependency detection; `DesignResult.interactions[]` seeding |
| `api_contracts[]`, `shared_data_models[]`, `event_contracts[]` | Boundary and contract obligations per component | Boundary-part detection (category `interaction-boundary`, `workflow-messaging`) |
| `quality_attributes` (numeric map — ArchitectureDesign types it as number values with no range; the 1–10 scale is the pattern schema's own convention) | Numeric priorities if the agent restates them | Weight prior when the description itself is silent on priorities (§8.2, blend factor 0.3) |
| `components[].technology_stack` (products, e.g. "FastAPI", "Kafka", "TypeScript") | Restated or passed as the optional `language` hint (single value: the one target language); standard/toolchain tokens ("C++20", "ES2017") also feed variant selection | Language binding (§8.4): per-language realization, or language-neutral guidance when the hint is absent/unrecognized; variant choice by the §8.4 precedence |

### 1.2 Optional structured hints

- `architecture_style` — one value of the architecture server's style enum (35 values, e.g. `event-driven`, `hexagonal`, `microservices`). Optional. **Precedence:** when the caller supplies a recognized value it *overrides* any style the Stage A brief extracts from the description text (§4); when absent (or unrecognized — §1.3 rule 3), style affinity follows the brief's extraction, and with no extraction there is no prior.
- `language` — at most one value of the `language` enum from `docs/pattern-schema.json` (`generic`, `c`, `cplusplus`, `csharp`, `go`, `java`, `javascript`, `kotlin`, `python`, `ruby`, `rust`, `scala`, `swift`, `typescript`). The agent submits one target language; the design patterns are returned realized in that language. When the value is absent, empty, unrecognized, or an explicit `generic`, the run proceeds in **generic mode** and the patterns are returned as language-neutral, abstract guidance (§1.3, §8.4).
- `paradigm` — at most one value of the `paradigm` enum from `docs/pattern-schema.json` (`oo`, `functional`, `concurrent`, `data-oriented`, `declarative`, `procedural`). Optional; narrows variant selection (§8.4) when a language entry carries multiple realizations and the description itself is paradigm-silent. Small closed enum by design: variant selection is expressed through *capability and style signals* (this hint, standard tokens in the description, free text in `focus`), NEVER through a catalog-wide `variant_id` parameter — variant ids are per-(pattern, language), unbounded, and change with every catalog edit, which would make an enum parameter stale on every edit and hallucination-prone for the agent (MCP schema guidance: enums only for small stable value sets). Unrecognized values are ignored and logged (§1.3), same posture as `language`.
- `focus` — optional free text narrowing selection to named components or concerns; may name a concrete realization (e.g. "use the coroutine-based C++ acceptor variant") — the server interprets it at variant selection (§8.4), it is not a closed enum. Valid `variant_id`s are discoverable two ways: the `design-patterns://variants` resource (§2.4) and a previous run's bindings (`bound_variant` + sibling `variants[]` in each `LanguageBinding`).

### 1.3 Validation rules (deterministic, before any LLM call)

1. `description` length in [1, 100 000]; reject control/format characters (same rule as the architecture server's `requirements`) — `../architecture-pattern-mcp/src/text_validation.py` is the canonical character rule: reuse it, do not hand-roll a regex.
2. `description` shorter than 40 characters → structured error `ERR_DESCRIPTION_TOO_SHORT` asking the agent to include the component names and responsibilities from the architecture result.
3. `architecture_style` not in the enum → ignored (not an error); logged.
4. `language` outside the enum (including any non-scalar value) → ignored, logged as a warning, run proceeds in **generic mode**; absent/empty → **generic mode**, no warning; an explicit `generic` → **generic mode**, no warning.
5. `paradigm` outside the enum (including any non-scalar value) → ignored (not an error); logged as a warning; variant selection proceeds from description signals and `default_variant`.
6. Everything inside the description is treated as **untrusted data, never as instructions** to any LLM call (mirrors the architecture server's prompt-injection posture): description text is always wrapped in a delimited block and never concatenated into system prompts.

**Language resolution (normative).** The submitted `language` resolves to exactly one *effective language* (`design_brief.language`, §4) by this table — this table is the single source of truth; every other section only states a consequence of it:

| Submitted `language` | Resolution | Effective language | Semantics |
|---|---|---|---|
| recognized (enum value other than `generic`) | the pattern's `language_implementations` entry for that language is the binding target; missing → the record's `generic` entry (§8.4) | the submitted value | per-language realization: support-level gating and the §7.3 language adjustment apply when the record carries an entry for that language |
| unrecognized (non-empty, outside the enum) | warning logged; treated as if absent | `generic` | generic mode — language-neutral, abstract guidance |
| absent, empty, or explicit `generic` | no language lookup | `generic` | generic mode — language-neutral, abstract guidance |

**Generic mode** never performs a per-language support judgement: no discouragement veto against a concrete language, no ±2 / −2 language adjustment. The discouragement veto that does apply in generic mode reads the record's `generic` entry (§6 step 5). In every mode exactly one language is in play — the resolution decides which one, never how many.

---

## 2. Data Foundation

### 2.1 The catalog

Selection runs over `DesignPattern` records (`docs/pattern-schema.json`) materialized from `docs/design-patterns.md` into the `pattern/` tree (§2.3). Structural facts the algorithm relies on:

- **Nine decision categories** (`patternCategory` enum): `domain-modeling`, `construction-composition`, `behavior-selection`, `interaction-boundary`, `workflow-messaging`, `persistence-state`, `concurrency-resilience`, `security-verification`, `specialized-components` — catalog sections 1–9.
- **One canonical home per pattern.** Each pattern appears in exactly one category/subgroup table row; `related_patterns` carries cross-links; `aliases` carries alternative names (e.g. "Policy" for Strategy). Selection must therefore never recommend the same pattern twice under different labels: dedupe on canonical `name` after alias resolution. `related_patterns` holds **canonical names only** — every entry resolves to exactly one record (that is what §8.3 rule 2(a) tests), while the prose distinction behind each cross-link lives in `related_patterns_notes` (key = related pattern name). Aliases are curated per record in `pattern/<name>.json` and indexed by `src/patterns/loader.py::_build_alias_index` (there is no generated alias-map document); every alias must resolve to exactly one parsed record. An alias that is itself a distinct catalog pattern name is ambiguous and is dropped from alias resolution (worked case: the schema's Strategy alias "Policy" collides with the catalog's own "Policy-based design", docs/design-patterns.md:231). If no alias data exists for a pattern, only its canonical name matches.
- **Facets** — `scope` / `suitable_scopes` / `unsuitable_scopes` (seven scope values: `value`, `object`, `module`, `boundary`, `message-flow`, `persistence-unit`, `task-lifetime`), `paradigm`, `provenance`, `language_implementations[]` (joined record) — are filters over the canonical home, never a second selection path.
- **Generic entry invariant.** Every `DesignPattern` record carries exactly one `language_implementations` entry per curated language and exactly one `generic` entry; the catalog materialiser rejects records that violate this (consistent with the schema's "`generic` entries cover the rest"). The `generic` entry is therefore always available as the binding target when no concrete-language entry applies (§8.4). In layout V8 the main record carries no `language_implementations` at all — the generic entry lives in `pattern/<name>.generic.json`, the concrete entries in their language files (§2.3).
- **`quality_attributes`** — numeric 1–10 scores; the fixed axes used for scoring are `testability`, `extensibility`, `maintainability`, `simplicity`, `performance`, `reliability`, `security`, `scalability` (eight design-level axes, all schema-required; four extended axes are additionally allowed — `observability`, `coupling`, `discoverability`, `type_safety` — and, like any axis beyond the eight, are ignored by the formula; unknown axis names are schema-rejected).
- **`avoid_when`** and `language_implementations[].support` (with REQUIRED `alternative` when `discouraged`) are negative gates (§8.4).

### 2.2 Retrieval index (built once at startup, idempotent warm-up)

- **Corpus unit:** one document per `DesignPattern` record (direct pattern-level indexing — the catalog itself is the decision vocabulary; no intermediate slug layer).
- **Document text:** `name` + `aliases` + `intent` + `problem` + `context` + `use_cases` (joined with separator tokens). `avoid_when` is excluded from the positive text and used only as a veto signal.
- **Document metadata:** `name`, `category`, `subgroup` (the catalog subsection title the row lives under), `scope`, `paradigm`, `provenance`, per-language `support` map (from the joined record, §2.3).
- **Two legs over the same node set:** a dense leg (embedder over document texts) and a BM25 leg (tokenized `name`+`aliases`+`problem`). Legs are built once at server startup; a failed warm-up refuses server start (fail-fast on embedder misconfiguration, retry-bounded by `embedder.document_embed_attempts`), mirroring the architecture server's `warmup_indexes` behavior. Warmup is bound to the lifecycle: it runs once in the FastMCP lifespan startup, before the first request is accepted; a failed warmup raises at startup.
- **Precomputed dense vectors (optional).** The dense leg's embeddings may be shipped as `data/dense-cache/{manifest.json,vectors.npy}`, baked into the image at `/app/data/dense-cache`. The cache is generated by `make docker-build` / `make docker-build-all` whenever it is absent (it is a build input of the image, so a freshly seeded cache volume starts warm — an *existing* volume keeps its own snapshot until it is reset, so a catalogue/model-changing rebuild owes one re-embed unless the volume is deleted first); `make embed-cache` forces regeneration, and `make docker-build-nocache` builds without it. A cache entry is accepted only when the metadata fingerprint — record count, canonical name order, `corpus_sha256` over the joined `document_text`s, configured model string, `text_instruction_sha256`, and (when the runtime declares an identity) `model_repo` / `onnx_variant` — AND a semantic probe — a stored probe vector must match a fresh embedding of the same probe text at cosine ≥ `dense_cache.probe_min_cosine` — both agree. Any mismatch re-embeds and rewrites the cache in `mode: auto`, refuses startup in `mode: require`, and is ignored in `mode: off`. The rewrite happens in the same call (`dense_cache.write`), so the next start reloads the vectors with zero corpus embed calls; both compose stacks mount a named volume over the path, so the snapshot survives container recreation and image upgrades, and a fresh volume is seeded from the image's baked copy. Write-back is best-effort: a failing store or probe logs `Dense cache write failed` and the run keeps the vectors it just embedded. The `--health` path (offline embedder) never reads the cache.

### 2.3 Language-file storage layout (V8) — normative

A `DesignPattern` is stored as a **split record**: the language-neutral core in `pattern/<name>.json` plus every realization — the generic one in `pattern/<name>.generic.json` and each concrete-language one in `pattern/<name>.<language>.json` — as colocated sibling files validated by their own schema (`docs/pattern-language-schema.json`). The main record declares the association in a `language_files` map with exactly 14 keys — declared in-record map, colocated files (layout **V8**). Everything downstream in this guide consumes the **joined** record (rule 4); the split is a storage- and materialization-level concern only.

1. **Main record.** `pattern/<name>.json`, validated by `docs/pattern-schema.json`. Carries the core and the `language_files` map — and **no** `language_implementations` field (the schema rejects unknown keys; rule 5 restates it): keys are exactly the 14 `language` enum values (`generic` + the 13 concrete), values are file names relative to the record's own directory; both constraints are schema-enforced. The pre-V8 inline generic entry now lives in `<name>.generic.json`.
2. **Language files.** One per enum language — generic and concrete alike: `pattern/<name>.<language>.json`, validated by `docs/pattern-language-schema.json`. Independent documents with their own `version` and identity fields (`pattern`, `language`); `support` (native/idiomatic/library/manual/discouraged), `implementation_notes`, optional `idiom_name`, `code_sketch`, `pitfalls`, `sources`; `discouraged` REQUIRES `alternative` (same negative gate as §2.1). `generic` is now itself a language-file language: exactly one generic file per record (`<name>.generic.json`), joining as the first implementation entry.
   A language MAY admit **multiple realizations (variants)** — same pattern, same language, different paradigm (pure-OO vs pure-functional vs concurrent style), different library/idiom (ACE vs Boost.Asio), different language-standard floor (`min_standard`: C++11/C++17/C++20, Java 8, ES2017…), different required features (`requires_features`: coroutines, concepts…). When they exist, the file carries a `variants` array (each variant: required `variant_id`/`name`/`paradigm`/`implementation_notes`, optional `min_standard`/`requires_features`/`idiom_name`/`code_sketch`/`pitfalls`/`sources`) plus a `default_variant` slug; the file's flat fields MUST equal the default variant's fields, so the flat form remains a complete summary of one realization. `variants` absent = exactly one implicit realization (the flat fields) — every pre-variants file stays valid unchanged. How other catalogs solve this: iluwatar ships named sibling implementations inside one pattern module (5+ Singleton variants); faif/python-patterns uses one file per variant; Refactoring.Guru keeps one example per language and loses the distinction. The layout keeps ONE file per language (the map key and the one-entry-per-language join invariant both demand it) and nests variants inside it.
3. **Relation.** The declared map is the join: read the main JSON, follow `language_files[lang]`. Three-way agreement is mandatory — map value == file name == `<name>.<language>.json`; the map key == the file's `language` field; the file's `pattern` field == the record's `name`.
4. **Join (materializer).** Load and validate the main record; load and validate the declared generic file first, strip its file-level fields `pattern`/`version`, and make it the first entry; then for each concrete language **in enum order**, load the declared file, validate it, and append its entry, stripping the file-level fields `pattern`/`version` and the record-level `language_files` map. `variants` and `default_variant` ride along inside the joined entry (present iff the language file declares variants). The joined record is structurally identical to an inline record with all 14 entries; the joined `language_implementations` array (generic first, then concrete in enum order) is an assembled in-memory shape — stored records never carry it. It is the only form the selection pipeline (§2.2 warm-up, stages A–H) ever sees.
5. **Integrity check (build/CI gate, fails loud).** For every main record: `language_implementations` is ABSENT from the main record; `language_files` has exactly the 14 keys (`generic` + the 13 concrete); every value resolves to an existing file that validates and agrees with the map (rule 3); no orphan `pattern/<stem>.<suffix>.json` without a map entry or a matching main record; when `variants` is present: `default_variant` resolves to one `variant_id`, `variant_id`s are unique, and every flat field the variants model shares with the default variant (`idiom_name`/`implementation_notes`/`code_sketch`/`pitfalls`/`sources`) equals it. Extension workflow: a new language extends the enum in `docs/pattern-schema.json` (§15 single source of truth), every record's map, and its files; a new pattern adds main record + map + files; the check fails until map, files, and convention all agree.
6. **Colocation disambiguation.** Kebab-case pattern names are dot-free, so in `pattern/` a main record has a dot-free stem while a language file's stem carries the `.language` suffix (`acceptor-connector.python`); the generic file's `.generic` suffix behaves the same way for this disambiguation. Consumers enumerating catalog records filter `pattern/*.json` accordingly; language files are reached through the map, never by glob.

### 2.4 Variant discovery resource — normative

The server exposes **one read-only MCP resource** for variant discovery: `design-patterns://variants`. It is a resource, not a tool: read-only catalog view, cacheable by clients, no tool-call permission, no side effects — and no required step in the agent workflow (progressive disclosure, not a mandatory pre-call).

- **Materialization.** Built at server startup from the joined records (§2.3 rule 4), AFTER the §2.3 integrity check passes — the resource can never expose a catalog state that failed validation; its payload changes only with the catalog and the server process, so client caches cannot go stale within a run.
- **Payload.** Grouped by language (variants differ between languages — the resource is the per-language truth): `language → pattern → {variant_id, name, paradigm, min_standard, requires_features, default (bool), one-line distinguishing note}`. Full per-variant `implementation_notes`/`code_sketch` stay in the catalog; the resource carries the choice surface only.
- **Resource template.** `design-patterns://variants/{pattern}` returns the per-pattern slice across languages for targeted reads; unknown pattern → structured error, never an empty 200.
- **Use.** Rare and optional: an agent (or a human user through the agent's prompt) stating a realization preference up front — "make it functional", "we're on C++03" — checks vocabulary before writing `focus`/`paradigm` hints. The common flow needs no resource read: the §8.4 ladder binds a variant and the result's own bindings expose the siblings (`bound_variant`, `variants[]`).
- **Deliberately NOT a pre-call enumeration step.** The client cannot know which pattern this server will select before the description is decomposed; forcing variant awareness before selection would invert the problem-first contract (pattern follows problem, realization follows toolchain). Discovery is therefore post-selection by default (result bindings) and vocabulary-only up front (this resource).

---

## 3. Pipeline Overview

```text
 Description (+hints) ──► A. NORMALIZE & EXTRACT ──► B. DECOMPOSE ──► per part ──┐
                                                                            ▼
                                    C. RETRIEVE (hybrid + rerank) ──► D. SCORE (weights → blend → adjustments)
                                                                            ▼
                                    E. SELECT (primary + supporting, thresholds, tie-break)
                                                                            ▼
                                    F. COMPOSE (parts → DesignResult candidate set)
                                                                            ▼
                                    G. GENERATE ──► H. EVALUATE ──► score ≥ 50 ? ──no──► REFINE (≤2 attempts, may re-select)
                                                                            │ yes
                                                                            ▼
                                                                       DesignResult
```

Stages A–E and the numeric gates of H are the selection process proper; G exists because selection is validated by what it feeds. A cooperative cancellation flag is checked between every stage; a cancelled run stops at the next boundary.

### 3.1 Deterministic vs LLM steps

| Stage | Deterministic | LLM |
|---|---|---|
| A. Normalize & extract | validation, normalization, architecture-field recognition | one structured call: `DesignBrief` (style/constraints/components restated) |
| B. Decompose | part-count bounds, merge rule, scope/category enum validation | part splitting (same call as A) |
| C. Retrieve | dense + BM25 legs, fusion, cross-encoder rerank blend, caps | — (cross-encoder is a model, but a fixed scoring model, not a prompted LLM) |
| D. Score | weight smoothing, `analysis_score`, blend, all adjustments, ranking, tie-breaks | one small call for all parts: a part-id → eight quality-axis weights map (depends on Stage B's output — it cannot run per brief) |
| E. Select | thresholds, near-tie detection, supporting caps, fallbacks | — |
| F. Compose | dominant-part election, dedupe, application-set assembly | — |
| G. Generate | schema validation, enum enforcement, retry-on-malformed | one call per attempt: applications/participants/bindings |
| H. Evaluate & refine | 0–100 score comparison, best-attempt retention, early stop, reselection trigger | one evaluation call per attempt |

Only four kinds of prompted LLM calls exist in the pipeline (A+B combined, batched weights, generate, evaluate). Everything else is arithmetic over catalog data and is unit-testable without a model.

### 3.2 Differences from the architecture server

The pipeline mirrors the architecture server in mechanics, but the following differences are deliberate; each is listed so nobody reasons by analogy where the analogy breaks:

- **(a) Decomposition into decision parts** (Stages A/B) — no analogue: the architecture server scores one pattern set per domain.
- **(b) `fit_score` and its adjustment ledger** (§7.3) — no analogue: the sibling has `blended_score` and `analysis_score` only.
- **(c) Bounded reselection rule** (§11 item 4) — the sibling's `design_loop` reuses the same selection across attempts.
- **(d) Per-category fallback patterns** (§12) — the sibling has one global fallback (`layered-monolith`, src/patterns/retriever.py:83).
- **(e) Index unit: one document per `DesignPattern`** (§2.2) — the sibling indexes `ArchitectureDomain` slugs. Both servers cap stage-1 recall per leg (§14 `dense_top_k`/`bm25_top_k`, same 20/20 defaults), so the fused pool stays bounded even though this server's indexed documents are the full pattern text rather than slugs.
- **(f) Eight quality axes** (§2.1) — the sibling uses six.
- **(g) Name `top_k_candidates`** (§14) — the sibling's is `top_k_patterns` (same default 5).
- **(h) Missing-axis default 5.0** (§7.2) — the sibling scores a missing attribute as 0.0; here an absent axis is unmeasured, not worst-case.
- **(i) `max_tries` 2** (§14) — aligned with the sibling's config default (src/config.py:219).

---

## 4. Stage A — Normalize & Extract (`DesignBrief`)

**Inputs:** validated `description`, optional hints.
**Outputs:** a `DesignBrief`: `restated_goal` (≤2 sentences), `architecture_style` (enum or null), `constraints[]` (verbatim quotes), `components[]` (name, type, responsibilities, technologies), `flows[]` (source→target, kind), `negative_signals[]` (explicit "must not / no X" statements), `language` (the **effective language** per §1.3 — the submitted enum value in recognized mode, `generic` in generic mode; always exactly one value). `paradigm` (the validated `paradigm` hint or null — §1.2; feeds variant selection at §8.4).

**Decision rules:**
1. One structured-output LLM call produces brief **and** parts (Stage B) together; the response must validate against the brief+parts schema or the call retries once, then the server falls back to a minimal brief with a single part covering the whole description (scope `module`, category hypothesis null). The fallback single part bypasses Stage B and the anti-scope guard entirely and feeds Stage C directly with scope `module` and category hypothesis null.
2. The system prompt embeds the category and scope enums derived from `docs/pattern-schema.json` (never hand-maintained lists — the architecture server had drift bugs this way) and instructs: *design-level decisions only; architectural styles, deployment topology and data-platform choices are out of scope and must not become parts*.
3. `negative_signals` are extracted verbatim (e.g. "no reflection", "must stay single-threaded", "avoid singletons") for the Stage C veto.
4. If the brief names architecture-level constructs only (e.g. only "use microservices"), the pipeline still guarantees ≥1 implementable part: the described component's own construction becomes a part in category `construction-composition`.
5. **Style precedence (deterministic).** When the caller supplied a recognized `architecture_style` hint (§1.2), `DesignBrief.architecture_style` is the hint — the brief's own extraction is discarded. Otherwise it is the style extracted from the description text, or null. The resolution is pure fallback arithmetic (`hint or extracted`), never a merge.

---

## 5. Stage B — Decompose into Decision Parts

**Inputs:** `DesignBrief`.
**Outputs:** 1–`max_parts` (default 8) `DecisionPart`s, each: `id` (kebab-case), `statement` (one-sentence decision problem, self-contained), `component_refs[]` (brief component names), `scope` (one of the seven scope enum values), `category_hypothesis` (one of the nine categories, or null), `constraints[]`, `priority` (0–1, exactly one part may be 1.0), `negative_signals[]` (inherited from brief when component-scoped).

**Decision rules:**
1. **One part = one primary-pattern decision.** A part is created for each distinct decision surface named in the brief (a component responsibility cluster, a boundary obligation, a workflow obligation, a runtime constraint) — not per component, not per sentence.
2. **Bounds.** Fewer than 1 or more than `max_parts` parts → deterministic repair: the `max_parts` lowest-priority parts sharing both `category_hypothesis` and `scope` are merged (statements concatenated with "; "), others kept. Contract-driven parts (§10.2) are exempt from the merge repair; if the count still exceeds `max_parts`, emit structured error `ERR_TOO_MANY_BOUNDARY_PARTS` listing the obligations — never silently collapse them.
3. **Scope validation.** `scope` not in the enum → falls back to `module`. `category_hypothesis` not in the enum → null.
4. **Priority.** Exactly one part is normalized to priority 1.0 (the decision the agent most needs answered); the LLM proposes, the deterministic check normalizes. If no part is at 1.0, the highest raw priority becomes 1.0; if several tie at 1.0, the first in pipeline order keeps 1.0 and the others are demoted to the next-highest distinct raw value.
5. **Anti-scope guard.** Parts whose statement only restates an architecture-level choice (naming something from the catalog's out-of-scope list: layered, hexagonal, clean, onion, microkernel, microservices versus monolith, API Gateway, service discovery, CQRS, Event Sourcing, Saga, Transactional Outbox, Strangler Fig, sharding, replication, change-data-capture, data engineering, autoscaling, fleet-level resiliency) are dropped — *unless* the catalog explicitly hosts a locally implementable part of it (Anti-Corruption Layer, Presentation Model, Poison Pill, Backpressure, Version Number), in which case the part is kept with that category hypothesis. The out-of-scope set is derived from the parsed Background section of `docs/design-patterns.md` ("Out of scope by design"), not a hand-maintained literal; likewise "catalogued locally" is decided by whether a parsed `DesignPattern` record exists, never by the prose example list.

---

## 6. Stage C — Hybrid Retrieval (per part)

This stage mirrors the architecture server's retrieval exactly in mechanics, with the corpus and queries swapped for design-pattern data.

**Inputs:** one `DecisionPart`; the prebuilt retrieval index.
**Outputs:** a ranked candidate list of `(DesignPattern, retrieval_score)` pairs, ≤ `top_k_candidates` (default 5) kept *after* scoring (Stage D), plus the raw pool for normalization.

**Algorithm:**

1. **Query construction.** Dense-leg query: `statement` + `constraints` + scope phrase (e.g. "task-lifetime"). BM25-leg query: the same text normalized (lowercase, hyphens→spaces) — different representations per leg is intentional (mirrors the architecture retriever).
2. **Two-leg retrieval.** Dense leg (embedder) and BM25 leg over the shared pattern-document node set; per-leg top-k = `dense_top_k` 20 / `bm25_top_k` 20 (§14), clamped to the corpus size, `0` = full corpus (lossless). The fused union is therefore at most `dense_top_k + bm25_top_k` names, which bounds the cross-encoder pool of step 4; the caps must satisfy `dense_top_k + bm25_top_k` ≤ reranker `max_batch_size` (48) so one part never spans more than one cross-encoder request.
3. **Fusion.** Relative-score fusion with leg weights `dense_weight` 0.7 / `bm25_weight` 0.3 (must sum to 1.0 ±1e-3; validated).
4. **Cross-encoder rerank.** Mandatory when the fused pool has >1 entry: score `(query = part statement, document = pattern document text)`; cap survivors at `rerank_top_n` 10 using the Vespa-style reciprocal-rank blend. `rerank_text` (default `full`) selects the document text: `capsule` = name + intent + problem + first two use cases — roughly a third of the characters and a measured 3.6–4.3x faster on the CPU sidecar (2026-09-25), but rankings are NOT equivalent (top-1 differs; top-5 overlap 2/5, top-10 7/10, top-20 18/20 on a 40-document live probe), so `capsule` is opt-in and requires re-validating selection quality before relying on it

 `RR(fused_rank) + RR(ce_rank)` with `RR(r) = 1/(60 + r − 1)` (r is 1-based; numerically identical to the architecture server's `reciprocal_rank_score`, src/patterns/retriever.py:118-126) — maximum attainable blend 2/60 ≈ 0.0333. The pool it scores is the step-2 fused union (≤ `dense_top_k + bm25_top_k`), never the whole catalog.

 The blend always runs (it protects consensus-backed patterns from cross-encoder outliers); it is order-equivalent to CE-only only when the fused and cross-encoder rankings agree on relative order — when they disagree, the blend favours the fused rank.
5. **Deterministic pre-filters (applied to the survivor pool, before scoring):**
 - **Scope veto:** drop patterns whose `unsuitable_scopes` contains `part.scope`.
 - **Negative-signal veto:** drop patterns whose canonical `name` or any resolved `alias` token-overlaps a `negative_signal` (e.g. signal "avoid singletons" vetoes `singleton` and alias "Borg"); the veto uses the canonical name plus resolved aliases only (§2.1 alias contract).
 - **Language discouragement veto:** drop patterns whose `language_implementations` entry for the effective language (`design_brief.language`) has `support = "discouraged"` — for a recognized language that is its entry, for `generic` it is the record's `generic` entry. A pattern with no entry for the effective language is *not* vetoed (absence is not discouragement). The dropped pattern's REQUIRED `alternative` string is recorded on the part's veto ledger for Stage G's `alternatives_considered`.
6. **Relevance floor.** If the survivor pool is empty **or** the best blend < `min_fusion_score` (default 0.0, validator caps it at 2/60) → category fallback (§10.3) and the part is flagged `is_fallback`.
7. **Alias dedupe.** Resolve aliases to canonical `name`; one entry per pattern. Dedupe uses the canonical name plus resolved aliases only (§2.1 alias contract); if no alias data exists for a pattern, only its canonical name matches.

---

## 7. Stage D — Requirements-Aware Scoring (per part)

**Inputs:** candidate pool with `retrieval_score` (the RR blend), the part, the effective language (recognized or generic mode), architecture style.
**Outputs:** every candidate annotated with `analysis_score`, `fusion_score_normalized`, `fit_score`, and an adjustment ledger.

### 7.1 Weight extraction (one small LLM call for all parts)

The call receives *only* every part's id, statement and constraints plus the brief constraints (no pattern data — keeps it cheap and unbiased) and returns a map from decision-part id to eight floats in [0, 1], one per design-level axis: `testability, extensibility, maintainability, simplicity, performance, reliability, security, scalability`. Every supplied part id must appear exactly once and no other key may; a missing or extra id is a validation error the repair retry sees. Batching the call does not couple the ratings — each part is rated from its own statement and constraints — and it replaces N provider round trips with one.

**Calibration anchors (embedded in the prompt, mirroring the architecture server):**

| Anchor | Meaning | Design-level evidence examples |
|---|---|---|
| 0.0 | explicitly excluded / anti-required | "no test harness", "must not add layers" |
| 0.1–0.2 | not mentioned (never all-zero) | — |
| 0.3–0.5 | passing/generic mention | "should be clean", "some tests" |
| 0.6–0.8 | clear priority with concrete target | "p99 < 50 ms on the enrich step", "every handler unit-testable" |
| 0.9–1.0 | non-negotiable | "zero duplicate writes", "PCI-DSS validation on every field" |

Hard rules: values in [0,1]; order of operations is propose → cap → max-normalize → smooth → score; reserve 1.0 for at most one or two axes per part, and if everything is critical, peak at 0.85–0.9 (src/pipeline.py:426-428) — then normalize so the maximum = 1.0 (ties may all reach 1.0); base every weight on a phrase actually present; numeric SLOs outrank vague adjectives; sparse/ambiguous parts get 0.1–0.2 on unmentioned axes.

**Deterministic post-processing (per part, after the batch):** parts whose vector came back all-zero are retried in one second batch containing exactly those parts; a vector that is still all-zero falls back to the unweighted mean (0.125 each, E4). Convex smoothing: `w'_k = α·w_k + (1−α)/8` with `weight_smoothing_alpha` α = 0.7. When the part carries no priority evidence *and* the agent forwarded architecture `quality_attributes` (numeric map), those values (max-normalized, so no range assumption is needed) are blended 0.7 extracted / 0.3 architectural prior before smoothing.

### 7.2 Deterministic scores

1. **`analysis_score`** (0–100): `10 × Σ_k w'_k · qa_k(pattern) / Σ_k w'_k` over the eight fixed axes, using the pattern's `quality_attributes`; a missing axis takes the neutral mid-scale default 5.0. This deliberately diverges from the architecture server, which scores a missing attribute as 0.0 (src/pipeline.py:1696); here an absent axis is *unmeasured*, and 0.0 would let record incompleteness masquerade as a quality judgement.
2. **`fusion_score_normalized`** (0–100): min–max normalization of `retrieval_score` across the part's candidate pool.
3. **`blended`**: `analysis_blend_weight · analysis_score + fusion_blend_weight · fusion_score_normalized` with defaults 0.7 / 0.3 (must sum to 1.0 ±1e-3; validated).

### 7.3 Alignment adjustments (deterministic, applied to `blended`, result clamped to [0, 100])

| Adjustment | Condition | Delta |
|---|---|---|
| Scope bonus | `part.scope ∈ pattern.suitable_scopes` | +3 |
| Category hypothesis match | `pattern.category == part.category_hypothesis` (non-null) | +5 |
| Category hypothesis mismatch | `pattern.category ≠ part.category_hypothesis` (non-null) | −10 |
| Style affinity match | `pattern.category` in the affinity set of the architecture style (§11.1) | +5 |
| Language support (recognized mode, and only when the record has an entry for the requested language) | `native` / `idiomatic` / `library` | +2 |
| | `manual` | −2 |
| | `discouraged` | already vetoed in Stage C — never reaches scoring |
| Language support (generic mode, or no entry for the requested language) | — | no adjustment |

Stacking is allowed (e.g. category match + style affinity + scope = +13); the clamp bounds the effect. Every applied delta is recorded in the adjustment ledger for the `DesignResult.overview.reasoning` trace and for tests.

---

## 8. Stage E — Ranking, Thresholds, Selection

**Inputs:** annotated candidates per part.
**Outputs:** per part — one primary pattern (or fallback/ambiguity outcome), 0–`max_supporting` supporting patterns, `alternatives_considered` seeds.

### 8.1 Ranking and tie-breaking (deterministic total order)

Sort candidates by: `fit_score` desc → `analysis_score` desc → `fusion_score_normalized` desc → catalog order (category rank 1–9, then subgroup order, then row order — the catalog file defines a stable canonical order) → `name` alphabetical. The chain guarantees a unique, reproducible ranking even when scores tie exactly.

### 8.2 Selection thresholds

| Threshold | Default | Rule |
|---|---|---|
| `primary_score_threshold` | 50 | the top candidate's `fit_score` must be ≥ 50 **and** its `analysis_score` ≥ 50 (dual gate) to become primary; otherwise category fallback (§10.3) with `is_fallback = true`. The dual gate **extends** the architecture server's single `style_score_threshold` (src/config.py:228-234, which gates on `analysis_score` only); the `fit_score` floor has no architecture-server analogue |
| `supporting_score_threshold` | 40 | a non-primary candidate is eligible as supporting only at `fit_score` ≥ 40 |
| `near_tie_margin` | 5 | every candidate within the margin of the leader's `fit_score` is flagged: the part is marked `ambiguous` and each such candidate MUST appear in the winning application's `alternatives_considered` with the delta; only the top two contend for primary |
| `max_supporting` | 3 | hard cap per part |
| `top_k_candidates` | 5 | candidates carried into Stage F per part (selection happens *after* scoring, never before — same ordering as the architecture server) |

### 8.3 Supporting-pattern addition rules

A candidate joins the primary's supporting set iff **all** hold:

1. `fit_score ≥ supporting_score_threshold` and it is not the primary (in this part or, after Stage F dedupe, in any part).
2. **Distinct-constraint rule:** it solves a constraint the primary demonstrably does not — evidenced either by (a) an edge from the primary's `related_patterns` plus a non-overlapping catalog subgroup, or (b) an explicit statement among the part's `constraints`. A candidate whose only justification restates the primary's problem is rejected ("supporting patterns only where they solve distinct constraints" — catalog usage rule 2).
3. **No known conflict** with the primary (conflict table maintained alongside the catalog; e.g. `singleton` vs `dependency-injection`, `service-locator` vs `dependency-injection`, `template-method` vs `strategy` for the same variation point, `mock` vs `stub` for the same test seam). Conflicting pairs may not both be selected for the same part.
4. **Same-subgroup limit:** two patterns from the same catalog subgroup may both be supporting only under rule 2(b) — an explicit, distinct, stated constraint each.
5. Not discouraged in the effective language (already vetoed upstream, §6 step 5); a pattern with no entry for that language passes this check.

Each supporting pattern's `role` string (required by `SupportingPattern` in the result schema) names the distinct constraint and the part: e.g. "bound hung geo-lookups (enrichment)".

### 8.4 Language handling

One language is in play per run: the effective language (`design_brief.language`, §1.3/§4) — the submitted language in recognized mode, `generic` in generic mode.

**Selection-time consequences.**
- *Discouragement veto:* applied in Stage C against the entry for the effective language (a concrete language's entry in recognized mode, the `generic` entry in generic mode). Vetoed patterns never reach scoring, and their REQUIRED `alternative` reaches `alternatives_considered` through the veto ledger (§6 step 5, §11 item 1).
- *Language adjustment:* the `+2 / −2` support adjustment (§7.3) applies in recognized mode only, and only when the record carries an entry for the requested language; a generic-mode lookup makes no support-level judgement.
- *Supporting eligibility:* §8.3 rule 5 reads the same entry.

**GENERATE-time binding (per application).** The bound entry is the pattern's `language_implementations` entry for the effective language; when the effective language is `generic`, or when a recognized language has no entry on the record, the bound entry is the record's `generic` entry (always present — §2.1 invariant). `LanguageBinding.language` states the bound entry's own `language` value (so a fallback binding reports `generic`), and `support`, `idiom_name`, `implementation_notes`, `code_sketch` and `pitfalls` come from that entry verbatim (`code_sketch` may be null; generic entries carry language-neutral guidance). When the bound entry carries `variants` (language file declared multiple realizations), GENERATE binds **exactly one** variant, chosen by this precedence (first match wins):

1. **Explicit naming** — `focus` quotes a `variant_id` verbatim (e.g. `ace-reactor`, "use the `ace-reactor` variant"). Exact `variant_id` match only; a paraphrase like "use the ACE variant" matches no id and falls through to the next layer.
2. **Paradigm hint** — the `paradigm` hint (§1.2) matches a variant's `paradigm`. Several variants sharing the paradigm → tie breaks at layer 3.
3. **Capability floor** — standard/toolchain tokens stated in the description (`components[].technology_stack`: "C++20", "Java 8", "ES2017"…) are matched against each variant's `min_standard`/`requires_features`; among the variants the tokens can satisfy, the highest floor wins (the toolchain actually runs it).
4. **Default** — `default_variant`.

The selected variant's fields replace the entry's flat fields in the `LanguageBinding` (equal to them by the §2.3 checker anyway — `default_variant`'s fields always are): the binding sets `bound_variant` to the chosen `variant_id` (machine-readable, quotable in a follow-up `focus` hint) and carries `variants[]`, a compact summary of every NON-selected sibling (`variant_id`, `name`, `paradigm`, `min_standard`, `requires_features` — no code sketches or notes), so the agent can judge floor/paradigm fit against the actual codebase and re-call with `focus` quoting a sibling's `variant_id` when the bound realization is wrong for the toolchain. Sibling exposure is what makes the layer-1 loop informed instead of blind: the agent sees the alternatives exactly when they become relevant — after selection, before code generation. Variant choice is a GENERATE-stage concern only: support, veto and adjustment (§7.3) stay per-language — `support` is a property of the language, not of a variant. There is deliberately **no `variant_id` tool parameter**: the client would have to know every catalog value (unbounded and churning with every edit) — capability/style signals plus this precedence keep the input schema small and stable, mirroring how CMake `CXX_STANDARD`/`cxx_std_17`, Cargo `edition` and tsconfig `target` let the user state a floor and let the tool resolve the implementation. Discovery of concrete ids is post-selection (the binding's own `bound_variant`/`variants[]`) or vocabulary-only via the §2.4 resource — never an input parameter. `LanguageBinding.alternative` is never populated: the Stage C veto guarantees no binding carries `support = "discouraged"`.

**Request echo.** `DesignResult.overview.languages` (array in `docs/result-schema.json`) carries exactly one element: the effective language.

**Deleted multi-language semantics.** Because exactly one language is requested, the former partial-discouragement rule, the worst-value support adjustment, and the best-support binding have no meaning and are gone; every question they answered is now answered by the single effective-language lookup above.

---

## 9. Stage F — Composition into the Selection Result

**Inputs:** per-part selections with scores and ledgers.
**Outputs:** the candidate set consumed by GENERATE, organized for `DesignResult` (`docs/result-schema.json`).

**Decision rules:**

1. **Dominant-part election (deterministic).** `dominance = part.priority × (fit_score(primary)/100)`. The part with the highest dominance defines `DesignOverview`: its primary becomes `overview.primary_pattern`, its category becomes `overview.category`. Ties break by `fit_score`, then `analysis_score`, then part order.
2. **`overview.supporting_patterns`** = union of all parts' supporting sets (deduped by canonical name, capped at `max_supporting` overall by per-part `fit_score` desc), each with its `role` string.
3. **Applications.** Every selected (part, pattern) pair — primaries of secondary parts included — becomes one `PatternApplication` seed: `id` = `{part-id}-{pattern-name}` (kebab-case, matches `^[a-z][a-z0-9_-]*$`), `pattern`, `category`, `name` (human-readable, from the part's component refs), `problem` (from the part statement + constraints).
4. **Dedupe.** The same pattern selected for two parts yields two applications (different problems) but appears once in `overview.supporting_patterns`. A pattern that is primary in one part and supporting in another keeps its primary application and the supporting reference resolves to that application's id.
5. **Interactions seed.** Brief `flows[]` between components touched by different applications pre-seed `DesignResult.interactions[]` (`source`/`target` = application ids) for Stage G to refine.
6. **Traceability.** `overview.reasoning` is assembled deterministically from the ledgers: part decomposition count, each part's `statement` (the sampled decomposition question, verbatim), per-part retrieval blend top-3, weights summary plus the complete eight-axis weight vector (§7.2 axis order), applied adjustment deltas, threshold decisions, fallback/ambiguity flags — so a caller can attribute run-to-run variation without the server log. No LLM narration is required for it.

---

## 10. Interaction with the Architecture-Level Context

### 10.1 Style → category affinity (soft prior, +5 via §7.3)

| Architecture style (`architecture_style` hint / brief) | Affinity categories |
|---|---|
| `event-driven`, `broker`, `enterprise-service-bus` | workflow-messaging, interaction-boundary |
| `event-sourcing` | persistence-state |
| `command-query-responsibility-segregation` | persistence-state, domain-modeling |
| `saga` | workflow-messaging |
| `microservices`, `service-oriented-architecture` | interaction-boundary, workflow-messaging, concurrency-resilience |
| `hexagonal` | domain-modeling, behavior-selection, interaction-boundary |
| `layered-monolith`, `modular-monolith`, `monolithic` | domain-modeling, construction-composition |
| `pipe-and-filter` | workflow-messaging, behavior-selection |
| `model-view-controller`, `presentation-abstraction-control` | specialized-components |
| `microkernel-plugin`, `reflection-architecture` | behavior-selection, construction-composition |
| `reactive-architecture`, `actor-based`, `half-sync-half-async`, `task-control-architecture` | concurrency-resilience, workflow-messaging |
| `serverless`, `edge-computing` | concurrency-resilience, interaction-boundary |
| `space-based` | concurrency-resilience, persistence-state |
| `data-mesh`, `lambda-architecture`, `kappa-architecture` | interaction-boundary, persistence-state |
| `api-gateway`, `backend-for-frontend`, `service-mesh` | interaction-boundary |
| `rule-based-system`, `aiml-centric` | behavior-selection, domain-modeling |
| `blockchain-based` | persistence-state, security-verification |
| `hybrid-cloud`, `multi-cloud` | interaction-boundary, security-verification |
| (any style not listed) | no bonus |

The table is a *prior*, never a filter: an `event-driven` description whose part is clearly about test isolation still selects from `security-verification`/`construction-composition` on merit.

### 10.2 Other context interactions

- **Components → parts:** each responsibility cluster the GENERATE stage must place code into becomes a part; participants' `maps_to` fields must name entities derivable from the brief's component names so the agent can locate them.
- **Relationships → interactions:** sync/async/event-stream edges become `DesignResult.interactions[]` between the applications that implement their endpoints; the selection stage only seeds them, GENERATE refines types (`notifies`, `delegates-to`, `invalidates`, …).
- **Contracts → boundary parts:** every `api_contracts[]`/`event_contracts[]` obligation on a named component forces at least one part with `category_hypothesis` in {`interaction-boundary`, `workflow-messaging`} and `scope` `boundary` or `message-flow`. Contract-driven parts are exempt from the §5 rule 2 merge repair; if the resulting part count exceeds `max_parts`, emit `ERR_TOO_MANY_BOUNDARY_PARTS` listing the obligations — never silently collapse them.
- **Quality attributes → weights:** used only as the 0.3-magnitude prior of §7.1 — architecture-level scores must not override design-level evidence in the description.
- **Exclusion discipline:** architecture-level names never surface as selected patterns (Stage B rule 5); the catalog's background section (strategic DDD, Client-Dispatcher-Server, Broker) is not selectable — those records carry no category and are excluded from the index.

---

## 11. Stage G/H — Validation of the Selection by Generation and Evaluation

Selection is only complete when the selected set survives generation and evaluation; the loop mirrors the architecture server's `design_loop`.

1. **GENERATE** produces the concrete `DesignResult`: applications' `solution`, `participants` (role → `maps_to` concrete entity), `interactions`, `implementation_steps`, per-application `LanguageBinding` (bound entry = the `language_implementations` entry for the effective language, else the record's `generic` entry; `language` = the bound entry's language; `support`, `idiom_name`, `implementation_notes`, `code_sketch` from that entry; `bound_variant` + sibling `variants[]` per §8.4 when the bound entry carries variants; no `alternative` — the Stage C veto guarantees no binding is `discouraged`), `test_notes` (from the pattern's test-relevant benefits and the part's `testability` weight), and `alternatives_considered` (runner-up per near-tie, plus every Stage C veto-ledger entry as "considered but discouraged in {entry language}; use {alternative}" — the slot holds the actual language of the entry that triggered the veto, `generic` included). Output must validate against `DesignResult`; enum drift (category, language, support level) and pattern names not resolving to catalog records trigger automatic retry.
2. **EVALUATE** scores the result 0–100 against the eight design axes *as weighted by the part weights* (a part weighted for simplicity must not be dinged for lacking extensibility), plus cross-application consistency: no duplicated decision ownership, no conflicting pair from §8.3 rule 3, every supporting pattern's `role` traceable to a stated constraint.
3. **REFINE loop:** up to `max_tries` 2 attempts; each retry embeds the previous evaluation's findings; **best-attempt retention** keeps the highest-scoring valid result (never the last); early stop when `overall ≥ min_quality_score` 50. Cancellation is checked between attempts.
4. **Reselection rule:** when evaluation findings show a *selection* defect (primary does not own the decision; a supporting pattern's constraint is unsolved; a conflicting pair slipped through), the retry re-enters **Stage E** with the offending candidate excluded and the next-ranked candidate promoted — a bounded, ledgered reselection, not a full pipeline rerun. At most one reselection per loop (not per attempt); exclusions are sticky — a candidate excluded by reselection may not be re-promoted within the same loop. `attempts > 1` and the reselection ledger surface in logs for the agent.
5. Precedence: best-attempt retention governs whenever any attempt validates; the terminal error applies only when every attempt fails validation — in that case the last validation error propagates as a structured tool error (never a silently degraded result).

---

## 12. Fallbacks, Errors, and Ambiguity Handling

| # | Condition | Behavior |
|---|---|---|
| E1 | description < 40 chars | `ERR_DESCRIPTION_TOO_SHORT` — no pipeline run |
| E2 | control/format characters | validation error at intake |
| E3 | brief/parts LLM call fails twice | minimal single-part brief (whole description, scope `module`, hypothesis null) — degraded but functional; bypasses Stage B and the anti-scope guard entirely and feeds Stage C directly |
| E4 | weight extraction all-zero in both batches | unweighted mean (0.125 per axis) for exactly the parts that stayed all-zero |
| E5 | retrieval pool empty or below `min_fusion_score` | category fallback pattern, `is_fallback = true`, fallback noted in `overview.reasoning` |
| E6 | top candidate fails the dual 50/50 gate | category fallback pattern for the part (same as E5 semantics per-part) |
| E7 | part in `specialized-components` reaches E5/E6 | **no universal fallback exists for this category** → structured error `ERR_NO_PRIMARY_PATTERN` listing the top-3 candidates with scores and asking the agent to name the concrete component type (UI, loop, parser) — never an invented recommendation |
| E8 | near-tie (≤ `near_tie_margin`) | selection proceeds with the deterministic tie-break, `ambiguous` flag set, runner-up forced into `alternatives_considered` |
| E9 | description names only architecture-level decisions | one synthetic `construction-composition` part for the named component (Stage A rule 4) |
| E10 | unknown `architecture_style` / `language` / `paradigm` | ignored/dropped, logged (warning for an unrecognized `language` or `paradigm`); style affinity simply absent; `language` → **generic mode**: the result states the patterns language-neutrally (§8.4); `paradigm` → variant selection proceeds from description signals and `default_variant` (§8.4) |
| E11 | cancellation requested | pipeline stops at next stage boundary; partial state discarded |
| E12 | GENERATE output invalid after all retries | last validation error propagates; no fallback result |

**Category fallback patterns** (the lowest-ceremony universally applicable pattern per category, chosen from each category's "Use when" rows):

| Category | Fallback primary |
|---|---|
| domain-modeling | `transaction-script` |
| construction-composition | `dependency-injection` |
| behavior-selection | `strategy` |
| interaction-boundary | `data-transfer-object` |
| workflow-messaging | `observer` |
| persistence-state | `repository` |
| concurrency-resilience | `timeout` |
| security-verification | `guard-clause` |
| specialized-components | — (E7 instead) |

A fallback primary never gains supporting patterns (no evidence base to justify them) and always carries `is_fallback` into the reasoning trace.

**Ambiguity is surfaced, never hidden:** near-ties (E8), fallbacks (E5/E6/E7), veto-ledger entries, and reselections are all visible in `overview.reasoning`, `alternatives_considered`, and logs — the agent can always see why a pattern won and what almost won.

---

## 13. Worked Example

**Input to this server** (description derived by the agent from an architecture-pattern-mcp result: style `event-driven`, components Kafka-ingest / JSON-parser / geo-enricher / InfluxDB-sink / S3-sink, quality attributes scalability 9.1, maintainability 8.2, relationship "geo-enricher → Redis (sync)", technologies TypeScript/Node):

> "Event-driven ETL (architecture style: event-driven). Component `geo-enricher` (TypeScript/Node worker) calls a Redis geo-lookup synchronously while enriching parsed sensor events; enrichment must stay within the stage's p99 latency budget, Redis transient failures and outages must not stall the pipeline, and duplicate deliveries from upstream retries must not produce duplicate sink writes. Also: the JSON parser must tolerate unknown fields from heterogeneous device firmware. Constraints: at-least-once delivery, p99 < 50 ms per event."

**Stage A/B** produce three parts:

| Part | Statement (abridged) | Scope | Category hypothesis | Priority |
|---|---|---|---|---|
| `enrichment-resilience` | bound and survive the synchronous Redis lookup under p99 | task-lifetime | concurrency-resilience | 1.0 |
| `sink-idempotency` | duplicate deliveries must not double-write sinks | message-flow | workflow-messaging | 0.8 |
| `parser-tolerance` | unknown fields must not break parsing | object | interaction-boundary | 0.7 |

**Stage C/D for `enrichment-resilience`.** Extracted (raw) weights: reliability 1.0, performance 0.8, simplicity 0.6, testability 0.5, maintainability 0.4, scalability 0.3, extensibility 0.2, security 0.2 → smoothed (α = 0.7, +0.0375 each): 0.7375, 0.5975, 0.4575, 0.3875, 0.3175, 0.2475, 0.1775, 0.1775 (Σ = 3.1). Candidates after rerank and vetoes (scope `task-lifetime` vetoes none of these; no negative signals):

| Pattern | analysis_score | retrieval blend | fusion_norm | blended | adjustments | fit |
|---|---|---|---|---|---|---|
| `timeout` | 76.4 | 0.0317 | 100.0 | 0.7·76.4 + 0.3·100 = 83.5 | scope +3, category +5, style(`event-driven`→concurrency-resilience not in affinity set) +0, lang `native` +2 | **93.5** |
| `retry-with-backoff` | 72.0 | 0.0300 | 89.8 | 77.3 | +3 +5 +2 | 87.3 |
| `circuit-breaker` | 72.0 | 0.0283 | 79.6 | 74.3 | +3 +5 +2 | 84.3 |
| `bulkhead` | 68.0 | 0.0260 | 65.9 | 67.4 | +3 +5 +2 | 77.4 |
| `guarded-suspension` | 61.0 | 0.0200 | 29.9 | 51.7 | +3 +5 +0 (style) −2 (`manual`) | 57.7 |

*(analysis_score example, `timeout`: Σ w'·qa = 23.70 over its eight axis scores; 10 × 23.70 / 3.1 = 76.4.)*

**Stage E.** Primary: `timeout` (fit 93.5 ≥ 50, analysis 76.4 ≥ 50 — dual gate passed). Near-tie check: 93.5 − 87.3 = 6.2 > 5 → not ambiguous. Supporting (each ≥ 40, each with a distinct stated constraint, no conflicts, ≤ 3): `retry-with-backoff` — role "absorb transient Redis failures without caller awareness (enrichment)"; `circuit-breaker` — role "stop calling Redis during an outage (enrichment)". `bulkhead` rejected: no stated capacity-isolation constraint for this part (distinct-constraint rule fails); `cache-aside` (fit below threshold after adjustments) becomes an `alternatives_considered` seed. Dominance 1.0 × 0.935 elects this part as dominant → `overview.primary_pattern = timeout`, `overview.category = concurrency-resilience`.

The other two parts select `idempotent-receiver` (primary, workflow-messaging) and `tolerant-reader` (primary, interaction-boundary) by the same arithmetic; they enter as `PatternApplication`s (`sink-idempotency-idempotent-receiver`, `parser-tolerance-tolerant-reader`) and — where they support the dominant decision's constraints — as `overview.supporting_patterns` entries. GENERATE then binds each application to the TypeScript entries of the selected patterns' `language_implementations` (e.g. `timeout` → idiom "`AbortSignal`/`Promise.race` timeout wrapper", support `native`), maps participants to concrete entities (`GeoEnrichmentClient`, `RedisGeoGateway`), seeds interactions from the architecture relationships, and EVALUATE closes the loop; a hypothetical evaluation finding "duplicate-write constraint unsolved by `timeout`'s supporting set" would trigger the Stage E reselection rule on the next attempt, promoting `retry-with-backoff`-adjacent candidates only for that constraint.

---

## 14. Configuration Parameters (defaults)

| Parameter | Default | Constraint / note |
|---|---|---|
| `dense_weight` / `bm25_weight` | 0.7 / 0.3 | sum to 1.0 ± 1e-3 (validated); < 0.05 on either leg logs a warning |
| `dense_cache.mode` / `dense_cache.path` / `dense_cache.write` | `auto` / `/app/data/dense-cache` / `true` | `auto`: miss re-embeds; `require`: miss raises `ERR_008`; `off`: cache never read or written; `write`: persist the re-embedded corpus for the next start (best-effort — a failed write never fails the start) |
| `dense_cache.probe_min_cosine` / `probe_timeout_seconds` | `0.999` / `120.0` | semantic probe gate; below threshold or unreachable inside the bound → cache miss + re-embed |
| `embedder.document_embed_attempts` / `document_embed_backoff_seconds` | `10` / `2.0` | bounded startup retry (exponential backoff capped at 30 s) on `embed_documents` only |
| `embedder.embed_batch_size` | `16` (compose dev stack: `8`) | must not exceed the sidecar's ORT cap (8 for this model) or the request outlives the client timeout |
| `rerank_top_n` | 10 | survivors after cross-encoder cut |
| `reranker.rerank_text` | `full` | document text fed to the cross-encoder (§6 step 4): `full` = §2.2 document text; `capsule` = name + intent + problem + first two use cases (~3x fewer characters, measured 3.6–4.3x faster on the CPU sidecar). Rankings are NOT equivalent (measured divergence, §6 step 4) — opt-in only |
| `reranker.timeout` / `reranker.max_batch_size` | `300.0` / `48` | per-batch HTTP bound must exceed one full batch (measured 51.3 s for 48 documents on the CPU sidecar) |
| `rrf_k` | 60 | RR blend constant; blend max 2/60 ≈ 0.0333 |
| `min_fusion_score` | 0.0 | ≤ 2/60 (validator + startup defense-in-depth check) |
| `top_k_candidates` | 5 | per part, applied after scoring |
| `dense_top_k` / `bm25_top_k` | 20 / 20 | per-leg stage-1 recall cap (§6 step 2); `0` = full corpus. Measured 2026-09-24 on the 438-document catalog: lossless recall fed 438 documents (10 cross-encoder chunks of ≤48, ~58 s each) into every decision part ≈ 9.8 min/part; 20/20 keeps the fused pool ≤ 40 = one chunk |
| `retrieval.max_concurrent_parts` | 1 | ceiling on parts whose fuse → rerank → blend tail runs concurrently in `retrieve_many` (the shared embed + FAISS leg is a single batched call and not bounded by this). Semantics-neutral: gather argument order preserves per-part result order. A/B-measured 2026-09-25 on the CPU sidecar with 8 full-text part queries (§14 latency notes): a full cap-1 `retrieve_many` pass takes 242.4 s (untimed warm-up pass 250.4 s) and the cap-2 leg fails outright — one full-text ≤40-doc request already saturates the sidecar's 16,384-token inference buffer, so a second concurrent request cannot get an inference permit and TEI answers HTTP 429 "Model is overloaded" (TEI forces `max_batch_requests=8`; the client deliberately has no retries), which fails the whole request. Default therefore 1; cap 2 is opt-in and only safe with `capsule` rerank text or a faster sidecar |
| `weight_smoothing_alpha` | 0.7 | convex smoothing of extracted weights |
| `analysis_blend_weight` / `fusion_blend_weight` | 0.7 / 0.3 | sum to 1.0 ± 1e-3 (validated) |
| `scope_bonus` / `category_match` / `category_mismatch` / `style_affinity` | +3 / +5 / −10 / +5 | fit clamped to [0, 100] after adjustments |
| `language_support_bonus` / `language_manual_penalty` | +2 / −2 | recognized mode only, and only when the record has an entry for the requested language; generic mode → no language adjustment; `discouraged` → Stage C veto |
| `primary_score_threshold` | 50 | dual gate: fit ≥ 50 **and** analysis ≥ 50 |
| `supporting_score_threshold` | 40 | per supporting candidate |
| `near_tie_margin` | 5 | ambiguity flag + forced `alternatives_considered` entry |
| `max_supporting` | 3 | per part |
| `max_parts` | 8 | decomposition bound with merge repair |
| `max_tries` | 2 | generate–evaluate attempts (best retained); architecture server config default (src/config.py:219) |
| `min_quality_score` | 50 | early-stop threshold, 0–100; 100 disables |
| `generator.config.timeout_seconds` | 120.0 | provider-boundary bound on every generator call *and* the intent-level deadline around it: a call that outlives it raises `ERR_009` and the documented E3/E4/E12 fallback applies. Transport retries are fixed at one initial request (`max_retries=1`); `validation.max_retries` governs the separate structured-output repair path |
| `generator.config.generation_batch_size` | 5 | applications per GENERATE provider call (§11 item 1). `seed.applications` are split into batches of this size, each carrying only the catalog slices its own applications reference; the concatenated `RawGeneration` is assembled against the full seed (batch order preserved). A batch failure propagates unchanged into the design-loop/E12 path (sibling in-flight batches are cancelled) |
| `generator.config.generation_max_concurrency` | 2 | ceiling on GENERATE batches in flight at once; results are concatenated in batch order, so the assembled `RawGeneration` is identical to the sequential loop. `1` = the pre-2026-09-25 sequential behavior |
| `generator.config.circuit_breaker_failure_threshold` | 3 | consecutive qualifying transport failures (timeout, connection error, HTTP 408/429/5xx, empty response) that open this agent's process-local circuit. Authentication/configuration 4xx never count |
| `generator.config.circuit_breaker_cooldown_seconds` | 60.0 | open-state rejection window before exactly one half-open probe is admitted; a probe success closes and resets the counter, a probe failure reopens |
| `tasks.deadline_seconds` | 1800.0 | complete execution budget of one tool call / one job run, independent of `generator.config.timeout_seconds`. On expiry the in-flight provider coroutine is cancelled and the run ends with `ERR_014` (`select_design_patterns`, `analyze`, `generate`, `evaluate`, `submit_design_job`). Implementation: the pipeline is a llama-index `Workflow` whose internal timeout is `deadline_seconds * 0.9`; the wrappers translate the resulting `WorkflowTimeoutError` into `TimeoutError`, which `run_with_deadline` labels `ERR_014`. Raise it explicitly for legitimately longer runs; never remove it |
| `reasoning.enabled` | `true` | master switch for the pre-LLM reasoning layer; `false` degrades every phase to the in-prompt thinking scaffold |
| `reasoning.per_phase.<phase>.enabled` | `true` (analyze, weights, generate, evaluate) | per-phase switch; a disabled phase contributes an empty reasoning context — no generator call, no reasoning subprocess, no cache entry. Overrides merge onto the shipped agenda/directive |
| `reasoning.per_phase.<phase>.pre_llm_thoughts` | 2 / 2 / 2 / 2 (analyze, weights, generate, evaluate) | reasoning steps before the phase's main LLM call (reduced 2026-09-25 from 4 / 3 / 3 / 3 — see the §14 latency notes) |
| `reasoning.per_phase.<phase>.tools` | `["shannon", "code"]` in every shipped phase | both scratchpad servers are enabled by default; the list alternates per step, so a phase with ≥ 2 thoughts invokes both. Startup health-checks exactly the tools the enabled phases use |
| `reasoning.max_total_steps` | 8 | ceiling on reasoning steps per phase run |
| `reasoning.max_injected_chars` | `null` (uncapped; `REASONING_MAX_INJECTED_CHARS`) | cap on the rendered reasoning trace injected into downstream prompts: when the render exceeds the cap it is cut and `(reasoning trace truncated at N chars)` appended; per-step thoughts inside the trace stay whole, only the downstream injection truncates. Cut input tokens on every downstream call (latency lever L6). The struct-`None` note: the wire config maps an unset env (`:-`) to `null` = uncapped |
| `validation.truncation_rerun` | `repair` (`VALIDATION_TRUNCATION_RERUN`) | what happens when a response is cut by the provider token ceiling (`finish_reason=length`) and does not parse (§4 rule 1a): `repair` (shipped default, behavior-neutral) = the truncated payload enters the normal repair retry; `boost` = truncation-first — NO repair re-call into the same ceiling, exactly ONE rerun with `truncation_boost_max_completion_tokens`, and a second truncation raises `LLMError(failure_kind="truncated")` (transport-class, never repair-prompted) |
| `validation.truncation_boost_max_completion_tokens` | `65536` (`VALIDATION_TRUNCATION_BOOST_TOKENS`) | completion budget of the boost rerun (MiniMax M2.x recommended cap); reaches the provider as litellm's `max_completion_tokens` |
| `prompts.stable_prefix_first` | `false` (`PROMPT_STABLE_PREFIX`) | L7 prompt order: reorders each operation's user prompt so the ≥512-char static response-contract block (identical across all calls of that operation) leads and the per-run payload (description, parts, result JSON, reasoning context) trails — maximizing MiniMax passive prompt-cache hits (≥512-token stable prefix). Content is identical in both modes, only ordering changes; prompt-cache adoption is judged by `usage.prompt_tokens_details.cached_tokens` when the provider exposes it |

All parameters are read at startup, validated by the same sum-to-one and range checks the architecture server uses, and every decision they gate is logged with its inputs (part id, candidate, raw scores, applied deltas) so any selection is reproducible from the logs.

Four bounds govern a run and must not be confused: the **provider-call timeout** (`generator.config.timeout_seconds`, 120 s) bounds one HTTP request to the model; the **structured-output repair budget** (`validation.max_retries`, 1) re-issues exactly one call and only for model-output failures — a provider timeout, provider exception, empty response or open circuit is re-raised immediately and never repair-prompted (and with `validation.truncation_rerun="boost"` a `finish_reason=length` truncation is handled by the truncation-first rerun instead of the repair loop); the **design-loop retry** (`max_tries`, 2) is the existing GENERATE→EVALUATE attempt boundary that owns E12; and the **whole-run deadline** (`tasks.deadline_seconds`, 1800 s) cancels whatever is still in flight, including the provider coroutine, and ends the run with `ERR_014`. Mechanically the whole-run deadline rides the pipeline's Workflow timeout (`deadline_seconds * 0.9`): expiry raises `WorkflowTimeoutError` inside the graph, each public wrapper re-raises it as `TimeoutError`, and the tool seam (`run_with_deadline`) maps that to `ERR_014`.

**Latency notes (2026-09-25).** Live A/B run on the CPU dev stack (8-part IoT/Kafka request, 777 chars): before (sequential part tails, sequential GENERATE batches, 4/3/3/3 reasoning steps) 1169 s; the part tails alone account for 315 s (8 × ~40 s) and the six sequential GENERATE batches for ~495 s. The cap-2 retrieval-tail leg fails outright against this sidecar (HTTP 429, one full-text request saturates the 16,384-token inference buffer — see the `retrieval.max_concurrent_parts` row), so the shipped default is 1; a full cap-1 `retrieve_many` pass measures 242.4 s (warm-up 250.4 s). `retrieval.max_concurrent_parts` 2 and `reranker.rerank_text: capsule` remain opt-in; `capsule` measured 3.6–4.3x faster on the sidecar but rankings diverge (see §6 step 4).

---

## 15. Determinism, Concurrency, and Testing Notes

- **Deterministic core.** Everything except the four LLM call kinds (§3.1) is pure arithmetic over catalog data: identical inputs, parameters and index → identical ranking, tie-breaks, thresholds, fallbacks. Retrieval is CPU + embedder HTTP and therefore runs off the event loop in a worker thread (mirrors the architecture server's `asyncio.to_thread` usage) so concurrent MCP requests do not stall.
- **Cancellation.** A cooperative token is checked between stages and between attempts; effects stop at the next boundary.
- **Workflow realization (2026-10-01).** `DesignPipeline` is a llama-index `Workflow` subclass: the five entry points start one event graph whose steps (`route`, `analyze_step`, `generate_step`, `evaluate_step`, `refine_step`) realize the §3 stage order, and the E12 loop back-edge is the `refine_step` → `generate_step` `SelectionReady` event instead of a Python loop. Observables are unchanged: signatures, wire contract, error codes, E12 semantics, cancellation and the deterministic core (§15 above).
- **Test strategy implied by this design.** The deterministic core is testable without any model: weight smoothing, `analysis_score`, blend, adjustments, ranking chain, dual gate, near-tie margin, supporting rules (distinct-constraint, conflict table, caps), dominant election, dedupe, fallbacks E5–E7, and the E7 no-fallback rule for `specialized-components` — plus the small pure functions around the LLM calls: part repair/merge, alias dedupe, priority normalisation, reselection ledger and stickiness, dominance tie-break, contract-part exemption, plus the single-language resolution: recognized / unrecognized→generic / absent→generic, the effective-language normalization into `design_brief.language`, the uniform discouragement veto (concrete and `generic` entries), the generic-mode no-adjustment rule, the binding-resolution order (requested-language entry → `generic` entry), and the one-element `overview.languages` echo. LLM-dependent stages are tested with recorded/fixture responses asserting schema validity and the deterministic post-processing around them.
- **Catalog sync.** The category/scope/language enums in prompts and validators are derived from `docs/pattern-schema.json` (single source of truth), and the style-affinity table keys off the architecture server's published style enum — drift between catalog and code is a validation failure, not a silent behavior change. The language-file schema (`docs/pattern-language-schema.json`) references the same enums, and the §2.3 integrity check is part of this gate. The `design-patterns://variants` resource (§2.4) is materialized from the joined records at startup, behind the same §2.3 gate.
