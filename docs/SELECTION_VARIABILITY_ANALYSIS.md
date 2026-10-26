# Selection Variability — Why Identical Requests Return Different Patterns

**Status: non-normative finding report.** `docs/implementation-guide.md` remains the normative
specification of the selection pipeline; this document records a measured property of the
*implemented* pipeline (2026-09-26, dev stack) that the guide states only implicitly.

**Revision (2026-09-26, later audit).** Every `file:line` citation, config claim and tool-metadata
claim below was re-verified against the source; all held. Four analytical claims did not and have
been corrected in place, each marked *Correction* or *an earlier draft*: the channel-2 effect size
(§2.2), the response-disclosure claim (§4.4), the part-id recurrence count (§1.2), and the
reasoning-trace cache reach (§2.5). A controlled pair of runs and a four-call sample were added
(§1.4), two new drift findings were added to §5, and §6 was re-ranked.

**Revision (2026-09-26, second audit).** An independent pass re-verified every citation, config
and tool-metadata claim (all held), reproduced the MiniMax API findings via its published schema,
and added a **second controlled pair** (runs G–H, §1.4): two further byte-identical requests,
sequential in the same process, results on disk. Eight identical calls have now produced **seven
different primaries in seven different categories**. The reselection path triggered in a third
independent run (§2.4), and the style prior flipped between the two newest runs (§2.2). §6 gained
two externally-grounded options (request-cache prior art; serving-side batch invariance) and the
provider no-`seed` finding is confirmed against MiniMax's own API reference.

**Revision (2026-09-26, third audit).** The three claims the remediation list leans on were
re-verified directly in the source — `src/tools/design.py:61` ("deterministic reasoning trace")
and `:71` (`idempotent_hint=True`), `src/design_normalization.py:115`
(`architecture_style=brief.architecture_style or style`) and `src/config.py:31-54` (no `seed`
field; `top_p`/`top_k` present but unread) — and all three still hold. *(2026-10-07: the
`design.py` findings were remediated — the description now says "traceable reasoning trail" and
`idempotent_hint=False`; the wording quoted here is the audited pre-fix state.)* The
externally grounded options were re-grounded against primary documentation instead of summaries
(LiteLLM, vLLM, one arXiv paper; full list with URLs in §8), which added the verbatim provider wording, a per-worker
cache caveat, and a third independent source for the temperature-0 claim. Structurally this
revision adds a *Verdict* paragraph, a derived inference in §2.4, a fifth caller consequence
(§4.5), a two-tier conclusion above the option list, and an explicit warning that §6.1 does not
address the dominant channel.

**Claim.** `select_design_patterns` is not a pure function of its request. It is a stochastic
sampler wrapped in a deterministic executor: every call re-samples the problem decomposition, the
per-part quality weights and the architecture-style prior (all LLM calls at `temperature=0.1`,
no seed), and the deterministic stages A–H then execute that sample faithfully. Two identical
requests are two draws from the same distribution.

Measured: **eight identical calls, seven different `overview.primary_pattern` values in seven
different categories** — `tag-dispatching` (behavior-selection), `content-based-serializer`
(interaction-boundary), `load-leveller` (concurrency-resilience), `intercepting-validator`
(security-verification, twice), `envelope-wrapper` (interaction-boundary),
`publish-subscribe-channel` (workflow-messaging), `publish-subscribe-channel` again in the second
controlled pair. Pattern-set Jaccard overlap across observed pairs: **0.30–0.53**. Both
**controlled pairs** of byte-identical requests issued sequentially in one process (§1.4) returned
different primaries in different categories, and each pair shared **at most one part id**.

**Verdict.** Not a defect in the executor. The deterministic core is verified faithful (§3), and
all four entropy channels are upstream of it (§2); a different answer per call is the faithful
execution of a different sample. What *is* defective is the contract: the tool description and
the `idempotent_hint` annotation promise a reproducibility the pipeline cannot deliver (§5), and
`src/design_normalization.py:115` inverts the documented `architecture_style` precedence. Neither
the observed variation nor the near-tie flips are bugs — the first is a sampler, the second is
§8.1's total order resolving inside a 1.22–4.55-point noise band. On this stack no sampler-side
or provider-side knob can make two identical requests return identical payloads (§6);
reproducibility here is a caching decision, not a tuning decision.

---

## 1. Evidence base

| Item | Value |
|---|---|
| Request | `examples/design_pattern_client.py` — the 777-char event-driven IoT/Kafka description, `language=python` |
| Server | container `design-pattern-mcp-dev`, host port 8062, one process throughout (no restart between calls) |
| Model | `minimax/MiniMax-M2.7` via `https://api.minimax.io/v1`, one LiteLLM client for all call kinds |
| Sampling | `temperature=0.1`, no `seed`; `top_p`/`top_k` configured but never reach the provider (§5) |

| Reasoning layer | `REASONING_ENABLED=true`, 2 thoughts for each of analyze/weights/generate/evaluate |
| Catalog | 438 main records / 6132 language entries at run time (2026-09; current catalog is 486 records), identical across all runs |

Eight observations, all in the same server process. Runs A–D precede the container replacement at
08:38:49 local and rest on opportunistically collected traffic. Runs E–H were all issued after
that replacement, in the current container instance: E–F are the first controlled pair (§1.4),
G–H the second, each pair sequential with nothing else in flight and both results written to
disk. E–F's server ledger is not preserved; G–H's full JSON artifacts are, at
`/tmp/varcheck/run{1,2}.json`.

- **Run A** (user, 05:54:03–06:01:53 UTC, 470 s) — 5 parts, attempt 1 score 69.0 → early stop.
- **Run B** (user, 06:13:16 UTC) — 8 parts, attempt 1 score 42.0 → reselection → attempt 2 score
  68.7 → attempt 2 retained.
- **Run C** (this analysis, 08:16:27–08:30:15 local, 832 s) — 7 parts, attempt 1 score 68.0 → early stop.
- **Run D** (this analysis, 08:30:19–08:40:45 local, 626 s) — 7 parts, attempt 1 score **51.5**
  → early stop by 1.5 points.
- **Run E** (controlled pair 1, 11:50:28 local, 625 s) — 5 parts, attempt 1 score **48.5** →
  reselection → attempt 2 score 73.0 → attempt 2 retained.
- **Run F** (controlled pair 1, ~12:01 local, 554 s) — 6 parts, attempt 1 score 67.5 → early stop.
- **Run G** (controlled pair 2, 12:26:48–12:39:21 UTC, 754 s) — 6 parts, attempt 1 score **42.3**
  → reselection → attempt 2 score **74.5** → attempt 2 retained.
- **Run H** (controlled pair 2, 12:39:58–12:48:25 UTC, 490 s) — 6 parts, attempt 1 score 58.5 →
  early stop by 8.5 points.

### 1.1 The eight answers

| # | parts | `overview.primary_pattern` | `overview.category` | applications | loop |
|---|---|---|---|---|---|
| A | 5 | `tag-dispatching` | `behavior-selection` | 12 | 1 attempt (69.0) |
| B | 8 | `content-based-serializer` | `interaction-boundary` | 12 (18 before reselection) | 2 attempts (42.0 → 68.7) |
| C | 7 | `load-leveller` | `concurrency-resilience` | 15 | 1 attempt (68.0) |
| D | 7 | `intercepting-validator` | `security-verification` | 17 | 1 attempt (51.5) |
| E | 5 | `envelope-wrapper` | `interaction-boundary` | 9 | 2 attempts (48.5 → 73.0, reselection) |
| F | 6 | `publish-subscribe-channel` | `workflow-messaging` | 15 | 1 attempt (67.5) |
| G | 6 | `publish-subscribe-channel` | `workflow-messaging` | 11 | 2 attempts (42.3 → 74.5, reselection) |
| H | 6 | `intercepting-validator` | `security-verification` | 12 | 1 attempt (58.5) |

Three knife-edges on the same 50-point bar. Run D scored 51.5; run E scored **48.5** — 1.5 points
short, which triggered a reselection, moved the dominant primary from `tolerant-reader` to
`envelope-wrapper`, and cut the result to 9 applications; run F, one sampling difference away,
scored 67.5 and never entered that path. Run G scored **42.3**, reselected, and its attempt 2
scored 74.5 — the highest of the eight runs, reachable only through the low first draw. Run H's
58.5 cleared the bar by 8.5 and stopped. The loop branch is not hypothetical: it fired in three of
eight runs, and which branch fires is itself sampled.

### 1.2 The eight decompositions

The part list is the retrieval query set (channel 1, §2.1). Eight calls, eight different lists:

| # | decision parts |
|---|---|
| A | `worker-per-type-extensibility`, `dispatch-content-based-routing`, `ingest-retry-dlq-handling`, `gateway-backpressure-handling`, `gateway-payload-validation` |
| B | `gateway-schema-validation-strategy`, `dispatch-routing-contract`, `worker-extensibility-mechanism`, `ingest-throughput-optimization`, `worker-horizontal-scaling-strategy`, `backpressure-mechanism`, `worker-enrichment-pattern`, `timeseries-persistence-strategy` |
| C | `worker-per-type-extensibility`, `gateway-schema-validation`, `worker-horizontal-scaling`, `backpressure-mechanism`, `retry-dlq-error-handling`, `content-based-routing`, `event-enrichment-strategy` |
| D | `dispatch-routing-extensibility`, `schema-validation-strategy`, `kafka-publish-with-backpressure`, `retry-dlq-handling`, `worker-enrichment-architecture`, `worker-horizontal-scaling`, `timeseries-persistence-model` |
| E | `gateway-payload-validation-strategy`, `event-publishing-to-message-broker`, `event-type-routing-extensibility`, `worker-normalization-enrichment-pattern`, `backpressure-and-retry-dlq-strategy` |
| F | `versioned-json-schema-validation`, `kafka-event-publishing`, `type-based-event-routing`, `event-normalization-enrichment`, `backpressure-mechanism`, `retry-dead-letter-handling` |
| G | `api-gateway-schema-validation`, `event-publishing-to-kafka`, `event-type-routing`, `worker-enrichment-logic`, `worker-horizontal-scaling`, `backpressure-and-retry-handling` |
| H | `payload-validation-schema`, `event-publishing-to-kafka`, `type-based-event-routing`, `worker-per-type-processing-extensibility`, `event-normalization-enrichment`, `retry-dead-letter-queue` |

Four ids recur across the eight runs, and none recurs with the same meaning:

- `worker-per-type-extensibility` in A and C — run A's pool for it contains `tag-dispatching`, run
  C's does not; run C selects `content-based-router` where run A selects `tag-dispatching`.
- `backpressure-mechanism` in B and C (and F, as a sixth-part id with a different neighbourhood).
- `worker-horizontal-scaling` in C, D and G — a recurrence an earlier draft of this report missed.
  Run C scopes it `task-lifetime` and selects `load-leveller`; run D scopes it `module` and
  selects `competing-consumers`; run G scopes it `task-lifetime` again and selects
  `load-leveller` — but from a different pool (thread-pool, master-worker as retrieval neighbours).
  Same id, different scope or pool, different result.
- `event-publishing-to-kafka` in G and H — the only id the second controlled pair shares (1 of 11
  distinct ids); run G selected `publish-subscribe-channel` for it from a fit-90.5 pool, run H
  also `publish-subscribe-channel` but from a different pool (fit 86.8, different neighbours).

Part ids are model-authored prose, not stable keys. The controlled pairs are the extreme case:
**E and F share no part id at all; G and H share exactly one of eleven.**

### 1.3 Per-part primaries, run A vs run B

```
A: worker-per-type-extensibility → tag-dispatching        (ambiguous, Δ1.33)
   dispatch-content-based-routing → content-based-router  (ambiguous, Δ1.57)
   ingest-retry-dlq-handling → invalid-message-channel
   gateway-backpressure-handling → backpressure
   gateway-payload-validation → content-based-serializer  (ambiguous, Δ1.26)

B: gateway-schema-validation-strategy → tolerant-reader
   dispatch-routing-contract → content-based-serializer   (ambiguous)
   worker-extensibility-mechanism → content-based-serializer
   ingest-throughput-optimization → publish-subscribe-channel
   worker-horizontal-scaling-strategy → competing-consumers
   backpressure-mechanism → poison-pill
   worker-enrichment-pattern → normalizer
   timeseries-persistence-strategy → serialized-lob
```

Only `content-based-serializer` is primary in both runs. Six patterns occur only in A
(`tag-dispatching`, `type-erasure`, `point-to-point-channel`, `message-expiration`, `watchman`,
`rate-limiting`), eight only in B (`competing-consumers`, `publish-subscribe-channel`,
`normalizer`, `serialized-lob`, `dead-letter-channel`, `load-leveller`, `event-message`,
`publisher-subscriber`).

Run D's primaries — `content-based-router`, `intercepting-validator`,
`publish-subscribe-channel`, `invalid-message-channel`, `event-based-asynchronous`,
`competing-consumers`, `serialized-lob` — share three patterns with B and three with C, and its
dominant (`intercepting-validator`) appears in neither A's nor B's returned set.

### 1.4 The two controlled pairs, and a four-call sample

Runs A–D were collected opportunistically and two of them overlapped other traffic. Runs **E**
and **F** are a clean pair: two byte-identical requests, issued **sequentially** in one server
process, no `architecture_style` supplied, both results written to disk. Runs **G** and **H**
repeat that protocol in the same container instance at 12:26–12:48 UTC.

| # | duration | parts | `primary_pattern` | category | score | applications | loop |
|---|---|---|---|---|---|---|---|
| E | 625 s | 5 | `envelope-wrapper` | interaction-boundary | 73.0 | 9 | 2 attempts (48.5 → 73.0), reselection |
| F | 554 s | 6 | `publish-subscribe-channel` | workflow-messaging | 67.5 | 15 | 1 attempt |
| G | 754 s | 6 | `publish-subscribe-channel` | workflow-messaging | 74.5 | 11 | 2 attempts (42.3 → 74.5), reselection |
| H | 490 s | 6 | `intercepting-validator` | security-verification | 58.5 | 12 | 1 attempt (58.5) |

Pair E/F: pattern-set Jaccard **0.375** (6 shared / 16 union); part-id overlap **zero**.
Pair G/H: Jaccard **0.375** again (6 shared / 16 union); part-id overlap **1/11**
(`event-publishing-to-kafka`). Different primaries, different categories — the first pair's
dominants disagree, and the second pair's dominant lands in a category no other run's returned
set contains as primary. Per part:

```
E: gateway-payload-validation-strategy     -> envelope-wrapper          (3-way near-tie)
   event-publishing-to-message-broker      -> publish-subscribe-channel
   event-type-routing-extensibility        -> content-based-router
   worker-normalization-enrichment-pattern -> normalizer
   backpressure-and-retry-dlq-strategy     -> invalid-message-channel

F: versioned-json-schema-validation        -> content-based-serializer
   kafka-event-publishing                  -> publish-subscribe-channel
   type-based-event-routing                -> content-based-serializer
   event-normalization-enrichment          -> content-enricher
   backpressure-mechanism                  -> backpressure
   retry-dead-letter-handling              -> invalid-message-channel

G: api-gateway-schema-validation           -> envelope-wrapper          (near-tie)
   event-publishing-to-kafka               -> publish-subscribe-channel
   event-type-routing                      -> message-dispatcher
   worker-enrichment-logic                 -> normalizer
   worker-horizontal-scaling               -> load-leveller             (near-tie)
   backpressure-and-retry-handling         -> poison-pill               (near-tie)

H: payload-validation-schema               -> intercepting-validator    (fit 90.1, clear win)
   event-publishing-to-kafka               -> publish-subscribe-channel
   type-based-event-routing                -> content-based-router
   worker-per-type-processing-extensibility -> composed-message-processor (near-tie)
   event-normalization-enrichment          -> content-enricher
   retry-dead-letter-queue                 -> poison-pill               (3-way near-tie)
```

Across the two pairs only two per-part primaries ever agree (`publish-subscribe-channel` for the
event-publishing part; `poison-pill` for the retry/backpressure part) and neither pair's dominant
carries over: E/F dominants disagree, and H's `intercepting-validator` — also D's dominant — does
not appear in E/F/G's returned sets at all. It is absent from G's own candidate set, in fact: G's
attempt-1 reselection *excluded* `intercepting-validator` as a selection defect, and the very next
byte-identical request elected it dominant.

A separate four-call sample in the same process shows the decomposition drift is not an artifact of
two draws. All four produced five parts, yet:

| pair | shared part ids | Jaccard |
|---|---|---|
| 1–2, 1–3, 2–3, 2–4, 3–4 | **0** | 0.00 |
| 1–4 | 1 | 0.11 |

and the first part's primary moved `intercepting-validator` → `envelope-wrapper` →
`intercepting-validator` → `tolerant-reader`.

---

## 2. Where the entropy enters

Four model-dependent inputs reach the selection, in pipeline order. Everything else — fusion,
RR blend, vetoes, `analysis_score`, `fit_score`, ranking, thresholds, supporting rules, dominant
election, language binding — is arithmetic over catalog data (`src/patterns/retriever.py`,
`src/pipeline.py`), and its tie-break chain ends in `(catalog order, name)`
(`src/pipeline.py:488-496`), so it is reproducible **given** its inputs.

### 2.1 Channel 1 — decomposition drift (dominant)

`src/pipeline.py:1730-1797` (`_stage_a_b`). One LLM call produces the `DesignBrief` **and** the
1–`max_parts` decision parts. The parts are the retrieval queries
(`PartQuery(statement=…, constraints=…, scope=…)`, `src/pipeline.py:1896-1904`), so a different
decomposition is a different question. Observed part counts for the identical 777-char request:
**5, 8, 7, 7, 5, 6, 6, 6** — and the part *lists* are close to disjoint. Over A–D no more than two
ids recur in any pair; the controlled pairs share zero ids (E/F) and one of eleven (G/H); in the
four-call sample of §1.4 a part id recurs in only one of six pairs. Part count also bounds the
number of applications (one application per (part, pattern) seed, `src/pipeline.py:823-841`):
5 parts → 12 apps (A) and 9 (E), 6 → 15 (F), 11 (G) and 12 (H), 7 → 15 (C) and 17 (D), 8 → 18.
The count is a floor, not a formula — reselection cuts it, which is why 5 parts can yield 9
applications and why G and H, with identical part counts, differ by one.

### 2.2 Channel 2 — brief `architecture_style` drift

The same A/B call also extracts `brief.architecture_style` from the description text, which feeds
the ±5 `style_affinity` adjustment (`src/pipeline.py:518-519`, table in `src/style_affinity.py`).

Adjustment ledger over all scored candidates:

| Run | candidates scored | with `style_affinity+5` | categories receiving it |
|---|---|---|---|
| A | 48 | **34** | `interaction-boundary`, `workflow-messaging` (= the `event-driven` row) |
| C | 127 | **0** | — |

The brief is not logged, but the ledger identifies it: replaying each run's own pool and weight
vectors through `score_candidates` with a candidate `architecture_style` reproduces the logged
`fit_score`s exactly for one style and misses by exactly 5 for the other.

| Run | replay with `architecture_style="event-driven"` | replay with `architecture_style=None` |
|---|---|---|
| A (5 pools, 48 candidates) | **max &#124;Δfit&#124; = 0.006**, none > 0.01 | max &#124;Δfit&#124; = 5.003, 34/48 off by 5 |
| C (6 clean pools, 48 candidates) | max &#124;Δfit&#124; = 5.005, 34/48 off by 5 | **max &#124;Δfit&#124; = 0.009**, none > 0.01 |

So run A's brief carried an `event-driven`-class style and run C's carried none (or an unlisted
one) — extracted by the same model from the same 777 characters. The same record
(`content-based-router`, category `workflow-messaging`) is scored with `style_affinity+5` in run A
and without it in run C.

**Correction — the ±5 is a category-group step, not a per-candidate perturbation.** The delta is
added to *every* candidate whose category is in the style's affinity set (`src/pipeline.py:518-519`;
35-row table in `src/style_affinity.py`). A uniform shift applied to both the leader and the
runner-up cancels in the difference, so the channel's power to decide an election depends entirely
on whether the two tied candidates straddle the affinity boundary. Measured by replaying through
the production `score_candidates` + `select_for_part`:

| Probe | Result |
|---|---|
| Whole catalog at run time (438 candidates) at one blend value, all 35 styles + `None` (438 candidates × 36 conditions) | **0** primary changes; the leader's category identical in every distinct-affinity style |
| Run C `gateway-schema-validation` (`intercepting-validator` security-verification vs `tolerant-reader` interaction-boundary) | **14/35** styles flip the primary |
| Run D `dispatch-routing-extensibility` (`content-based-router` workflow-messaging vs `content-based-serializer` interaction-boundary) | **11/35** |
| Run C `worker-per-type-extensibility` | **6/35** |
| Run D `retry-dlq-handling` (`invalid-message-channel` vs `dead-letter-channel`, **both** workflow-messaging) | **0/35** |

The correct effect statement is therefore: for a **cross-category** near-tie the pair's gap moves by
exactly 5; for a **same-category** near-tie it moves by 0. An earlier draft compared the ±5 against
the observed 1.22–4.55 near-tie margins and concluded the effect "exceeds" them — that sets a
candidate-level magnitude against a leader/runner-up *difference* and is the wrong frame. The
channel is decisive for cross-category contests and inert for same-category ones.

In the controlled pairs the channel was **active in E, F and G** although the caller supplied no
`architecture_style`: `style_affinity+5` appears in their ledgers — and **absent in H**, whose
adjustment ledgers carry no `style_affinity` entry at all, from the same 777 characters minutes
later. It is not a rare event, and it flips on its own. (Fixed for style-pinning callers on
2026-09-26 — §6.1; the channel remains live for callers that omit `architecture_style`.)

### 2.3 Channel 3 — weight-vector drift

`src/pipeline.py:254-321`: one batched LLM call returns eight axis weights per part;
`analysis_score = 10 · Σ w'·qa / Σ w'` (`src/pipeline.py:437-442`), so the ranking follows whichever
axis dominates. Measured: for the recurring part id the vectors were near-identical across runs
(differing only in `scalability`, 0.5819 vs 0.1153), i.e. at `temperature=0.1` the model usually
reproduces its mode — but when the vector does move, it is decisive (§3).

### 2.4 Channel 4 — evaluate score and loop branch

`src/pipeline.py:1563-1645`. The EVALUATE call produces the 0–100 score that decides:

- early stop (`score ≥ min_quality_score` 50) vs a second attempt;
- **reselection**: when a finding is a *selection* defect and the score is below the bar, the
  offending patterns become sticky exclusions and Stage E re-runs over the remaining candidates
  (`src/pipeline.py:1908-1952`);
- which attempt is returned (**best-attempt retention**, not the last).

Run B is the worked case: attempt 1 scored 42.0, reselection excluded
`content-based-router, event-message, poison-pill, publisher-subscriber, tolerant-reader`, and the
returned set shrank from 18 to 12 applications with a different supporting set. The same run would
have returned the 18-application set had attempt 1 scored ≥50.

Run G replicates it from the clean-pair protocol: attempt 1 scored **42.3**, reselection excluded
`competing-consumers, content-based-router, content-enricher, intercepting-validator,
invalid-message-channel`, and attempt 2 scored **74.5** — the best score of all eight runs,
produced *only because* the first draw fell below the bar. Run H, one draw later, scored 58.5 and
stopped immediately. Three of eight runs entered the loop; the branch itself is sampled, and it
moves the returned set by up to 6 applications and the score by up to 32 points.

*Derived inference.* Because the loop retains the best attempt rather than the last, the returned
score is a property of the **path** taken, not of the selection's intrinsic quality. Run G's first
draw scored 42.3 and its second attempt 74.5 — the highest of the eight — while run A's single
attempt scored 69.0 on a comparable part count. A caller therefore cannot read the score as a
quality signal comparable across calls: it may be high because the reselection path fired, not
because the selection was better. §4.1 and §4.5 are the only stable readings of either number.

### 2.5 Prompt-context drift (upstream of 1–4)

The server is itself an MCP client of the two structured-reasoning scratchpads
(`shannonthinking`, `code-reasoning`; `src/reasoning/client.py`, config §`reasoning`). The
thoughts are authored by the server's own LLM (`operation="reasoning:<phase>:step:<n>"`, same
temperature, no seed) and appended to each phase's user prompt
(`src/agent.py:316-318`; `src/reasoning/prompts.py:62-71`). Consequences:

- 8 extra sampled provider calls per run; run A issued 14 provider calls in total
  (8 reasoning + brief_and_parts + weights + 3 generate + 1 evaluate).
- A failed/short draft step breaks the loop early, so the *number* of reasoning steps — and hence
  the injected prompt text — varies per run.
- `analyze` and `generate` are listed in `_CACHEABLE_PHASES` (`src/reasoning/prompts.py:142`) and
  are cached in-process by payload digest (`src/reasoning/client.py:298-307`); `weights` and
  `evaluate` are not. **But only `analyze` actually hits across runs.** Its payload is
  `{"description": text}` (`src/pipeline.py:1404`), constant for an identical request. Over four
  calls in one process `analyze` logged `duration_ms=27989` every time, to the millisecond, while
  `weights` varied (25939 / 33708 / 28429 / 23965). `generate` **never** hit — 32630 / 38869 / 47201
  / 29359, all different — because its payload embeds the composed selection, so it misses whenever
  the decomposition moves. An earlier draft implied `generate` was effectively cached; it is not,
  and caching it buys nothing across runs.
  Consequence: call #1 in a process differs from calls #2..n by more than sampling noise.

---

## 3. Controlled experiment — the arithmetic is faithful, its input is not

Replay of each run's logged candidate pools through the **production** functions
(`score_candidates`, `select_for_part`, `src/pipeline.py:420,537`) reproduces the logged
`fit_score`s to **max &#124;Δ&#124; = 0.006** once the run's own weight vector and the correct
`architecture_style` are supplied; the residual is consistent with the 6-decimal rounding of
`retrieval_blend` in the log. The server log therefore contains enough information to re-derive
every selection decision offline.

> Reproducibility caveat (second audit): the logs backing the §3 replay figures and the §2.2
> ledger table were transient (see §7) and no longer exist; the figures rest on the analysis
> performed while they were live plus the recovered `DesignResult` artifacts. They are consistent
> with every verifiable citation and with the fresh runs, but the replay itself is not currently
> repeatable. The second pair's artifacts (`/tmp/varcheck/run{1,2}.json`) *are* preserved and
> carry per-part fit/blend/adjustment ledgers sufficient for an independent replay.

Holding run A's pool for part `worker-per-type-extensibility` fixed and swapping **only** the
weight vector — both vectors taken from run A itself — flips the primary:

```
run A own vector            (extensibility .7375, scalability .5819)
    tag-dispatching        fit 80.482   → PRIMARY tag-dispatching  (ambiguous, Δ1.33)
    content-based-router   fit 79.159

run A sibling vector        (extensibility .7375, maintainability/simplicity .5625)
    content-based-router   fit 76.038   → PRIMARY content-based-router  (ambiguous)
    tag-dispatching        fit 76.008      (margin: 0.030 points)
```

Across all 13 weight vectors observed live, applied to run A's five pools, **2 of 5 parts change
their winner** (`worker-per-type-extensibility`: `tag-dispatching` 9/13 vs
`content-based-router` 4/13; `gateway-payload-validation`: `content-based-serializer` 12/13 vs
`tolerant-reader` 1/13).

The near-tie margin is 5 and measured leader/runner-up gaps were **1.22–4.55** fit points on a
0–100 scale, with 3 of 5 parts flagged `ambiguous` in run A. The §8.1 total order guarantees a
*unique* ranking, never a *stable* one: the election is decided inside the sampling noise band.

---

## 4. Consequences for a calling agent

1. **Repeat calls are samples, not retries.** A second call is not a re-execution of the first;
   it is an independent draw. Differences are expected, not an error signal.
2. **The near-tie siblings are the real choice surface.** When `ambiguous` is set, the runner-up
   is a coin-flip sibling at the same evidence level; `alternatives_considered` carries the delta.
   A caller wanting stability should pin a run (cache the result) rather than re-call.
3. **Ids are model-authored.** Part ids (`kebab(raw.id)`) and the application ids derived from
   them churn between runs; do not cache or reference applications by id across calls.
4. **The response discloses most of the sample, not all of it.** An earlier draft claimed the
   decomposition and the weights were response-invisible and attributable only from the server log.
   That is wrong. `_reasoning` (`src/pipeline.py:967-1019`) assembles `overview.reasoning` from the
   ledgers and already emits, per part: the part id and scope, the primary, the retrieval top-3 with
   blends and `fit`s, `weights_source` with the top-3 axes, the dual-gate thresholds, the supporting
   set, the ambiguity and veto flags, and the full adjustment ledger for the primary and supporting
   candidates. A caller can already see which parts were produced, what each won on, and by how
   much. Genuinely missing: the part *statements*, the complete 8-axis weight vectors (only the top 3
   are printed), and the scored-but-unselected candidates. Those remain server-log only
   (`Part weights extracted`, `Candidate scored`, `Near-tie flagged (E8)`).
   **Fixed 2026-09-26 (§6.3):** the part statements and the complete 8-axis vectors are now
   appended to `overview.reasoning`; of the three gaps only the scored-but-unselected candidates
   remain server-log only.
5. **The returned score is not a stability signal.** It is the best of up to two sampled evaluate
   calls on a path that may or may not have reselected (§2.4, and the inference derived there).
   Two runs of comparable quality can differ by up to 32 points, and the highest score observed
   (74.5) came from a run whose *first* draw fell below the bar. Use the score to judge one pinned
   result; never to compare two results, and never as evidence that the tool is deterministic.

---

## 5. Documentation drift found while measuring

| Location | Claim / state | Finding |
|---|---|---|
| `src/tools/design.py:61` | tool description: "…with per-application language bindings and a **deterministic** reasoning trace" | the payload is not reproducible |
| `src/tools/design.py:71` | `idempotent_hint=True` on `select_design_patterns` | same request → different payload |
| `docs/implementation-guide.md` §15 | "Deterministic core … identical inputs, parameters and index → identical ranking, tie-breaks, thresholds, fallbacks" | true and correctly scoped (non-LLM core only), but the guide never states that two identical *requests* may return different selections |
| `docs/implementation-guide.md` §14 | parameter table | `generator.config.temperature` is not documented at all |
| `config/config.json:14-15`, `src/config.py:38-39` | `top_p`, `top_k` | dead knobs: the llama-index LiteLLM wrapper has no `top_p` field (pydantic `extra=ignore`) and nothing reads `top_k`; neither reaches the provider, so neither can be used to stabilise output |
| `src/config.py:31-54` | generator inner config | no `seed` field exists at any layer, so there is no determinism switch to flip |
| `docker/docker-compose.yml:97-98` | `REASONING_RETRY_ENABLED/THOUGHTS` pins | inert: no pipeline call site requests a `retry` phase, and `get_strategy` silently falls back to the first `per_phase` entry for unknown phases |
| `src/design_normalization.py:115` | `architecture_style=brief.architecture_style or style` | **precedence inversion**: the model's guess overrides the caller's own `architecture_style` parameter. Guide line 29 says the hint applies "when present" and style affinity is derived from description text "when absent" — the code does the opposite. Correcting the order removes channel 2 from the sampled set for every caller who supplies the parameter (§6.1), at zero LLM cost **Fixed 2026-09-26: the resolution is now `style or brief.architecture_style` (caller wins); row retained as the historical finding.** |
| `src/tools/design.py:71`, `src/tools/analyze.py:69`, `src/tools/evaluate.py`, `src/tools/generate.py` | `idempotent_hint=True` on all four pipeline tools | the sibling `olk/architecture-pattern-mcp` (`../architecture-pattern-mcp/src/tools/design.py:207`, comment *"non-idempotent: each run creates a new LLM pipeline"*) and `olk/agent-pattern-mcp` declare `idempotent_hint=False` on the same tools, and their READMEs state "Not idempotent." In fairness, MCP's normative schema defines `idempotentHint` as "no additional effect on its environment" — not output equality — and every annotation is an untrusted hint, so this is not a spec violation. It is nonetheless the only one of the three servers claiming it |
| MiniMax OpenAI-compatible API (`platform.minimax.io/docs/api-reference/text-openai-api`) | provider surface for this deployment | the published Chat Completions schema has **no `seed` parameter at all** and `n` supports only 1; some OpenAI parameters are documented as silently ignored. So even adding a `seed` field to `GeneratorInnerConfig` could not reach the provider: provider-side determinism is unavailable on this stack, not merely unused |

---

## 6. Options if reproducibility is required

Ranked by measured effect per unit of cost. This ordering changed after the §2.2 correction and
after checking what the tool can already control; the second audit added items 4's prior art and
new item 8.

**Conclusion first.** The options split into two tiers that must not be conflated. *Everything
else* (items 1–3 and 5–7) changes the sampled distribution or the disclosure, never the
guarantee. *Repeatability*: item 4 is the only measure on this stack that makes two identical
requests return byte-identical payloads; item 8 achieves the same only by changing the model and
the serving stack. Consequently §6.1 below — the cheapest change by an order of magnitude — does
**not** make the tool reproducible. It removes channel 2 (§2.2) for callers that pass
`architecture_style`, while the dominant channel 1 (§2.1) — the one that produced the 0-of-10 and
1-of-11 part-id overlaps — is untouched by it. Treat option 1 as a spec fix that happens to remove
one entropy source, not as a determinism fix.

1. **Honour the caller's `architecture_style`** — flip `src/design_normalization.py:115` from
   `brief.architecture_style or style` to `style or brief.architecture_style`. One line, no LLM
   cost, and it removes channel 2 from the sampled set for every caller that supplies the
   parameter. It is also a spec fix (§5), independent of determinism. Nothing else here comes
   close per line changed. **Applied 2026-09-26** — `src/design_normalization.py` now resolves
   `style or brief.architecture_style`; regression-pinned in `tests/unit/test_input_validation.py`
   and end-to-end in `tests/unit/test_style_precedence.py`.
2. **Stop over-claiming** — set `idempotent_hint=False` on the four pipeline tools and delete
   "deterministic" from the tool description. Both public siblings already do this. Zero cost.
3. **Surface the rest of the sample** — append the part statements and the full 8-axis weight
   vectors to `overview.reasoning`. It is already a free-form ledger string, so this needs no
   schema change, only a guide/code pair. Makes variation attributable from the client side (§4.4).
   **Applied 2026-09-26** — `_reasoning` (`src/pipeline.py`) now appends, per part,
   `statement: <verbatim part statement>` and `weights vector: axis=value …` with all eight axes
   in §7.2 order (`none` when no vector exists, e.g. hand-built selections); regression-pinned in
   `tests/unit/test_composition.py`.
4. **Request-keyed result cache** — hash the canonicalised request together with a
   model / prompt / catalog version digest and serve repeats from it. This is the only measure that
   makes repeats byte-identical. MCP's caching utility explicitly excludes `tools/call` (it covers
   `server/discover`, `tools/list`, `prompts/list`, `resources/list`, `resources/templates/list`,
   `resources/read`), so the cache must be application-level, and it needs a TTL because the
   catalog changes. This is the industry-standard consistency pattern for exactly this problem:
   LiteLLM's proxy ships an exact-match cache keyed on the full request (`cache_response_in_memory`
   / Redis), LangChain exposes the same via `set_llm_cache`, GPTCache predates both, and AWS's
   caching guidance explicitly sells response caching as a *consistency* mechanism ("identical
   outputs for identical inputs"). One warning from the same literature: use **exact** matching.
   Semantic caches (similarity-threshold replay) are documented by LiteLLM as inappropriate for
   agentic/multi-turn traffic — a near-duplicate request would get a stale answer replayed, which
   for a selection tool is worse than variation.
   Verified wording (third audit): LiteLLM's caching page states that the exact-match caches
   ("`redis`, `s3`, `gcs`, `local`, `disk`") "key on a hash of the whole request, so any change to
   the conversation is a miss", and its semantic-caching page carries the warning verbatim —
   "Semantic caching is designed for single-shot prompts. On multi-turn or agentic traffic it will
   replay stale responses" — closing with the recommendation to "use an exact-match cache (`type:
   redis`) instead … so any change to the conversation is a cache miss and stale replays cannot
   happen". Two design consequences for this server: (a) cache the **`DesignResult`**, not the
   provider exchanges — per-call caching is already half-implemented in the reasoning layer and
   hits unevenly (`analyze` hits, `weights`/`generate` never do, §2.5), so per-call caches give
   partial coverage and a still-partly-sampled run, whereas the result cache is all-or-nothing by
   construction; (b) an in-process cache lives inside one worker process — LiteLLM documents the
   same trap for its own proxy, where four workers keep four separate caches — so a horizontally
   scaled deployment needs a shared store or the hit rate divides by the worker count.
5. **`REASONING_ENABLED=false`** — removes 8 sampled prompt-context calls per run. Note it also
   removes the `analyze` cache hit, so it does not stabilise the main call's prompt; it only
   removes upstream noise.
6. **Freeze the selection inputs as fixtures** — record the parts and weight vectors of one run and
   replay them; the deterministic core then reproduces to max &#124;Δ&#124; = 0.006 (§3). This is
   what `tests/unit/` already does with fakes, and it is the right tool for tests and demos.
7. **Temperature 0** — *demoted from first place.* It cannot touch channel 1, which is the dominant
   one, and it cannot touch the style step. Provider documentation rules out relying on it: OpenAI
   calls its `seed` "best effort… **Determinism is not guaranteed**" (and has since marked the
   parameter deprecated/beta, monitored via `system_fingerprint`) and its own worked
   temperature-0 example returns five non-identical outputs; Anthropic states that "even with
   temperature set to 0, the results will not be fully deterministic" and documents no `seed`
   parameter at all; SGLang's FAQ says results "are not deterministic, even with a temperature of
   0" and attributes ~95 % of it to dynamic batching. Treat it as variance reduction, never as a
   fix.
   Third audit adds an independent quantification: arXiv 2408.04667 ("Non-Determinism of
   "Deterministic" LLM Settings") fixes temperature 0, top-p 1 *and* the seed and still measures
   varying outputs on hosted endpoints, while its control run — Llama3-8B on local GPUs with no
   serving optimizations — comes out deterministic. That localises the residual cause to batching
   and prefix caching, i.e. the mechanism item 8 addresses, and it is why adding a `seed` field
   would not have been a fix even on a provider that accepts one.
8. **Serving-side batch invariance (self-hosting only)** — the recent batch-invariant-kernel work
   (Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 2025) has landed in vLLM
   as `VLLM_BATCH_INVARIANT=1`: it makes completions bitwise-identical regardless of batch
   composition, the cause SGLang attributes ~95 % of temperature-0 nondeterminism to. vLLM's own
   documentation is explicit that this is opt-in and the price deliberate: "vLLM does not
   guarantee the reproducibility of the results by default, for the sake of performance"; batch
   invariance "may impact performance compared to the default non-deterministic mode. This
   trade-off is intentional to guarantee reproducibility"; and in online serving mode "you can
   only enable batch invariance". This is the
   only *true* provider-side determinism path, and it requires serving an open-weights model
   yourself. The widely cited ~30–60 % throughput cost is the Thinking Machines Lab measurement,
   not a vLLM figure — vLLM documents the trade-off without quantifying it — and api.minimax.io
   cannot offer any of it (no `seed`, no such flag; §5). Relevant only if some consumer ever needs
   replay-for-audit selection; for that case it beats every option above except the cache at
   making runs identical, but it changes the model, not just the serving.

Not measured here: the aggregate contribution of each channel — only channels 1 and 2 were observed
to differ decisively in the samples taken, and §2.2 now bounds channel 2's reach to cross-category
near-ties. The style-flip rates in §2.2 come from four near-tie pairs reconstructed at 4-decimal
precision, so they are an estimate with wide error bars, not a population rate. Self-consistency /
best-of-N (sample N selections, vote) is deliberately absent from this list: it can raise
*quality* but not *reproducibility* — each draw still varies, MiniMax's `n` supports only 1 so it
means N full pipeline calls, and the pipeline's own evaluate loop is already a weak two-sample
version of it.

---

## 7. Reproduction recipe

```bash
# 1. two identical blocking calls, sequentially (the dev client waits for the full pipeline;
#    measured 470-832 s per call on this dev stack, i.e. 8-14 minutes)
make client-design                          # writes the DesignResult JSON to stdout
make client-design

# 2. the server-side ledger for each call (JSON records go to stderr, not stdout)
docker logs --since 60m design-pattern-mcp-dev 2>&1 >/dev/null | python3 -c '
import json,sys
for line in sys.stdin:
    if not line.startswith("{"): continue
    r=json.loads(line); m=r.get("message")
    if m in ("Analyze phase started","Decomposition repaired","Part weights extracted",
             "Part selected","Near-tie flagged (E8)","Selection composed",
             "Attempt evaluated","Reselection ledger","Selection run completed"):
        print(r["timestamp"][11:19], m, {k:v for k,v in r.items()
              if k not in ("timestamp","level","logger","message")})'
```

Expect: different part ids/counts, different primaries for the same part id, and possibly a
two-attempt loop with reselection. The per-part `Candidate scored` records carry the full
`fit_score`/`analysis_score`/adjustment ledger needed to replay the deterministic core (§3).

This recipe was re-run against the dev stack on 2026-09-26 and works verbatim (68 records matched);
the second controlled pair (§1.4) was produced with the same `make client-design` call twice, redirecting
stdout to a file per run. Two operational notes:

- **Segment by part-id set, not by time window.** Concurrent requests interleave in one log — two
  of the first four runs above overlapped. The `Analyze phase started` → `Decomposition repaired` pair
  identifies a run, but attribution must be confirmed against the emitted part ids.
- **The log is transient.** The container instance that produced runs A–D was replaced at 08:38:49
  local and that ledger is no longer retrievable; only the recovered `DesignResult` artefacts
  survive. Capture `docker logs` to a file before analysing it. Runs E–F's ledger is likewise not
  preserved, so §1.4 and §3 rest on artefacts plus a same-process log sample; runs G–H's `DesignResult`
  JSON artifacts are preserved at `/tmp/varcheck/run{1,2}.json` (their client stderr in
  `run{1,2}.log`, timestamps in `run{1,2}.start`/`.end`), but their server-side ledger was not
  captured either — redirect `docker logs` to a file if you intend to replay (§3).

---

## 8. External sources consulted (third audit)

Primary documentation re-read on 2026-09-26 to ground §6.4, §6.7 and §6.8. Sources already cited
inline in §5 (MCP tool-annotations schema, SGLang determinism FAQ, MiniMax API reference) are
listed there and are not repeated.

| Source | What it confirms here |
|---|---|
| [LiteLLM — caching](https://docs.litellm.ai/docs/proxy/caching) | exact-match caches "key on a hash of the whole request, so any change to the conversation is a miss"; in-memory caches are per worker process |
| [LiteLLM — semantic caching](https://docs.litellm.ai/docs/proxy/caching_semantic) | "designed for single-shot prompts"; on "multi-turn or agentic traffic it will replay stale responses"; the documented remedy is the exact-match Redis cache |
| [LiteLLM — provider pass-through params (Replicate)](https://docs.litellm.ai/docs/providers/replicate) | `seed` is a provider pass-through, not a LiteLLM feature — a `seed` field stays a dead knob on any provider whose schema omits it (MiniMax, §5) |
| [vLLM — reproducibility](https://docs.vllm.ai/en/latest/usage/reproducibility.html) | non-determinism is the default "for the sake of performance"; in online mode "you can only enable batch invariance" |
| [vLLM — batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/) | `VLLM_BATCH_INVARIANT=1` exists and decouples output from batch size and order; the performance trade-off is "intentional" and unquantified |
| [arXiv 2408.04667 — Non-Determinism of "Deterministic" LLM Settings](https://arxiv.org/html/2408.04667v5) | temperature 0 with a fixed seed still varies on hosted endpoints; an unoptimized local-GPU control run is deterministic, localising the cause to batching / prefix caching |
