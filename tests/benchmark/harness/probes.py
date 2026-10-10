"""Stage-attribution probes: seam wrappers that time every injected dependency call (plan §2.2).

Why proxies and not instrumentation: the pipeline is fully constructor-injected
(``DesignPipeline.__init__`` at ``src/pipeline.py:1294``, ``RetrievalIndex.build``), so wrapping
the four seam objects the harness hands in captures build-time and request-time calls alike —
zero ``src/`` edits, zero monkeypatching.

Stage buckets (plan §3):

- ``build.embed_documents`` — index build, reported separately, excluded from per-request sums;
- per request: ``embed.query``, ``rerank``, ``llm.analyze`` (``extract_brief_and_parts``),
  ``llm.weights`` (``extract_weights_batch``), ``llm.generate``, ``llm.evaluate`` and
  ``reasoning.<phase>``.

Every record carries ``ok``/``error`` so a failed call still attributes its wall clock, and the
per-stage **call count** distinguishes "fewer calls" from "faster calls" when comparing arms.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

#: Index-build stages: recorded, but never part of a per-request aggregate.
BUILD_STAGES: tuple[str, ...] = ("build.embed_documents",)
STAGE_BUILD_EMBED_DOCUMENTS = BUILD_STAGES[0]

STAGE_EMBED_QUERY = "embed.query"
STAGE_RERANK = "rerank"
STAGE_LLM_ANALYZE = "llm.analyze"
STAGE_LLM_WEIGHTS = "llm.weights"
STAGE_LLM_GENERATE = "llm.generate"
STAGE_LLM_EVALUATE = "llm.evaluate"

#: Offline mode substitutes a scripted Stage-C index for the real retrieval pipeline; the stage
#: name says so, so an offline record can never be mistaken for a measured retrieval latency.
STAGE_RETRIEVAL_SCRIPTED = "retrieval.scripted"

#: Per-request stage buckets in report order (build stages deliberately absent).
REQUEST_STAGES: tuple[str, ...] = (
    STAGE_EMBED_QUERY,
    STAGE_RERANK,
    STAGE_LLM_ANALYZE,
    STAGE_LLM_WEIGHTS,
    STAGE_LLM_GENERATE,
    STAGE_LLM_EVALUATE,
    STAGE_RETRIEVAL_SCRIPTED,
)

#: Below this many samples a percentile is reported as ``null`` with a note (never NaN, plan §4).
MIN_PERCENTILE_SAMPLES = 4

#: ``statistics.quantiles(n=20, method="inclusive")`` cut-point indices: p50 = 10/20, p95 = 19/20.
_QUANTILE_INDEX_P50 = 9
_QUANTILE_INDEX_P95 = 18


def reasoning_stage(phase: str) -> str:
    """Stage bucket for one reasoning phase (``analyze``/``weights``/``generate``/...)."""
    return f"reasoning.{phase}"


def _llm_profile(inner: Any, operation: str) -> dict[str, Any]:
    """Best-effort per-call LLM data volume (Step 0b), read WITHOUT touching ``src/``.

    The production ``DesignAgent`` measures and logs each transport call at INFO
    (``Provider call succeeded`` with ``prompt_chars``/``response_chars`` in the log record's
    ``extra``). At harness runtime a module-level store in
    :mod:`tests.benchmark.harness.log_capture` remembers the last finished call per operation;
    a structured-output repair re-call is flagged by the ``LLM %s output failed validation``
    WARNING that preceded it (schema→operation mapping). Values attach AFTER the wrapped call
    finishes; ``None``/absent when the inner client never logged (scripted doubles, offline) —
    sparse data never fabricates numbers.
    """
    del inner  # selection happens via the capture module, not the proxy's polymorphic inner
    from .log_capture import last_call_meta

    meta = last_call_meta(operation)
    if meta is None:
        return {}
    payload: dict[str, Any] = {}
    if meta.get("prompt_chars") is not None:
        payload["prompt_chars"] = int(meta["prompt_chars"])
    if meta.get("response_chars") is not None:
        payload["response_chars"] = int(meta["response_chars"])
    if meta.get("repair") is True:
        payload["repair"] = True
    if meta.get("cached_tokens") is not None:
        payload["cached_tokens"] = int(meta["cached_tokens"])
    return payload


@dataclass(frozen=True)
class StageRecord:
    """One timed seam call."""

    stage: str
    op: str
    started_ns: int
    duration_ms: float
    ok: bool
    meta: dict[str, Any] = field(default_factory=dict)


class StageRecorder:
    """Append-only timing sink shared by every proxy of one run."""

    def __init__(self) -> None:
        self._records: list[StageRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[StageRecord]:
        """All records in call order (live view; use :meth:`snapshot` to freeze a slice)."""
        return self._records

    def now_ns(self) -> int:
        """Monotonic clock reading for a caller that wants to time its own boundary."""
        return time.perf_counter_ns()

    def mark(
        self,
        *,
        stage: str,
        op: str,
        started_ns: int,
        ok: bool,
        meta: dict[str, Any] | None = None,
    ) -> StageRecord:
        """Record one finished call; ``duration_ms`` is derived from the caller's start stamp."""
        record = StageRecord(
            stage=stage,
            op=op,
            started_ns=started_ns,
            duration_ms=(time.perf_counter_ns() - started_ns) / 1_000_000.0,
            ok=ok,
            meta=dict(meta or {}),
        )
        self._records.append(record)
        return record

    def snapshot(self) -> list[StageRecord]:
        """A copy of every record so far (caller keeps this as a per-request slice)."""
        return list(self._records)


async def _timed(
    recorder: StageRecorder,
    *,
    stage: str,
    op: str,
    meta: dict[str, Any],
    call: Callable[[], Awaitable[Any]],
    profile: Callable[[], dict[str, Any]] | None = None,
) -> Any:
    """Await ``call()``, recording its duration and outcome even on failure.

    ``profile`` (Step-0b) runs AFTER ``call()`` returns, so per-call LLM data captured from the
    ``src.*`` log seam during the call can be attached to the same record; it never runs on a
    failed call (the failure meta carries the error instead).
    """
    started_ns = recorder.now_ns()
    try:
        result = await call()
    except BaseException as exc:  # noqa: BLE001 - attribution must survive any seam failure
        recorder.mark(
            stage=stage,
            op=op,
            started_ns=started_ns,
            ok=False,
            meta={**meta, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    if profile is not None:
        with contextlib.suppress(Exception):  # capture must never fail attribution
            meta = {**meta, **profile()}
    recorder.mark(stage=stage, op=op, started_ns=started_ns, ok=True, meta=meta)
    return result


class EmbedderTimingProxy:
    """Timing wrapper for an ``EmbedderProtocol`` (``src/patterns/embedder.py``)."""

    def __init__(self, inner: Any, recorder: StageRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    @property
    def inner(self) -> Any:
        """The wrapped embedder (identity/logging; never unwrap for the pipeline)."""
        return self._inner

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Index-build document embedding (recorded as a build stage)."""
        return cast(
            "list[list[float]]",
            await _timed(
                self._recorder,
                stage=STAGE_BUILD_EMBED_DOCUMENTS,
                op="embed_documents",
                meta={"texts": len(texts)},
                call=lambda: self._inner.embed_documents(texts),
            ),
        )

    async def embed_query(self, text: str) -> list[float]:
        """Single query embedding (per request, §6 step 2)."""
        return cast(
            "list[float]",
            await _timed(
                self._recorder,
                stage=STAGE_EMBED_QUERY,
                op="embed_query",
                meta={"texts": 1},
                call=lambda: self._inner.embed_query(text),
            ),
        )

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Batched query embedding (the pipeline's shared per-request round trip)."""
        return cast(
            "list[list[float]]",
            await _timed(
                self._recorder,
                stage=STAGE_EMBED_QUERY,
                op="embed_queries",
                meta={"texts": len(texts)},
                call=lambda: self._inner.embed_queries(texts),
            ),
        )


class RerankerTimingProxy:
    """Timing wrapper for a ``RerankerProtocol`` (``src/patterns/safe_tei_rerank.py``)."""

    def __init__(self, inner: Any, recorder: StageRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[tuple[int, float]]:
        """One cross-encoder pass (§6 step 4)."""
        return cast(
            "list[tuple[int, float]]",
            await _timed(
                self._recorder,
                stage=STAGE_RERANK,
                op="rerank",
                meta={"documents": len(documents), "top_n": top_n},
                call=lambda: self._inner.rerank(query, documents, top_n),
            ),
        )


class RetrievalIndexTimingProxy:
    """Timing wrapper for the *scripted* Stage-C index double used by offline mode.

    The double replaces the whole retrieval pipeline (dense leg, BM25, fusion, rerank), so the
    only honest thing to record is the call itself — under a stage name that cannot be confused
    with a measured retrieval latency (:data:`STAGE_RETRIEVAL_SCRIPTED`).
    """

    def __init__(self, inner: Any, recorder: StageRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    async def retrieve_many(self, queries: Sequence[Any]) -> list[list[Any]]:
        """One batched per-part retrieval (plan §6 step 1-4 replacement)."""
        return cast(
            "list[list[Any]]",
            await _timed(
                self._recorder,
                stage=STAGE_RETRIEVAL_SCRIPTED,
                op="retrieve_many",
                meta={"queries": len(queries)},
                call=lambda: self._inner.retrieve_many(queries),
            ),
        )

    async def retrieve(self, **kwargs: Any) -> list[Any]:
        """Single-part retrieval (``RetrievalIndex.retrieve`` keyword contract)."""
        return cast(
            "list[Any]",
            await _timed(
                self._recorder,
                stage=STAGE_RETRIEVAL_SCRIPTED,
                op="retrieve",
                meta={"statement_chars": len(str(kwargs.get("statement", "")))},
                call=lambda: self._inner.retrieve(**kwargs),
            ),
        )


class LLMTimingProxy:
    """Timing wrapper for ``LLMClientProtocol`` (the four prompted call kinds, §3.1)."""

    def __init__(self, inner: Any, recorder: StageRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    async def extract_brief_and_parts(
        self, description: str, hints: Any, *, reasoning_context: str = ""
    ) -> Any:
        """Stage A+B (``llm.analyze``)."""
        return await _timed(
            self._recorder,
            stage=STAGE_LLM_ANALYZE,
            op="extract_brief_and_parts",
            meta={
                "description_chars": len(description),
                "reasoning_context_chars": len(reasoning_context),
            },
            call=lambda: self._inner.extract_brief_and_parts(
                description, hints, reasoning_context=reasoning_context
            ),
            profile=lambda: _llm_profile(self._inner, "brief_and_parts"),
        )

    async def extract_weights_batch(
        self, parts: Sequence[Any], brief: Any, *, reasoning_context: str = ""
    ) -> Any:
        """Stage D weights (``llm.weights``); one call per part batch (§7.1)."""
        return await _timed(
            self._recorder,
            stage=STAGE_LLM_WEIGHTS,
            op="extract_weights_batch",
            meta={
                "parts": len(parts),
                "reasoning_context_chars": len(reasoning_context),
            },
            call=lambda: self._inner.extract_weights_batch(
                parts, brief, reasoning_context=reasoning_context
            ),
            profile=lambda: _llm_profile(self._inner, "weights"),
        )

    async def generate(self, seed: Any, *, reasoning_context: str = "") -> Any:
        """One GENERATE batch call (``llm.generate``, §11 item 1)."""
        applications = getattr(seed, "applications", None)
        if applications is None and isinstance(seed, dict):
            applications = seed.get("applications")
        return await _timed(
            self._recorder,
            stage=STAGE_LLM_GENERATE,
            op="generate",
            meta={
                "applications": len(applications or ()),
                "reasoning_context_chars": len(reasoning_context),
            },
            call=lambda: self._inner.generate(seed, reasoning_context=reasoning_context),
            profile=lambda: _llm_profile(self._inner, "generate"),
        )

    async def evaluate(
        self, result: Any, parts: Sequence[Any], *, reasoning_context: str = ""
    ) -> Any:
        """EVALUATE call (``llm.evaluate``, §11 item 2)."""
        return await _timed(
            self._recorder,
            stage=STAGE_LLM_EVALUATE,
            op="evaluate",
            meta={
                "parts": len(parts),
                "reasoning_context_chars": len(reasoning_context),
            },
            call=lambda: self._inner.evaluate(result, parts, reasoning_context=reasoning_context),
            profile=lambda: _llm_profile(self._inner, "evaluate"),
        )


class ReasoningTimingProxy:
    """Timing wrapper for ``ReasoningClient``'s two pipeline-visible members.

    The pipeline reads ``.enabled`` (``src/pipeline.py:1338``) and calls
    ``run_pre_llm(phase, payload)`` (``src/pipeline.py:1341``) — nothing else, so the proxy
    exposes exactly those and delegates the rest.
    """

    def __init__(self, inner: Any, recorder: StageRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    @property
    def enabled(self) -> bool:
        """Master switch, mirrored from the wrapped client."""
        return bool(self._inner.enabled)

    async def run_pre_llm(self, phase: str, payload: dict[str, Any]) -> Any:
        """One pre-LLM reasoning loop for ``phase`` (internal drafts count toward this stage)."""
        return await _timed(
            self._recorder,
            stage=reasoning_stage(phase),
            op="run_pre_llm",
            meta={"payload_keys": sorted(payload)},
            call=lambda: self._inner.run_pre_llm(phase, payload),
        )

    def __getattr__(self, item: str) -> Any:
        """Delegate anything else (``health_check``, ``close``) to the wrapped client."""
        return getattr(self._inner, item)


def percentile_fields(values: Sequence[float]) -> dict[str, Any]:
    """p50/p95 of ``values`` via the stdlib inclusive quantiles, or ``null`` fields when sparse.

    ``statistics.quantiles(n=20, method="inclusive")`` yields the 19 cut points at ``i/20``;
    p50 is index 9 and p95 index 18 (plan §4). Fewer than :data:`MIN_PERCENTILE_SAMPLES`
    samples report ``None`` plus an explanatory note — never NaN.
    """
    samples = len(values)
    if samples < MIN_PERCENTILE_SAMPLES:
        return {
            "samples": samples,
            "p50_ms": None,
            "p95_ms": None,
            "note": f"{samples} sample(s): p50/p95 need at least {MIN_PERCENTILE_SAMPLES}",
        }
    cuts = statistics.quantiles(list(values), n=20, method="inclusive")
    return {
        "samples": samples,
        "p50_ms": round(cuts[_QUANTILE_INDEX_P50], 3),
        "p95_ms": round(cuts[_QUANTILE_INDEX_P95], 3),
        "note": None,
    }


def stage_table(records: Sequence[StageRecord]) -> dict[str, dict[str, Any]]:
    """Group ``records`` by stage: call count, failures, total ms, per-call p50/p95 and ops.

    Per-stage **data volume** (Step-0b): ``chars`` = mean per-call prompt_chars/response_chars,
    ``chars_per_s`` = derived per-call throughput (prompt+response chars over duration), only
    for records that carry them (LLM seam). ``repair_calls`` counts records marked
    ``repair=True`` (the structured-output repair re-call, second transport call of one
    ``validate_with_retries`` attempt pair).
    """
    table: dict[str, dict[str, Any]] = {}
    for record in records:
        entry = table.setdefault(
            record.stage,
            {
                "count": 0,
                "failures": 0,
                "total_ms": 0.0,
                "ops": {},
                "durations": [],
                "prompt_chars": [],
                "response_chars": [],
                "repair_calls": 0,
                "cached_tokens": [],
            },
        )
        entry["count"] += 1
        if not record.ok:
            entry["failures"] += 1
        entry["total_ms"] = round(entry["total_ms"] + record.duration_ms, 3)
        entry["ops"][record.op] = entry["ops"].get(record.op, 0) + 1
        entry["durations"].append(record.duration_ms)
        meta = record.meta
        if isinstance(meta.get("prompt_chars"), int) and meta["prompt_chars"] > 0:
            entry["prompt_chars"].append(int(meta["prompt_chars"]))
        if isinstance(meta.get("response_chars"), int) and meta["response_chars"] > 0:
            entry["response_chars"].append(int(meta["response_chars"]))
        if meta.get("repair") is True:
            entry["repair_calls"] += 1
        if isinstance(meta.get("cached_tokens"), int) and meta["cached_tokens"] is not None:
            entry["cached_tokens"].append(int(meta["cached_tokens"]))
    for entry in table.values():
        entry.update(percentile_fields(entry.pop("durations")))
        entry["prompt_chars_mean"] = round_mean_or_none(entry["prompt_chars"])
        entry.pop("prompt_chars")
        entry["response_chars_mean"] = round_mean_or_none(entry["response_chars"])
        entry.pop("response_chars")
        entry["repair_calls"] = int(entry["repair_calls"])
        entry["cached_tokens_mean"] = round_mean_or_none(entry["cached_tokens"])
        entry.pop("cached_tokens")
        entry["chars_per_s_mean"] = chars_per_s(entry)
    return table


def round_mean_or_none(values: list[int]) -> float | None:
    """Mean of non-empty int list, else ``None`` (keeps sparse stages null, plan §4 style)."""
    if not values:
        return None
    return round(sum(values) / len(values), 1)


def chars_per_s(entry: dict[str, Any]) -> float | None:
    """Derived per-stage LLM throughput: (prompt + response chars) / call seconds, mean over
    calls, ``None`` when the stage carries no LLM-seam char capture or a non-positive latency.

    Requires BOTH per-call chars present AND duration > 0 (entry fields already validated).
    """
    prompt = entry.get("prompt_chars_mean")
    response = entry.get("response_chars_mean")
    total_ms = entry.get("total_ms")
    calls = entry.get("count")
    if prompt is None and response is None:
        return None
    if not total_ms or not calls or total_ms <= 0.0 or calls <= 0:
        return None
    chars = (prompt or 0.0) + (response or 0.0)
    if chars <= 0.0:
        return None
    seconds_per_call = (total_ms / calls) / 1000.0
    return round(float(chars) / float(seconds_per_call), 1)


def request_stage_total_ms(table: dict[str, dict[str, Any]]) -> float:
    """Σ total_ms over every non-build stage (index build excluded, plan §3)."""
    return round(
        sum(float(entry["total_ms"]) for stage, entry in table.items() if stage not in BUILD_STAGES),
        3,
    )


def residual_ms(e2e_ms: float, table: dict[str, dict[str, Any]]) -> float:
    """``pipeline-residual = e2e − Σ(request stages)``: harness/loop overhead outside the seams."""
    return round(e2e_ms - request_stage_total_ms(table), 3)
