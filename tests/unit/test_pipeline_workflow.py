"""Workflow-realization tests for :class:`~src.pipeline.DesignPipeline` (E11).

The pipeline is a llama-index ``Workflow`` subclass; these tests pin the
conversion-specific observables that the existing suites do not cover: the single
``PipelineDone`` stop event across all five entry modes, the E12 loop back-edge
through the event graph, the internal ``WorkflowTimeoutError`` → ``TimeoutError``
translation, and concurrent runs on one pipeline instance.
"""

from __future__ import annotations

import asyncio
import logging

from collections.abc import Sequence

import pytest
from llama_index.core.workflow import WorkflowTimeoutError

from src.agent import DesignHints
from src.config import ServerConfig
from src.errors import (
    ERROR_CANCELLED,
    ERROR_GENERATION_FAILED,
    ERROR_LLM_PROVIDER,
    DesignPatternError,
)
from src.pipeline import (
    CancellationToken,
    DesignPipeline,
    PipelineDone,
    assemble_result,
    build_generation_seed,
)
from src.patterns.loader import PatternLoader
from src.schemas import (
    AnalysisResult,
    DesignBrief,
    DecisionPart,
    DesignEvaluation,
    DesignResult,
    GenerationSeed,
    RawBriefAndParts,
    RawBriefComponent,
    RawDecisionPart,
    RawDesignBrief,
    RawGeneration,
    RequirementWeights,
)
from tests.unit.fakes import ScriptedLLM, StubHit, StubIndex, evaluation_with


class _SlowBriefLLM(ScriptedLLM):
    """extract_brief_and_parts sleeps ``delay`` seconds, everything else scripted."""

    def __init__(self, delay: float) -> None:
        super().__init__(brief_and_parts_result=_brief_and_parts(), weights={"reliability": 1.0})
        self.delay = delay

    async def extract_brief_and_parts(
        self, description: str, hints: DesignHints, *, reasoning_context: str = ""
    ) -> RawBriefAndParts:
        await asyncio.sleep(self.delay)
        self.calls.append("extract_brief_and_parts")
        assert self.brief_and_parts_result is not None
        return self.brief_and_parts_result


class _NeverReturningLLM(ScriptedLLM):
    """Exactly one LLM method never returns; the rest are scripted normally."""

    def __init__(self, method: str) -> None:
        super().__init__(brief_and_parts_result=_brief_and_parts(), weights={"reliability": 1.0})
        self.method = method

    def _block(self) -> None:
        async def _hang() -> None:
            await asyncio.sleep(3600)

        self._task = asyncio.ensure_future(_hang())  # pragma: no cover - cancelled with the loop

    async def extract_brief_and_parts(
        self, description: str, hints: DesignHints, *, reasoning_context: str = ""
    ) -> RawBriefAndParts:
        if self.method == "extract_brief_and_parts":
            self._block()
            await self._task
        return await super().extract_brief_and_parts(description, hints, reasoning_context=reasoning_context)

    async def extract_weights_batch(
        self, parts: Sequence[DecisionPart], brief: DesignBrief, *, reasoning_context: str = ""
    ) -> dict[str, RequirementWeights]:
        if self.method == "extract_weights_batch":
            self._block()
            await self._task
        return await super().extract_weights_batch(parts, brief, reasoning_context=reasoning_context)

    async def generate(self, seed: GenerationSeed, *, reasoning_context: str = "") -> RawGeneration:
        if self.method == "generate":
            self._block()
            await self._task
        return await super().generate(seed, reasoning_context=reasoning_context)

    async def evaluate(
        self, result: DesignResult, parts: Sequence[DecisionPart], *, reasoning_context: str = ""
    ) -> DesignEvaluation:
        if self.method == "evaluate":
            self._block()
            await self._task
        return await super().evaluate(result, parts, reasoning_context=reasoning_context)


class _RecordingEvaluateLLM(ScriptedLLM):
    """Records which decision parts reach the LLM EVALUATE call."""

    def __init__(self) -> None:
        super().__init__(brief_and_parts_result=_two_part_brief_and_parts(), weights={"reliability": 1.0})
        self.recorded_part_ids: list[str] = []

    async def evaluate(
        self, result: DesignResult, parts: Sequence[DecisionPart], *, reasoning_context: str = ""
    ) -> DesignEvaluation:
        self.recorded_part_ids = [part.id for part in parts]
        return DesignEvaluation(score=100.0)


class _BatchFailLLM(ScriptedLLM):
    """First GENERATE call fails transport-style; the second hangs until cancelled."""

    def __init__(self) -> None:
        super().__init__(brief_and_parts_result=_two_part_brief_and_parts(), weights={"reliability": 1.0})
        self.generate_calls: list[int] = []
        self.sibling_cancelled = False

    async def generate(self, seed: GenerationSeed, *, reasoning_context: str = "") -> RawGeneration:
        self.generate_calls.append(len(self.generate_calls) + 1)
        if len(self.generate_calls) == 1:
            raise DesignPatternError(ERROR_LLM_PROVIDER, "scripted batch failure")
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.sibling_cancelled = True
        raise  # noqa: PLE0704 deliberate re-raise outside handler propagates cancellation

DESCRIPTION = (
    "Component GeoEnricher calls a Redis geo-lookup synchronously while enriching sensor events; "
    "enrichment must stay within the p99 latency budget and survive Redis outages."
)


def _brief_and_parts() -> RawBriefAndParts:
    return RawBriefAndParts(
        design_brief=RawDesignBrief(
            restated_goal="harden the enrichment stage",
            components=[RawBriefComponent(name="GeoEnricher", type="worker", responsibilities=["enrich events"])],
        ),
        decision_parts=[
            RawDecisionPart(
                id="enrichment-resilience",
                statement="bound the synchronous Redis lookup so enrichment survives an outage",
                component_refs=["GeoEnricher"],
                scope="task-lifetime",
                category_hypothesis="concurrency-resilience",
                priority=1.0,
            )
        ],
    )


def _pipeline(
    catalog: PatternLoader, llm: ScriptedLLM, index: StubIndex, **config: object
) -> DesignPipeline:
    server_config = ServerConfig.model_validate(config) if config else ServerConfig()
    return DesignPipeline(
        config=server_config,
        catalog=catalog,
        index=index,  # type: ignore[arg-type]
        llm=llm,
    )


async def test_run_design_golden_through_event_graph(catalog: PatternLoader) -> None:
    """run_design walks analyze → generate → evaluate → loop through the event graph."""
    llm = ScriptedLLM(
        brief_and_parts_result=_brief_and_parts(),
        weights={"reliability": 1.0},
    )
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1), StubHit("bulkhead", 0.02, 2, 2)])
    result = await _pipeline(catalog, llm, index).run_design(description=DESCRIPTION)
    assert isinstance(result.design.overview.primary_pattern, str)
    assert result.design.overview.primary_pattern  # non-empty
    assert result.attempts >= 1
    assert result.design.overview.score == pytest.approx(result.evaluation.score)
    assert "attempts=" in result.design.overview.reasoning.splitlines()[0]


async def test_loop_back_edge_fail_then_pass(catalog: PatternLoader) -> None:
    """E12 back-edge: a failing attempt 1 re-enters generate via SelectionReady for attempt 2."""
    llm = ScriptedLLM(
        brief_and_parts_result=_brief_and_parts(),
        weights={"reliability": 1.0},
        generation=RawGeneration(applications=[]),  # answers no seed → validation failure
        generate_from_seed=False,
    )
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    pipeline = _pipeline(catalog, llm, index)
    with pytest.raises(DesignPatternError):
        await pipeline.run_design(description=DESCRIPTION)
    assert llm.calls.count("generate") == ServerConfig().loop.max_tries


async def test_best_attempt_retention_through_loop(catalog: PatternLoader) -> None:
    """§11: best-scoring attempt wins even when the second attempt scores lower."""
    llm = ScriptedLLM(
        brief_and_parts_result=_brief_and_parts(),
        weights={"reliability": 1.0},
        evaluation_sequence=[DesignEvaluation(score=80.0), DesignEvaluation(score=10.0)],
    )
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    result = await _pipeline(
        catalog, llm, index, loop={"min_quality_score": 100}
    ).run_design(description=DESCRIPTION)
    assert result.attempts == 2
    assert result.evaluation.score == pytest.approx(80.0)
    assert "best attempt=1" in result.design.overview.reasoning.splitlines()[0]


async def test_internal_timeout_translates_to_timeout_error(catalog: PatternLoader) -> None:
    """The workflow's internal timeout surfaces as TimeoutError from the wrapper (ERR_014 upstream)."""

    class _NeverReturningLLM(ScriptedLLM):
        async def extract_brief_and_parts(
            self, description: str, hints: object, *, reasoning_context: str = ""
        ) -> RawBriefAndParts:
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

    llm = _NeverReturningLLM(brief_and_parts_result=_brief_and_parts())
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    pipeline = _pipeline(catalog, llm, index, tasks={"deadline_seconds": 0.05})
    with pytest.raises(TimeoutError):
        await pipeline.run_design(description=DESCRIPTION)


async def test_mode_routing_analyze_stops_after_analysis(catalog: PatternLoader) -> None:
    """analyze mode emits PipelineDone right after ANALYZE — generate is never called."""
    llm = ScriptedLLM(brief_and_parts_result=_brief_and_parts(), weights={"reliability": 1.0})
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    analysis = await _pipeline(catalog, llm, index).analyze(
        description=DESCRIPTION, architecture_style="layered", language="python"
    )
    assert isinstance(analysis, AnalysisResult)
    assert analysis.decision_parts
    assert "generate" not in llm.calls


async def test_mode_routing_generate_single_shot(catalog: PatternLoader) -> None:
    """generate mode enters mid-graph at GenerateEntry and stops after one GENERATE."""
    llm = ScriptedLLM(brief_and_parts_result=_brief_and_parts(), weights={"reliability": 1.0})
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    pipeline = _pipeline(catalog, llm, index)
    analysis = await pipeline.analyze(description=DESCRIPTION)
    result = await pipeline.generate(analysis)
    assert isinstance(result, DesignResult)
    assert result.overview.primary_pattern
    assert llm.calls.count("generate") == 1
    assert "evaluate" not in llm.calls


async def test_mode_routing_evaluate_direct(catalog: PatternLoader) -> None:
    """evaluate mode enters at EvaluateEntry (mid-graph, E11) and returns the evaluation."""
    llm = ScriptedLLM(brief_and_parts_result=_brief_and_parts(), weights={"reliability": 1.0})
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    pipeline = _pipeline(catalog, llm, index)
    analysis = await pipeline.analyze(description=DESCRIPTION)
    result = await pipeline.generate(analysis)
    evaluation = await pipeline.evaluate(result, analysis.decision_parts)
    assert isinstance(evaluation, DesignEvaluation)
    assert evaluation.score >= 0.0


async def test_mode_routing_design_loop_mid_graph(catalog: PatternLoader) -> None:
    """design_loop mode enters at SelectionReady with the passed analysis (no re-ANALYZE)."""
    llm = ScriptedLLM(brief_and_parts_result=_brief_and_parts(), weights={"reliability": 1.0})
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    pipeline = _pipeline(catalog, llm, index)
    analysis = await pipeline.analyze(description=DESCRIPTION)
    analyze_calls = llm.calls.count("extract_brief_and_parts")
    result = await pipeline.design_loop(analysis)
    assert llm.calls.count("extract_brief_and_parts") == analyze_calls
    assert result.attempts >= 1
    assert result.design.overview.primary_pattern


async def test_concurrent_runs_on_one_pipeline(catalog: PatternLoader) -> None:
    """E9b: two concurrent run_design invocations on one instance both settle."""
    llm = ScriptedLLM(brief_and_parts_result=_brief_and_parts(), weights={"reliability": 1.0})
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    pipeline = _pipeline(catalog, llm, index)
    results = await asyncio.gather(
        pipeline.run_design(description=DESCRIPTION),
        pipeline.run_design(description=DESCRIPTION),
    )
    for result in results:
        assert result.attempts >= 1
        assert result.design.overview.primary_pattern


# ---------------------------------------------------------------------------
# Mutant-killing tests for the Workflow conversion (see verify/mutmut-equivalents.md
# and the conversion plan .omp/plans/DESIGN_PIPELINE_WORKFLOW_PLAN.md).
# ---------------------------------------------------------------------------


async def test_timeout_boundary_factor(catalog: PatternLoader) -> None:
    """The Workflow timeout is tasks.deadline_seconds * 0.9 (kills factor + floor mutants).

    With deadline 1.0 s the internal timeout is 0.9 s. A 1.0 s LLM sleep survives the
    internal timeout (→ TimeoutError); any mutant producing an internal bound ≥ the sleep
    (÷0.9 → 1.11 s, ×1.9 → 1.9 s, dropped default → 45 s) lets the pipeline complete.
    """
    llm = _SlowBriefLLM(delay=1.0)
    pipeline = _pipeline(catalog, llm, StubIndex(), tasks={"deadline_seconds": 1.0})
    with pytest.raises(TimeoutError) as excinfo:
        await pipeline.run_design(description=DESCRIPTION)
    assert "timed out" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, WorkflowTimeoutError)


@pytest.mark.parametrize(
    ("wrapper", "blocking_method"),
    [
        ("analyze", "extract_brief_and_parts"),
        ("generate", "generate"),
        ("design_loop", "generate"),
        ("evaluate", "evaluate"),
    ],
)
async def test_timeout_message_wrappers(
    catalog: PatternLoader, wrapper: str, blocking_method: str
) -> None:
    """Every wrapper translates WorkflowTimeoutError → TimeoutError mentioning the timeout."""
    # p1 builds the mid-graph inputs (analysis, result) with a normally scripted LLM.
    llm_p1 = ScriptedLLM(brief_and_parts_result=_brief_and_parts(), weights={"reliability": 1.0})
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    p1 = _pipeline(catalog, llm_p1, index)
    analysis = await p1.analyze(description=DESCRIPTION)
    result = await p1.generate(analysis)

    llm_p2 = _NeverReturningLLM(blocking_method)
    p2 = _pipeline(catalog, llm_p2, index, tasks={"deadline_seconds": 0.05})
    call = getattr(p2, wrapper)
    if wrapper == "analyze":
        args: tuple[object, ...] = ()
        kwargs: dict[str, object] = {"description": DESCRIPTION}
    elif wrapper == "evaluate":
        args = (result, analysis.decision_parts)
        kwargs = {}
    else:
        args = (analysis,)
        kwargs = {}
    with pytest.raises(TimeoutError) as excinfo:
        await call(*args, **kwargs)
    assert "timed out" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, WorkflowTimeoutError)


async def test_paradigm_and_focus_drive_variant_binding(catalog: PatternLoader) -> None:
    """Variant selection honours paradigm and focus hint (kills variant-path mutants)."""
    description = (
        "Component EventBus notifies subscribers when sensor events arrive; producers must "
        "not know their consumers and subscriptions change at runtime."
    )
    part = RawDecisionPart(
        id="event-dispatch",
        statement="dispatch domain events to subscribers without coupling producers to consumers",
        component_refs=["EventBus"],
        scope="runtime",
        category_hypothesis="workflow-messaging",
        priority=1.0,
    )
    brief_and_parts = RawBriefAndParts(
        design_brief=RawDesignBrief(
            restated_goal="dispatch domain events",
            components=[RawBriefComponent(name="EventBus", type="worker", responsibilities=["dispatch"])],
        ),
        decision_parts=[part],
    )
    llm = ScriptedLLM(brief_and_parts_result=brief_and_parts, weights={"reliability": 1.0})
    index = StubIndex(hits=[StubHit("observer", 0.0317, 1, 1)])
    pipeline = _pipeline(catalog, llm, index)
    base = await pipeline.run_design(description=description, language="python")
    bound = [a.language_binding.bound_variant for a in base.design.applications]
    assert bound, "expected at least one bound variant"
    assert all(v == "callable-registry" for v in bound), bound

    functional = await pipeline.run_design(
        description=description, language="python", paradigm="functional"
    )
    assert [a.language_binding.bound_variant for a in functional.design.applications] == [
        "signal-dispatcher"
    ]

    focused = await pipeline.run_design(
        description=description,
        language="python",
        focus="prefer the signal-dispatcher variant",
    )
    assert [a.language_binding.bound_variant for a in focused.design.applications] == [
        "signal-dispatcher"
    ]


async def test_analyze_forwards_hints(catalog: PatternLoader) -> None:
    """analyze forwards style/paradigm/focus into the brief (hint wins over raw §4 rule 6)."""
    llm = ScriptedLLM(brief_and_parts_result=_brief_and_parts(), weights={"reliability": 1.0})
    pipeline = _pipeline(catalog, llm, StubIndex())
    analysis = await pipeline.analyze(
        description=DESCRIPTION,
        architecture_style="event-driven",
        paradigm="procedural",
        focus="stay with the callable-registry variant",
    )
    assert analysis.focus == "stay with the callable-registry variant"
    assert analysis.design_brief.paradigm == "procedural"
    assert analysis.design_brief.architecture_style == "event-driven"
    default = await pipeline.analyze(description=DESCRIPTION)
    assert default.focus is None


async def test_cancellation_at_stage_boundaries(catalog: PatternLoader) -> None:
    """Pre-cancelled tokens abort at the documented stage boundaries (E10)."""
    llm = ScriptedLLM(brief_and_parts_result=_brief_and_parts(), weights={"reliability": 1.0})
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    pipeline = _pipeline(catalog, llm, index)
    token = CancellationToken()
    token.cancel()
    with pytest.raises(DesignPatternError) as excinfo:
        await pipeline.analyze(description=DESCRIPTION, cancellation=token)
    assert excinfo.value.code == ERROR_CANCELLED
    assert "cancelled at stage boundary: stage B" in excinfo.value.detail
    assert llm.calls.count("extract_brief_and_parts") == 1

    analysis = await pipeline.analyze(description=DESCRIPTION)

    generate_token = CancellationToken()
    generate_token.cancel()
    with pytest.raises(DesignPatternError) as excinfo:
        await pipeline.generate(analysis, cancellation=generate_token)
    assert excinfo.value.code == ERROR_CANCELLED
    assert "stage G" in excinfo.value.detail
    assert "generate" not in llm.calls

    loop_token = CancellationToken()
    loop_token.cancel()
    with pytest.raises(DesignPatternError) as excinfo:
        await pipeline.design_loop(analysis, cancellation=loop_token)
    assert excinfo.value.code == ERROR_CANCELLED
    assert "cancelled at stage boundary" in excinfo.value.detail  # boundary is "attempt 2"


def _two_part_brief_and_parts() -> RawBriefAndParts:
    """Two decision parts so GENERATE fans out into two batches (batch_size=1)."""
    return RawBriefAndParts(
        design_brief=RawDesignBrief(
            restated_goal="cache repeated lookups",
            components=[RawBriefComponent(name="GeoEnricher", type="worker", responsibilities=["enrich"])],
        ),
        decision_parts=[
            RawDecisionPart(
                id="enrichment-resilience",
                statement="bound the synchronous Redis lookup so enrichment survives an outage",
                component_refs=["GeoEnricher"],
                scope="task-lifetime",
                category_hypothesis="concurrency-resilience",
                priority=1.0,
            ),
            RawDecisionPart(
                id="lookup-latency",
                statement="cache repeated geo lookups so repeated enrichments stay within budget",
                component_refs=["GeoEnricher"],
                scope="task-lifetime",
                category_hypothesis="persistence-state",
                priority=0.9,
            ),
        ],
    )


async def test_evaluate_forwards_decision_parts(catalog: PatternLoader) -> None:
    """Direct evaluate forwards the caller's parts to the LLM (kills parts→None mutant)."""
    llm = _RecordingEvaluateLLM()
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1), StubHit("cache-aside", 0.03, 2, 2)])
    pipeline = _pipeline(catalog, llm, index)
    analysis = await pipeline.analyze(description=DESCRIPTION)
    result = await pipeline.generate(analysis)
    evaluation = await pipeline.evaluate(result, analysis.decision_parts)
    assert isinstance(evaluation, DesignEvaluation)
    assert evaluation.score == 100.0
    assert llm.recorded_part_ids == ["enrichment-resilience", "lookup-latency"]


async def test_batch_failure_cancels_siblings_and_reraises(catalog: PatternLoader) -> None:
    """A failing GENERATE batch cancels in-flight siblings before re-raising."""
    llm = _BatchFailLLM()
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1), StubHit("cache-aside", 0.03, 2, 2)])
    pipeline = _pipeline(
        catalog,
        llm,
        index,
        generator={"config": {"generation_batch_size": 1}},
    )
    analysis = await pipeline.analyze(description=DESCRIPTION)
    with pytest.raises(DesignPatternError) as excinfo:
        await pipeline.generate(analysis)
    assert excinfo.value.code == ERROR_LLM_PROVIDER
    assert llm.generate_calls == [1, 2]
    assert llm.sibling_cancelled is True


async def test_assemble_result_validation_error_code(catalog: PatternLoader) -> None:
    """Empty raw generation surfaces ERR_004 with the validation prefix (kills assemble mutants)."""
    llm = ScriptedLLM(brief_and_parts_result=_brief_and_parts(), weights={"reliability": 1.0})
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    pipeline = _pipeline(catalog, llm, index)
    analysis = await pipeline.analyze(description=DESCRIPTION)
    composed = pipeline._compose(analysis)  # noqa: SLF001 - ruff SLF family not selected
    seed = build_generation_seed(
        composed=composed,
        selections=analysis.selections,
        brief=analysis.design_brief,
        catalog=catalog,
        focus=analysis.focus,
        technology_tokens=analysis.technology_tokens,
    )
    with pytest.raises(DesignPatternError) as excinfo:
        assemble_result(
            seed=seed,
            raw=RawGeneration(applications=[]),
            composed=composed,
            brief=analysis.design_brief,
            catalog=catalog,
            config=ServerConfig(),
        )
    assert excinfo.value.code == ERROR_GENERATION_FAILED
    assert excinfo.value.detail.startswith("GENERATE output failed validation: ")
    assert "no generated application for this seed" in excinfo.value.detail


async def test_pipeline_done_mode_tag(catalog: PatternLoader) -> None:
    """PipelineDone payload_as enforces the declared payload type per mode."""
    analysis_done = PipelineDone(
        result=AnalysisResult(design_brief=DesignBrief(restated_goal="x")), mode="analyze"
    )
    assert analysis_done.mode == "analyze"
    assert analysis_done.payload_as(AnalysisResult) is analysis_done._get_result()  # noqa: SLF001

    evaluation_done = PipelineDone(result=DesignEvaluation(score=100.0), mode="generate")
    with pytest.raises(AssertionError) as excinfo:
        evaluation_done.payload_as(AnalysisResult)
    message = str(excinfo.value)
    assert "DesignEvaluation" in message
    assert "AnalysisResult" in message


async def test_finish_header_and_findings_exact(catalog: PatternLoader) -> None:
    """_finish renders the exact attempts header and per-attempt finding lines."""
    llm = ScriptedLLM(
        brief_and_parts_result=_brief_and_parts(),
        weights={"reliability": 1.0},
        generation_sequence=[RawGeneration(applications=[])],
    )
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    pipeline = _pipeline(catalog, llm, index, loop={"min_quality_score": 100})
    result = await pipeline.run_design(description=DESCRIPTION)
    lines = result.design.overview.reasoning.splitlines()
    assert lines[0] == (
        "attempts=2 (max_tries=2), best attempt=2 (score 100.0, min_quality_score 100), reselection=no"
    )
    assert lines[1].startswith("1: failed error GENERATE output failed validation: ")
    assert lines[1].endswith("enrichment-resilience-timeout: no generated application for this seed")
    assert lines[2] == "2: valid score 100.0"


async def test_finish_log_record(caplog: pytest.LogCaptureFixture, catalog: PatternLoader) -> None:
    """_finish logs the attempts log as structured extras on 'Selection run completed'."""
    llm = ScriptedLLM(
        brief_and_parts_result=_brief_and_parts(),
        weights={"reliability": 1.0},
        generation_sequence=[RawGeneration(applications=[])],
    )
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])
    pipeline = _pipeline(catalog, llm, index, loop={"min_quality_score": 100})
    with caplog.at_level(logging.INFO, logger="src.pipeline"):
        await pipeline.run_design(description=DESCRIPTION)
    record = next(r for r in caplog.records if r.getMessage() == "Selection run completed")
    assert record.attempts[0]["status"] == "failed"
    assert record.attempts[0]["failure_kind"] is None
    assert record.attempts[1]["status"] == "valid"
    assert record.attempts[1]["score"] == 100.0
    assert record.best_score == 100.0
    assert record.reselected is False


async def test_reselection_header_reports_yes(catalog: PatternLoader) -> None:
    """E12 reselection flips the reselection token in the overview reasoning header."""
    llm = ScriptedLLM(
        brief_and_parts_result=_brief_and_parts(),
        weights={"reliability": 1.0},
        evaluation_sequence=[
            evaluation_with("timeout", score=20.0),
            DesignEvaluation(score=60.0),
        ],
    )
    hits = [StubHit("timeout", 0.0317, 1, 1), StubHit("retry-with-backoff", 0.0300, 2, 2)]
    result = await _pipeline(catalog, llm, index=StubIndex(hits=hits), loop={"min_quality_score": 100}).run_design(
        description=DESCRIPTION
    )
    assert result.design.overview.primary_pattern == "retry-with-backoff"
    assert "reselection=yes" in result.design.overview.reasoning


async def test_default_guard_keeps_catalogued_part(catalog: PatternLoader) -> None:
    """Default anti-scope guard keeps a catalogued part and fills its category (init mutants)."""
    part = RawDecisionPart(
        id="legacy-isolation",
        statement="introduce microservices and put an ACL in front of the legacy inventory",
        component_refs=[],
        scope="task-lifetime",
        category_hypothesis=None,
        priority=1.0,
    )
    brief_and_parts = RawBriefAndParts(
        design_brief=RawDesignBrief(
            restated_goal="isolate the legacy inventory",
            components=[
                RawBriefComponent(name="InventoryFacade", type="boundary", responsibilities=["isolate"])
            ],
        ),
        decision_parts=[part],
    )
    llm = ScriptedLLM(brief_and_parts_result=brief_and_parts, weights={"reliability": 1.0})
    index = StubIndex(hits=[StubHit("anti-corruption-layer", 0.0317, 1, 1)])
    pipeline = _pipeline(catalog, llm, index)
    analysis = await pipeline.analyze(description=DESCRIPTION)
    assert analysis.decision_parts[0].id == "legacy-isolation"
    acl = catalog.get_by_name("anti-corruption-layer")
    assert acl is not None
    assert analysis.decision_parts[0].category_hypothesis == acl.category
