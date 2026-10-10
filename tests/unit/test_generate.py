"""``generate_design`` tool surface: both input paths and the ERR_003 rejections.
The analysis path feeds a real pipeline ``analyze`` payload; the explicit path names catalog
patterns directly. Unknown names, a missing input and a missing description are the documented
``ERR_003`` failures.
"""

from __future__ import annotations

import asyncio

import pytest
from fastmcp.exceptions import ToolError

from src.errors import DesignPatternError
from src.config import ServerConfig
from src.pipeline import DesignPipeline
from src.tools.analyze import AnalyzeDesignTool
from src.tools.generate import GenerateDesignTool
from src.schemas import RawBriefAndParts, RawDecisionPart, RawDesignBrief
from tests.unit.fakes import ScriptedLLM, StubHit, StubIndex

DESCRIPTION = (
    "Component GeoEnricher calls a Redis geo-lookup synchronously while enriching sensor events; "
    "enrichment must stay within the p99 latency budget and survive Redis outages."
)


def _scripted_llm() -> ScriptedLLM:
    return ScriptedLLM(
        brief_and_parts_result=RawBriefAndParts(
            design_brief=RawDesignBrief(
                restated_goal="harden the enrichment stage",
                components=[
                    {"name": "GeoEnricher", "type": "worker", "responsibilities": ["enrich events"]}
                ],
                constraints=["p99 < 50 ms per event"],
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
        ),
        weights={"reliability": 1.0},
    )


def _tools(catalog) -> tuple[AnalyzeDesignTool, GenerateDesignTool]:
    pipeline = DesignPipeline(
        config=ServerConfig(),
        catalog=catalog,
        index=StubIndex(hits=[StubHit(name="timeout", retrieval_score=5.0)]),  # type: ignore[arg-type]
        llm=_scripted_llm(),
    )
    return AnalyzeDesignTool(pipeline=pipeline), GenerateDesignTool(
        pipeline=pipeline, pattern_loader=catalog
    )


async def test_generate_from_an_analysis_payload_materializes_the_primary(catalog) -> None:
    """Analysis path: the application realizes the selected primary, bound to the language."""
    analyze, generate = _tools(catalog)

    analysis = await analyze.analyze(DESCRIPTION, language="python")
    design = await generate.generate(analysis=analysis)

    assert design["applications"], "the design carries at least one application"
    application = design["applications"][0]
    assert application["pattern"] == analysis["selections"]["enrichment-resilience"]["primary"]
    assert application["language_binding"]["language"] == "python"


async def test_explicit_patterns_materialize_without_an_analysis(catalog) -> None:
    """Explicit path: named patterns become applications directly."""
    _, generate = _tools(catalog)

    design = await generate.generate(patterns=["timeout"], description=DESCRIPTION)

    assert design["overview"]["primary_pattern"] == "timeout"
    assert [application["pattern"] for application in design["applications"]] == ["timeout"]


async def test_unknown_pattern_name_is_a_structured_error(catalog) -> None:
    """An unresolvable pattern name is rejected with ERR_003."""
    _, generate = _tools(catalog)

    with pytest.raises(ToolError, match="ERR_003"):
        await generate.generate(patterns=["no-such-pattern-xyz"], description=DESCRIPTION)


async def test_missing_input_is_a_structured_error(catalog) -> None:
    """Neither an analysis payload nor explicit patterns is rejected with ERR_003."""
    _, generate = _tools(catalog)

    with pytest.raises(ToolError, match="ERR_003"):
        await generate.generate()


async def test_patterns_without_description_is_a_structured_error(catalog) -> None:
    """The explicit path requires the system description; its absence is ERR_003."""
    _, generate = _tools(catalog)

    with pytest.raises(ToolError, match="ERR_003"):
        await generate.generate(patterns=["timeout"])


# --------------------------------------------------------------------------------------
# GENERATE batching (§11 item 1 + generator.config.generation_batch_size)
# --------------------------------------------------------------------------------------


def _analysis_with_applications(catalog, count: int):
    """An analysis whose composition yields exactly ``count`` applications (one per part)."""
    from src.schemas import (
        AnalysisResult,
        DecisionPart,
        DesignBrief,
        PartSelection,
    )

    names = list(catalog.records)[:count]
    parts = []
    selections = {}
    for index, name in enumerate(names):
        part = DecisionPart(
            id=f"part-{index}",
            statement=f"realize {name} for component {index}",
            component_refs=[f"Component{index}"],
            scope="module",
            category_hypothesis=catalog.records[name].category,  # type: ignore[arg-type]
            priority=1.0 - index / 100,
        )
        parts.append(part)
        selections[part.id] = PartSelection(
            part=part,
            primary=name,
            primary_category=catalog.records[name].category,  # type: ignore[arg-type]
        )
    return AnalysisResult(
        design_brief=DesignBrief(
            restated_goal="realize twelve applications",
            components=[
                {"name": f"Component{index}", "type": "worker", "responsibilities": ["work"]}
                for index in range(count)
            ],
        ),
        decision_parts=parts,
        selections=selections,
    )


class _RecordingLLM(ScriptedLLM):
    """ScriptedLLM that records every seed it was handed, in call order."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003 - test double
        super().__init__(**kwargs)
        self.seeds = []

    async def generate(self, seed, *, reasoning_context: str = ""):  # noqa: ANN001, ANN201
        from src.schemas import RawGeneration

        self.seeds.append(seed)
        if self.fail_on_call is not None and len(self.seeds) == self.fail_on_call:
            from src.errors import LLMError

            raise LLMError(
                provider="minimax",
                model="minimax/MiniMax-M2.7",
                failure_kind="timeout",
                provider_message="LLM provider call timed out after 120s",
                extra={"operation": "generate"},
            )
        from tests.unit.fakes import generation_for_seeds

        return generation_for_seeds(seed) if self.generate_from_seed else RawGeneration()


def _pipeline_with(catalog, llm, *, batch_size: int = 5):
    from src.pipeline import DesignPipeline

    return DesignPipeline(
        config=ServerConfig.model_validate(
            {"generator": {"config": {"generation_batch_size": batch_size}}}
        ),
        catalog=catalog,
        index=StubIndex(hits=[StubHit(name="timeout", retrieval_score=5.0)]),  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
    )


async def test_twelve_applications_are_generated_in_three_ordered_batches(catalog) -> None:
    """generation_batch_size=5 splits 12 applications into 5 + 5 + 2 sequential calls."""
    llm = _RecordingLLM()
    pipeline = _pipeline_with(catalog, llm, batch_size=5)
    analysis = _analysis_with_applications(catalog, 12)

    design = await pipeline.generate(analysis)

    assert [len(seed.applications) for seed in llm.seeds] == [5, 5, 2]
    ordered_ids = [application.id for seed in llm.seeds for application in seed.applications]
    assert ordered_ids == [application.id for application in design.applications]
    assert len(design.applications) == 12, "every application reaches assemble_result"
    assert len({application.id for application in design.applications}) == 12
    for seed in llm.seeds:
        referenced = {application.pattern for application in seed.applications}
        assert set(seed.catalog_slices) == referenced, "a batch carries only its own catalog slices"


async def test_batch_order_is_the_input_order(catalog) -> None:
    """The concatenated RawGeneration follows the seed's input order, not the call order."""
    llm = _RecordingLLM()
    pipeline = _pipeline_with(catalog, llm, batch_size=5)
    analysis = _analysis_with_applications(catalog, 12)

    design = await pipeline.generate(analysis)

    assert [application.id for application in design.applications] == [
        application.id for seed in llm.seeds for application in seed.applications
    ]


async def test_batches_run_bounded_concurrent_and_concatenate_in_order(catalog) -> None:
    """At most ``generation_max_concurrency`` batches are in flight; output is batch-ordered."""
    inflight = 0
    peak = 0

    class _SlowLLM(_RecordingLLM):
        async def generate(self, seed, *, reasoning_context: str = ""):  # noqa: ANN001, ANN201
            nonlocal inflight, peak
            inflight += 1
            peak = max(peak, inflight)
            try:
                await asyncio.sleep(0.01)  # yield so admitted batches overlap
                return await super().generate(seed, reasoning_context=reasoning_context)
            finally:
                inflight -= 1

    llm = _SlowLLM()
    pipeline = DesignPipeline(
        config=ServerConfig.model_validate(
            {"generator": {"config": {"generation_batch_size": 5, "generation_max_concurrency": 2}}}
        ),
        catalog=catalog,
        index=StubIndex(hits=[StubHit(name="timeout", retrieval_score=5.0)]),  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
    )
    analysis = _analysis_with_applications(catalog, 12)

    design = await pipeline.generate(analysis)

    assert peak == 2, "two batches must overlap under the cap"
    assert [application.id for application in design.applications] == [
        application.id for seed in llm.seeds for application in seed.applications
    ], "concatenation follows batch order regardless of completion order"
    assert [len(seed.applications) for seed in llm.seeds] == [5, 5, 2]


async def test_batch_failure_cancels_inflight_siblings(catalog) -> None:
    """A failing batch propagates ERR_009 and no generate() call outlives the gather."""
    entered: set[int] = set()
    cancelled: list[bool] = []

    class _HangingThenFailingLLM(_RecordingLLM):
        async def generate(self, seed, *, reasoning_context: str = ""):  # noqa: ANN001, ANN201
            batch_index = len(self.seeds)
            self.seeds.append(seed)
            entered.add(batch_index)
            if batch_index == 0:
                await asyncio.sleep(0.01)  # let the sibling suspend in Event().wait() first
                from src.errors import LLMError

                raise LLMError(
                    provider="minimax",
                    model="minimax/MiniMax-M2.7",
                    failure_kind="timeout",
                    provider_message="LLM provider call timed out after 120s",
                    extra={"operation": "generate"},
                )
            try:
                await asyncio.Event().wait()  # hang until gather cancels us
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            from src.schemas import RawGeneration

            return RawGeneration()

    llm = _HangingThenFailingLLM()
    pipeline = DesignPipeline(
        config=ServerConfig.model_validate(
            {"generator": {"config": {"generation_batch_size": 5, "generation_max_concurrency": 2}}}
        ),
        catalog=catalog,
        index=StubIndex(hits=[StubHit(name="timeout", retrieval_score=5.0)]),  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
    )
    analysis = _analysis_with_applications(catalog, 12)

    with pytest.raises(DesignPatternError) as exc_info:
        await pipeline.generate(analysis)

    assert exc_info.value.code == "ERR_009"
    assert cancelled, "the hanging sibling batch must be cancelled, not left in flight"


async def test_empty_application_list_keeps_a_single_call(catalog) -> None:
    """The empty-application behavior is unchanged: one call with the seed as handed."""
    from src.schemas import GenerationSeed

    llm = _RecordingLLM()
    pipeline = _pipeline_with(catalog, llm, batch_size=5)

    raw = await pipeline._generate_in_batches(
        GenerationSeed(restated_goal="nothing to realize"), reasoning_context=""
    )

    assert raw.applications == []
    assert len(llm.seeds) == 1
    assert llm.seeds[0].applications == []
