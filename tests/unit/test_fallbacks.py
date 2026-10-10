"""§12 — fallback and error paths E3, E5, E6, E12 plus the §11 loop rules.

Each test drives the real pipeline with the injected seams: a scripted LLM decides what the model
"said", a stub index decides what retrieval "found", and the assertions read the documented
fallback semantics out of the result.
"""

from __future__ import annotations

import asyncio

import pytest

from src.config import ServerConfig
from src.errors import ERROR_GENERATION_FAILED, DesignPatternError
from src.reasoning.schemas import ReasoningTrace
from src.tools.design import pipeline_result_to_output
from src.pipeline import DesignPipeline
from src.schemas import (
    DesignEvaluation,
    RawBriefAndParts,
    RawDecisionPart,
    RawDesignBrief,
    RawGeneration,
)
from tests.unit.fakes import ScriptedLLM, StubHit, StubIndex, evaluation_with


def _brief_and_parts(*parts: RawDecisionPart) -> RawBriefAndParts:
    return RawBriefAndParts(
        design_brief=RawDesignBrief(
            restated_goal="harden the enrichment stage",
            components=[{"name": "GeoEnricher", "type": "worker", "responsibilities": ["enrich events"]}],
        ),
        decision_parts=list(parts),
    )


DESCRIPTION = (
    "Component GeoEnricher calls a Redis geo-lookup synchronously while enriching sensor events; "
    "enrichment must stay within the p99 latency budget and survive Redis outages."
)


def _part(part_id: str = "enrichment-resilience", category: str = "concurrency-resilience") -> RawDecisionPart:
    return RawDecisionPart(
        id=part_id,
        statement="bound the synchronous Redis lookup so enrichment survives an outage",
        component_refs=["GeoEnricher"],
        scope="task-lifetime",
        category_hypothesis=category,
        priority=1.0,
    )


def _pipeline(catalog, llm, index, *, config: ServerConfig | None = None) -> DesignPipeline:
    return DesignPipeline(
        config=config or ServerConfig(),
        catalog=catalog,
        index=index,  # type: ignore[arg-type]
        llm=llm,
    )


async def test_e3_minimal_brief_on_double_llm_failure(catalog) -> None:
    """E3: a failing brief call degrades to one whole-description part and still selects."""
    llm = ScriptedLLM(
        fail_brief=True,
        weights={
            "reliability": 1.0, "performance": 0.8, "simplicity": 0.6, "testability": 0.5,
            "maintainability": 0.4, "scalability": 0.3, "extensibility": 0.2, "security": 0.2,
        },
    )
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1), StubHit("bulkhead", 0.02, 2, 2)])
    result = pipeline_result_to_output(await _pipeline(catalog, llm, index).run_design(description=DESCRIPTION))
    assert llm.calls[0] == "extract_brief_and_parts"
    # the fallback part is the whole description in module scope with no category hypothesis
    application_ids = [application["id"] for application in result["applications"]]
    assert application_ids == ["whole-description-timeout"]
    assert "whole-description" in result["overview"]["reasoning"]


async def test_e5_empty_retrieval_pool_falls_back_and_flags(catalog) -> None:
    """E5: an empty pool yields the §12 category fallback, flagged in the reasoning trace."""
    llm = ScriptedLLM(brief_and_parts_result=_brief_and_parts(_part()), weights={"reliability": 1.0})
    result = pipeline_result_to_output(await _pipeline(catalog, llm, StubIndex(hits=[])).run_design(description=DESCRIPTION))
    assert result["applications"][0]["pattern"] == "timeout"  # §12 fallback for concurrency-resilience
    assert "fallback primary" in result["overview"]["reasoning"]
    assert result["overview"]["supporting_patterns"] == []


async def test_e12_generation_failure_propagates(catalog) -> None:
    """E12: when every attempt produces an invalid payload, the last error propagates."""
    llm = ScriptedLLM(
        brief_and_parts_result=_brief_and_parts(_part()),
        weights={"reliability": 1.0},
        generation=RawGeneration(applications=[]),  # answers no seed → validation failure
        generate_from_seed=False,
    )
    index = StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1), StubHit("bulkhead", 0.02, 2, 2)])
    with pytest.raises(DesignPatternError) as excinfo:
        await _pipeline(catalog, llm, index).run_design(description=DESCRIPTION)
    assert excinfo.value.code == ERROR_GENERATION_FAILED
    assert llm.calls.count("generate") == ServerConfig().loop.max_tries


async def test_min_quality_score_early_stop(catalog) -> None:
    """§11 item 3: a first attempt scoring ≥ ``min_quality_score`` stops the loop."""
    llm = ScriptedLLM(
        brief_and_parts_result=_brief_and_parts(_part()),
        weights={"reliability": 1.0},
        evaluation=DesignEvaluation(score=50.0),
    )

    llm = ScriptedLLM(
        brief_and_parts_result=_brief_and_parts(_part()),
        weights={"reliability": 1.0},
        evaluation=DesignEvaluation(score=50.0),
    )
    result = pipeline_result_to_output(await _pipeline(catalog, llm, StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])).run_design(
        description=DESCRIPTION
    ))
    assert llm.calls.count("generate") == 1
    assert result["overview"]["score"] == pytest.approx(50.0)


async def test_best_attempt_is_retained_not_the_last(catalog) -> None:
    """§11 item 3: the highest-scoring valid attempt wins, even when a later one scores lower."""
    llm = ScriptedLLM(
        brief_and_parts_result=_brief_and_parts(_part()),
        weights={"reliability": 1.0},
        evaluation_sequence=[DesignEvaluation(score=80.0), DesignEvaluation(score=10.0)],
    )
    config = ServerConfig.model_validate({"loop": {"min_quality_score": 100}})
    result = pipeline_result_to_output(await _pipeline(catalog, llm, StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)]), config=config).run_design(
        description=DESCRIPTION
    ))
    assert llm.calls.count("generate") == 2
    assert result["overview"]["score"] == pytest.approx(80.0)
    assert "best attempt=1" in result["overview"]["reasoning"]


async def test_reselection_promotes_the_next_ranked_candidate(catalog) -> None:
    """§11 item 4: a selection defect re-enters Stage E once, with sticky exclusions."""
    llm = ScriptedLLM(
        brief_and_parts_result=_brief_and_parts(_part()),
        weights={"reliability": 1.0},
        evaluation_sequence=[
            evaluation_with("timeout", score=20.0),
            DesignEvaluation(score=60.0),
        ],
    )
    config = ServerConfig.model_validate({"loop": {"min_quality_score": 100}})
    hits = [StubHit("timeout", 0.0317, 1, 1), StubHit("retry-with-backoff", 0.0300, 2, 2)]
    result = pipeline_result_to_output(await _pipeline(catalog, llm, StubIndex(hits=hits), config=config).run_design(description=DESCRIPTION))
    assert result["applications"][0]["pattern"] == "retry-with-backoff"
    assert "sticky exclusions" in result["overview"]["reasoning"]


async def test_cancellation_stops_at_the_next_boundary(catalog) -> None:
    """E11: a cancelled token aborts the run with the documented code before any generation."""
    from src.pipeline import CancellationToken

    llm = ScriptedLLM(brief_and_parts_result=_brief_and_parts(_part()), weights={"reliability": 1.0})
    token = CancellationToken()
    token.cancel()
    with pytest.raises(DesignPatternError) as excinfo:
        await _pipeline(catalog, llm, StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)])).run_design(
            description=DESCRIPTION, cancellation=token
        )
    assert excinfo.value.code == "ERR_012"
    assert "generate" not in llm.calls


async def test_too_many_boundary_parts_is_rejected(catalog) -> None:
    """§5 rule 2: unmergeable contract-driven parts raise the documented error."""
    from src.pipeline import repair_parts

    raw_parts = [
        RawDecisionPart(
            id=f"boundary-{index}",
            statement=f"honour contract obligation {index}",
            scope="module",
            category_hypothesis="interaction-boundary",
            priority=1.0,
            is_contract_part=True,
        )
        for index in range(3)
    ]
    with pytest.raises(DesignPatternError) as excinfo:
        repair_parts(
            raw_parts,
            max_parts=2,
            scopes=catalog.enums.scopes,
            categories=catalog.enums.categories,
            inherited_negative_signals=[],
        )
    assert excinfo.value.code == "ERR_005"
    assert len(excinfo.value.extra["obligations"]) == 3


async def test_merge_repair_combines_same_category_and_scope(catalog) -> None:
    """§5 rule 2: over-limit parts sharing category and scope merge instead of failing."""
    from src.pipeline import repair_parts

    raw_parts = [
        RawDecisionPart(
            id=f"part-{index}",
            statement=f"decision {index} for the same surface",
            scope="module",
            category_hypothesis="behavior-selection",
            priority=0.5 - index * 0.1,
        )
        for index in range(3)
    ]
    parts = repair_parts(
        raw_parts,
        max_parts=2,
        scopes=catalog.enums.scopes,
        categories=catalog.enums.categories,
        inherited_negative_signals=[],
    )
    assert len(parts) == 2
    merged = next(part for part in parts if ";" in part.statement)
    assert merged.statement.count(";") == 1


async def test_e9_architecture_only_part_is_replaced_by_a_construction_part(catalog) -> None:
    """E9 / §5 rule 5: an architecture-level-only brief still yields one implementable part."""
    llm = ScriptedLLM(
        brief_and_parts_result=RawBriefAndParts(
            design_brief=RawDesignBrief(
                restated_goal="split the system into services",
                components=[{"name": "IngestPipeline", "type": "service", "responsibilities": []}],
            ),
            decision_parts=[
                RawDecisionPart(
                    id="architecture-choice",
                    statement="use microservices for the ingest pipeline",
                    scope="boundary",
                    category_hypothesis="interaction-boundary",
                    priority=1.0,
                )
            ],
        ),
        weights={"reliability": 1.0},
    )
    index = StubIndex(hits=[StubHit("dependency-injection", 0.0317, 1, 1)])
    result = pipeline_result_to_output(await _pipeline(catalog, llm, index).run_design(description=DESCRIPTION))
    application = result["applications"][0]
    assert application["id"] == "construct-ingestpipeline-dependency-injection"
    assert application["category"] == "construction-composition"
    assert "1 decision part(s) decomposed" in result["overview"]["reasoning"]


class _CancellingReasoning:
    """Reasoning seam cancelled mid-phase, as ``cancel_design``/a disconnect does."""

    enabled = True

    async def run_pre_llm(self, phase: str, payload: dict[str, object]) -> ReasoningTrace:
        raise asyncio.CancelledError


async def test_cancellation_is_not_degraded_by_the_reasoning_block(catalog) -> None:
    """§12: the degradation contract covers *failures*; a cancellation must propagate.

    Swallowing it (the old UnboundLocalError path) let a cancelled job keep running the
    phase without a trace instead of stopping the task.
    """
    pipeline = DesignPipeline(
        config=ServerConfig(),
        catalog=catalog,
        index=StubIndex(hits=[StubHit("timeout", 0.0317, 1, 1)]),
        llm=ScriptedLLM(),
        reasoning=_CancellingReasoning(),  # type: ignore[arg-type]
    )

    with pytest.raises(asyncio.CancelledError):
        await pipeline._reasoning_block("analyze", {"part": "p1"})
