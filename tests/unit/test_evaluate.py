"""``evaluate_design`` tool surface: scoring contract, gates and advisory integrity findings.

A design drafted by the generate path scores cleanly (consistent design, empty integrity);
bare and malformed payloads hit the documented ERR_004/ERR_002 gates; an injected conflict
pair surfaces as an advisory finding without failing the call.
"""

from __future__ import annotations

import copy

import pytest
from fastmcp.exceptions import ToolError

from src.config import ServerConfig
from src.pipeline import DesignPipeline
from src.tools.evaluate import EvaluateDesignTool
from src.tools.generate import GenerateDesignTool
from src.schemas import DesignEvaluation, RawBriefAndParts, RawDecisionPart, RawDesignBrief
from src.schemas.enums import QUALITY_AXES
from tests.unit.fakes import ScriptedLLM, StubHit, StubIndex

DESCRIPTION = (
    "Component GeoEnricher calls a Redis geo-lookup synchronously while enriching sensor events; "
    "enrichment must stay within the p99 latency budget and survive Redis outages."
)

AXIS_SCORES = {
    "testability": 75.0,
    "extensibility": 70.0,
    "maintainability": 72.0,
    "simplicity": 80.0,
    "performance": 71.0,
    "reliability": 90.0,
    "security": 68.0,
    "scalability": 66.0,
}


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
        evaluation=DesignEvaluation(score=88.0, axis_scores=dict(AXIS_SCORES)),
    )


def _tools(catalog) -> tuple[GenerateDesignTool, EvaluateDesignTool]:
    pipeline = DesignPipeline(
        config=ServerConfig(),
        catalog=catalog,
        index=StubIndex(hits=[StubHit(name="timeout", retrieval_score=5.0)]),  # type: ignore[arg-type]
        llm=_scripted_llm(),
    )
    return (
        GenerateDesignTool(pipeline=pipeline, pattern_loader=catalog),
        EvaluateDesignTool(pipeline=pipeline),
    )


async def test_generated_design_scores_with_the_documented_contract(catalog) -> None:
    """Happy path: 0-100 score, summary agreement, per-axis metrics, no integrity findings."""
    generate, evaluate = _tools(catalog)

    design = await generate.generate(patterns=["timeout"], description=DESCRIPTION)
    result = await evaluate.evaluate(design_json=design, requirements=DESCRIPTION)

    assert 0.0 <= result["score"] <= 100.0
    assert result["summary"]["overall_score"] == result["score"]
    assert set(result["metrics"]) <= set(QUALITY_AXES)
    assert result["integrity_findings"] == []


async def test_design_without_applications_is_rejected(catalog) -> None:
    """ERR_004: a bare overview with no application is not enough to evaluate."""
    _, evaluate = _tools(catalog)

    with pytest.raises(ToolError, match="ERR_004"):
        await evaluate.evaluate(
            design_json={
                "overview": {
                    "reasoning": "a bare overview without applications",
                    "primary_pattern": "timeout",
                    "category": "concurrency-resilience",
                },
                "applications": [],
            },
            requirements=DESCRIPTION,
        )


async def test_malformed_design_payload_is_rejected(catalog) -> None:
    """ERR_002: a payload without an overview object is malformed at the boundary."""
    _, evaluate = _tools(catalog)

    with pytest.raises(ToolError, match="ERR_002"):
        await evaluate.evaluate(design_json={"applications": []}, requirements=DESCRIPTION)


async def test_conflicting_pair_is_reported_as_advisory_integrity(catalog) -> None:
    """An injected conflicting pair is flagged but never fails the request."""
    generate, evaluate = _tools(catalog)

    design = await generate.generate(patterns=["timeout"], description=DESCRIPTION)
    conflicted = copy.deepcopy(design)
    conflicted["applications"].append(
        {
            "id": "app-di",
            "pattern": "dependency-injection",
            "name": "wiring for GeoEnricher",
            "problem": "components hard-wire their collaborators",
            "solution": "inject the collaborators",
        }
    )
    conflicted["applications"].append(
        {
            "id": "app-singleton",
            "pattern": "singleton",
            "name": "shared state for GeoEnricher",
            "problem": "state must be shared globally",
            "solution": "one shared instance",
        }
    )

    result = await evaluate.evaluate(design_json=conflicted, requirements=DESCRIPTION)

    findings = "\n".join(result["integrity_findings"])
    assert "conflicting pattern pair" in findings
    assert "dependency-injection" in findings and "singleton" in findings
