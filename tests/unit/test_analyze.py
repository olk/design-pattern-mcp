"""``analyze_design`` tool surface: ANALYZE over the real catalog with scripted doubles.

Each test drives the tool's handler function directly (``await tool.analyze(...)``) with the
injected seams — a scripted LLM decides what the model "said", a stub index decides what
retrieval "found" — and asserts the documented payload and error semantics.
"""

from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError

from src.config import ServerConfig
from src.pipeline import DesignPipeline
from src.tools.analyze import AnalyzeDesignTool
from src.schemas import RawBriefAndParts, RawDecisionPart, RawDesignBrief
from tests.unit.fakes import ScriptedLLM, StubHit, StubIndex

PART_ID = "enrichment-resilience"
PART_CATEGORY = "concurrency-resilience"

DESCRIPTION = (
    "Component GeoEnricher calls a Redis geo-lookup synchronously while enriching sensor events; "
    "enrichment must stay within the p99 latency budget and survive Redis outages."
)


def _part() -> RawDecisionPart:
    return RawDecisionPart(
        id=PART_ID,
        statement="bound the synchronous Redis lookup so enrichment survives an outage",
        component_refs=["GeoEnricher"],
        scope="task-lifetime",
        category_hypothesis=PART_CATEGORY,
        priority=1.0,
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
            decision_parts=[_part()],
        ),
        weights={"reliability": 1.0},
    )


def _tool(catalog, index: StubIndex) -> AnalyzeDesignTool:
    pipeline = DesignPipeline(
        config=ServerConfig(),
        catalog=catalog,
        index=index,  # type: ignore[arg-type]
        llm=_scripted_llm(),
    )
    return AnalyzeDesignTool(pipeline=pipeline)


async def test_analyze_selects_the_scripted_primary_over_the_real_catalog(catalog) -> None:
    """Happy path: retrieval finds ``timeout``, so the payload reports it as the primary."""
    tool = _tool(catalog, StubIndex(hits=[StubHit(name="timeout", retrieval_score=5.0)]))

    payload = await tool.analyze(DESCRIPTION)

    assert any(part["id"] == PART_ID for part in payload["decision_parts"])
    selection = payload["selections"][PART_ID]
    assert selection["primary"] == "timeout"
    assert payload["selected_patterns"] == ["timeout"]
    assert PART_CATEGORY in payload["categories"]
    assert payload["is_fallback"] is False


async def test_short_description_is_a_structured_error(catalog) -> None:
    """E1: the tool maps the intake failure to a ToolError carrying ERR_001."""
    tool = _tool(catalog, StubIndex(hits=[StubHit(name="timeout", retrieval_score=5.0)]))

    with pytest.raises(ToolError, match="ERR_001"):
        await tool.analyze("too short")


async def test_empty_retrieval_pool_falls_back_and_the_payload_flags_it(catalog) -> None:
    """E5: the §12 category fallback primary is selected and the whole payload flags fallback."""
    tool = _tool(catalog, StubIndex(hits=[]))

    payload = await tool.analyze(DESCRIPTION)

    selection = payload["selections"][PART_ID]
    assert selection["primary"] == "timeout"  # §12 fallback for concurrency-resilience
    assert selection["is_fallback"] is True
    assert payload["selected_patterns"] == ["timeout"]
    assert payload["is_fallback"] is True
