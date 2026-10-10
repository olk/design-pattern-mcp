"""§4 rule 6 end-to-end: the caller's architecture_style hint, not the brief's extraction,
drives the §7.3 style_affinity prior (SELECTION_VARIABILITY_ANALYSIS.md §6.1)."""

from __future__ import annotations

from src.config import ServerConfig
from src.pipeline import DesignPipeline
from src.schemas import RawBriefAndParts, RawDecisionPart, RawDesignBrief
from src.tools.design import pipeline_result_to_output
from tests.unit.fakes import ScriptedLLM, StubHit, StubIndex


async def test_style_affinity_follows_the_caller_hint_end_to_end(catalog) -> None:
    """microservices (caller) covers workflow-messaging; hexagonal (brief's guess) does not."""
    llm = ScriptedLLM(
        brief_and_parts_result=RawBriefAndParts(
            design_brief=RawDesignBrief(
                restated_goal="publish domain events",
                architecture_style="hexagonal",  # the model's extraction — must lose
            ),
            decision_parts=[
                RawDecisionPart(
                    id="event-publishing",
                    statement="publish domain events to the broker",
                    component_refs=[],
                    scope="message-flow",
                    category_hypothesis="workflow-messaging",
                    priority=1.0,
                )
            ],
        ),
        weights={"reliability": 1.0},
    )
    pipeline = DesignPipeline(
        config=ServerConfig(),
        catalog=catalog,
        index=StubIndex(hits=[StubHit(name="publish-subscribe-channel", retrieval_score=5.0)]),  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
    )

    result = pipeline_result_to_output(
        await pipeline.run_design(
            description="An EventPublisher publishes domain events to the broker; every subscriber receives each event.",
            architecture_style="microservices",
        )
    )

    assert result["overview"]["primary_pattern"] == "publish-subscribe-channel"
    assert "style_affinity+5" in result["overview"]["reasoning"]
