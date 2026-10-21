"""Ch2 substitution contract: confident decision-model choices override the LLM style.

A fake DecisionModelClient is injected into DesignPipeline; no network. Proves
same-description → same-style → same affinity set, the caller-hint win, and the
override-on-confident-answer-only degrade rules.
"""

from __future__ import annotations

from typing import Any

from src.config import ServerConfig
from src.decisions.client import StyleDecision
from src.patterns.loader import PatternLoader
from src.pipeline import DesignPipeline
from src.schemas import DesignBrief
from src.style_affinity import STYLE_AFFINITY, affinity_for
from tests.unit.fakes import ScriptedLLM, StubIndex


class _FakeDecisionClient:
    def __init__(self, decision: StyleDecision | None) -> None:
        self._decision = decision
        self.calls: list[str] = []

    def enabled(self) -> bool:
        return True

    async def choose_style(self, description: str) -> StyleDecision | None:
        self.calls.append(description)
        return self._decision

    async def close(self) -> None:
        pass


def _pipeline(catalog: PatternLoader, fake: Any, **config: Any) -> DesignPipeline:
    server_config = ServerConfig.model_validate(config) if config else ServerConfig()
    return DesignPipeline(
        config=server_config,
        catalog=catalog,
        index=StubIndex(),  # type: ignore[arg-type]
        llm=ScriptedLLM(),
        decision_client=fake,
    )


def _brief(style: str | None = "microservices") -> DesignBrief:
    return DesignBrief(restated_goal="test the style substitution", architecture_style=style)


DESCRIPTION = "Event-driven IoT ingestion with Kafka feeding a worker pool for enrichment"


async def test_confident_choice_overrides_and_stabilizes_affinity(catalog: PatternLoader) -> None:
    fake = _FakeDecisionClient(StyleDecision(style="event-driven", confidence=0.9, probabilities={"event-driven": 0.9}))
    pipeline = _pipeline(catalog, fake)
    first = await pipeline._apply_decision_style(DESCRIPTION, None, _brief())
    second = await pipeline._apply_decision_style(DESCRIPTION, None, _brief())
    assert first.architecture_style == "event-driven"
    assert second.architecture_style == "event-driven"
    assert affinity_for(first.architecture_style) == affinity_for(second.architecture_style)
    assert affinity_for(first.architecture_style) == STYLE_AFFINITY["event-driven"]


async def test_caller_hint_wins_without_calling_client(catalog: PatternLoader) -> None:
    fake = _FakeDecisionClient(StyleDecision(style="event-driven", confidence=0.9, probabilities={"event-driven": 0.9}))
    pipeline = _pipeline(catalog, fake)
    result = await pipeline._apply_decision_style(DESCRIPTION, "microservices", _brief("microservices"))
    assert result.architecture_style == "microservices"
    assert fake.calls == []


async def test_low_confidence_keeps_llm_style(catalog: PatternLoader) -> None:
    fake = _FakeDecisionClient(StyleDecision(style="event-driven", confidence=0.3, probabilities={"event-driven": 0.3}))
    pipeline = _pipeline(catalog, fake)
    result = await pipeline._apply_decision_style(DESCRIPTION, None, _brief("microservices"))
    assert result.architecture_style == "microservices"
    assert len(fake.calls) == 1


async def test_none_choice_keeps_llm_style(catalog: PatternLoader) -> None:
    fake = _FakeDecisionClient(StyleDecision(style=None, confidence=0.95, probabilities={"none": 0.95}))
    pipeline = _pipeline(catalog, fake)
    result = await pipeline._apply_decision_style(DESCRIPTION, None, _brief("microservices"))
    assert result.architecture_style == "microservices"


async def test_client_failure_keeps_llm_style(catalog: PatternLoader) -> None:
    fake = _FakeDecisionClient(None)
    pipeline = _pipeline(catalog, fake)
    result = await pipeline._apply_decision_style(DESCRIPTION, None, _brief("microservices"))
    assert result.architecture_style == "microservices"


async def test_no_client_keeps_llm_style(catalog: PatternLoader) -> None:
    pipeline = DesignPipeline(
        config=ServerConfig(),
        catalog=catalog,
        index=StubIndex(),  # type: ignore[arg-type]
        llm=ScriptedLLM(),
    )
    result = await pipeline._apply_decision_style(DESCRIPTION, None, _brief("microservices"))
    assert result.architecture_style == "microservices"


async def test_floor_is_configurable(catalog: PatternLoader) -> None:
    fake = _FakeDecisionClient(StyleDecision(style="event-driven", confidence=0.3, probabilities={"event-driven": 0.3}))
    pipeline = _pipeline(catalog, fake, decisions={"style_confidence_floor": 0.2})
    result = await pipeline._apply_decision_style(DESCRIPTION, None, _brief("microservices"))
    assert result.architecture_style == "event-driven"
