"""Weights-tail, retrieval and fallback contracts inside the pipeline (§7.1, §8).

The deterministic core decides: every weights proposal goes through the same §7.1 tail, the
retrieved pool reaches scoring unfiltered, and a pool that survives no veto takes the documented
E5 fallback.
"""

from __future__ import annotations

import logging

import pytest

from src.config import ServerConfig
from src.patterns.loader import PatternLoader
from src.pipeline import DesignPipeline
from src.schemas import DecisionPart, DesignBrief
from src.schemas.enums import QUALITY_AXES
from tests.unit.fakes import ScriptedLLM, StubHit, StubIndex


def _parts() -> list[DecisionPart]:
    return [
        DecisionPart(id="p1", statement="Cache invalidation for computed order views", priority=2.0),
        DecisionPart(id="p2", statement="Pluggable payment provider selection", priority=1.0),
    ]


def _brief() -> DesignBrief:
    return DesignBrief(restated_goal="serving views without stale data")


def _pipeline(
    catalog: PatternLoader,
    *,
    hits: list[StubHit] | None = None,
    llm: ScriptedLLM | None = None,
) -> DesignPipeline:
    return DesignPipeline(
        config=ServerConfig(),
        catalog=catalog,
        index=StubIndex(hits=hits or [StubHit("repository", 0.0317, 1, 1)]),  # type: ignore[arg-type]
        llm=llm or ScriptedLLM(weights={"testability": 1.0}),
    )


async def test_llm_path_labels_the_e4_mean(catalog: PatternLoader) -> None:
    """All-zero twice on the LLM path keeps the documented E4 mean label."""
    pipeline = _pipeline(catalog, llm=ScriptedLLM(weights={}))
    selections = await pipeline._select_all(_parts()[:1], _brief(), None)
    assert selections["p1"].weights_source == "e4_mean"
    # The uniform mean is max-normalized to 1.0 on every axis, then smoothed.
    assert all(value == pytest.approx(0.7375) for value in selections["p1"].weights.values())


async def test_weights_tail_logs_the_source_and_peak(catalog: PatternLoader, caplog: pytest.LogCaptureFixture) -> None:
    """A flat LLM vector takes the architecture prior and the log names source and raw peak."""
    llm = ScriptedLLM(weights=dict.fromkeys(QUALITY_AXES, 0.1))
    brief = DesignBrief(restated_goal="g", quality_attributes={"scalability": 1.0})
    pipeline = _pipeline(catalog, llm=llm)
    with caplog.at_level(logging.INFO, logger="src.pipeline"):
        await pipeline._select_all(_parts()[:1], brief, None)
    record = next(r for r in caplog.records if r.getMessage() == "Part weights extracted")
    assert record.weights_source == "llm+architecture_prior"  # type: ignore[attr-defined]
    assert record.peak == pytest.approx(0.1)  # type: ignore[attr-defined]
    assert record.part_id == "p1"  # type: ignore[attr-defined]


async def test_select_all_keeps_every_retrieved_hit_whatever_the_hypothesis(
    catalog: PatternLoader,
) -> None:
    """Retrieval is not filtered by the part's category hypothesis: the fused pool reaches scoring."""
    pipeline = _pipeline(
        catalog,
        hits=[StubHit("repository", 0.0317, 1, 1), StubHit("strategy", 0.0317, 2, 2)],
    )
    part = _parts()[0].model_copy(update={"category_hypothesis": "persistence-state"})
    selection = (await pipeline._select_all([part], _brief(), None))["p1"]
    candidate_names = {candidate.name for candidate in selection.candidates}
    assert {"repository", "strategy"} <= candidate_names


async def test_select_all_falls_back_when_every_hit_is_vetoed(catalog: PatternLoader) -> None:
    """A vetoed-out pool takes the E5 fallback with the documented reason."""
    index = StubIndex(hits=[StubHit("repository", 0.0317, 1, 1)])  # type: ignore[arg-type]
    pipeline = DesignPipeline(
        config=ServerConfig(),
        catalog=catalog,
        index=index,  # type: ignore[arg-type]
        llm=ScriptedLLM(weights={"testability": 1.0}),
    )
    part = _parts()[0].model_copy(
        update={"negative_signals": ["avoid repository"], "category_hypothesis": "persistence-state"}
    )
    selection = (await pipeline._select_all([part], _brief(), None))["p1"]
    assert selection.is_fallback is True
    assert "pool empty after the veto pre-filters" in (selection.fallback_reason or "")


def test_select_for_part_keeps_the_documented_e5_reason(catalog: PatternLoader) -> None:
    """With no caller reason, the empty pool reports the fixed E5 text verbatim."""
    from src.pipeline import select_for_part

    selection = select_for_part(
        part=_parts()[0].model_copy(update={"category_hypothesis": "persistence-state"}),
        candidates=[],
        record_of=catalog.records,
        catalog=catalog,
        brief=_brief(),
        config=ServerConfig(),
        veto_ledger=[],
    )
    assert selection.fallback_reason == "no candidate survived retrieval and the veto pre-filters (E5)"
