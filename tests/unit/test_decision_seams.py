"""Decision-seam contract inside the pipeline: J4 weights and J2 category narrowing.

The two seams share one property: the decision model proposes, the deterministic core
still decides. Weights go through the *same* §7.1 tail as the LLM path, and category
narrowing can only ever remove candidates from an already-retrieved pool — never
starve a part.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from src.config import ServerConfig
from src.patterns.loader import PatternLoader
from src.patterns.retriever import RetrievalHit
from src.pipeline import DesignPipeline, _decision_part_state
from src.schemas import BriefComponent, DecisionPart, DesignBrief
from src.schemas.enums import QUALITY_AXES
from tests.unit.fakes import ScriptedLLM, StubHit, StubIndex


class _FakeTacit:
    """Scripted stand-in for :class:`src.decisions.tacit.TacitClient` (no I/O)."""

    def __init__(
        self,
        vectors: list[list[dict[str, float] | None]] | None = None,
        category: tuple[str | None, float] = ("domain-modeling", 0.9),
        gate: float | None = None,
    ) -> None:
        self._vectors = list(vectors or [])
        self._category = category
        self._gate = gate
        self.calls: list[list[str]] = []
        self.category_calls: list[str] = []

    def enabled(self) -> bool:
        return True

    async def score_axes_many(
        self, states: list[str], axes: tuple[str, ...] = QUALITY_AXES
    ) -> list[dict[str, float] | None]:
        self.calls.append(list(states))
        scripted = self._vectors.pop(0) if self._vectors else [None] * len(states)
        return scripted

    async def choose_category(self, state: str) -> tuple[str | None, float]:
        self.category_calls.append(state)
        return self._category

    async def gate(self, state: str) -> float | None:
        return self._gate

    async def close(self) -> None:
        pass


class _ScriptedCategoryTacit(_FakeTacit):
    """Category answers consumed in call order (one per part)."""

    def __init__(self, answers: list[tuple[str | None, float]]) -> None:
        super().__init__()
        self._answers = list(answers)

    async def choose_category(self, state: str) -> tuple[str | None, float]:
        self.category_calls.append(state)
        return self._answers.pop(0)


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
    weights_client: Any = None,
    category_client: Any = None,
    hits: list[StubHit] | None = None,
    llm: ScriptedLLM | None = None,
    **config: Any,
) -> DesignPipeline:
    return DesignPipeline(
        config=ServerConfig.model_validate(config) if config else ServerConfig(),
        catalog=catalog,
        index=StubIndex(hits=hits or [StubHit("repository", 0.0317, 1, 1)]),  # type: ignore[arg-type]
        llm=llm or ScriptedLLM(weights={"testability": 1.0}),
        weights_client=weights_client,
        category_client=category_client,
    )


def _hit(name: str) -> RetrievalHit:
    return RetrievalHit(name=name, retrieval_score=0.0317, fused_rank=1, rerank_rank=1)


# -- J4: weights --------------------------------------------------------------------


async def test_weights_seam_uses_decision_model_and_the_shared_tail(catalog: PatternLoader) -> None:
    """A Tacit vector is smoothed exactly like an LLM vector and labelled as its source."""
    llm = ScriptedLLM(weights={"testability": 1.0})
    vector = dict.fromkeys(QUALITY_AXES, 0.0)
    vector["testability"] = 1.0
    client = _FakeTacit(vectors=[[vector, vector]])
    pipeline = _pipeline(
        catalog,
        weights_client=client,
        llm=llm,
        decisions={"weights": {"enabled": True}},
    )
    selections = await pipeline._select_all(_parts(), _brief(), None)
    first = selections["p1"]
    assert first.weights_source == "decision-model"
    # smoothed = 0.7 * normalized + 0.3 / 8 (the shipped alpha = 0.7): 0.7375 / 0.0375.
    assert first.weights["testability"] == pytest.approx(0.7375)
    assert first.weights["simplicity"] == pytest.approx(0.0375)
    assert llm.calls == []  # the LLM never saw a weights call
    await client.close()


async def test_weights_seam_unusable_vector_falls_back_to_the_llm(catalog: PatternLoader) -> None:
    llm = ScriptedLLM(weights={"testability": 1.0})
    client = _FakeTacit(vectors=[[None, None], [None, None]])
    pipeline = _pipeline(
        catalog,
        weights_client=client,
        llm=llm,
        decisions={"weights": {"enabled": True}},
    )
    selections = await pipeline._select_all(_parts(), _brief(), None)
    assert selections["p1"].weights_source == "llm"
    assert llm.calls == ["extract_weights_batch:p1,p2"]
    # One retry for exactly the parts that failed, then the LLM path.
    assert len(client.calls) == 2
    assert len(client.calls[1]) == 2


async def test_weights_seam_all_zero_vector_falls_back_to_the_llm(catalog: PatternLoader) -> None:
    """An all-zero Tacit answer is unusable, not an E4 vector: the LLM owns that path."""
    zeros = dict.fromkeys(QUALITY_AXES, 0.0)
    llm = ScriptedLLM(weights={"scalability": 1.0})
    client = _FakeTacit(vectors=[[zeros], [zeros]])
    pipeline = _pipeline(
        catalog,
        weights_client=client,
        llm=llm,
        decisions={"weights": {"enabled": True}},
    )
    selections = await pipeline._select_all(_parts()[:1], _brief(), None)
    assert selections["p1"].weights_source == "llm"
    assert llm.calls == ["extract_weights_batch:p1"]
    assert selections["p1"].weights["scalability"] == pytest.approx(0.7375)


async def test_weights_seam_retry_recovers_a_part(catalog: PatternLoader) -> None:
    vector = dict.fromkeys(QUALITY_AXES, 0.0)
    vector["reliability"] = 1.0
    llm = ScriptedLLM(weights={"testability": 1.0})
    client = _FakeTacit(vectors=[[None], [vector]])
    pipeline = _pipeline(
        catalog,
        weights_client=client,
        llm=llm,
        decisions={"weights": {"enabled": True}},
    )
    selections = await pipeline._select_all(_parts()[:1], _brief(), None)
    assert selections["p1"].weights_source == "decision-model"
    assert selections["p1"].weights["reliability"] == pytest.approx(0.7375)
    assert llm.calls == []


async def test_weights_seam_disabled_keeps_the_llm_path(catalog: PatternLoader) -> None:
    llm = ScriptedLLM(weights={"testability": 1.0})
    client = _FakeTacit(vectors=[[dict.fromkeys(QUALITY_AXES, 1.0)]])
    pipeline = _pipeline(catalog, weights_client=client, llm=llm)
    selections = await pipeline._select_all(_parts()[:1], _brief(), None)
    assert selections["p1"].weights_source == "llm"
    assert client.calls == []
    assert llm.calls == ["extract_weights_batch:p1"]


# -- J2: category narrowing ---------------------------------------------------------


async def test_category_narrowing_filters_on_a_confident_choice(catalog: PatternLoader) -> None:
    client = _FakeTacit(category=("persistence-state", 0.9))
    pipeline = _pipeline(catalog, category_client=client, decisions={"category": {"enabled": True}})
    parts = _parts()[:1]
    narrowed, overrides = await pipeline._narrow_by_category(
        parts, _brief(), [[_hit("repository"), _hit("strategy"), _hit("observer")]]
    )
    assert [hit.name for hit in narrowed[0]] == ["repository"]
    assert overrides == {"p1": "persistence-state"}
    assert client.category_calls == [_decision_part_state(_brief(), parts[0])]


async def test_category_narrowing_below_floor_keeps_the_pool(catalog: PatternLoader) -> None:
    client = _FakeTacit(category=("persistence-state", 0.2))
    pipeline = _pipeline(catalog, category_client=client, decisions={"category": {"enabled": True}})
    narrowed, overrides = await pipeline._narrow_by_category(
        _parts()[:1], _brief(), [[_hit("repository"), _hit("strategy")]]
    )
    assert [hit.name for hit in narrowed[0]] == ["repository", "strategy"]
    assert overrides == {}


async def test_category_narrowing_never_empties_the_pool(catalog: PatternLoader) -> None:
    """A filter that keeps nothing is discarded — the unfiltered pool stands."""
    client = _FakeTacit(category=("security-verification", 0.95))
    pipeline = _pipeline(catalog, category_client=client, decisions={"category": {"enabled": True}})
    narrowed, overrides = await pipeline._narrow_by_category(
        _parts()[:1], _brief(), [[_hit("repository"), _hit("strategy")]]
    )
    assert [hit.name for hit in narrowed[0]] == ["repository", "strategy"]
    assert overrides == {}


async def test_category_narrowing_no_category_keeps_the_pool(catalog: PatternLoader) -> None:
    client = _FakeTacit(category=(None, 0.0))
    pipeline = _pipeline(catalog, category_client=client, decisions={"category": {"enabled": True}})
    narrowed, overrides = await pipeline._narrow_by_category(_parts()[:1], _brief(), [[_hit("repository")]])
    assert [hit.name for hit in narrowed[0]] == ["repository"]
    assert overrides == {}


async def test_category_narrowing_disabled_never_calls_the_client(catalog: PatternLoader) -> None:
    client = _FakeTacit(category=("persistence-state", 0.9))
    pipeline = _pipeline(catalog, category_client=client)
    narrowed, overrides = await pipeline._narrow_by_category(
        _parts()[:1], _brief(), [[_hit("repository"), _hit("strategy")]]
    )
    assert [hit.name for hit in narrowed[0]] == ["repository", "strategy"]
    assert overrides == {}
    assert client.category_calls == []


async def test_category_narrowing_keeps_names_the_catalog_does_not_know(catalog: PatternLoader) -> None:
    client = _FakeTacit(category=("persistence-state", 0.9))
    pipeline = _pipeline(catalog, category_client=client, decisions={"category": {"enabled": True}})
    narrowed, overrides = await pipeline._narrow_by_category(
        _parts()[:1], _brief(), [[_hit("repository"), _hit("no-such-record")]]
    )
    assert [hit.name for hit in narrowed[0]] == ["repository", "no-such-record"]
    assert overrides == {"p1": "persistence-state"}


async def test_category_narrowing_reaches_the_selection(catalog: PatternLoader) -> None:
    """End to end through the pipeline: the narrowed pool feeds Stage E unchanged otherwise."""
    client = _FakeTacit(category=("persistence-state", 0.9))
    pipeline = _pipeline(
        catalog,
        category_client=client,
        hits=[StubHit("repository", 0.0317, 1, 1), StubHit("strategy", 0.0317, 2, 2)],
        decisions={"category": {"enabled": True}},
    )
    selections = await pipeline._select_all(_parts()[:1], _brief(), None)
    candidate_names = {candidate.name for candidate in selections["p1"].candidates}
    assert "strategy" not in candidate_names
    assert "repository" in candidate_names
# -- the shared §7.1 tail: prior blend, E4 label, and the state contract --------------


async def test_weights_seam_flat_vector_takes_the_architecture_prior(
    catalog: PatternLoader,
) -> None:
    """A flat Tacit vector (peak <= 0.2) blends the catalog prior exactly like the LLM path."""
    flat = dict.fromkeys(QUALITY_AXES, 0.1)
    client = _FakeTacit(vectors=[[flat]])
    brief = DesignBrief(
        restated_goal="serving views without stale data",
        quality_attributes={"scalability": 1.0, "testability": 0.5},
    )
    pipeline = _pipeline(
        catalog, weights_client=client, decisions={"weights": {"enabled": True}}
    )
    selections = await pipeline._select_all(_parts()[:1], brief, None)
    selection = selections["p1"]
    assert selection.weights_source == "decision-model+architecture_prior"
    # The prior is max-normalized over the axes it names: scalability 1.0, testability 0.5.
    # raw = 0.7 * 0.1 + 0.3 * prior, so named priorities outrank the unnamed axes.
    assert selection.weights["scalability"] > selection.weights["testability"]
    assert selection.weights["testability"] > selection.weights["security"]


async def test_llm_path_labels_the_e4_mean(catalog: PatternLoader) -> None:
    """All-zero twice on the LLM path keeps the documented E4 mean label."""
    pipeline = _pipeline(catalog, llm=ScriptedLLM(weights={}))
    selections = await pipeline._select_all(_parts()[:1], _brief(), None)
    assert selections["p1"].weights_source == "e4_mean"
    # The uniform mean is max-normalized to 1.0 on every axis, then smoothed.
    assert all(value == pytest.approx(0.7375) for value in selections["p1"].weights.values())


async def test_weights_seam_without_a_client_keeps_the_llm_path(
    catalog: PatternLoader,
) -> None:
    """The seam flag alone is not enough: an unwired client leaves the LLM path untouched."""
    llm = ScriptedLLM(weights={"testability": 1.0})
    pipeline = _pipeline(catalog, llm=llm, decisions={"weights": {"enabled": True}})
    selections = await pipeline._select_all(_parts()[:1], _brief(), None)
    assert selections["p1"].weights_source == "llm"
    assert llm.calls == ["extract_weights_batch:p1"]


async def test_weights_seam_retries_only_the_failed_parts(
    catalog: PatternLoader,
) -> None:
    good = dict.fromkeys(QUALITY_AXES, 0.0)
    good["reliability"] = 1.0
    client = _FakeTacit(vectors=[[good, None], [good]])
    pipeline = _pipeline(
        catalog, weights_client=client, decisions={"weights": {"enabled": True}}
    )
    selections = await pipeline._select_all(_parts(), _brief(), None)
    assert selections["p1"].weights_source == "decision-model"
    assert selections["p2"].weights_source == "decision-model"
    # First call: both parts; retry: exactly the part that failed (p2).
    assert [len(states) for states in client.calls] == [2, 1]
    assert client.calls[1] == [_decision_part_state(_brief(), _parts()[1])]


def test_decision_part_state_carries_the_brief_and_the_part() -> None:
    """The state is the serving bytes: goal, components, statement, scope, constraints."""
    brief = DesignBrief(
        restated_goal="serve views without stale data",
        components=[BriefComponent(name="Api"), BriefComponent(name="Cache")],
    )
    part = DecisionPart(
        id="p1",
        statement="invalidate cached views on write",
        scope="task-lifetime",
        constraints=["p99 < 50 ms", "no cross-region traffic", "keep the cache warm"],
    )
    assert _decision_part_state(brief, part).splitlines() == [
        "System goal: serve views without stale data",
        "Components: Api, Cache",
        "Decision part: invalidate cached views on write",
        "Scope: task-lifetime",
        "Constraints: p99 < 50 ms; no cross-region traffic; keep the cache warm",
    ]
    long_part = part.model_copy(update={"statement": "x" * 5000})
    assert len(_decision_part_state(brief, long_part)) <= 1500
def test_decision_part_state_caps_components_and_constraints() -> None:
    """The 8-component / 4-constraint caps keep the state bounded before the 1500-char cut."""
    brief = DesignBrief(
        restated_goal="goal",
        components=[BriefComponent(name=f"C{index}") for index in range(10)],
    )
    part = DecisionPart(
        id="p1", statement="s", constraints=[f"c{index}" for index in range(6)]
    )
    state = _decision_part_state(brief, part)
    assert "Components: C0, C1, C2, C3, C4, C5, C6, C7" in state
    assert "C8" not in state
    assert state.endswith("Constraints: c0; c1; c2; c3")


async def test_category_narrowing_logs_the_kept_and_pool_sizes(
    catalog: PatternLoader, caplog: pytest.LogCaptureFixture
) -> None:
    client = _FakeTacit(category=("persistence-state", 0.9))
    pipeline = _pipeline(
        catalog,
        category_client=client,
        decisions={"category": {"enabled": True}},
    )
    with caplog.at_level(logging.DEBUG, logger="src.pipeline"):
        await pipeline._narrow_by_category(
            _parts()[:1],
            _brief(),
            [[_hit("repository"), _hit("strategy"), _hit("observer")]],
        )
    record = next(
        r for r in caplog.records if r.getMessage() == "Category narrowing applied"
    )
    assert record.part_id == "p1"  # type: ignore[attr-defined]
    assert record.category == "persistence-state"  # type: ignore[attr-defined]
    assert record.kept == 1  # type: ignore[attr-defined]
    assert record.pool == 3  # type: ignore[attr-defined]
    assert record.confidence == pytest.approx(0.9)  # type: ignore[attr-defined]


async def test_category_narrowing_logs_the_empty_filter_fallback(
    catalog: PatternLoader, caplog: pytest.LogCaptureFixture
) -> None:
    client = _FakeTacit(category=("security-verification", 0.95))
    pipeline = _pipeline(
        catalog,
        category_client=client,
        decisions={"category": {"enabled": True}},
    )
    with caplog.at_level(logging.DEBUG, logger="src.pipeline"):
        await pipeline._narrow_by_category(
            _parts()[:1], _brief(), [[_hit("repository"), _hit("strategy")]]
        )
    record = next(
        r
        for r in caplog.records
        if r.getMessage()
        == "Category narrowing kept nothing; using the unfiltered pool"
    )
    assert record.part_id == "p1"  # type: ignore[attr-defined]
    assert record.category == "security-verification"  # type: ignore[attr-defined]
    assert record.pool == 2  # type: ignore[attr-defined]


async def test_category_narrowing_logs_the_skipped_reason(
    catalog: PatternLoader, caplog: pytest.LogCaptureFixture
) -> None:
    client = _FakeTacit(category=("persistence-state", 0.2))
    pipeline = _pipeline(
        catalog,
        category_client=client,
        decisions={"category": {"enabled": True}},
    )
    with caplog.at_level(logging.DEBUG, logger="src.pipeline"):
        await pipeline._narrow_by_category(
            _parts()[:1], _brief(), [[_hit("repository")]]
        )
    record = next(r for r in caplog.records if r.getMessage() == "Category narrowing skipped")
    assert record.part_id == "p1"  # type: ignore[attr-defined]
    assert record.category == "persistence-state"  # type: ignore[attr-defined]
    assert record.confidence == pytest.approx(0.2)  # type: ignore[attr-defined]


async def test_weights_seam_retry_logs_the_failed_part_ids(
    catalog: PatternLoader, caplog: pytest.LogCaptureFixture
) -> None:
    good = dict.fromkeys(QUALITY_AXES, 0.0)
    good["reliability"] = 1.0
    client = _FakeTacit(vectors=[[good, None], [good]])
    pipeline = _pipeline(
        catalog,
        weights_client=client,
        decisions={"weights": {"enabled": True}},
    )
    with caplog.at_level(logging.INFO, logger="src.pipeline"):
        await pipeline._select_all(_parts(), _brief(), None)
    record = next(
        r
        for r in caplog.records
        if r.getMessage() == "Tacit weights unusable; retrying those parts"
    )
    assert record.part_ids == ["p2"]  # type: ignore[attr-defined]


async def test_weights_tail_logs_the_source_and_peak(
    catalog: PatternLoader, caplog: pytest.LogCaptureFixture
) -> None:
    flat = dict.fromkeys(QUALITY_AXES, 0.1)
    client = _FakeTacit(vectors=[[flat]])
    brief = DesignBrief(restated_goal="g", quality_attributes={"scalability": 1.0})
    pipeline = _pipeline(
        catalog,
        weights_client=client,
        decisions={"weights": {"enabled": True}},
    )
    with caplog.at_level(logging.INFO, logger="src.pipeline"):
        await pipeline._select_all(_parts()[:1], brief, None)
    record = next(
        r for r in caplog.records if r.getMessage() == "Part weights extracted"
    )
    assert record.weights_source == "decision-model+architecture_prior"  # type: ignore[attr-defined]
    assert record.peak == pytest.approx(0.1)  # type: ignore[attr-defined]
    assert record.part_id == "p1"  # type: ignore[attr-defined]


async def test_category_narrowing_replaces_a_conflicting_hypothesis(catalog: PatternLoader) -> None:
    """The narrowed category becomes the part's effective one for scoring and fallback.

    The 7.3 adjustment scores a hit +5 when its category equals ``part.category_hypothesis``
    and -10 otherwise, and E5/E7 take the fallback primary from that same hypothesis. A
    narrowed pool scored against the old hypothesis would therefore take -10 on every
    survivor and fall back to a pattern from outside the pool.
    """
    client = _FakeTacit(category=("persistence-state", 0.9))
    pipeline = _pipeline(
        catalog,
        category_client=client,
        hits=[StubHit("repository", 0.0317, 1, 1), StubHit("strategy", 0.0317, 2, 2)],
        decisions={"category": {"enabled": True}},
    )
    part = _parts()[0].model_copy(update={"category_hypothesis": "behavior-selection"})
    selection = (await pipeline._select_all([part], _brief(), None))["p1"]
    assert selection.part.category_hypothesis == "persistence-state"
    assert selection.primary == "repository"
    labels = [adjustment.label for adjustment in selection.candidates[0].adjustments]
    assert "category_match" in labels
    assert "category_mismatch" not in labels


async def test_category_narrowing_at_exactly_the_floor_narrows(catalog: PatternLoader) -> None:
    """The floor is inclusive: confidence == confidence_floor is enough to act."""
    client = _FakeTacit(category=("persistence-state", 0.5))
    pipeline = _pipeline(catalog, category_client=client, decisions={"category": {"enabled": True}})
    narrowed, overrides = await pipeline._narrow_by_category(
        _parts()[:1], _brief(), [[_hit("repository"), _hit("strategy")]]
    )
    assert [hit.name for hit in narrowed[0]] == ["repository"]
    assert overrides == {"p1": "persistence-state"}


async def test_category_narrowing_continues_after_a_skipped_part(catalog: PatternLoader) -> None:
    """One low-confidence part must not stop the remaining parts from narrowing."""
    client = _ScriptedCategoryTacit([("persistence-state", 0.2), ("persistence-state", 0.9)])
    pipeline = _pipeline(catalog, category_client=client, decisions={"category": {"enabled": True}})
    hits = [[_hit("repository"), _hit("strategy")], [_hit("repository"), _hit("strategy")]]
    narrowed, overrides = await pipeline._narrow_by_category(_parts(), _brief(), hits)
    assert [hit.name for hit in narrowed[0]] == ["repository", "strategy"]  # skipped
    assert [hit.name for hit in narrowed[1]] == ["repository"]  # narrowed
    assert overrides == {"p2": "persistence-state"}


async def test_category_narrowing_continues_after_an_empty_filter(catalog: PatternLoader) -> None:
    """A filter that keeps nothing falls back for that part only."""
    client = _ScriptedCategoryTacit([("security-verification", 0.9), ("persistence-state", 0.9)])
    pipeline = _pipeline(catalog, category_client=client, decisions={"category": {"enabled": True}})
    hits = [[_hit("repository"), _hit("strategy")], [_hit("repository"), _hit("strategy")]]
    narrowed, overrides = await pipeline._narrow_by_category(_parts(), _brief(), hits)
    assert [hit.name for hit in narrowed[0]] == ["repository", "strategy"]  # kept unfiltered
    assert [hit.name for hit in narrowed[1]] == ["repository"]
    assert overrides == {"p2": "persistence-state"}


async def test_category_narrowing_logs_only_a_real_replacement(
    catalog: PatternLoader, caplog: pytest.LogCaptureFixture
) -> None:
    """``replaced_hypothesis`` names the discarded hypothesis, and is None when it agrees."""
    conflicting = _parts()[0].model_copy(update={"category_hypothesis": "behavior-selection"})
    agreeing = _parts()[0].model_copy(update={"category_hypothesis": "persistence-state"})
    client = _FakeTacit(category=("persistence-state", 0.9))
    pipeline = _pipeline(catalog, category_client=client, decisions={"category": {"enabled": True}})
    with caplog.at_level(logging.INFO, logger="src.pipeline"):
        await pipeline._narrow_by_category([conflicting], _brief(), [[_hit("repository")]])
        await pipeline._narrow_by_category([agreeing], _brief(), [[_hit("repository")]])
    applied = [r for r in caplog.records if r.getMessage() == "Category narrowing applied"]
    assert applied[0].replaced_hypothesis == "behavior-selection"  # type: ignore[attr-defined]
    assert applied[1].replaced_hypothesis is None  # type: ignore[attr-defined]


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


async def test_category_narrowing_refuses_a_length_mismatch(catalog: PatternLoader) -> None:
    """``strict=True`` on the part/hit zip is a guard: a caller-side mismatch must be loud."""
    client = _FakeTacit(category=("persistence-state", 0.9))
    pipeline = _pipeline(catalog, category_client=client, decisions={"category": {"enabled": True}})
    with pytest.raises(ValueError):
        await pipeline._narrow_by_category(_parts(), _brief(), [[_hit("repository")]])


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
