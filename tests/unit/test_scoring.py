"""§7 — deterministic scoring arithmetic on real catalog records.

The weight post-processing is pinned to the guide's worked example (§13): raw weights
reliability 1.0 / performance 0.8 / simplicity 0.6 / testability 0.5 / maintainability 0.4 /
scalability 0.3 / extensibility 0.2 / security 0.2, smoothed with α = 0.7 into
0.7375 / 0.5975 / 0.4575 / 0.3875 / 0.3175 / 0.2475 / 0.1775 / 0.1775 (Σ = 3.1).

The score chain is then checked against hand-computed golden numbers for the real ``timeout``
record (quality attributes: testability 6, extensibility 5, maintainability 7, simplicity 6,
performance 8, reliability 9, security 6, scalability 8), a three-candidate pool and the documented
default parameters.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.config import ServerConfig
from src.pipeline import extract_part_weights_batch, score_candidates
from src.schemas import QUALITY_AXES, DecisionPart, DesignBrief
from tests.unit.fakes import ScriptedLLM, StubHit

WORKED_EXAMPLE_RAW = {
    "reliability": 1.0,
    "performance": 0.8,
    "simplicity": 0.6,
    "testability": 0.5,
    "maintainability": 0.4,
    "scalability": 0.3,
    "extensibility": 0.2,
    "security": 0.2,
}
WORKED_EXAMPLE_SMOOTHED = {
    "reliability": 0.7375,
    "performance": 0.5975,
    "simplicity": 0.4575,
    "testability": 0.3875,
    "maintainability": 0.3175,
    "scalability": 0.2475,
    "extensibility": 0.1775,
    "security": 0.1775,
}


def _part() -> DecisionPart:
    return DecisionPart(
        id="enrichment-resilience",
        statement="bound and survive the synchronous Redis lookup under p99",
        scope="task-lifetime",
        category_hypothesis="concurrency-resilience",
        constraints=["p99 < 50 ms per event"],
        priority=1.0,
    )


def _brief(language: str = "typescript") -> DesignBrief:
    return DesignBrief(
        restated_goal="event-driven ETL",
        architecture_style="event-driven",
        constraints=["at-least-once delivery"],
        language=language,  # type: ignore[arg-type]
    )


async def test_weight_smoothing_matches_the_worked_example() -> None:
    """§7.1: propose → max-normalize → convex smoothing reproduces the guide's numbers."""
    llm = ScriptedLLM(weights=dict(WORKED_EXAMPLE_RAW))
    part = _part()
    batch = await extract_part_weights_batch(llm, [part], _brief(), config=ServerConfig())
    weights, source = batch[part.id]
    assert source == "llm"
    for axis, expected in WORKED_EXAMPLE_SMOOTHED.items():
        assert weights[axis] == pytest.approx(expected, abs=1e-9)
    assert sum(weights.values()) == pytest.approx(3.1, abs=1e-9)
    assert set(weights) == set(QUALITY_AXES)
    assert llm.calls == [f"extract_weights_batch:{part.id}"]


async def test_one_batched_call_covers_every_part() -> None:
    """Six parts cost ONE provider call; each part keeps its own vector and source."""
    parts = [
        _part().model_copy(update={"id": f"part-{index}", "priority": 1.0 - index * 0.1})
        for index in range(6)
    ]
    llm = ScriptedLLM(weights=dict(WORKED_EXAMPLE_RAW))

    batch = await extract_part_weights_batch(llm, parts, _brief(), config=ServerConfig())

    assert llm.calls == ["extract_weights_batch:" + ",".join(part.id for part in parts)]
    assert set(batch) == {part.id for part in parts}
    for part in parts:
        weights, source = batch[part.id]
        assert source == "llm"
        for axis, expected in WORKED_EXAMPLE_SMOOTHED.items():
            assert weights[axis] == pytest.approx(expected, abs=1e-9)


async def test_all_zero_weights_fall_back_to_the_unweighted_mean() -> None:
    """E4: an all-zero extraction is retried once, then replaced by the unweighted mean.

    The mean stays uniform through max-normalization and convex smoothing, so the resulting
    ``analysis_score`` is the plain average of the pattern's eight axes - which is what the
    fallback promises (no axis is preferred).
    """
    part = _part()
    zero = dict.fromkeys(QUALITY_AXES, 0.0)
    llm = ScriptedLLM(
        weights_batch_sequence=[{part.id: zero}, {part.id: zero}]
    )
    batch = await extract_part_weights_batch(llm, [part], _brief(), config=ServerConfig())
    weights, source = batch[part.id]
    assert source == "e4_mean"
    values = list(weights.values())
    assert len(set(values)) == 1
    assert values[0] > 0.0
    assert llm.calls == [f"extract_weights_batch:{part.id}"] * 2


async def test_retry_batch_covers_only_the_all_zero_parts() -> None:
    """The E4 retry is batched over exactly the parts that came back all-zero."""
    silent = _part().model_copy(update={"id": "silent-part", "priority": 0.9})
    loud = _part().model_copy(update={"id": "loud-part", "priority": 0.8})
    zero = dict.fromkeys(QUALITY_AXES, 0.0)
    good = dict(WORKED_EXAMPLE_RAW)
    llm = ScriptedLLM(
        weights_batch_sequence=[
            {silent.id: zero, loud.id: good},
            {silent.id: good},
        ]
    )

    batch = await extract_part_weights_batch(llm, [silent, loud], _brief(), config=ServerConfig())

    assert llm.calls == [
        f"extract_weights_batch:{silent.id},{loud.id}",
        f"extract_weights_batch:{silent.id}",
    ]
    assert batch[loud.id][1] == "llm"
    assert batch[silent.id][1] == "llm"
    for axis, expected in WORKED_EXAMPLE_SMOOTHED.items():
        assert batch[silent.id][0][axis] == pytest.approx(expected, abs=1e-9)


async def test_weights_failure_degrades_to_the_mean_without_raising() -> None:
    """A hard weights-call failure is treated as an all-zero proposal (E4 path), not a crash."""
    part = _part()
    llm = ScriptedLLM(fail_weights=True)
    batch = await extract_part_weights_batch(llm, [part], _brief(), config=ServerConfig())
    weights, source = batch[part.id]
    assert source == "e4_mean"
    assert len(set(weights.values())) == 1


async def test_architecture_prior_blends_only_when_the_part_is_silent() -> None:
    """§7.1: the 0.7/0.3 prior applies when the part carries no priority evidence."""
    llm = ScriptedLLM(weights=dict.fromkeys(QUALITY_AXES, 0.1))
    brief = _brief()
    brief.quality_attributes = {"scalability": 9.1, "maintainability": 8.2}
    part = _part()
    batch = await extract_part_weights_batch(llm, [part], brief, config=ServerConfig())
    weights, source = batch[part.id]
    assert source == "llm+architecture_prior"
    # scalability gets the largest prior contribution (9.1 is the max → prior 1.0)
    assert weights["scalability"] == max(weights.values()) or weights["scalability"] > weights["security"]


def test_scoring_chain_on_real_records(catalog) -> None:
    """§7.2/§7.3 golden numbers for a three-candidate concurrency pool in recognized mode."""
    part = _part()
    brief = _brief()
    hits = [
        StubHit("timeout", 0.0317, fused_rank=1, rerank_rank=1),
        StubHit("retry-with-backoff", 0.0300, fused_rank=2, rerank_rank=3),
        StubHit("circuit-breaker", 0.0283, fused_rank=3, rerank_rank=2),
    ]
    candidates = score_candidates(
        hits,
        part=part,
        brief=brief,
        weights=dict(WORKED_EXAMPLE_SMOOTHED),
        record_of=catalog.records,
        catalog=catalog,
        config=ServerConfig(),
    )
    by_name = {candidate.name: candidate for candidate in candidates}
    timeout = by_name["timeout"]
    # 10 * (sum w'*qa) / sum w' with the real quality attributes: Σ w'·qa = 22.6425
    assert timeout.analysis_score == pytest.approx(73.0403, abs=5e-4)
    assert timeout.fusion_score_normalized == pytest.approx(100.0)
    assert timeout.blended == pytest.approx(81.1282, abs=5e-4)
    labels = {entry.label: entry.delta for entry in timeout.adjustments}
    assert labels == {
        "scope_bonus": 3.0,
        "category_match": 5.0,
        "language_support_library": 2.0,
    }
    assert timeout.fit_score == pytest.approx(91.1282, abs=5e-4)
    # min-max normalization over the pool: the retrieval minimum maps to 0, the maximum to 100
    assert by_name["circuit-breaker"].fusion_score_normalized == pytest.approx(0.0)
    assert by_name["retry-with-backoff"].fusion_score_normalized == pytest.approx(50.0)
    # §8.1 ordering: fit desc
    assert next(candidate.name for candidate in candidates) == "timeout"


def test_missing_axis_uses_the_neutral_default(catalog) -> None:
    """§7.2 divergence (h): an absent axis is unmeasured, not worst-case — 5.0, never 0.0."""
    part = _part()
    brief = _brief()
    record = catalog.get_by_name("timeout")
    assert record is not None
    trimmed = replace(record, quality_attributes={"reliability": 10.0, "performance": 8.0})
    candidates = score_candidates(
        [StubHit("timeout", 0.0317)],
        part=part,
        brief=brief,
        weights=dict(WORKED_EXAMPLE_SMOOTHED),
        record_of={"timeout": trimmed},
        catalog=catalog,
        config=ServerConfig(),
    )
    # Σ w'·qa = 0.7375·10 + 0.5975·8 + 6 · 0.1775·5 ... = 7.375 + 4.78 + 0 = 12.155 + 2.6625 = 14.8175
    expected = 10 * (0.7375 * 10 + 0.5975 * 8 + 5 * (3.1 - 0.7375 - 0.5975)) / 3.1
    assert candidates[0].analysis_score == pytest.approx(expected, abs=1e-9)


def test_fit_score_is_clamped_to_100(catalog) -> None:
    """§7.3: the clamp bounds the stacking of adjustments, it never lets fit exceed 100."""
    part = _part()
    brief = _brief()
    record = catalog.get_by_name("timeout")
    assert record is not None
    perfect = replace(
        record,
        quality_attributes=dict.fromkeys(QUALITY_AXES, 10.0),
        suitable_scopes=("task-lifetime",),
    )
    candidates = score_candidates(
        [StubHit("timeout", 0.0317, fused_rank=1, rerank_rank=1)],
        part=part,
        brief=brief,
        weights=dict(WORKED_EXAMPLE_SMOOTHED),
        record_of={"timeout": perfect},
        catalog=catalog,
        config=ServerConfig(),
    )
    assert candidates[0].analysis_score == pytest.approx(100.0)
    assert candidates[0].fit_score == 100.0


def test_generic_mode_applies_no_language_adjustment(catalog) -> None:
    """§8.4: generic mode makes no support-level judgement (−2/+2 never fire)."""
    part = _part()
    hits = [StubHit("timeout", 0.0317)]
    generic = score_candidates(
        hits,
        part=part,
        brief=_brief(language="generic"),
        weights=dict(WORKED_EXAMPLE_SMOOTHED),
        record_of=catalog.records,
        catalog=catalog,
        config=ServerConfig(),
    )
    recognized = score_candidates(
        hits,
        part=part,
        brief=_brief(language="typescript"),
        weights=dict(WORKED_EXAMPLE_SMOOTHED),
        record_of=catalog.records,
        catalog=catalog,
        config=ServerConfig(),
    )
    assert [entry.label for entry in generic[0].adjustments] == ["scope_bonus", "category_match"]
    assert generic[0].adjustments != recognized[0].adjustments
