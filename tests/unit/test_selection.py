"""§8 — ranking, thresholds, near-ties, supporting rules and the E7 no-fallback error.

Candidates are constructed with explicit fit/analysis values so each rule is exercised in isolation
against real catalog records (their categories, subgroups, conflicts and related_patterns are the
inputs the rules actually read).
"""

from __future__ import annotations

import pytest

from src.config import ServerConfig
from src.errors import ERROR_NO_PRIMARY_PATTERN, DesignPatternError
from src.pipeline import select_for_part
from src.schemas import AnnotatedCandidate, DecisionPart, DesignBrief, PartSelection


def _candidate(
    name: str, fit: float, *, analysis: float = 80.0, fusion: float = 50.0, order: int = 0
) -> AnnotatedCandidate:
    return AnnotatedCandidate(
        name=name,
        category="concurrency-resilience",
        retrieval_score=0.03,
        analysis_score=analysis,
        fusion_score_normalized=fusion,
        blended=fit,
        fit_score=fit,
        catalog_order=order,
    )


def _part(
    *,
    scope: str = "module",
    category: str | None = "concurrency-resilience",
    constraints: list[str] | None = None,
    part_id: str = "enrichment-resilience",
) -> DecisionPart:
    return DecisionPart(
        id=part_id,
        statement="bound the synchronous lookup",
        scope=scope,  # type: ignore[arg-type]
        category_hypothesis=category,  # type: ignore[arg-type]
        constraints=constraints or [],
        priority=1.0,
    )


def _select(part: DecisionPart, candidates: list[AnnotatedCandidate], catalog, config=None) -> PartSelection:
    return select_for_part(
        part=part,
        candidates=candidates,
        record_of=catalog.records,
        catalog=catalog,
        brief=DesignBrief(restated_goal="goal"),
        config=config or ServerConfig(),
        veto_ledger=[],
    )


def test_dual_gate_promotes_the_leader(catalog) -> None:
    """§8.2: fit ≥ 50 and analysis ≥ 50 → the leader is the primary."""
    selection = _select(_part(), [_candidate("timeout", 91.1), _candidate("bulkhead", 60.0)], catalog)
    assert selection.primary == "timeout"
    assert selection.primary_category == "concurrency-resilience"
    assert selection.is_fallback is False


def test_dual_gate_failure_falls_back_to_the_category_pattern(catalog) -> None:
    """§8.2 / E6: a leader below either threshold degrades to the §12 fallback."""
    selection = _select(_part(), [_candidate("timeout", 49.9), _candidate("bulkhead", 30.0)], catalog)
    assert selection.is_fallback is True
    assert selection.primary == "timeout"  # §12 fallback for concurrency-resilience
    assert "dual gate" in (selection.fallback_reason or "")
    assert selection.supporting == []


def test_analysis_only_failure_also_gates(catalog) -> None:
    """The dual gate is a conjunction: a high fit with a weak analysis score still fails."""
    selection = _select(
        _part(), [_candidate("timeout", 95.0, analysis=49.0), _candidate("bulkhead", 40.0)], catalog
    )
    assert selection.is_fallback is True


def test_specialized_components_has_no_fallback(catalog) -> None:
    """E7: ``specialized-components`` raises with the top-3 candidates instead of inventing one."""
    part = _part(category="specialized-components", part_id="ui-widget")
    with pytest.raises(DesignPatternError) as excinfo:
        _select(part, [_candidate("timeout", 20.0), _candidate("bulkhead", 10.0)], catalog)
    assert excinfo.value.code == ERROR_NO_PRIMARY_PATTERN
    assert excinfo.value.extra["part_id"] == "ui-widget"
    assert len(excinfo.value.extra["candidates"]) == 2


def test_empty_pool_is_e5_with_fallback(catalog) -> None:
    """E5: an empty pool still yields a primary — the category fallback, flagged."""
    selection = _select(_part(), [], catalog)
    assert selection.is_fallback is True
    assert selection.primary == "timeout"
    assert "veto pre-filters" in (selection.fallback_reason or "")


def test_near_tie_flags_and_forces_the_runner_up(catalog) -> None:
    """E8: a runner-up within the margin marks the part ambiguous and seeds the alternatives."""
    selection = _select(
        _part(),
        [_candidate("timeout", 91.1), _candidate("retry-with-backoff", 87.3), _candidate("bulkhead", 70.0)],
        catalog,
    )
    assert selection.ambiguous is True
    assert "retry-with-backoff" in selection.near_tie
    assert "bulkhead" not in selection.near_tie
    assert any(entry.startswith("retry-with-backoff") and "near-tie" in entry for entry in selection.alternatives)


def test_no_near_tie_beyond_the_margin(catalog) -> None:
    """A gap larger than ``near_tie_margin`` leaves the part unambiguous."""
    selection = _select(
        _part(), [_candidate("timeout", 91.1), _candidate("retry-with-backoff", 85.0)], catalog
    )
    assert selection.ambiguous is False
    assert selection.near_tie == []


def test_supporting_requires_the_threshold(catalog) -> None:
    """§8.3 rule 1: a candidate below ``supporting_score_threshold`` never joins the set."""
    selection = _select(
        _part(constraints=["retry-with-backoff absorbs transient failures"]),
        [_candidate("timeout", 91.1), _candidate("retry-with-backoff", 39.9)],
        catalog,
    )
    assert selection.supporting == []


def test_supporting_accepts_a_constraint_named_candidate(catalog) -> None:
    """§8.3 rule 2(b): an explicit stated constraint that names the candidate justifies it."""
    selection = _select(
        _part(constraints=["retry-with-backoff absorbs transient Redis failures"]),
        [_candidate("timeout", 91.1), _candidate("retry-with-backoff", 70.0)],
        catalog,
    )
    assert [choice.pattern for choice in selection.supporting] == ["retry-with-backoff"]
    role = selection.supporting[0].role
    assert "retry-with-backoff absorbs transient Redis failures" in role
    assert "(enrichment-resilience)" in role


def test_supporting_rejects_a_candidate_without_distinct_constraint(catalog) -> None:
    """§8.3 rule 2: a candidate whose only justification is unrelated to a stated constraint fails."""
    selection = _select(
        _part(constraints=["the write path must stay idempotent"]),
        [_candidate("timeout", 91.1), _candidate("bulkhead", 70.0)],
        catalog,
    )
    assert selection.supporting == []


def test_conflicting_pair_is_rejected(catalog) -> None:
    """§8.3 rule 3: the conflict table forbids pairing the same variation point."""
    part = _part(category="construction-composition", part_id="wiring")
    selection = _select(
        part,
        [_candidate("singleton", 91.1), _candidate("dependency-injection", 80.0)],
        catalog,
    )
    assert selection.primary == "singleton"
    assert selection.supporting == []


def test_same_subgroup_needs_a_stated_constraint(catalog) -> None:
    """§8.3 rule 4: same-subgroup pairs may both be supporting only under rule 2(b)."""
    timeout = catalog.get_by_name("timeout")
    circuit_breaker = catalog.get_by_name("circuit-breaker")
    assert timeout is not None and circuit_breaker is not None
    assert timeout.subgroup == circuit_breaker.subgroup  # "Dependency failure and overload"
    without_constraint = _select(
        _part(),
        [_candidate("timeout", 91.1), _candidate("circuit-breaker", 80.0)],
        catalog,
    )
    assert without_constraint.supporting == []
    with_constraint = _select(
        _part(constraints=["circuit-breaker stops calling Redis during an outage"]),
        [_candidate("timeout", 91.1), _candidate("circuit-breaker", 80.0)],
        catalog,
    )
    assert [choice.pattern for choice in with_constraint.supporting] == ["circuit-breaker"]


def test_supporting_cap_is_max_supporting(catalog) -> None:
    """§8.2: ``max_supporting`` is a hard cap per part."""
    config = ServerConfig.model_validate({"selection": {"max_supporting": 2}})
    candidates = [
        _candidate("timeout", 91.1),
        _candidate("retry-with-backoff", 80.0),
        _candidate("circuit-breaker", 78.0),
        _candidate("bulkhead", 76.0),
    ]
    constraints = [
        "retry-with-backoff absorbs transient failures",
        "circuit-breaker stops calling a failing dependency",
        "bulkhead isolates the capacity of each caller",
    ]
    selection = _select(_part(constraints=constraints), candidates, catalog, config=config)
    assert len(selection.supporting) == 2


def test_ranking_chain_breaks_ties_deterministically(catalog) -> None:
    """§8.1: equal fit scores fall through analysis → fusion_norm → catalog order → name."""
    selection = _select(
        _part(),
        [
            _candidate("retry-with-backoff", 80.0, analysis=70.0, order=9),
            _candidate("timeout", 80.0, analysis=70.0, fusion=90.0, order=5),
            _candidate("bulkhead", 80.0, analysis=70.0, fusion=90.0, order=2),
        ],
        catalog,
    )
    assert selection.primary == "bulkhead"  # lower catalog order index wins the tie
