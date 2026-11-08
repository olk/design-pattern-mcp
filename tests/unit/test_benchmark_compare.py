"""Repeat-stability legs of the benchmark comparison contract.

``_repeat_stability`` measures how often a scenario's primary pattern survives its own
repeats; the style A/B gates on the candidate being non-inferior there. These tests pin the
three states that matter: a mixed split, a split with no repeated scenario (``None``, never a
fabricated 1.0), and the verdict consequences of both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.benchmark.harness import compare


def _arm(name: str, records: list[dict[str, Any]]) -> compare.Arm:
    summary = {
        "quality": {
            "overall": {
                "hit_rate": 0.5,
                "hit_rate_wilson": {"low": 0.3, "high": 0.7},
            }
        }
    }
    return compare.Arm(name, Path("."), {"mode": "e2e"}, summary, records)


def _record(scenario_id: str, repeat: int, pattern: str, split: str = "holdout") -> dict[str, Any]:
    return {
        "scenario": {"scenario_id": scenario_id, "split": split},
        "repeat": repeat,
        "quality": {"final_pattern": pattern},
    }


def test_repeat_stability_counts_only_repeated_scenarios() -> None:
    """A 3-repeat scenario is one observation; a single-record scenario is not evidence."""
    arm = _arm(
        "baseline",
        [
            _record("a", 0, "x"),
            _record("a", 1, "x"),
            _record("a", 2, "x"),
            _record("b", 0, "x"),
            _record("b", 1, "y"),
            _record("b", 2, "x"),
            _record("c", 0, "z"),
        ],
    )
    stability = compare._repeat_stability(arm)
    assert stability["holdout"] == {"scenarios": 2, "stable": 1, "final_pattern_repeat_agreement": 0.5}
    assert stability["train"] == {"scenarios": 0, "stable": 0, "final_pattern_repeat_agreement": None}


def test_verdict_requires_repeat_non_inferiority() -> None:
    """The flag turns a stable gain into a loss when the candidate repeats itself less."""
    baseline = _arm(
        "baseline",
        [_record("a", repeat, "x") for repeat in range(2)] + [_record("b", repeat, "x") for repeat in range(2)],
    )
    candidate = _arm(
        "candidate",
        [_record("a", repeat, "x") for repeat in range(2)] + [_record("b", 0, "x"), _record("b", 1, "y")],
    )
    p95_a = {"samples": 6, "p95_ms": 100.0}
    p95_b = {"samples": 6, "p95_ms": 80.0}
    gated = compare._verdict(baseline, candidate, p95_a, p95_b, require_repeat_stability=True)
    assert gated["outcome"] == "candidate-loss"
    assert gated["rule"] == compare._RULE_REPEAT_STABILITY
    assert any("repeat agreement" in reason for reason in gated["reasons"])
    assert gated["conditions"]["holdout_final_pattern_repeat_agreement"] == {"baseline": 1.0, "candidate": 0.5}
    ungated = compare._verdict(baseline, candidate, p95_a, p95_b)
    assert ungated["outcome"] == "candidate-win"
    assert ungated["rule"] == compare._RULE


def test_verdict_needs_repeats_for_the_stability_leg() -> None:
    """Without >= 2 repeats per holdout scenario the leg cannot be judged, so it fails closed."""
    baseline = _arm("baseline", [_record("a", 0, "x"), _record("b", 0, "x")])
    candidate = _arm("candidate", [_record("a", 0, "x"), _record("b", 0, "x")])
    verdict = compare._verdict(
        baseline,
        candidate,
        {"samples": 6, "p95_ms": 100.0},
        {"samples": 6, "p95_ms": 80.0},
        require_repeat_stability=True,
    )
    assert verdict["outcome"] == "candidate-loss"
    assert any("needs >= 2 repeats" in reason for reason in verdict["reasons"])
