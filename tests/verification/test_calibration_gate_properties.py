"""L2 oracles for the Ch2 style calibration gate (scripts/decision_models/style_head_calibrate.py).

The gate decides whether a confidence floor may ship, so its arithmetic must hold for any
replay table, not just the corpus that happened to be measured:

  CG-1  monotonicity: acted count and hits are non-increasing as the floor rises.
  CG-2  recommendation honesty: a recommended floor clears all three bars and no lower
        tabulated floor does.

The calibrator is a build-time script (not an importable package), loaded by path exactly as
``tests/unit/test_style_head_calibrate.py`` loads it.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.unit.conftest import REPO_ROOT

_FLOORS: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "decision_models" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calibrator = _load("style_head_calibrate_properties", "style_head_calibrate.py")

_choices: st.SearchStrategy[str] = st.sampled_from(["a", "none"])
_labels: st.SearchStrategy[str] = st.sampled_from(["a", "b", "none"])
_row = st.fixed_dictionaries(
    {
        "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "choice": _choices,
        "label": _labels,
    }
)
_floors = st.lists(st.sampled_from(_FLOORS), min_size=1, max_size=len(_FLOORS), unique=True).map(sorted)


def _pairs(rows: Sequence[dict[str, Any]]) -> list[tuple[float, bool, str]]:
    return [(float(row["confidence"]), row["choice"] == row["label"], row["choice"]) for row in rows]


def _qualifies(row: dict[str, Any], accuracy: float, rate: float, min_acted: int) -> bool:
    return (
        row["acted_accuracy"] is not None
        and int(row["acted"]) >= min_acted
        and float(row["acted_accuracy"]) >= accuracy
        and float(row["acted_rate"]) >= rate
    )


class TestCG1Monotonicity:
    @given(rows=st.lists(_row, min_size=1, max_size=40), floors=_floors)
    @settings(max_examples=60, deadline=None)
    def test_acted_and_hits_never_rise_with_the_floor(self, rows: list[dict[str, Any]], floors: list[float]) -> None:
        pairs = _pairs(rows)
        table = calibrator.summarize(pairs, floors, 10)["floors"]
        acted = [int(row["acted"]) for row in table]
        assert acted == sorted(acted, reverse=True)
        hits = [
            sum(
                1
                for confidence, correct, choice in pairs
                if confidence >= row["floor"] and choice != "none" and correct
            )
            for row in table
        ]
        assert hits == sorted(hits, reverse=True)


class TestCG2RecommendationHonesty:
    @given(
        rows=st.lists(_row, min_size=1, max_size=40),
        floors=_floors,
        accuracy=st.sampled_from([0.5, 0.8]),
        rate=st.sampled_from([0.1, 0.3]),
        min_acted=st.sampled_from([1, 5, 10]),
    )
    @settings(max_examples=80, deadline=None)
    def test_recommended_floor_clears_the_bars_and_is_the_lowest(
        self,
        rows: list[dict[str, Any]],
        floors: list[float],
        accuracy: float,
        rate: float,
        min_acted: int,
    ) -> None:
        table = calibrator.summarize(_pairs(rows), floors, 10)["floors"]
        recommended = calibrator.recommend(table, accuracy, rate, min_acted)
        if recommended["floor"] is None:
            return
        ordered = sorted(table, key=lambda row: row["floor"])
        chosen = next(row for row in ordered if row["floor"] == recommended["floor"])
        assert _qualifies(chosen, accuracy, rate, min_acted)
        for row in ordered:
            if row["floor"] < chosen["floor"]:
                assert not _qualifies(row, accuracy, rate, min_acted)
