"""The style-head floor calibrator: /v1/systemone replay, floor table, consistency probes.

``scripts/decision_models/style_head_calibrate.py`` replays the holdout rows of
``qa/typed/style.jsonl`` through the running linear style head and fits the Ch2
``decisions.style.confidence_floor``. Everything networked is stubbed here; what is
asserted is the *measurement*: the floor table's acted-accuracy stratification, the
Wilson bands, the repeat/order probes, the served-artifact provenance, and that the
``make style-head-calibrate`` recipe parses against the script's own parser.

The script is loaded by path (it is not an importable package), the same way
``test_client_examples.py`` loads the example clients.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from src.decisions import seam_prompts
from tests.unit.conftest import REPO_ROOT


SCRIPTS = REPO_ROOT / "scripts" / "decision_models"


def _load(name: str, filename: str) -> types.ModuleType:
    """Import one script by path (registered in sys.modules so dataclasses resolve)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module



def test_style_corpus_matches_the_served_prompt_bytes() -> None:
    """Trainer asks exactly what the server asks: qid, instructions, criteria, state.

    ``criteria`` is compared as an ORDERED mapping (``list(row) == list(criteria)``), not
    just as a dict: ``==`` on dicts ignores insertion order, so a re-ordering would pass
    dict equality while serving the linear head a criteria serialisation it never trained on.
    """
    from src.decisions.client import STYLE_QUESTION_ID, build_style_criteria

    style_path = REPO_ROOT / "qa" / "typed" / "style.jsonl"
    criteria = build_style_criteria()
    rows = [json.loads(line) for line in style_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "qa/typed/style.jsonl must not be empty"
    corpus = json.loads((REPO_ROOT / "tests" / "benchmark" / "scenarios" / "full.json").read_text(encoding="utf-8"))
    scenarios = {item["scenario_id"]: item for item in corpus.get("scenarios", corpus)}
    for row in rows:
        scenario_id = row["source"]["scenario_id"]
        assert row["id"] == f"{scenario_id}:style"
        assert row["state"] == scenarios[scenario_id]["description"]
        assert row["instructions"] == seam_prompts.STYLE_INSTRUCTIONS
        assert list(row["criteria"]) == list(criteria)
        assert row["criteria"] == criteria
        assert row["label"] in criteria
        assert row["source"]["split"] == scenarios[scenario_id]["split"]
    assert {row["source"]["scenario_id"] for row in rows} == set(scenarios)
    assert STYLE_QUESTION_ID == "architecture_style"



@pytest.fixture(scope="module")
def calibrator() -> types.ModuleType:
    return _load("style_head_calibrate", "style_head_calibrate.py")


def test_calibration_metrics_match_definitions(calibrator: types.ModuleType) -> None:
    """ECE = binned |acc − conf|, Brier = mean (conf − outcome)²."""
    pairs = [(0.9, True), (0.9, False), (0.1, False), (0.1, False)]
    assert calibrator.brier_score(pairs) == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)
    ece = calibrator.expected_calibration_error(pairs, 10)
    assert ece == pytest.approx(0.5 * abs(0.5 - 0.9) + 0.5 * abs(0.0 - 0.1))


def test_calibration_table_reports_override_accuracy(calibrator: types.ModuleType) -> None:
    """Acted accuracy is over overrides only; confident ``none`` answers are stratified out."""
    pairs = [
        (0.9, True, "hexagonal"),
        (0.8, True, "hexagonal"),
        (0.4, False, "hexagonal"),
        (0.2, False, "hexagonal"),
    ]
    report = calibrator.summarize(pairs, [0.5, 0.85], 10)
    assert report["n"] == 4
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["override_rate"] == pytest.approx(1.0)
    by_floor = {row["floor"]: row for row in report["floors"]}
    assert by_floor[0.5]["acted"] == 2
    assert by_floor[0.5]["acted_accuracy"] == pytest.approx(1.0)
    assert by_floor[0.85]["acted"] == 1
    assert report["confidence"]["min"] == pytest.approx(0.2)
    assert report["confidence"]["max"] == pytest.approx(0.9)


def test_calibration_table_stratifies_abstentions(calibrator: types.ModuleType) -> None:
    """A confident ``none`` is a seam no-op: counted separately, never in the enable bar."""
    pairs = [(0.9, True, "hexagonal"), (0.9, True, "none"), (0.9, False, "none")]
    by_floor = {row["floor"]: row for row in calibrator.summarize(pairs, [0.5], 10)["floors"]}
    row = by_floor[0.5]
    assert row["acted"] == 1
    assert row["acted_rate"] == pytest.approx(1 / 3)
    assert row["acted_accuracy"] == pytest.approx(1.0)
    assert row["none_acted"] == 2
    assert row["none_accuracy"] == pytest.approx(0.5)
    low, high = row["acted_accuracy_ci95"]
    assert low <= 1.0 <= high


def test_calibration_table_bands_small_samples(calibrator: types.ModuleType) -> None:
    """Every floor row carries a 95% Wilson interval; an empty override set spans [0, 1]."""
    pairs = [(0.9, True, "none")]
    by_floor = {row["floor"]: row for row in calibrator.summarize(pairs, [0.5], 10)["floors"]}
    row = by_floor[0.5]
    assert row["acted"] == 0
    assert row["acted_accuracy"] is None
    assert row["acted_accuracy_ci95"] == [0.0, 1.0]
    low, high = calibrator.wilson_interval(24, 30)
    assert 0.0 < low < 0.8 < high < 1.0


def test_calibration_metric_selects_the_floored_number(calibrator: types.ModuleType) -> None:
    answer = {"choice": "x", "confidence": 0.3, "probabilities": {"x": 0.6, "y": 0.4}}
    assert calibrator.metric_of(answer, "confidence") == pytest.approx(0.3)
    assert calibrator.metric_of(answer, "top_prob") == pytest.approx(0.6)


def test_calibration_replays_holdout_only_by_default(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay asks the served question shape and scores choice vs label."""
    seen: list[dict[str, Any]] = []
    path = tmp_path / "style.jsonl"
    rows = [
        {
            "id": "s1:style",
            "state": "system one",
            "instructions": "Which style?",
            "criteria": {"a": "A", "b": "B"},
            "label": "a",
            "source": {"split": "holdout", "scenario_id": "s1"},
        },
        {
            "id": "s2:style",
            "state": "system two",
            "instructions": "Which style?",
            "criteria": {"a": "A", "b": "B"},
            "label": "b",
            "source": {"split": "train", "scenario_id": "s2"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def _fake_ask(
        base_url: str,
        model: str,
        row: dict[str, Any],
        question_id: str,
        timeout: float,
        criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        seen.append({"model": model, "state": row["state"], "qid": question_id, "criteria": criteria})
        return {"choice": "a", "confidence": 0.7, "probabilities": {"a": 0.7, "b": 0.3}}

    monkeypatch.setattr(calibrator, "ask", _fake_ask)
    assert calibrator.main(["--style", str(path), "--base-url", "http://x/v1", "--floors", "0.5"]) == 0
    assert [entry["state"] for entry in seen] == ["system one"]
    assert seen[0]["qid"] == "architecture_style"


def test_recommend_picks_the_lowest_qualifying_floor(calibrator: types.ModuleType) -> None:
    """The recommendation is the lowest floor clearing both bars, not the most accurate one."""
    pairs = [
        (0.9, True, "hexagonal"),
        (0.8, True, "hexagonal"),
        (0.7, False, "hexagonal"),
        (0.6, False, "hexagonal"),
        (0.2, False, "hexagonal"),
    ]
    table = calibrator.summarize(pairs, [0.5, 0.65, 0.85], 10)["floors"]
    # 0.5 acts on four rows at 0.50 accuracy (fails the 0.6 bar); 0.65 acts on two of
    # three at 0.667 — so the recommendation is the lowest floor that clears both.
    # min_acted=1: this test targets floor selection, not the small-n guard.
    recommended = calibrator.recommend(table, 0.6, 0.4, 1)
    assert recommended["floor"] == pytest.approx(0.65)
    assert "acted accuracy" in recommended["reason"]


def test_recommend_refuses_a_floor_below_the_min_acted_count(calibrator: types.ModuleType) -> None:
    """A perfect floor on too few overrides never qualifies, however accurate."""
    pairs = [(0.9, True, "hexagonal"), (0.9, True, "hexagonal")]
    table = calibrator.summarize(pairs, [0.5], 10)["floors"]
    assert calibrator.recommend(table, 0.8, 0.1, 10)["floor"] is None
    assert calibrator.recommend(table, 0.8, 0.1, 2)["floor"] == pytest.approx(0.5)
    qualified = calibrator.recommend(table, 0.8, 0.1, 2)
    assert qualified["acted"] == 2
    assert qualified["acted_accuracy_ci95"][0] <= 1.0 <= qualified["acted_accuracy_ci95"][1]


def test_recommend_reports_nothing_when_no_floor_qualifies(calibrator: types.ModuleType) -> None:
    pairs = [(0.9, False, "hexagonal"), (0.8, False, "hexagonal"), (0.7, False, "hexagonal")]
    table = calibrator.summarize(pairs, [0.5, 0.85], 10)["floors"]
    recommended = calibrator.recommend(table, 0.9, 0.3)
    assert recommended["floor"] is None
    assert "cannot be enabled safely" in recommended["reason"]


def test_recommend_reports_only_without_thresholds(calibrator: types.ModuleType) -> None:
    table = calibrator.summarize([(0.9, True, "hexagonal")], [0.5], 10)["floors"]
    assert calibrator.recommend(table, None, None)["floor"] is None
    assert calibrator.recommend(table, 0.5, None)["floor"] is None


def test_calibration_main_fails_when_no_floor_qualifies(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corpus whose head is never right must not yield an enableable floor."""
    path = tmp_path / "style.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "s1:style",
                "state": "system one",
                "instructions": "Which style?",
                "criteria": {"a": "A", "b": "B"},
                "label": "a",
                "source": {"split": "holdout", "scenario_id": "s1"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "b", "confidence": 0.9, "probabilities": {"a": 0.1, "b": 0.9}},
    )
    report = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(path),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--report",
            str(report),
        ]
    )
    assert code == 2
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["recommended"]["floor"] is None
    assert written["floors"][0]["acted_accuracy"] == pytest.approx(0.0)


def _write_labels_source(path: Path) -> str:
    """A minimal STYLE_LABELS module; returns its sha256 for the test ledger."""
    source = 'STYLE_LABELS: dict[str, tuple[str, str]] = {\n    "s1": ("a", "rationale one"),\n}\n'
    path.write_text(source, encoding="utf-8")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _write_style_row(path: Path, rows: int = 1) -> None:
    """One acted override per row (all choice ``a``); ``rows`` clears the min-acted bar."""
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"s{index}:style",
                    "state": f"system {index}",
                    "instructions": "Which style?",
                    "criteria": {"a": "A", "b": "B"},
                    "label": "a",
                    "source": {"split": "holdout", "scenario_id": f"s{index}"},
                }
            )
            for index in range(1, rows + 1)
        ),
        encoding="utf-8",
    )


def test_calibration_main_refuses_to_recommend_without_human_review(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The table prints, but an unreviewed ledger blocks the floor and fails the run."""
    style = tmp_path / "style.jsonl"
    _write_style_row(style, rows=10)
    digest = _write_labels_source(tmp_path / "labels.py")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"status": "unreviewed", "reviews": {"style": []}}), encoding="utf-8")
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}},
    )
    report = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--labels",
            str(tmp_path / "labels.py"),
            "--review",
            str(review),
            "--report",
            str(report),
        ]
    )
    assert code == 2
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["label_review"]["certified"] is False
    assert written["recommended"]["floor"] is None
    assert "not 'reviewed'" in written["recommended"]["reason"]
    assert written["floors"][0]["acted_accuracy"] == pytest.approx(1.0)
    assert digest  # the ledger below pins these exact bytes


def test_calibration_main_recommends_once_the_ledger_certifies(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full coverage at the recorded sha unlocks the recommendation the table supports."""
    style = tmp_path / "style.jsonl"
    _write_style_row(style, rows=10)
    digest = _write_labels_source(tmp_path / "labels.py")
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "status": "reviewed",
                "reviewers": ["ada"],
                "reviews": {
                    "style": [
                        {
                            "scenario_id": "s1",
                            "label": "a",
                            "reviewer": "ada",
                            "rationale_ok": True,
                            "labels_sha256": digest,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}},
    )
    report = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--labels",
            str(tmp_path / "labels.py"),
            "--review",
            str(review),
            "--report",
            str(report),
        ]
    )
    assert code == 0
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["label_review"]["certified"] is True
    assert written["recommended"]["floor"] == pytest.approx(0.5)


def test_calibration_main_voids_coverage_when_labels_change(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label edit after review changes the bytes, so the recorded sha no longer certifies."""
    style = tmp_path / "style.jsonl"
    _write_style_row(style, rows=10)
    digest = _write_labels_source(tmp_path / "labels.py")
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "status": "reviewed",
                "reviews": {
                    "style": [
                        {
                            "scenario_id": "s1",
                            "label": "a",
                            "reviewer": "ada",
                            "rationale_ok": True,
                            "labels_sha256": digest,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    with (tmp_path / "labels.py").open("a", encoding="utf-8") as handle:
        handle.write("# post-review tweak\n")
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}},
    )
    report = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--labels",
            str(tmp_path / "labels.py"),
            "--review",
            str(review),
            "--report",
            str(report),
        ]
    )
    assert code == 2
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["recommended"]["floor"] is None



def test_calibrator_none_choice_matches_the_served_vocabulary(calibrator: types.ModuleType) -> None:
    """The stratified ``none`` must be the seam's own abstention key, not a lookalike."""
    assert calibrator.NONE_CHOICE == seam_prompts.STYLE_NONE_KEY



def test_repeat_consistency_counts_flips_and_acted_floor_crossings(calibrator: types.ModuleType) -> None:
    """A choice that moves across identical replays is a determinism defect, not noise.

    The probe reports it three ways because they fail differently: a flip at all
    (``style_choice_repeat_agreement``), the confidence spread (a head whose softmax is
    near-tied), and per-floor acted-set flips (a row that crosses the flap the floor sits on).
    """
    stable = {"scenario_id": "s1", "label": "a", "choice": "a", "confidence": 0.9}
    flips = {"scenario_id": "s2", "label": "b", "choice": "b", "confidence": 0.55}
    flips_again = {**flips, "choice": "a", "confidence": 0.75}
    straddle = {"scenario_id": "s3", "label": "a", "choice": "a", "confidence": 0.45}
    straddle_high = {**straddle, "confidence": 0.65}
    report = calibrator.consistency(
        [[stable, flips, straddle], [stable, flips_again, straddle_high]],
        [0.5, 0.7],
    )
    assert report is not None
    assert report["replays"] == 2
    # s1 and s3 keep their choice; only s2 flips — s3 moves on confidence alone, which the
    # acted-flip columns catch at the floors it straddles.
    assert report["stable_rows"] == 2
    assert report["style_choice_repeat_agreement"] == pytest.approx(2 / 3)
    assert report["unstable_rows"] == ["s2"]
    assert report["acted_flips"] == {"0.5": 1, "0.7": 1}
    assert report["max_confidence_spread"] == pytest.approx(0.2)


def test_repeat_consistency_needs_two_replays(calibrator: types.ModuleType) -> None:
    """One replay measures nothing; the block is None rather than a perfect score."""
    row = {"scenario_id": "s1", "label": "a", "choice": "a", "confidence": 0.9}
    assert calibrator.consistency([[row]], [0.5]) is None


def test_permute_criteria_keeps_the_answer_set_and_only_reorders_it(calibrator: types.ModuleType) -> None:
    """A permutation must preserve every item: reordering an answer set may not edit it."""
    import random

    criteria = {f"style-{index}": f"gloss {index}" for index in range(36)}
    shuffled = calibrator.permute_criteria(criteria, random.Random(7))
    assert list(shuffled) != list(criteria)
    assert dict(shuffled) == dict(criteria)
    assert list(shuffled) == list(calibrator.permute_criteria(criteria, random.Random(7)))


def test_order_invariance_separates_flips_from_canonical_drift(calibrator: types.ModuleType) -> None:
    """Order agreement is the gated number; canonical drift is reported beside it.

    A head can be perfectly order-stable yet disagree with the canonical run (all its
    permutations answer the same wrong way), which the agreement number alone would hide.
    """
    run_a = [
        {"scenario_id": "s1", "choice": "a", "confidence": 0.9},
        {"scenario_id": "s2", "choice": "a", "confidence": 0.8},
    ]
    run_b = [
        {"scenario_id": "s1", "choice": "a", "confidence": 0.9},
        {"scenario_id": "s2", "choice": "b", "confidence": 0.7},
    ]
    drift = [
        {"scenario_id": "s1", "choice": "b", "confidence": 0.6},
        {"scenario_id": "s2", "choice": "b", "confidence": 0.7},
    ]
    report = calibrator.order_invariance([run_a, run_a], ["a", "a"])
    assert report is not None
    assert report["style_choice_order_agreement"] == pytest.approx(1.0)
    assert report["canonical_agreement"] == pytest.approx(1.0)
    unstable = calibrator.order_invariance([run_a, run_b], ["a", "a"])
    assert unstable is not None
    assert unstable["style_choice_order_agreement"] == pytest.approx(0.5)
    assert unstable["unstable_rows"] == ["s2"]
    drifted = calibrator.order_invariance([drift, drift], ["a", "a"])
    assert drifted is not None
    assert drifted["style_choice_order_agreement"] == pytest.approx(1.0)
    assert drifted["canonical_agreement"] == pytest.approx(0.0)
    assert calibrator.order_invariance([], ["a"]) is None


def test_calibration_main_records_both_probes_in_the_report(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--repeats``/``--permutations`` write the two consistency blocks into the artifact."""
    style = tmp_path / "style.jsonl"
    _write_style_row(style, rows=4)
    _write_labels_source(tmp_path / "labels.py")
    seen_criteria: list[dict[str, Any] | None] = []

    def _ask(base_url: str, model: str, row: dict[str, Any], question_id: str, timeout: float, criteria: Any = None) -> dict[str, Any]:
        seen_criteria.append(criteria)
        return {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}}

    monkeypatch.setattr(calibrator, "ask", _ask)
    report_path = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--repeats",
            "3",
            "--permutations",
            "2",
            "--report",
            str(report_path),
        ]
    )
    assert code == 0
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["consistency"]["replays"] == 3
    assert written["consistency"]["style_choice_repeat_agreement"] == pytest.approx(1.0)
    assert written["order_invariance"]["orderings"] == 2
    assert written["order_invariance"]["style_choice_order_agreement"] == pytest.approx(1.0)
    # 4 rows x (3 repeats + 2 permutations) = 20 calls, and only the permuted runs override.
    assert len(seen_criteria) == 20
    assert sum(1 for criteria in seen_criteria if criteria is not None) == 8
    assert all(sorted(criteria) == ["a", "b"] for criteria in seen_criteria if criteria is not None)  # type: ignore[arg-type]


def test_calibration_main_refuses_a_floor_on_a_nondeterministic_head(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A floor fitted on a head that flips on identical bytes is a threshold on noise."""
    style = tmp_path / "style.jsonl"
    rows = 10  # clears the default --min-acted=10 guard, so only the repeat probe can block
    _write_style_row(style, rows=rows)
    digest = _write_labels_source(tmp_path / "labels.py")
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "status": "reviewed",
                "reviewers": ["ada"],
                "reviews": {
                    "style": [
                        {
                            "scenario_id": "s1",
                            "label": "a",
                            "reviewer": "ada",
                            "rationale_ok": True,
                            "labels_sha256": digest,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    calls = {"n": 0}

    def _flipping(base_url: str, model: str, row: dict[str, Any], question_id: str, timeout: float, criteria: Any = None) -> dict[str, Any]:
        # First replay all correct (so the floor table alone clears every bar); the second
        # replay flips every row. Only the repeat probe may then block the recommendation.
        calls["n"] += 1
        return {"choice": "a" if calls["n"] <= rows else "b", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}}

    monkeypatch.setattr(calibrator, "ask", _flipping)
    report_path = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--labels",
            str(tmp_path / "labels.py"),
            "--review",
            str(review),
            "--repeats",
            "2",
            "--report",
            str(report_path),
        ]
    )
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["recommended"]["floor"] is None
    assert "flip choice across identical replays" in written["recommended"]["reason"]
    assert written["consistency"]["stable_rows"] < written["consistency"]["rows"]
    assert code == 2


def test_calibration_main_tolerates_one_flip_where_the_gate_does(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal bar is the seam's 0.99 target, not 1.0 — a PASSING arm must exit 0.

    Two contradicting verdicts on one measurement are worse than either alone: if the script
    refused on any flip, a 199/200 arm (agreement 0.995, objective PASS) would exit 2 and fail
    ``make style-head-calibrate``'s exit-0 requirement. The bar is inclusive, so on 200 rows it
    admits 2 flips (exactly 0.99) and rejects 3. On the real 138-row instrument it admits 1.
    """
    style = tmp_path / "style.jsonl"
    rows = 200
    _write_style_row(style, rows=rows)
    digest = _write_labels_source(tmp_path / "labels.py")
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "status": "reviewed",
                "reviewers": ["ada"],
                "reviews": {
                    "style": [
                        {
                            "scenario_id": "s1",
                            "label": "a",
                            "reviewer": "ada",
                            "rationale_ok": True,
                            "labels_sha256": digest,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    def _run(flips: int) -> tuple[int, dict[str, Any]]:
        calls = {"n": 0}

        def _ask(base_url: str, model: str, row: dict[str, Any], question_id: str, timeout: float, criteria: Any = None) -> dict[str, Any]:
            # Replay 1 is perfect (the floor table is computed from it); replay 2 flips
            # ``flips`` rows.
            calls["n"] += 1
            positional = calls["n"] - rows
            flip = 0 < positional <= flips
            return {
                "choice": "b" if flip else "a",
                "confidence": 0.9,
                "probabilities": {"a": 0.9, "b": 0.1},
            }

        monkeypatch.setattr(calibrator, "ask", _ask)
        report_path = tmp_path / f"calibration-{flips}.json"
        code = calibrator.main(
            [
                "--style",
                str(style),
                "--base-url",
                "http://x/v1",
                "--floors",
                "0.5",
                "--min-acted-accuracy",
                "0.8",
                "--min-acted-rate",
                "0.1",
                "--labels",
                str(tmp_path / "labels.py"),
                "--review",
                str(review),
                "--repeats",
                "2",
                "--report",
                str(report_path),
            ]
        )
        return code, json.loads(report_path.read_text(encoding="utf-8"))

    code, written = _run(1)
    assert written["consistency"]["style_choice_repeat_agreement"] == pytest.approx(199 / 200)
    assert written["recommended"]["floor"] == pytest.approx(0.5)
    assert code == 0

    # Exactly at the bar (198/200 == 0.99) still recommends: the comparison is inclusive.
    code, written = _run(2)
    assert written["consistency"]["style_choice_repeat_agreement"] == pytest.approx(0.99)
    assert written["recommended"]["floor"] == pytest.approx(0.5)
    assert code == 0

    code, written = _run(3)
    assert written["consistency"]["style_choice_repeat_agreement"] == pytest.approx(197 / 200)
    assert written["recommended"]["floor"] is None
    assert "flip choice across identical replays" in written["recommended"]["reason"]
    assert code == 2


def test_calibration_accepts_exactly_the_argv_the_make_rule_passes(calibrator: types.ModuleType) -> None:
    """The ``style-head-calibrate`` recipe must parse — including the inspected provenance flags.

    A ``make -n`` run only PRINTS the recipe, so an option the script never defined is
    invisible until the first real run, hours into a GPU session. This pins the contract:
    every flag in the Makefile's ``style-head-calibrate`` target parses, and the provenance values
    land in the report instead of being dropped.
    """
    argv = [
        "--style",
        "qa/typed/style.jsonl",
        "--base-url",
        "http://127.0.0.1:18800/v1",
        "--floors",
        *["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"],  # unquoted in the recipe: shell-split
        "--metric",
        "confidence",
        "--min-acted-accuracy",
        "0.8",
        "--min-acted-rate",
        "0.3",
        "--min-acted",
        "10",
        "--repeats",
        "5",
        "--permutations",
        "5",
        "--report",
        "models/style-head-calibration.json",
        "--served-container",
        "design-pattern-style-head-dev",
        "--served-image",
        "sha256:abc",
        "--served-image-tag",
        "design-pattern-style-head:1.0.2",
        "--served-container-started-at",
        "2026-11-13T10:00:00Z",
        "--served-model-repo",
        "built-in-image (qa/typed/style.jsonl)",
    ]
    args = calibrator.parse_args(argv)
    assert args.repeats == 5
    assert args.permutations == 5
    assert args.floors == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert args.served_image == "sha256:abc"
    assert args.served_model_repo == "built-in-image (qa/typed/style.jsonl)"



def test_calibration_report_carries_the_served_artifact(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A docker TAG is not a served artifact; the report pins the image ID and start time."""
    style = tmp_path / "style.jsonl"
    _write_style_row(style, rows=2)
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}},
    )
    report_path = tmp_path / "calibration.json"
    calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://127.0.0.1:18800/v1",
            "--floors",
            "0.5",
            "--report",
            str(report_path),
            "--served-container",
            "design-pattern-style-head-dev",
            "--served-image",
            "sha256:deadbeef",
            "--served-image-tag",
            "design-pattern-style-head:1.0.2",
            "--served-container-started-at",
            "2026-11-13T10:00:00Z",
            "--served-model-repo",
            "built-in-image (qa/typed/style.jsonl)",
        ]
    )
    served = json.loads(report_path.read_text(encoding="utf-8"))["served"]
    assert served == {
        "base_url": "http://127.0.0.1:18800/v1",
        "container": "design-pattern-style-head-dev",
        "image_id": "sha256:deadbeef",
        "image_tag": "design-pattern-style-head:1.0.2",
        "container_started_at": "2026-11-13T10:00:00Z",
        "model_repo": "built-in-image (qa/typed/style.jsonl)",
    }



def test_calibration_report_carries_per_row_decisions(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The arm comparison is PAIRED per row; aggregates cannot reconstruct which rows moved.

    ``E2 - E1`` has to name flipped rows and per-class deltas, so each report must carry the
    decision it made for each scenario — with the identity to pair on, the label to score
    against, and the number the floor was fitted on.
    """
    style = tmp_path / "style.jsonl"
    _write_style_row(style, rows=3)
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "a", "confidence": 0.8, "probabilities": {"a": 0.8, "b": 0.2}},
    )
    report_path = tmp_path / "calibration.json"
    calibrator.main(["--style", str(style), "--base-url", "http://x/v1", "--floors", "0.5", "--report", str(report_path)])
    rows = json.loads(report_path.read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 3
    assert [row["scenario_id"] for row in rows] == ["s1", "s2", "s3"]
    for row in rows:
        assert set(row) == {"scenario_id", "label", "choice", "confidence"}
        assert row["label"] == "a" and row["choice"] == "a"
        assert row["confidence"] == pytest.approx(0.8)



def test_style_corpus_holdout_is_family_disjoint_and_not_paraphrased() -> None:
    """The holdout only means something if it is uncontaminated.

    The gate's central claim is "this head generalizes to unseen families". That is a
    property of the DATA, not of the trainer, so a later re-split could silently destroy it
    while every other test stayed green. Three separate leakage routes are checked:

    1. family — the corpus carries a ``family`` field; a family spanning the split means the
       head was trained on siblings of the rows it is scored on;
    2. scenario identity — the same scenario must not appear twice;
    3. paraphrase — a near-identical description is leakage even under a different id, so the
       highest 5-gram Jaccard between any train x holdout pair must stay well below the
       duplication threshold (it measures 0.148; 0.80 is a generous ceiling).
    """
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "qa" / "typed" / "style.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train = [row for row in rows if row["source"]["split"] == "train"]
    holdout = [row for row in rows if row["source"]["split"] == "holdout"]
    assert (len(train), len(holdout)) == (719, 138)

    corpus = json.loads((REPO_ROOT / "tests" / "benchmark" / "scenarios" / "full.json").read_text(encoding="utf-8"))
    scenarios = {item["scenario_id"]: item for item in corpus.get("scenarios", corpus)}

    # 1. No family may straddle the split.
    train_families = {scenarios[row["source"]["scenario_id"]]["family"] for row in train}
    holdout_families = {scenarios[row["source"]["scenario_id"]]["family"] for row in holdout}
    assert not (train_families & holdout_families), f"families straddle the split: {train_families & holdout_families}"

    # 2. No shared scenario identity.
    train_ids = {row["source"]["scenario_id"] for row in train}
    assert not (train_ids & {row["source"]["scenario_id"] for row in holdout})

    # 3. No paraphrase duplication.
    def shingles(text: str, width: int = 5) -> set[str]:
        words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
        return {" ".join(words[index : index + width]) for index in range(max(0, len(words) - width + 1))}

    train_shingles = [shingles(row["state"]) for row in train]
    holdout_shingles = [shingles(row["state"]) for row in holdout]
    worst = 0.0
    for held in holdout_shingles:
        for seen in train_shingles:
            if not held or not seen:
                continue
            intersection = len(held & seen)
            if intersection:
                worst = max(worst, intersection / len(held | seen))
    assert worst < 0.80, f"a holdout state paraphrases a train state (5-gram Jaccard {worst:.3f})"
