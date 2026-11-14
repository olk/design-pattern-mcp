"""The linear style head's trainer: split placement, label mapping, and the build gate.

The trainer runs inside ``docker/Dockerfile.style-head`` and is the only producer of the
head bytes, so a wrong split placement would score the head on rows it trained on, and a
column-index label mapping would report a plausible accuracy over permuted answers. Both
mistakes are invisible in the build log — the gate would still print a number — which is
why they are pinned here instead.

scikit-learn is not a dependency of this repo (it lives only in the sidecar image), so
``main`` is exercised with stub modules injected into ``sys.modules`` and a stubbed
``train_pipeline``: what is asserted is the trainer's control flow, the artifacts it writes,
and its exit code, not sklearn's arithmetic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from src.decisions import seam_prompts
from tests.unit.conftest import REPO_ROOT


SCRIPTS = REPO_ROOT / "scripts" / "decision_models"
SEAM_PROMPTS = REPO_ROOT / "src" / "decisions" / "seam_prompts.py"


def _load(name: str, filename: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer() -> types.ModuleType:
    return _load("train_style_head", "train_style_head.py")


class _StubPipeline:
    """A fitted pipeline stand-in: fixed per-state scores in the MODEL's class order."""

    def __init__(self, model_classes: list[str], by_state: dict[str, dict[str, float]]) -> None:
        self.classes_ = list(model_classes)
        self._by_state = by_state

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[self._by_state[text].get(label, 0.0) for label in self.classes_] for text in texts]


def _row(row_id: str, split: str, label: str, state: str | None = None) -> dict[str, Any]:
    return {
        "id": row_id,
        "state": state or f"state of {row_id}",
        "label": label,
        "source": {"split": split},
    }


def test_canonical_classes_end_with_the_none_key(trainer: types.ModuleType) -> None:
    """The served answer space is the shared vocabulary plus ``none``, in that order."""
    classes = trainer.canonical_classes(seam_prompts)
    assert classes == [*seam_prompts.STYLE_VOCABULARY, seam_prompts.STYLE_NONE_KEY]


def test_split_placement_takes_the_holdout_out_of_training(trainer: types.ModuleType) -> None:
    """Rows partition by corpus split, and an empty side is a refusal, not a silent 0.0."""
    classes = trainer.canonical_classes(seam_prompts)
    rows = [
        _row("a", "train", "microservices"),
        _row("b", "train", "hexagonal"),
        _row("c", "holdout", "microservices"),
    ]
    train, evaluate = trainer.split_rows(rows, "holdout", classes)
    assert [row["id"] for row in train] == ["a", "b"]
    assert [row["id"] for row in evaluate] == ["c"]
    with pytest.raises(SystemExit) as no_train:
        trainer.split_rows([_row("c", "holdout", "microservices")], "holdout", classes)
    assert "no training rows" in str(no_train.value)
    with pytest.raises(SystemExit) as no_eval:
        trainer.split_rows([_row("a", "train", "microservices")], "holdout", classes)
    assert "no rows in the 'holdout' split" in str(no_eval.value)


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_row("x", "train", "not-a-style"), id="label-outside-the-class-set"),
        pytest.param(_row("x", "train", "microservices", state="  "), id="blank-state"),
        pytest.param({"id": "x", "state": "s", "label": "microservices"}, id="no-source-split"),
    ],
)
def test_a_row_the_head_cannot_learn_is_a_refusal(trainer: types.ModuleType, row: dict[str, Any]) -> None:
    """Dropping a bad row would train a head that cannot answer the served question."""
    classes = trainer.canonical_classes(seam_prompts)
    with pytest.raises(SystemExit):
        trainer.split_rows([_row("ok", "train", "microservices"), row], "holdout", classes)


def test_probabilities_map_onto_the_served_order_by_label(trainer: types.ModuleType) -> None:
    """``classes_`` is alphabetical, so column mapping would permute 36 answers into each other.

    Here the model's two columns are in the REVERSE of the served order, and a third served
    class is unknown to the model — the two ways an index-based mapping goes wrong.
    """
    served = ["layered-monolith", "event-driven", "microservices"]
    aligned = trainer.align_probabilities(["event-driven", "layered-monolith"], [0.9, 0.1], served)
    assert aligned == {"layered-monolith": 0.1, "event-driven": 0.9, "microservices": 0.0}
    assert list(aligned) == served
    assert trainer.top_label(aligned, served) == "event-driven"


def test_accuracy_and_majority_are_scored_on_the_eval_shard(trainer: types.ModuleType) -> None:
    """Accuracy is hits/rows over held-out rows, with the majority share as the margin base."""
    classes = trainer.canonical_classes(seam_prompts)
    rows = [
        _row("a", "holdout", "microservices", state="s1"),
        _row("b", "holdout", "microservices", state="s2"),
        _row("c", "holdout", "hexagonal", state="s3"),
    ]
    pipeline = _StubPipeline(
        ["hexagonal", "microservices"],
        {
            "s1": {"microservices": 0.8},
            "s2": {"microservices": 0.6},
            "s3": {"microservices": 0.7},
        },
    )
    summary = trainer.accuracy_summary(pipeline, rows, classes)
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["majority_share"] == pytest.approx(2 / 3)
    assert summary["per_class"]["microservices"] == [2, 2]
    assert summary["per_class"]["hexagonal"] == [0, 1]
    # Only classes the shard actually contains appear, so the report stays readable.
    assert "none" not in summary["per_class"]


@pytest.mark.parametrize(
    ("accuracy", "majority", "gate", "margin", "clears"),
    [
        pytest.param(0.833, 0.094, 0.65, 0.10, True, id="measured-linear"),
        pytest.param(0.507, 0.094, 0.65, 0.10, False, id="rejected-neural-run-below-bar"),
        pytest.param(0.70, 0.62, 0.65, 0.10, False, id="clears-the-bar-not-the-margin"),
    ],
)
def test_the_gate_needs_both_the_absolute_bar_and_the_margin(
    trainer: types.ModuleType,
    accuracy: float,
    majority: float,
    gate: float,
    margin: float,
    clears: bool,
) -> None:
    """A head that only matches the majority label is a class prior wearing an accuracy."""
    verdict = trainer.gate_failure(accuracy, majority, gate, margin)
    assert (verdict is None) is clears


def _install_fake_modules(trainer: types.ModuleType, monkeypatch: pytest.MonkeyPatch, dump_calls: list[Path]) -> None:
    """``joblib``/``sklearn`` are image-only deps; the trainer imports them inside ``main``."""

    def _dump(_pipeline: Any, path: Path, compress: int = 0) -> None:
        assert compress == 3
        path.write_bytes(b"stub-head")
        dump_calls.append(path)

    joblib = types.ModuleType("joblib")
    joblib.dump = _dump  # type: ignore[attr-defined]
    sklearn = types.ModuleType("sklearn")
    sklearn.__version__ = "test-0.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "joblib", joblib)
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    # The pre-registered LogisticRegression arm is measured beside the shipped head on every
    # build; it needs sklearn too, so it is stubbed with the number it actually produced.
    monkeypatch.setattr(trainer, "reference_logreg_accuracy", lambda *args, **kwargs: 0.4348)


def _write_corpus(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_main_refuses_a_head_below_the_gate_and_still_writes_the_evidence(
    trainer: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 2 with a FAIL line; the artifacts stay behind so the log can be diagnosed."""
    dump_calls: list[Path] = []
    _install_fake_modules(trainer, monkeypatch, dump_calls)
    rows = [
        _row("t1", "train", "microservices", state="t1"),
        _row("e1", "holdout", "microservices", state="e1"),
        _row("e2", "holdout", "hexagonal", state="e2"),
    ]
    corpus = _write_corpus(tmp_path / "style.jsonl", rows)
    monkeypatch.setattr(
        trainer,
        "train_pipeline",
        lambda train, seed: _StubPipeline(
            ["hexagonal", "microservices"],
            {"e1": {"hexagonal": 0.9}, "e2": {"hexagonal": 0.9}},
        ),
    )
    out_dir = tmp_path / "model"
    code = trainer.main(
        [
            "--style",
            str(corpus),
            "--seam-prompts",
            str(SEAM_PROMPTS),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 2
    assert dump_calls == [out_dir / "style_head.joblib"]
    report = json.loads((out_dir / "style-head-report.json").read_text(encoding="utf-8"))
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["majority_share"] == pytest.approx(0.5)
    assert report["eval_rows"] == 2
    assert report["train_rows"] == 1
    assert report["eval_split"] == "holdout"
    assert report["classes"] == trainer.canonical_classes(seam_prompts)
    assert report["per_class"] == {"microservices": [0, 1], "hexagonal": [1, 1]}
    # The recipe correction is auditable from the artifact: the shipped estimator and the
    # plan's original arm travel together, so neither number can be read without the other.
    assert report["estimator"] == trainer.ESTIMATOR
    assert report["preregistered"]["estimator"] == trainer.PREREGISTERED_ESTIMATOR
    assert report["preregistered"]["accuracy"] == pytest.approx(0.4348)
    assert "style-head-gate: FAIL" in capsys.readouterr().err


def test_main_passes_a_head_that_clears_both_bars(
    trainer: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 0 with the PASS line the Dockerfile's build gate is read by."""
    dump_calls: list[Path] = []
    _install_fake_modules(trainer, monkeypatch, dump_calls)
    rows = [
        _row("t1", "train", "microservices", state="t1"),
        _row("e1", "holdout", "microservices", state="e1"),
        _row("e2", "holdout", "hexagonal", state="e2"),
    ]
    corpus = _write_corpus(tmp_path / "style.jsonl", rows)
    monkeypatch.setattr(
        trainer,
        "train_pipeline",
        lambda train, seed: _StubPipeline(
            ["hexagonal", "microservices"],
            {"e1": {"microservices": 0.9}, "e2": {"hexagonal": 0.9}},
        ),
    )
    out_dir = tmp_path / "model"
    code = trainer.main(["--style", str(corpus), "--seam-prompts", str(SEAM_PROMPTS), "--out-dir", str(out_dir)])
    assert code == 0
    out = capsys.readouterr().out
    assert "style-head-gate: PASS holdout=1.0000" in out
    assert "sklearn test-0.0" in out
