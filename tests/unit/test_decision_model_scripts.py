"""Wiring tests for the style-label derivation script.

``derive_style_labels.py`` writes the typed style corpus the linear Ch2 style head trains on;
what is asserted here is the *wiring*: a row accepts only a label inside the served vocabulary
and only criteria that match it (``src/decisions/seam_prompts.py``).
"""

from __future__ import annotations

import importlib.util
import sys
import types

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


@pytest.fixture(scope="module")
def labeler() -> types.ModuleType:
    return _load("derive_style_labels", "derive_style_labels.py")


def test_style_rows_accept_a_label_inside_the_served_vocabulary(labeler: types.ModuleType) -> None:
    """The derivation accepts the shipped corpus: criteria keys ARE the served vocabulary."""
    scenario_id = next(iter(labeler.STYLE_LABELS))
    scenario = {"scenario_id": scenario_id, "description": "a system", "split": "holdout"}
    criteria = {style: seam_prompts.STYLE_GLOSSES[style] for style in seam_prompts.STYLE_VOCABULARY}
    criteria[seam_prompts.STYLE_NONE_KEY] = seam_prompts.STYLE_NONE_GLOSS
    row = labeler.to_style_row(scenario, criteria, "qa/qa.jsonl", "abc123")
    assert row["id"] == f"{scenario_id}:style"
    assert row["label"] == labeler.STYLE_LABELS[scenario_id][0]
    assert set(row["criteria"]) - {seam_prompts.STYLE_NONE_KEY} == set(seam_prompts.STYLE_VOCABULARY)


def test_style_rows_reject_a_label_outside_the_served_vocabulary(
    labeler: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A style the seam can never offer must fail at derivation, not three runs later.

    Deriving the stage-2 shortlist from a §10.1 category left 16 of 22 holdout labels
    unofferable; the labels themselves were never checked against the served answer space.
    """
    scenario_id = next(iter(labeler.STYLE_LABELS))
    monkeypatch.setitem(labeler.STYLE_LABELS, scenario_id, ("not-a-served-style", "because"))
    scenario = {"scenario_id": scenario_id, "description": "a system", "split": "holdout"}
    criteria = {style: seam_prompts.STYLE_GLOSSES[style] for style in seam_prompts.STYLE_VOCABULARY}
    criteria["not-a-served-style"] = "a style the gateway never offers"
    criteria[seam_prompts.STYLE_NONE_KEY] = seam_prompts.STYLE_NONE_GLOSS
    with pytest.raises(SystemExit, match="outside the served style vocabulary"):
        labeler.to_style_row(scenario, criteria, "qa/qa.jsonl", "abc123")


def test_style_rows_reject_criteria_drifted_from_the_served_vocabulary(labeler: types.ModuleType) -> None:
    """The row's criteria must restate the served vocabulary exactly, not a subset."""
    scenario_id = next(iter(labeler.STYLE_LABELS))
    scenario = {"scenario_id": scenario_id, "description": "a system", "split": "holdout"}
    criteria = {style: seam_prompts.STYLE_GLOSSES[style] for style in seam_prompts.STYLE_VOCABULARY}
    criteria[seam_prompts.STYLE_NONE_KEY] = seam_prompts.STYLE_NONE_GLOSS
    label = labeler.STYLE_LABELS[scenario_id][0]
    dropped = next(style for style in seam_prompts.STYLE_VOCABULARY if style != label)
    criteria.pop(dropped)
    with pytest.raises(SystemExit, match="criteria keys drifted"):
        labeler.to_style_row(scenario, criteria, "qa/qa.jsonl", "abc123")
