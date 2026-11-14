"""The linear style head's serving wire: label mapping, confidence, and request refusal.

``scripts/decision_models/style_head_serve.py`` runs in the CPU sidecar image, where
scikit-learn is installed and the MCP server is not; the pure functions here are what a
wrong probability column mapping or a permissive request path would break, and both fail
silently in production (the seam's degrade-never-raise contract turns a mis-answer into an
injected wrong style). The pipeline is a stub because the assertions are about the mapping,
not about scikit-learn.

Loaded by path, like the other image-build scripts (``test_style_head_calibrate.py``).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

import pytest

from tests.unit.conftest import REPO_ROOT


SCRIPTS = REPO_ROOT / "scripts" / "decision_models"


def _load(name: str, filename: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def serving() -> types.ModuleType:
    return _load("style_head_serve", "style_head_serve.py")


class _StubPipeline:
    """``predict_proba`` columns in the MODEL's (alphabetical) order, not the served order."""

    def __init__(self, model_classes: list[str], row: dict[str, float]) -> None:
        self.classes_ = list(model_classes)
        self._row = row

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[self._row.get(label, 0.0) for label in self.classes_] for _ in texts]


def _service(serving: types.ModuleType, model_classes: list[str] | None = None, **row: float) -> Any:
    classes = ("beta", "alpha", "gamma")
    pipeline = _StubPipeline(model_classes or ["alpha", "beta"], row)
    return serving.StyleHeadService(pipeline=pipeline, classes=classes, eval_split="holdout")


def test_confidence_is_top_minus_the_mean_of_the_rest(serving: types.ModuleType) -> None:
    """The floor the seam fits is TypeSafe confidence, and the choice follows label order.

    The model's ``classes_`` is ``["alpha", "beta"]`` while the served order is
    ``("beta", "alpha", "gamma")``, so a column-index mapping would hand ``alpha``'s mass to
    ``beta`` and answer the wrong style with full confidence. ``gamma`` is absent from the
    model entirely: a legal served answer worth 0.0, and part of the mean of the rest.
    """
    service = _service(serving, alpha=0.2, beta=0.7)
    answer = service.answer("a checkout service with flash sales")
    assert answer["choice"] == "beta"
    assert list(answer["probabilities"]) == ["beta", "alpha", "gamma"]
    assert answer["probabilities"] == {"beta": 0.7, "alpha": 0.2, "gamma": 0.0}
    assert answer["confidence"] == pytest.approx(0.7 - (0.2 + 0.0) / 2)


def test_a_tie_resolves_to_the_earlier_served_option(serving: types.ModuleType) -> None:
    """Order is the served order, so a tie cannot depend on dict insertion of the model."""
    service = _service(serving, alpha=0.5, beta=0.5)
    assert service.answer("anything")["choice"] == "beta"


def test_a_criteria_set_that_differs_from_the_trained_classes_is_422(
    serving: types.ModuleType,
) -> None:
    """The head is criteria-blind, so a drifted criterion set is drift, not a request."""
    service = _service(serving, alpha=0.2, beta=0.7)
    body = {
        "model": "style-linear",
        "state": "a checkout service with flash sales",
        "questions": {
            "architecture_style": {
                "type": "choice",
                "criteria": {"beta": "b gloss", "alpha": "a gloss"},
            }
        },
    }
    status, payload = serving.answer_systemone(service, body)
    assert status == 422
    assert "criteria/class drift" in payload["error"]
    # The message names what moved, so an operator can tell a stale image from a stale server.
    assert "gamma" in payload["error"]


def test_matching_criteria_and_non_dict_criteria_are_answered(serving: types.ModuleType) -> None:
    """Only a dict criteria map can be compared; anything else keeps the answer path."""
    service = _service(serving, alpha=0.2, beta=0.7)
    matching = {
        "state": "a checkout service",
        "questions": {"q": {"type": "choice", "criteria": {"alpha": "a", "beta": "b", "gamma": "c"}}},
    }
    status, payload = serving.answer_systemone(service, matching)
    assert status == 200
    assert payload["answers"]["q"]["choice"] == "beta"
    no_criteria = {"state": "a checkout service", "questions": {"q": {"type": "choice"}}}
    status, _ = serving.answer_systemone(service, no_criteria)
    assert status == 200


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("not-an-object", id="body-not-object"),
        pytest.param({"questions": {"q": {"type": "choice"}}}, id="state-missing"),
        pytest.param({"state": "", "questions": {"q": {}}}, id="state-empty"),
        pytest.param({"state": "   ", "questions": {"q": {}}}, id="state-blank"),
        pytest.param({"state": "a checkout service"}, id="questions-missing"),
        pytest.param({"state": "a checkout service", "questions": {}}, id="questions-empty"),
        pytest.param({"state": "a checkout service", "questions": {"q": "choice"}}, id="question-not-object"),
    ],
)
def test_a_malformed_request_is_400(serving: types.ModuleType, body: object) -> None:
    """Every rejection path answers 400 with a reason: a 200 here would act on nothing."""
    service = _service(serving, alpha=0.2, beta=0.7)
    status, payload = serving.answer_systemone(service, body)
    assert status == 400
    assert payload["error"]
