"""Tool-boundary adapters (DP-7): dict ⇄ pydantic forms and the advisory integrity checks.

Pure unit tests — no pipeline, no LLM, no retrieval; just the conversion contract and the
deterministic findings of :mod:`src.design_validation`.
"""

from __future__ import annotations

import pytest

from src.errors import MalformedDesignOverviewError
from src.schemas import AnalysisResult, DesignBrief, DesignResult
from src.tools._adapters import (
    analysis_to_pydantic,
    design_from_dict,
    design_integrity_findings,
    design_to_pydantic,
)

MINIMAL_DESIGN: dict = {
    "overview": {
        "reasoning": "one decision, one pattern",
        "primary_pattern": "timeout",
        "category": "concurrency-resilience",
    },
    "applications": [
        {
            "id": "app-1",
            "pattern": "timeout",
            "name": "bounded geo-lookup",
            "problem": "the lookup may hang",
            "solution": "bound the wait",
        }
    ],
}


def _design(**overrides) -> DesignResult:
    payload = {**MINIMAL_DESIGN, **overrides}
    return design_from_dict(payload)


def test_design_from_dict_rejects_a_missing_overview() -> None:
    """A payload without an overview object is malformed at the boundary."""
    with pytest.raises(MalformedDesignOverviewError, match="overview"):
        design_from_dict({"applications": []})


def test_design_from_dict_rejects_a_non_dict_overview() -> None:
    """An overview that is not an object is malformed at the boundary."""
    with pytest.raises(MalformedDesignOverviewError, match="overview"):
        design_from_dict({"overview": "bare text"})


def test_design_from_dict_returns_a_valid_result_for_a_minimal_payload() -> None:
    """The minimal result-schema-shaped payload validates into a strict DesignResult."""
    design = design_from_dict(MINIMAL_DESIGN)

    assert isinstance(design, DesignResult)
    assert design.overview.primary_pattern == "timeout"
    assert [application.pattern for application in design.applications] == ["timeout"]


def test_analysis_to_pydantic_accepts_a_dict_and_returns_instances_verbatim() -> None:
    """Dicts validate into AnalysisResult; instances pass through unchanged."""
    analysis = AnalysisResult(
        design_brief=DesignBrief(restated_goal="harden the enrichment stage"),
        effective_language="generic",
    )
    from_dict = analysis_to_pydantic(analysis.model_dump(mode="json", by_alias=True))

    assert isinstance(from_dict, AnalysisResult)
    assert from_dict == analysis
    assert analysis_to_pydantic(analysis) is analysis


def test_design_to_pydantic_accepts_both_representations() -> None:
    """A DesignResult passes through; the dict form validates into an equal model."""
    design = _design()

    assert design_to_pydantic(design) is design
    assert design_to_pydantic(MINIMAL_DESIGN) == design


def test_integrity_findings_flag_an_empty_design() -> None:
    """A design without applications carries the documented emptiness finding."""
    empty = design_to_pydantic({**MINIMAL_DESIGN, "applications": []})

    assert design_integrity_findings(empty) == ["the design carries no pattern application"]


def test_integrity_findings_flag_a_dominant_pattern_that_is_not_applied() -> None:
    """The overview's primary must be realized by an application."""
    design = design_from_dict(
        {
            **MINIMAL_DESIGN,
            "overview": {**MINIMAL_DESIGN["overview"], "primary_pattern": "strategy"},
        }
    )

    findings = design_integrity_findings(design)

    assert any("'strategy'" in finding and "not applied" in finding for finding in findings)


def test_integrity_findings_flag_duplicated_application_ids() -> None:
    """Two applications sharing an id are flagged once for the duplication."""
    duplicated = {**MINIMAL_DESIGN, "applications": [MINIMAL_DESIGN["applications"][0]] * 2}
    design = design_from_dict(duplicated)

    findings = design_integrity_findings(design)

    assert any("duplicated application id 'app-1'" in finding for finding in findings)


def test_integrity_findings_flag_a_conflicting_pair() -> None:
    """A §8.3 rule 3 conflict pair among the applications is flagged."""
    singleton = {
        "id": "app-1",
        "pattern": "singleton",
        "name": "shared state",
        "problem": "state must be global",
        "solution": "one shared instance",
    }
    injection = {
        "id": "app-2",
        "pattern": "dependency-injection",
        "name": "wiring",
        "problem": "hard-wired collaborators",
        "solution": "inject them",
    }
    design = design_from_dict(
        {
            **MINIMAL_DESIGN,
            "overview": {**MINIMAL_DESIGN["overview"], "primary_pattern": "singleton"},
            "applications": [singleton, injection],
        }
    )

    findings = design_integrity_findings(design)

    conflict = next(
        (finding for finding in findings if "conflicting pattern pair" in finding), None
    )
    assert conflict is not None
    assert "singleton" in conflict and "dependency-injection" in conflict
