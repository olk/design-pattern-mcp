"""§1.3 — input validation (E1/E2) and the language/paradigm/style resolution tables.

The resolution table is the single source of truth for the effective language: recognized values
pass through, an unrecognized value warns and degrades to generic mode, and absent/empty/explicit
``generic`` are generic mode without a warning.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from src.errors import ERROR_DESCRIPTION_TOO_SHORT, DesignPatternError
from src.pipeline import (
    MIN_DESCRIPTION_LENGTH,
    build_brief,
    resolve_language,
    resolve_paradigm,
    resolve_style,
    validate_description,
)
from src.schemas import RawBriefAndParts, RawDesignBrief
from src.text_validation import (
    DOMAIN_MAX_LENGTH,
    FREETEXT_MAX_LENGTH,
    PATTERN_NAME_MAX_LENGTH,
    DomainName,
    PatternName,
    PrintableText,
    _domain_validator,
    _freetext_validator,
    _pattern_name_validator,
    ensure_printable_text,
)

LANGUAGES = (
    "generic", "c", "cplusplus", "csharp", "go", "java", "javascript",
    "kotlin", "python", "ruby", "rust", "scala", "swift", "typescript",
)
PARADIGMS = ("oo", "functional", "concurrent", "data-oriented", "declarative", "procedural")


def test_description_shorter_than_40_chars_is_e1() -> None:
    """E1: too-short descriptions fail before any LLM call and name the error code."""
    with pytest.raises(DesignPatternError) as excinfo:
        validate_description("too short")
    assert excinfo.value.code == ERROR_DESCRIPTION_TOO_SHORT
    assert "component names" in (excinfo.value.hint or "")


def test_description_length_is_measured_after_stripping() -> None:
    """§1.3 rule 1: the length bound applies to the stripped text."""
    body = "x" * MIN_DESCRIPTION_LENGTH
    assert validate_description(f"  {body}  ") == body
    with pytest.raises(DesignPatternError):
        validate_description(f"  {body[:-1]}  ")


def test_control_and_format_characters_are_rejected() -> None:
    """§1.3 rule 1: the canonical character rule rejects control/format characters."""
    with pytest.raises(ValueError, match="disallowed character"):
        ensure_printable_text("a zero\u200bwidth space", field="description")
    with pytest.raises(ValueError, match="disallowed character"):
        ensure_printable_text("null\u0000byte", field="description")
    assert ensure_printable_text("line\nbreak\tok", field="description") == "line\nbreak\tok"


def test_description_length_ceiling() -> None:
    """§1.3 rule 1: >100 000 characters is rejected with the documented message."""
    with pytest.raises(ValueError, match="exceeds maximum length"):
        ensure_printable_text("a" * (FREETEXT_MAX_LENGTH + 1), field="description")


def test_language_resolution_recognized_and_absent() -> None:
    """§1.3 table rows 1 and 3: recognized passes through, absent/empty/explicit generic is generic."""
    assert resolve_language("rust", LANGUAGES) == "rust"
    assert resolve_language(None, LANGUAGES) == "generic"
    assert resolve_language("", LANGUAGES) == "generic"
    assert resolve_language("   ", LANGUAGES) == "generic"
    assert resolve_language("generic", LANGUAGES) == "generic"


def test_language_resolution_unrecognized_degrades_to_generic(caplog: pytest.LogCaptureFixture) -> None:
    """§1.3 table row 2 / E10: an unrecognized language warns and runs in generic mode."""
    with caplog.at_level("WARNING"):
        assert resolve_language("cobol", LANGUAGES) == "generic"
    assert any("generic mode" in record.message for record in caplog.records)


def test_language_resolution_rejects_non_scalar_values() -> None:
    """§1.3 rule 4: a non-scalar value (coerced to text by the boundary) is simply unrecognized."""
    assert resolve_language("['python', 'rust']", LANGUAGES) == "generic"
    assert resolve_language("python,rust", LANGUAGES) == "generic"


def test_paradigm_resolution_drops_unknown_values(caplog: pytest.LogCaptureFixture) -> None:
    """§1.3 rule 5 / E10: an off-enum paradigm is dropped with a warning, never an error."""
    assert resolve_paradigm("functional", PARADIGMS) == "functional"
    assert resolve_paradigm(None, PARADIGMS) is None
    with caplog.at_level("WARNING"):
        assert resolve_paradigm("quantum", PARADIGMS) is None
    assert any("paradigm" in record.message for record in caplog.records)


def test_style_resolution_ignores_unknown_styles(caplog: pytest.LogCaptureFixture) -> None:
    """§1.3 rule 3 / E10: an unknown architecture style is ignored, not an error."""
    assert resolve_style("event-driven") == "event-driven"
    assert resolve_style("microservices") == "microservices"
    with caplog.at_level("INFO"):
        assert resolve_style("vibe-architecture") is None


def test_brief_carries_exactly_one_effective_language() -> None:
    """§4: the brief's language is the resolved effective language, never the raw hint."""
    brief = build_brief(
        RawBriefAndParts(design_brief=RawDesignBrief(restated_goal="goal", language=None)),
        language="generic",
        paradigm=None,
        style="event-driven",
        description="a description long enough to matter for the pipeline",
    )
    assert brief.language == "generic"
    assert brief.architecture_style == "event-driven"
    assert brief.paradigm is None


def test_caller_style_hint_overrides_the_briefs_extracted_style() -> None:
    """§4 rule 6: a recognized caller hint wins over the model's extracted style (guide §1.2)."""
    brief = build_brief(
        RawBriefAndParts(
            design_brief=RawDesignBrief(
                restated_goal="goal",
                architecture_style="contract-styles",  # the model's off-enum guess
            ),
        ),
        language="generic",
        paradigm=None,
        style="microservices",  # the caller's recognized hint
        description="a description long enough to matter for the pipeline",
    )
    assert brief.architecture_style == "microservices"


def test_brief_style_used_only_when_the_caller_hint_is_absent() -> None:
    """§4 rule 6 fallback leg: no caller hint → the extracted style still drives affinity."""
    brief = build_brief(
        RawBriefAndParts(
            design_brief=RawDesignBrief(restated_goal="goal", architecture_style="event-driven"),
        ),
        language="generic",
        paradigm=None,
        style=None,
        description="a description long enough to matter for the pipeline",
    )
    assert brief.architecture_style == "event-driven"


class _AnnotatedPayload(BaseModel):
    """Carries the three annotated string types at Field boundaries (enforcement layer 1)."""

    text: PrintableText
    name: PatternName
    domain: DomainName


def test_annotated_types_strip_whitespace_on_the_happy_path() -> None:
    payload = _AnnotatedPayload(text="  keep me  ", name="  timeout  ", domain="  payments  ")
    assert (payload.text, payload.name, payload.domain) == ("keep me", "timeout", "payments")


def test_annotated_types_name_their_own_field_in_the_error() -> None:
    """Layer 1: each wrapper passes its own field name into the shared validator."""
    with pytest.raises(ValidationError, match="value contains disallowed character"):
        _AnnotatedPayload(text="zero\u200bwidth", name="timeout", domain="payments")
    with pytest.raises(ValidationError, match="name contains disallowed character"):
        _AnnotatedPayload(text="ok", name="zero\u200bwidth", domain="payments")
    with pytest.raises(ValidationError, match="domain contains disallowed character"):
        _AnnotatedPayload(text="ok", name="timeout", domain="zero\u200bwidth")


def test_annotated_types_line_break_policy_is_per_field() -> None:
    """Free text keeps line breaks; the name and domain fields do not (allow_line_breaks)."""
    assert _AnnotatedPayload(text="a\nb", name="timeout", domain="payments").text == "a\nb"
    with pytest.raises(ValidationError, match="name contains disallowed character"):
        _AnnotatedPayload(text="a\nb", name="a\nb", domain="payments")
    with pytest.raises(ValidationError, match="domain contains disallowed character"):
        _AnnotatedPayload(text="a\nb", name="timeout", domain="a\nb")


def test_wrappers_pass_their_documented_length_bounds() -> None:
    """``PatternName`` caps at 100 in the wrapper; ``Domain``/free text only in their ``Field``.

    The two layers are deliberate: ``DOMAIN_MAX_LENGTH`` (200) is enforced by the annotation,
    while the wrapper keeps the free-text ceiling — a wrapper-side cap would be a second,
    drifting limit.
    """
    assert _pattern_name_validator("x" * PATTERN_NAME_MAX_LENGTH) == "x" * PATTERN_NAME_MAX_LENGTH
    with pytest.raises(ValueError, match=f"exceeds maximum length of {PATTERN_NAME_MAX_LENGTH} characters"):
        _pattern_name_validator("x" * (PATTERN_NAME_MAX_LENGTH + 1))
    over_domain = "x" * (DOMAIN_MAX_LENGTH + 1)
    assert _domain_validator(over_domain) == over_domain
    assert _freetext_validator("x" * FREETEXT_MAX_LENGTH) == "x" * FREETEXT_MAX_LENGTH
    with pytest.raises(ValueError, match=f"exceeds maximum length of {FREETEXT_MAX_LENGTH} characters"):
        _freetext_validator("x" * (FREETEXT_MAX_LENGTH + 1))
