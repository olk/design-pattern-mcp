"""§4 rule 1 — the structured-output repair retry applies to model-output failures only.

A provider timeout, provider exception, empty response or open circuit is a transport failure:
re-issuing the identical call only burns another provider-call budget, so ``LLMError`` is
re-raised untouched. Exhausted output validation, in contrast, raises ``LLMError`` with
``failure_kind="validation"`` plus the operation metadata a log line needs.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.errors import ERROR_LLM_PROVIDER, LLMError
from src.validation import validate_with_retries


class _Payload(BaseModel):
    """Minimal response schema: one integer field."""

    value: int


async def test_malformed_output_is_repaired_once_with_the_validation_error() -> None:
    """A ValueError from the first call triggers exactly one repair carrying the failure."""
    repairs: list[str] = []

    async def initial() -> _Payload:
        raise ValueError("no JSON object found in model response (12 chars)")

    async def repair(system_prompt: str, user_prompt: str) -> _Payload:
        repairs.append(user_prompt)
        assert system_prompt == "sys"
        return _Payload(value=7)

    result = await validate_with_retries(
        initial,
        repair,
        _Payload,
        operation="weights",
        max_retries=1,
        system_prompt="sys",
        user_prompt="usr",
    )

    assert result.value == 7
    assert len(repairs) == 1
    assert "Previous validation error" in repairs[0]
    assert "no JSON object found in model response" in repairs[0]


async def test_schema_violation_is_repaired() -> None:
    """A Pydantic ValidationError is a model-output failure and earns a repair."""

    async def initial() -> _Payload:
        return _Payload.model_validate({"value": "not-a-number"})

    async def repair(_system: str, _user: str) -> _Payload:
        return _Payload(value=1)

    assert (await validate_with_retries(initial, repair, _Payload, operation="generate")).value == 1


async def test_provider_timeout_is_never_repair_prompted() -> None:
    """A 120 s provider timeout must not trigger a second identical provider call."""
    repairs: list[str] = []

    async def initial() -> _Payload:
        raise LLMError(
            provider="minimax",
            model="minimax/MiniMax-M2.7",
            failure_kind="timeout",
            provider_message="LLM provider call timed out after 120s",
            extra={"operation": "generate", "elapsed_ms": 120001.0, "prompt_chars": 4096},
        )

    async def repair(_system: str, user_prompt: str) -> _Payload:
        repairs.append(user_prompt)
        return _Payload(value=1)

    with pytest.raises(LLMError) as exc_info:
        await validate_with_retries(initial, repair, _Payload, operation="generate", max_retries=3)

    assert repairs == [], "a transport failure must not spend the repair budget"
    error = exc_info.value
    assert error.code == ERROR_LLM_PROVIDER
    assert error.failure_kind == "timeout"
    assert error.provider == "minimax"
    assert error.model == "minimax/MiniMax-M2.7"
    assert error.extra["operation"] == "generate"
    assert error.extra["elapsed_ms"] == 120001.0


async def test_provider_failure_during_repair_propagates_immediately() -> None:
    """A transport failure on the repair call aborts instead of burning the rest of the budget."""
    attempts: list[str] = []

    async def initial() -> _Payload:
        attempts.append("initial")
        raise ValueError("malformed")

    async def repair(_system: str, _user: str) -> _Payload:
        attempts.append("repair")
        raise LLMError(failure_kind="empty_response", provider_message="empty response")

    with pytest.raises(LLMError) as exc_info:
        await validate_with_retries(initial, repair, _Payload, operation="evaluate", max_retries=3)

    assert attempts == ["initial", "repair"]
    assert exc_info.value.failure_kind == "empty_response"


async def test_exhausted_validation_reports_operation_attempts_and_elapsed() -> None:
    """The aggregate failure is a validation failure with the operation metadata attached."""
    attempts = 0

    async def initial() -> _Payload:
        nonlocal attempts
        attempts += 1
        raise ValueError("no JSON object found in model response")

    async def repair(_system: str, _user: str) -> _Payload:
        nonlocal attempts
        attempts += 1
        raise ValueError("still no JSON object found in model response")

    with pytest.raises(LLMError) as exc_info:
        await validate_with_retries(initial, repair, _Payload, operation="brief_and_parts", max_retries=3)

    error = exc_info.value
    assert attempts == 4, "one initial call plus three repair attempts"
    assert error.code == ERROR_LLM_PROVIDER
    assert error.failure_kind == "validation"
    assert error.extra["operation"] == "brief_and_parts"
    assert error.extra["attempts"] == 4
    assert error.extra["elapsed_ms"] >= 0.0
    assert "still no JSON object found" in error.extra["last_validation_error"]
    assert "brief_and_parts" not in error.extra["last_validation_error"]


async def test_error_serialization_carries_the_failure_class() -> None:
    """``to_dict`` keeps the base payload and adds provider/model/failure_kind."""
    error = LLMError(
        provider="minimax",
        model="minimax/MiniMax-M2.7",
        failure_kind="circuit_open",
        provider_message="provider circuit is open",
        hint="wait for the cooldown",
        extra={"operation": "generate", "retry_after_seconds": 42.5},
    )

    payload = error.to_dict()

    assert payload["code"] == ERROR_LLM_PROVIDER
    assert payload["provider"] == "minimax"
    assert payload["model"] == "minimax/MiniMax-M2.7"
    assert payload["failure_kind"] == "circuit_open"
    assert payload["hint"] == "wait for the cooldown"
    assert payload["extra"]["retry_after_seconds"] == 42.5
