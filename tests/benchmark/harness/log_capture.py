"""Step-0b LLM-seam data capture: a logging.Handler on ``src.*`` loggers (no ``src/`` edits).

The production seam ``DesignAgent._complete`` (``src/agent.py``) logs every transport call at
INFO/WARNING under the ``src.agent`` / ``src.validation`` loggers, and the record ``extra``
carries exactly the data volume the latency-reduction plan's Step 0 needs:

- ``Provider call succeeded`` (INFO, ``src.agent``) — ``operation``, ``prompt_chars``,
  ``response_chars``, ``elapsed_ms``;
- ``LLM <Schema> output failed validation (attempt N/M)`` (WARNING, ``src.validation``) — the
  ground truth that a repair re-call is happening.

This module installs one handler that keeps the *last finished call per operation*; the timing
proxies in :mod:`tests.benchmark.harness.probes` read it (``_llm_profile``) when the wrapped
protocol method returns and attach ``prompt_chars``/``response_chars``/``repair``/``cached_tokens``
to the stage record. Sparse data (offline scripted seams, disabled INFO level) stays absent —
never fabricated.

Lifecycle: :func:`install` once per arm run (idempotent), :func:`last_call_meta` is read by the
proxies. ``cached_tokens``: the llama-index ``LiteLLM`` wrapper exposes
``prompt_tokens/completion_tokens/total_tokens`` in ``ChatResponse.additional_kwargs`` but NOT
``prompt_tokens_details.cached_tokens`` at any attribute of the protocol seam, so the value is
recorded ``None`` here (documented best-effort, plan Step 0 allows exactly that).
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

#: ``src.agent`` INFO lines and ``src.validation`` WARNING lines are the contract.
_CAPTURE_ROOTS: tuple[str, ...] = ("src",)

#: Marks the next ``Provider call succeeded`` for the operation as a repair second-call.
_VALIDATION_ATTEMPT_RE = re.compile(
    r"LLM (?P<schema>\S+) output failed validation \(attempt (?P<n>\d+)/(?P<m>\d+)\)"
)

# The validation WARNING names the response SCHEMA, the transport INFO line names the OPERATION
# (src/validation.py logs no extra): map the four shipped schemas to their operations.
_SCHEMA_TO_OPERATION: dict[str, str] = {
    "RawBriefAndParts": "brief_and_parts",
    "RequirementWeightsBatch": "weights",
    "RawGeneration": "generate",
    "DesignEvaluation": "evaluate",
}

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"handler": None}


def _handler() -> _LLMCaptureHandler | None:
    return _STATE["handler"]  # type: ignore[no-any-return]


def _set_handler(handler: _LLMCaptureHandler | None) -> None:
    _STATE["handler"] = handler


# modules own the state; tests reset via reset() only
_LAST_CALL: dict[str, dict[str, Any]] = {}
_PENDING_REPAIR: set[str] = set()


class _LLMCaptureHandler(logging.Handler):
    """Store per-operation LLM seam facts from ``src.*`` log lines; never mutates logging flow."""

    def emit(self, record: logging.LogRecord) -> None:
        """Record the relevant lines; swallow every error (a capture bug must not fail a run)."""
        try:
            msg = record.getMessage()
            extra = getattr(record, "__dict__", {})
            with _LOCK:
                if record.name == "src.validation":
                    match = _VALIDATION_ATTEMPT_RE.search(msg)
                    if match is not None:
                        operation = _SCHEMA_TO_OPERATION.get(
                            match.group("schema"), str(extra.get("operation", match.group("schema")))
                        )
                        _PENDING_REPAIR.add(operation)
                    return
                # src.agent (and agents with the same line contract)
                if msg == "Provider call succeeded":
                    operation = str(extra.get("operation", ""))
                    entry: dict[str, Any] = {}
                    if isinstance(extra.get("prompt_chars"), int):
                        entry["prompt_chars"] = int(extra["prompt_chars"])
                    if isinstance(extra.get("response_chars"), int):
                        entry["response_chars"] = int(extra["response_chars"])
                    if operation in _PENDING_REPAIR:
                        entry["repair"] = True
                        _PENDING_REPAIR.discard(operation)
                    else:
                        entry.pop("repair", None)
                    if operation:
                        _LAST_CALL[operation] = entry
        except Exception:  # noqa: BLE001 - capture must never break the instrumented run
            pass


def install() -> _LLMCaptureHandler:
    """Attach the capture handler to the ``src`` logger roots (idempotent, one arm run).

    Also raises the ``src.agent`` logger to INFO (the transport line lives there): the harness
    root sits at WARNING via ``--log-level``, which would otherwise gate the INFO record before
    any handler sees it. Log verbosity only — no pipeline behavior change.
    """
    existing = _handler()
    if existing is not None:
        return existing
    handler = _LLMCaptureHandler(level=logging.INFO)
    for root in _CAPTURE_ROOTS:
        logging.getLogger(root).addHandler(handler)
    logging.getLogger("src.agent").setLevel(logging.INFO)
    _set_handler(handler)
    return handler


def last_call_meta(operation: str | None = None) -> dict[str, Any] | None:
    """Latest captured call, scoped by ``operation`` when given, else the newest entry.

    Concurrent batches of the SAME operation (``generation_max_concurrency`` > 1) can interleave:
    the profile read then may attach a sibling batch's chars to this batch's record — same op,
    same stage, documented imprecision (the log seam carries no per-call id). Cross-operation
    reads are exact.
    """
    with _LOCK:
        if operation is not None:
            return dict(_LAST_CALL[operation]) if operation in _LAST_CALL else None
        if not _LAST_CALL:
            return None
        # Insertion order under the lock is call order; the last inserted op is the newest call.
        newest_op = next(reversed(_LAST_CALL))
        return dict(_LAST_CALL[newest_op])


def reset() -> None:
    """Detach the handler and clear state (used between arm runs and in the selfcheck)."""
    existing = _handler()
    if existing is not None:
        for root in _CAPTURE_ROOTS:
            logging.getLogger(root).removeHandler(existing)
        _set_handler(None)
    with _LOCK:
        _LAST_CALL.clear()
        _PENDING_REPAIR.clear()
