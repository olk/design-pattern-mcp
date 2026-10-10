"""The per-agent provider circuit breaker and its failure classification.

The clock is injected, so cooldown behavior is exact rather than timing-dependent. The status
classification is the reason auth/configuration 4xx never open the circuit: they repeat
identically until the operator changes something, so failing fast beats waiting out a cooldown.
"""

from __future__ import annotations

import pytest

from src.provider_health import (
    ProviderCircuitBreaker,
    extract_status_code,
    is_circuit_qualifying_status,
)


class _Clock:
    """Injectable monotonic clock the tests advance by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_constructor_rejects_unusable_bounds() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        ProviderCircuitBreaker(failure_threshold=0, cooldown_seconds=60.0)
    with pytest.raises(ValueError, match="cooldown_seconds"):
        ProviderCircuitBreaker(failure_threshold=3, cooldown_seconds=0.0)


def test_closed_circuit_admits_every_call_and_never_reports_a_retry() -> None:
    breaker = ProviderCircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)

    decision = breaker.before_call()

    assert decision.allowed is True
    assert decision.state == "closed"
    assert decision.retry_after_seconds == 0.0
    assert decision.transitioned is False


def test_three_consecutive_failures_open_the_circuit() -> None:
    clock = _Clock()
    breaker = ProviderCircuitBreaker(3, 60.0, clock=clock)

    for _ in range(2):
        breaker.before_call()
        breaker.record_failure()
        assert breaker.state == "closed", "below the threshold the circuit stays closed"

    breaker.before_call()
    breaker.record_failure()

    assert breaker.state == "open"
    assert breaker.consecutive_failures == 3


def test_open_circuit_rejects_locally_with_the_remaining_cooldown() -> None:
    clock = _Clock()
    breaker = ProviderCircuitBreaker(3, 60.0, clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(20.0)

    decision = breaker.before_call()

    assert decision.allowed is False
    assert decision.state == "open"
    assert decision.retry_after_seconds == pytest.approx(40.0)
    assert decision.transitioned is False


def test_cooldown_admits_exactly_one_half_open_probe() -> None:
    clock = _Clock()
    breaker = ProviderCircuitBreaker(3, 60.0, clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(60.0)

    probe = breaker.before_call()
    concurrent = breaker.before_call()

    assert probe.allowed is True
    assert probe.state == "half_open"
    assert probe.transitioned is True, "open -> half_open is the transition the log line reports"
    assert concurrent.allowed is False, "only one probe may be in flight"
    assert concurrent.state == "half_open"


def test_half_open_probe_failure_reopens_immediately() -> None:
    clock = _Clock()
    breaker = ProviderCircuitBreaker(3, 60.0, clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(60.0)

    breaker.before_call()
    breaker.record_failure()

    assert breaker.state == "open"
    assert breaker.before_call().allowed is False


def test_success_closes_the_circuit_and_resets_the_counter() -> None:
    breaker = ProviderCircuitBreaker(3, 60.0, clock=_Clock())
    breaker.record_failure()
    breaker.record_failure()

    breaker.record_success()

    assert breaker.state == "closed"
    assert breaker.consecutive_failures == 0
    assert breaker.before_call().allowed is True


def test_half_open_probe_success_closes_the_circuit() -> None:
    clock = _Clock()
    breaker = ProviderCircuitBreaker(3, 60.0, clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(60.0)
    breaker.before_call()

    breaker.record_success()

    assert breaker.state == "closed"
    assert breaker.consecutive_failures == 0
    assert breaker.before_call().state == "closed"


@pytest.mark.parametrize("status", [None, 408, 429, 500, 502, 503, 504])
def test_transport_failures_qualify_for_the_circuit(status: int | None) -> None:
    assert is_circuit_qualifying_status(status) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_configuration_and_auth_4xx_never_trip_the_circuit(status: int) -> None:
    assert is_circuit_qualifying_status(status) is False


def test_status_extraction_covers_the_wrapper_shapes() -> None:
    class _WithStatus(Exception):
        status_code = 429

    class _WithResponse(Exception):
        class response:  # noqa: N801 - provider SDK shape
            status_code = 503

    class _WithCode(Exception):
        code = 401

    assert extract_status_code(_WithStatus()) == 429
    assert extract_status_code(_WithResponse()) == 503
    assert extract_status_code(_WithCode()) == 401
    assert extract_status_code(ValueError("connection reset")) is None
