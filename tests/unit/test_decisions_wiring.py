"""Which client the Ch2 style seam gets (``style_client_for``): the linear head, always.

A provider the config accepts but the server never wires is a seam that silently never
answers: the degrade-never-raise contract turns a missing or mis-typed client into "the LLM
style stands", which is indistinguishable in the logs from a healthy model that keeps landing
below its floor. The mapping is one line in ``src/server.py``, and before this test nothing
covered it — the wire pairing was asserted only through the config's own URL guard.

Every seam here is built through ``ServerConfig``, so the test also proves the pairing the
wiring depends on still validates: a ``linear`` seam on a gateway root never reaches the
selector at all, and a ``tacit`` style seam is rejected at load.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import DecisionSeamConfig, ServerConfig
from src.decisions.client import DecisionModelClient
from src.server import style_client_for

#: The endpoint the style seam actually serves, so the seam under test is a real pairing.
ENDPOINTS: dict[str, str] = {
    "linear": "http://design-pattern-style-head:8800/v1",
}


def _seam(provider: str, enabled: bool = True) -> DecisionSeamConfig:
    return ServerConfig.model_validate(
        {"decisions": {"style": {"enabled": enabled, "provider": provider, "base_url": ENDPOINTS[provider]}}}
    ).decisions.style


def test_the_style_seam_gets_the_linear_client() -> None:
    """The style seam is served only by the linear head's ``/v1/systemone`` wire."""
    assert isinstance(style_client_for(_seam("linear")), DecisionModelClient)


def test_a_tacit_style_seam_is_rejected_at_load() -> None:
    """``provider=tacit`` on style has no client now, so the config must refuse it."""
    with pytest.raises(ValidationError):
        ServerConfig.model_validate(
            {"decisions": {"style": {"provider": "tacit", "base_url": "http://design-pattern-tacit:8100"}}}
        )


def test_the_client_follows_the_seam_switch() -> None:
    """``enabled()`` is what the pipeline gates on, so it must track the seam's own switch."""
    assert style_client_for(_seam("linear", enabled=True)).enabled() is True
    assert style_client_for(_seam("linear", enabled=False)).enabled() is False
