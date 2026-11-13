"""Which client the Ch2 style seam gets for each ``provider`` (``style_client_for``).

A provider the config accepts but the server never wires is a seam that silently never
answers: the degrade-never-raise contract turns a missing or mis-typed client into "the LLM
style stands", which is indistinguishable in the logs from a healthy model that keeps landing
below its floor. The mapping is one line in ``src/server.py``, and before this test nothing
covered it — the three-wire split was asserted only through the config's own URL guard.

Every seam here is built through ``ServerConfig``, so the test also proves the pairing the
wiring depends on still validates: a ``clm``/``linear`` seam on a gateway root (or a ``tacit``
seam on a ``/v1`` root) never reaches the selector at all.
"""

from __future__ import annotations

import pytest

from src.config import DecisionSeamConfig, ServerConfig
from src.decisions.client import DecisionModelClient
from src.decisions.tacit import TacitClient
from src.server import style_client_for

#: The endpoint each provider actually serves, so the seam under test is a real pairing.
ENDPOINTS: dict[str, str] = {
    "tacit": "http://design-pattern-tacit:8100",
    "clm": "http://design-pattern-clm:8700/v1",
    "linear": "http://design-pattern-style-head:8800/v1",
}


def _seam(provider: str, enabled: bool = True) -> DecisionSeamConfig:
    return ServerConfig.model_validate(
        {"decisions": {"style": {"enabled": enabled, "provider": provider, "base_url": ENDPOINTS[provider]}}}
    ).decisions.style


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        pytest.param("tacit", TacitClient, id="tacit-gateway"),
        pytest.param("clm", DecisionModelClient, id="clm-systemone"),
        pytest.param("linear", DecisionModelClient, id="linear-systemone"),
    ],
)
def test_each_provider_gets_its_own_wire(provider: str, expected: type) -> None:
    """``clm`` and ``linear`` share the TypeSafe client; ``tacit`` keeps its own wire."""
    assert isinstance(style_client_for(_seam(provider)), expected)


def test_the_linear_arm_is_a_distinct_client_from_the_tacit_arm() -> None:
    """The A/B's two arms must not be served by the same client class by accident."""
    assert type(style_client_for(_seam("linear"))) is not type(style_client_for(_seam("tacit")))


@pytest.mark.parametrize("provider", ["tacit", "clm", "linear"])
def test_the_client_follows_the_seam_switch(provider: str) -> None:
    """``enabled()`` is what the pipeline gates on, so it must track the seam's own switch."""
    assert style_client_for(_seam(provider, enabled=True)).enabled() is True
    assert style_client_for(_seam(provider, enabled=False)).enabled() is False
