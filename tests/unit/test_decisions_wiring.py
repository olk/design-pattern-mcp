"""Which client the Ch2 style seam gets (``style_client_for``): the linear head, always.

A provider the config accepts but the server never wires is a seam that silently never
answers: the degrade-never-raise contract turns a missing or mis-typed client into "the LLM
style stands", which is indistinguishable in the logs from a healthy model that keeps landing
below its floor. The mapping is one line in ``src/server.py``; these tests pin it, the config
pairing it depends on, and the startup report's shape.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from src.config import DecisionSeamConfig, ServerConfig
from src.decisions.client import DecisionModelClient
from src.server import health_check, style_client_for
from tests.unit.conftest import REPO_ROOT

#: The endpoint the style seam actually serves, so the seam under test is a real pairing.
ENDPOINT = "http://design-pattern-style-head:8800/v1"


def _seam(enabled: bool = True) -> DecisionSeamConfig:
    return ServerConfig.model_validate(
        {"decisions": {"style": {"enabled": enabled, "base_url": ENDPOINT}}}
    ).decisions.style


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """``health_check`` builds a server, whose constructor rewrites the root logger."""
    root_logger = logging.getLogger()
    handlers = root_logger.handlers[:]
    level = root_logger.level
    yield
    root_logger.handlers[:] = handlers
    root_logger.setLevel(level)


def test_the_style_seam_gets_the_linear_client() -> None:
    """The style seam is served only by the linear head's ``/v1/systemone`` wire."""
    assert isinstance(style_client_for(_seam()), DecisionModelClient)


def test_a_non_linear_provider_is_rejected_at_load() -> None:
    """A provider without a client must be refused at load, not silently unwired."""
    with pytest.raises(ValidationError):
        ServerConfig.model_validate({"decisions": {"style": {"provider": "hosted", "base_url": ENDPOINT}}})


def test_a_non_v1_base_url_is_rejected_at_load() -> None:
    """A gateway-style root would 404 under /systemone, so the config refuses it."""
    with pytest.raises(ValidationError):
        ServerConfig.model_validate({"decisions": {"style": {"base_url": "http://decision-gateway:8100"}}})


def test_the_client_follows_the_seam_switch() -> None:
    """``enabled()`` is what the pipeline gates on, so it must track the seam's own switch."""
    assert style_client_for(_seam(enabled=True)).enabled() is True
    assert style_client_for(_seam(enabled=False)).enabled() is False


async def test_health_check_reports_only_the_style_seam() -> None:
    """``--health`` prints one seam entry; that shape is what operators read after the removal."""
    report = await health_check(str(REPO_ROOT / "config" / "config.json"))
    assert set(report["decisions"]) == {"style"}
    assert report["decisions"]["style"].startswith("ok")
