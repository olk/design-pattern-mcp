"""§14 startup guard — ``DesignPipeline.validate`` fails fast on inconsistent state.

Pinned directly (rather than left to the integration smoke test) so the guard's
mutants stay classified: ``tests/unit/test_protocol_smoke.py`` is deliberately
outside the mutation test selection because it costs ~137 s of the ~179 s clean
suite while exercising happy paths.
"""

from __future__ import annotations

import inspect

import pytest

from src.anti_scope import AntiScopeGuard
from src.config import RetrievalConfig, ServerConfig
from src.errors import ERROR_GENERATION_FAILED, ERROR_NO_PRIMARY_PATTERN, DesignPatternError
from src.patterns.loader import PatternLoader
from src.pipeline import DesignPipeline
from tests.unit.fakes import ScriptedLLM, StubIndex


class _EmptyCatalog:
    """PatternLoader double reporting no records (the fatal startup condition)."""

    records: tuple[object, ...] = ()


def _pipeline(config: ServerConfig, catalog: object) -> DesignPipeline:
    return DesignPipeline(
        config=config,
        catalog=catalog,  # type: ignore[arg-type]
        index=StubIndex(),  # type: ignore[arg-type]
        llm=ScriptedLLM(),
        guard=AntiScopeGuard((), catalog),  # type: ignore[arg-type]
    )


def test_validate_accepts_a_consistent_startup(catalog: PatternLoader) -> None:
    _pipeline(ServerConfig(), catalog).validate()  # must not raise


def test_validate_signature_keeps_the_documented_keyword_defaults(catalog: PatternLoader) -> None:
    """The llama-index base class validates resource configs by default, resources not at all."""
    _pipeline(ServerConfig(), catalog).validate()  # body coverage: mutmut selects this test
    params = inspect.signature(DesignPipeline.validate).parameters
    assert set(params) == {"self", "validate_resource_configs", "validate_resources"}
    assert params["validate_resource_configs"].default is True
    assert params["validate_resources"].default is False


def test_validate_rejects_an_empty_catalog() -> None:
    with pytest.raises(DesignPatternError) as excinfo:
        _pipeline(ServerConfig(), _EmptyCatalog()).validate()
    assert excinfo.value.code == ERROR_NO_PRIMARY_PATTERN
    assert excinfo.value.detail == "catalog is empty"


def _unvalidated_floor(value: float) -> ServerConfig:
    """A config whose fusion floor never passed the model validator.

    ``RetrievalConfig`` already rejects ``min_fusion_score > 2/rrf_k``, so the pipeline-side
    check is defence in depth (config built by hand, or by a future bypass) — reached here by
    assigning after construction, which pydantic v2 does not re-validate.
    """
    config = ServerConfig(retrieval=RetrievalConfig(rrf_k=60))
    config.retrieval.min_fusion_score = value
    return config


def test_validate_rejects_a_min_fusion_score_above_the_blend_maximum(catalog: PatternLoader) -> None:
    """§6: the reciprocal-rank blend tops out at 2/rrf_k, so a floor above it can never pass."""
    with pytest.raises(DesignPatternError) as excinfo:
        _pipeline(_unvalidated_floor(0.04), catalog).validate()
    assert excinfo.value.code == ERROR_GENERATION_FAILED
    assert excinfo.value.detail == "min_fusion_score exceeds the reciprocal-rank blend maximum 2/rrf_k"


def test_validate_accepts_the_boundary_at_the_blend_maximum(catalog: PatternLoader) -> None:
    """The documented 1e-3 headroom means exactly 2/rrf_k + 1e-3 is still reachable."""
    _pipeline(_unvalidated_floor(2.0 / 60 + 1e-3), catalog).validate()  # must not raise
