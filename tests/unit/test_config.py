"""Configuration contract: §14 defaults, validation invariants and the env-expansion wiring.

The config file is the deployment unit; every parameter the pipeline gates a decision on is
validated at load time, so a broken configuration must fail loudly rather than silently change
a selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import ConfigManager, ServerConfig
from src.config_expansion import expand_env, expand_env_in_obj
from tests.unit.conftest import REPO_ROOT


class TestDefaults:
    def test_validation_defaults_are_single_repair(self) -> None:
        config = ServerConfig.model_validate({})
        assert config.validation.retry_on_fail is True
        assert config.validation.max_retries == 1

    def test_litellm_log_level_default(self) -> None:
        assert ServerConfig.model_validate({}).litellm_log_level == "WARNING"

    def test_effective_parameters_snapshot_covers_every_gate(self) -> None:
        snapshot = ServerConfig.model_validate({}).effective_parameters()
        for key in (
            "min_fusion_score",
            "primary_score_threshold",
            "max_tries",
            "min_quality_score",
            "dense_top_k",
            "bm25_top_k",
            "max_concurrent_parts",
            "rerank_top_n",
            "rerank_text",
            "reasoning_enabled",
            "reasoning_max_total_steps",
            "reasoning_phases",
            "generator_timeout_seconds",
            "decisions_style_enabled",
            "decisions_style_model",
            "decisions_weights_enabled",
            "decisions_category_enabled",
            "decisions_reasoning_gate_enabled",
            "reranker_provider",
        ):
            assert key in snapshot

    def test_decisions_defaults_are_per_seam_and_off(self) -> None:
        decisions = ServerConfig.model_validate({}).decisions
        for seam in (decisions.style, decisions.weights, decisions.category, decisions.reasoning_gate):
            assert seam.enabled is False
            assert seam.api_key_env is None
        assert decisions.style.provider == "clm"
        assert decisions.style.model == "clm-latest"
        assert decisions.style.base_url == "http://design-pattern-clm:8700/v1"
        assert decisions.style.timeout_seconds == 10.0
        assert decisions.weights.provider == "tacit"
        assert decisions.weights.model == "tacit-4b"
        assert decisions.weights.base_url == "http://design-pattern-tacit:8100"
        assert decisions.weights.timeout_seconds == 30.0
        assert decisions.category.confidence_floor == 0.5
        assert decisions.reasoning_gate.timeout_seconds == 5.0

    def test_partial_seam_override_keeps_that_seams_own_endpoint(self) -> None:
        """A partial override must not fall back onto the generic (CLM) seam defaults."""
        decisions = ServerConfig.model_validate({"decisions": {"weights": {"enabled": True}}}).decisions
        assert decisions.weights.enabled is True
        assert decisions.weights.provider == "tacit"
        assert decisions.weights.base_url == "http://design-pattern-tacit:8100"
        assert decisions.weights.model == "tacit-4b"
        assert decisions.weights.timeout_seconds == 30.0
        # The siblings keep their own defaults untouched.
        assert decisions.style.provider == "clm"
        assert decisions.style.model == "clm-latest"
        assert decisions.reasoning_gate.timeout_seconds == 5.0

    def test_decisions_snapshot_values(self) -> None:
        snapshot = ServerConfig.model_validate({}).effective_parameters()
        assert snapshot["decisions_style_enabled"] is False
        assert snapshot["decisions_style_model"] == "clm-latest"
        assert snapshot["decisions_weights_enabled"] is False
        assert snapshot["decisions_category_enabled"] is False
        assert snapshot["decisions_reasoning_gate_enabled"] is False
        assert snapshot["reranker_provider"] == "tei"

    def test_reranker_provider_defaults_to_tei(self) -> None:
        reranker = ServerConfig.model_validate({}).reranker
        assert reranker.provider == "tei"
        assert reranker.clm_base_url == "http://design-pattern-clm:8700/v1"
        assert reranker.clm_timeout_seconds == 60.0

    def test_reranker_provider_rejects_unknown_value(self) -> None:
        with pytest.raises(ValidationError):
            ServerConfig.model_validate({"reranker": {"provider": "bge"}})

    def test_dense_cache_defaults(self) -> None:
        cache = ServerConfig.model_validate({}).dense_cache
        assert cache.mode == "auto"
        assert cache.path == "/app/data/dense-cache"
        assert cache.write is True
        assert cache.probe_timeout_seconds == 120.0
        assert cache.probe_min_cosine == 0.999

    def test_dense_cache_rejects_unknown_keys(self) -> None:
        with pytest.raises(ValidationError):
            ServerConfig.model_validate({"dense_cache": {"nope": 1}})

    def test_dense_cache_mode_literal(self) -> None:
        with pytest.raises(ValidationError):
            ServerConfig.model_validate({"dense_cache": {"mode": "bogus"}})

    def test_shipped_config_validates(self) -> None:
        loaded = ConfigManager.load_config("config/config.json")
        config = ServerConfig.model_validate(loaded)
        assert config.dense_cache.mode == "auto"
        assert config.embedder.config.document_embed_attempts >= 1
        assert config.embedder.config.document_embed_backoff_seconds > 0.0

    def test_effective_parameters_includes_dense_cache_keys(self) -> None:
        snapshot = ServerConfig.model_validate({}).effective_parameters()
        assert "dense_cache_mode" in snapshot
        assert "dense_cache_path" in snapshot

    def test_retrieval_top_k_defaults_and_bounds(self) -> None:
        retrieval = ServerConfig.model_validate({}).retrieval
        assert retrieval.dense_top_k == 20
        assert retrieval.bm25_top_k == 20
        assert ServerConfig.model_validate({"retrieval": {"dense_top_k": 0}}).retrieval.dense_top_k == 0
        with pytest.raises(ValidationError):
            ServerConfig.model_validate({"retrieval": {"bm25_top_k": -1}})

    def test_reasoning_snapshot_records_every_phase(self) -> None:
        snapshot = ServerConfig.model_validate({}).effective_parameters()

        assert snapshot["reasoning_enabled"] is True
        assert snapshot["reasoning_max_total_steps"] == 8
        assert snapshot["generator_timeout_seconds"] == 120.0
        phases = snapshot["reasoning_phases"]
        assert set(phases) == {"analyze", "weights", "generate", "evaluate"}
        for phase, strategy in phases.items():
            assert strategy["enabled"] is True, phase
            assert strategy["tools"] == ["shannon", "code"], phase
        assert {phase: strategy["pre_llm_thoughts"] for phase, strategy in phases.items()} == {
            "analyze": 2,
            "weights": 2,
            "generate": 2,
            "evaluate": 2,
        }

    def test_partial_per_phase_override_keeps_agenda_and_directive(self) -> None:
        shipped = ServerConfig.model_validate({}).reasoning.get_strategy("weights")

        config = ServerConfig.model_validate(
            {"reasoning": {"per_phase": {"weights": {"enabled": False, "pre_llm_thoughts": 1}}}}
        )

        strategy = config.reasoning.get_strategy("weights")
        assert strategy.enabled is False
        assert strategy.pre_llm_thoughts == 1
        assert strategy.step_agenda == shipped.step_agenda
        assert strategy.system_directive == shipped.system_directive

    def test_provider_reliability_defaults(self) -> None:
        config = ServerConfig.model_validate({})
        assert config.generator.config.generation_batch_size == 5
        assert config.generator.config.generation_max_concurrency == 2
        assert config.retrieval.max_concurrent_parts == 1
        assert config.generator.config.circuit_breaker_failure_threshold == 3
        assert config.generator.config.circuit_breaker_cooldown_seconds == 60.0
        assert config.tasks.deadline_seconds == 1800.0
        assert config.generator.config.timeout_seconds == 120.0, (
            "the per-provider-call bound is unchanged and independent of the whole-run deadline"
        )

    def test_provider_reliability_effective_parameter_keys(self) -> None:
        snapshot = ServerConfig.model_validate({}).effective_parameters()
        assert snapshot["generator_generation_batch_size"] == 5
        assert snapshot["generator_circuit_breaker_failure_threshold"] == 3
        assert snapshot["generator_circuit_breaker_cooldown_seconds"] == 60.0
        assert snapshot["tasks_deadline_seconds"] == 1800.0

    @pytest.mark.parametrize(
        "payload",
        [
            {"generator": {"config": {"generation_batch_size": 0}}},
            {"generator": {"config": {"circuit_breaker_failure_threshold": 0}}},
            {"generator": {"config": {"circuit_breaker_cooldown_seconds": 0.0}}},
            {"tasks": {"deadline_seconds": 0.0}},
            {"tasks": {"deadline_seconds": -1.0}},
        ],
    )
    def test_provider_reliability_bounds(self, payload: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            ServerConfig.model_validate(payload)


class TestValidation:
    def test_unknown_top_level_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ServerConfig.model_validate({"nope": 1})

    def test_validation_block_rejects_unknown_keys(self) -> None:
        with pytest.raises(ValidationError):
            ServerConfig.model_validate({"validation": {"retries": 2}})

    def test_max_retries_bounds(self) -> None:
        assert ServerConfig.model_validate({"validation": {"max_retries": 0}}).validation.max_retries == 0
        with pytest.raises(ValidationError):
            ServerConfig.model_validate({"validation": {"max_retries": 11}})

    def test_retrieval_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match=r"sum to 1.0"):
            ServerConfig.model_validate({"retrieval": {"dense_weight": 0.5, "bm25_weight": 0.2}})

    def test_min_fusion_score_may_not_exceed_blend_maximum(self) -> None:
        with pytest.raises(ValidationError, match="2/rrf_k"):
            ServerConfig.model_validate({"retrieval": {"min_fusion_score": 1.0}})


class TestEnvExpansion:
    def test_env_placeholder_with_default(self) -> None:
        assert expand_env("{env:NOT_SET_ANYWHERE:-fallback}") == "fallback"

    def test_env_placeholder_uses_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DPML_TEST_ENV_VALUE", "set")
        assert expand_env("{env:DPML_TEST_ENV_VALUE:-fallback}") == "set"

    def test_recursive_expansion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DPML_TEST_ENV_VALUE", "42")
        expanded = expand_env_in_obj({"a": ["{env:DPML_TEST_ENV_VALUE:-0}"]})
        assert expanded == {"a": ["42"]}


class TestRepositoryConfig:
    def test_repository_config_loads_and_validates(self) -> None:
        raw = json.loads((REPO_ROOT / "config" / "config.json").read_text(encoding="utf-8"))
        config = ServerConfig.model_validate(expand_env_in_obj(raw))
        assert config.port > 0
        assert config.retrieval.dense_weight + config.retrieval.bm25_weight == pytest.approx(1.0)

    def test_repository_config_ships_both_tools_for_every_phase(self) -> None:
        raw = json.loads((REPO_ROOT / "config" / "config.json").read_text(encoding="utf-8"))
        config = ServerConfig.model_validate(expand_env_in_obj(raw))

        assert set(config.reasoning.per_phase) == {
            "analyze",
            "weights",
            "generate",
            "evaluate",
        }
        for phase, strategy in config.reasoning.per_phase.items():
            assert strategy.enabled is True, phase
            assert strategy.tools == ["shannon", "code"], phase
            assert strategy.step_agenda, phase

    def test_config_manager_reads_the_repository_config(self) -> None:
        loaded = ConfigManager.load_config(str(Path(REPO_ROOT) / "config" / "config.json"))
        assert "generator" in loaded

    def test_repository_config_ships_the_provider_reliability_placeholders(self) -> None:
        raw = json.loads((REPO_ROOT / "config" / "config.json").read_text(encoding="utf-8"))
        generator = raw["generator"]["config"]
        assert generator["generation_batch_size"] == ("{env:GENERATOR_GENERATION_BATCH_SIZE:-5}")
        assert generator["circuit_breaker_failure_threshold"] == (
            "{env:GENERATOR_CIRCUIT_BREAKER_FAILURE_THRESHOLD:-3}"
        )
        assert generator["circuit_breaker_cooldown_seconds"] == (
            "{env:GENERATOR_CIRCUIT_BREAKER_COOLDOWN_SECONDS:-60.0}"
        )
        assert raw["tasks"]["deadline_seconds"] == ("{env:TASKS_DEADLINE_SECONDS:-1800.0}")

    @pytest.mark.parametrize(
        "variable",
        [
            "GENERATOR_GENERATION_BATCH_SIZE",
            "GENERATOR_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
            "GENERATOR_CIRCUIT_BREAKER_COOLDOWN_SECONDS",
            "TASKS_DEADLINE_SECONDS",
        ],
    )
    def test_provider_reliability_env_placeholders_resolve(
        self, variable: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = json.loads((REPO_ROOT / "config" / "config.json").read_text(encoding="utf-8"))
        monkeypatch.setenv(variable, "7" if "BATCH" in variable or "THRESHOLD" in variable else "11.5")

        config = ServerConfig.model_validate(expand_env_in_obj(raw))

        if variable.endswith("BATCH_SIZE"):
            assert config.generator.config.generation_batch_size == 7
        elif variable.endswith("THRESHOLD"):
            assert config.generator.config.circuit_breaker_failure_threshold == 7
        elif variable.endswith("COOLDOWN_SECONDS"):
            assert config.generator.config.circuit_breaker_cooldown_seconds == 11.5
        else:
            assert config.tasks.deadline_seconds == 11.5

    def test_compose_files_pass_the_provider_reliability_knobs(self) -> None:
        """Both deployment units must forward the knobs or the config placeholders are unreachable."""
        dev = (REPO_ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
        prod = (REPO_ROOT / "systemd" / "docker-compose.yml").read_text(encoding="utf-8")

        for name in (
            "GENERATOR_GENERATION_BATCH_SIZE",
            "GENERATOR_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
            "GENERATOR_CIRCUIT_BREAKER_COOLDOWN_SECONDS",
            "TASKS_DEADLINE_SECONDS",
        ):
            assert name in dev, name
            assert name in prod, name
