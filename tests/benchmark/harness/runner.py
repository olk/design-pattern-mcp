"""Run one benchmark arm end to end: scenarios → records → summary → report (plan §3, §4).

The runner owns the *run*: it loads the corpus, builds the mode's seam wiring (offline scripts or
the live in-process pipeline), times each scenario at the ``run_design`` boundary, computes the
quality block from the real ``AnalysisResult``/``PipelineResult``, and writes the run directory
(``manifest.json``, ``scenarios/<id>.json``, ``summary.json``, ``report.md``).

Stage attribution comes from :mod:`probes` — the runner never instruments ``src/``; it only
measures boundaries it owns (the per-scenario wall clock) and stages the proxies recorded.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.agent import build_llm_client
from src.config import ServerConfig
from src.patterns.embedder import build_embedder, embedder_identity
from src.patterns.loader import PatternLoader
from src.patterns.retriever import RetrievalIndex
from src.patterns.safe_tei_rerank import build_reranker
from src.pipeline import DesignPipeline, compose_selection
from src.decisions.client import DecisionModelClient
from src.reasoning.client import ReasoningClient
from src.schemas import AnalysisResult, PipelineResult

from . import config as cfg
from . import log_capture
from . import scoring
from .offline import OfflineScript, build_pipeline as build_offline_pipeline, build_script
from .probes import (
    REQUEST_STAGES,
    EmbedderTimingProxy,
    LLMTimingProxy,
    ReasoningTimingProxy,
    RerankerTimingProxy,
    StageRecord,
    StageRecorder,
    percentile_fields,
    residual_ms,
    stage_table,
)

HARNESS_VERSION = cfg.HARNESS_VERSION


@dataclass
class RunOptions:
    """One benchmark arm invocation (CLI surface of :mod:`main`)."""

    mode: str
    scenarios_path: Path = cfg.DEFAULT_SCENARIOS
    out_dir: Path | None = None
    split: str = "all"
    limit: int | None = None
    repeats: int = 1
    flips: tuple[str, ...] = ()
    warmup: bool = True
    run_id: str | None = None


class ScenarioExecutor(Protocol):
    """Runs one scenario against one seam wiring and returns the pipeline observables."""

    name: str
    recorder: StageRecorder

    def describe(self) -> dict[str, Any]:
        """Mode-specific provenance block for the manifest."""
        ...

    async def run(
        self, scenario: cfg.Scenario
    ) -> tuple[AnalysisResult, PipelineResult | None, list[StageRecord], dict[str, Any]]:
        """Execute ``scenario``; return (analysis, pipeline result or None, stage slice, provenance)."""
        ...


class OfflineExecutor:
    """Scripted retrieval + scripted LLM over the real catalog (deterministic, no network)."""

    name = "offline"

    def __init__(
        self,
        *,
        config: ServerConfig,
        catalog: PatternLoader,
        recorder: StageRecorder,
        flips: frozenset[str],
    ) -> None:
        self._config = config
        self._catalog = catalog
        self._recorder = recorder
        self._flips = flips
        self.recorder = recorder

    def describe(self) -> dict[str, Any]:
        """Seam provenance: the two doubles and the disabled reasoning client."""
        return {
            "seams": {
                "index": "tests.unit.fakes.StubIndex (scripted label pool, stage retrieval.scripted)",
                "llm": "tests.unit.fakes.ScriptedLLM (uniform weights, seed-derived generation)",
                "reasoning": "disabled (reasoning=None)",
            },
            "flips": sorted(self._flips),
            "note": (
                "offline stage attribution covers the scripted retrieval call plus llm.*/reasoning.* "
                "stages only; embed/rerank are exercised by live mode"
            ),
        }

    async def run(
        self, scenario: cfg.Scenario
    ) -> tuple[AnalysisResult, PipelineResult | None, list[StageRecord], dict[str, Any]]:
        """Run one scenario through the real pipeline over the scenario's scripted pool."""
        script: OfflineScript = build_script(scenario, self._catalog, flip=scenario.scenario_id in self._flips)
        pipeline = build_offline_pipeline(script, config=self._config, catalog=self._catalog, recorder=self._recorder)
        analysis, result, slice_records = await _execute_pipeline(pipeline, scenario, self._recorder)
        return analysis, result, slice_records, {"script": script.describe()}


class LiveExecutor:
    """Real seams in-process: TEI embedder/reranker, LiteLLM generator, reasoning client."""

    name = "live"

    def __init__(
        self,
        *,
        config: ServerConfig,
        catalog: PatternLoader,
        recorder: StageRecorder,
        pipeline: DesignPipeline,
        reasoning_health: dict[str, str] | None,
        index_source: str,
    ) -> None:
        self._config = config
        self._catalog = catalog
        self._recorder = recorder
        self._pipeline = pipeline
        self._reasoning_health = reasoning_health
        self._index_source = index_source
        self.recorder = recorder

    @classmethod
    async def create(cls, *, config: ServerConfig, catalog: PatternLoader, recorder: StageRecorder) -> LiveExecutor:
        """Build the live wiring exactly like ``DesignPatternServer._initialize``, then wrap it.

        The dense-cache precondition is checked up front: without a warm cache the index build
        re-embeds 486 documents through the CPU sidecar (~15-20 min), which is never what a
        latency comparison wants. ``DENSE_CACHE_MODE=off`` opts into that.
        """
        if config.dense_cache.mode != "off" and not cfg.dense_cache_ready(config.dense_cache.path):
            raise RuntimeError(
                f"dense cache not ready at {config.dense_cache.path!r} "
                "(manifest.json + vectors.npy missing): run `make dense-cache`, point "
                "DENSE_CACHE_PATH at a warm cache, or set "
                "DENSE_CACHE_MODE=off to re-embed at start"
            )
        embedder = EmbedderTimingProxy(build_embedder(config.embedder), recorder)
        reranker = RerankerTimingProxy(build_reranker(config.reranker), recorder)
        index = await RetrievalIndex.build(
            catalog,
            embedder,  # type: ignore[arg-type]  # duck-typed proxy over EmbedderProtocol
            reranker,  # type: ignore[arg-type]  # duck-typed proxy over RerankerProtocol
            config.retrieval,
            config.reranker.rerank_top_n,
            rerank_text=config.reranker.rerank_text,
            dense_cache=config.dense_cache,
            identity=embedder_identity(config.embedder),
        )
        agent = build_llm_client(config)
        llm = LLMTimingProxy(agent, recorder)
        reasoning_inner: ReasoningClient | None = None
        reasoning_health: dict[str, str] | None = None
        if config.reasoning.enabled:
            reasoning_inner = ReasoningClient(config.reasoning, agent)  # type: ignore[arg-type]  # runtime is DesignAgent
            reasoning_health = await reasoning_inner.health_check()
        pipeline = DesignPipeline(
            config=config,
            catalog=catalog,
            index=index,
            llm=llm,  # type: ignore[arg-type]  # duck-typed proxy over LLMClientProtocol
            reasoning=(
                ReasoningTimingProxy(reasoning_inner, recorder)  # type: ignore[arg-type]
                if reasoning_inner is not None
                else None
            ),
            decision_client=DecisionModelClient(config.decisions)
            if config.decisions.enabled and os.environ.get(config.decisions.api_key_env)
            else None,
        )
        pipeline.validate()
        return cls(
            config=config,
            catalog=catalog,
            recorder=recorder,
            pipeline=pipeline,
            reasoning_health=reasoning_health,
            index_source=f"RetrievalIndex.build (dense cache: {config.dense_cache.mode} @ {config.dense_cache.path})",
        )

    def describe(self) -> dict[str, Any]:
        """Seam provenance: real objects behind timing proxies plus the reasoning health map."""
        return {
            "seams": {
                "index": self._index_source,
                "llm": f"{type(self._pipeline._llm).__name__} over {self._config.generator.provider}:{self._config.generator.config.model}",
                "reasoning": (
                    "ReasoningClient (enabled)"
                    if self._config.reasoning.enabled
                    else "disabled via REASONING_ENABLED=false"
                ),
                "decisions": (
                    f"DecisionModelClient ({self._config.decisions.model})"
                    if self._pipeline._decision_client is not None
                    else f"unconfigured ({self._config.decisions.api_key_env} unset)"
                    if self._config.decisions.enabled
                    else "disabled via DECISIONS_ENABLED=false"
                ),
            },
            "reasoning_health": self._reasoning_health,
        }

    async def run(
        self, scenario: cfg.Scenario
    ) -> tuple[AnalysisResult, PipelineResult | None, list[StageRecord], dict[str, Any]]:
        """Run one scenario in-process; failures propagate (no silent degradation)."""
        analysis, result, slice_records = await _execute_pipeline(self._pipeline, scenario, self._recorder)
        return analysis, result, slice_records, {}


class AnalyzeExecutor(LiveExecutor):
    """Live seams, ANALYZE phase only: the arm a generation-skipping default would ship.

    The wiring is :class:`LiveExecutor`'s (real TEI embedder/reranker, real generator for the
    brief/parts and requirement weights, optional reasoning); only the phase scope differs:
    :meth:`DesignPipeline.analyze` runs and ``design_loop`` does not. Hit/recall stay fully
    observable because they come from the analysis; the dominant pattern is the deterministic
    §9 composition over those selections, exactly the value the full arm's ``final_pattern``
    carries when no loop reselection happens. The design/evaluation block is ``null``, never a
    zero.
    """

    name = "analyze"

    async def run(self, scenario: cfg.Scenario) -> tuple[AnalysisResult, None, list[StageRecord], dict[str, Any]]:
        """Run ANALYZE only; failures propagate to the runner's failure policy."""
        analysis, slice_records = await _execute_analyze_pipeline(self._pipeline, scenario, self._recorder)
        return analysis, None, slice_records, {"phases": ["analyze"]}

    def describe(self) -> dict[str, Any]:
        """Seam provenance: the live wiring plus the reduced phase scope."""
        description = super().describe()
        description["seams"]["phases"] = (
            "analyze only via DesignPipeline.analyze (no generate/evaluate; "
            "pipeline block and evaluation null, hit/recall/score observable)"
        )
        return description


async def _execute_pipeline(
    pipeline: DesignPipeline, scenario: cfg.Scenario, recorder: StageRecorder
) -> tuple[AnalysisResult, PipelineResult, list[StageRecord]]:
    """``run_design`` decomposed into its two public phases, plus the stage slice it produced.

    ``DesignPipeline.run_design`` is exactly ``analyze`` + ``design_loop`` (``src/pipeline.py:1358``);
    calling them explicitly keeps the Stage A-E observables the quality metrics read without
    changing what executes.
    """
    offset = len(recorder)
    analysis = await pipeline.analyze(
        description=scenario.description,
        architecture_style=scenario.architecture_style,
        language=scenario.language,
        paradigm=scenario.paradigm,
        focus=scenario.focus,
    )
    result = await pipeline.design_loop(analysis)
    return analysis, result, recorder.snapshot()[offset:]


async def _execute_analyze_pipeline(
    pipeline: DesignPipeline, scenario: cfg.Scenario, recorder: StageRecorder
) -> tuple[AnalysisResult, list[StageRecord]]:
    """ANALYZE only: the same request construction and phase as :func:`_execute_pipeline`.

    The arm is a strict phase subset of a ``live`` run: intake, brief/parts, retrieval, weights,
    reasoning and deterministic selection all run; the generate–evaluate loop does not. Failures
    propagate, exactly as in the full arm.
    """
    offset = len(recorder)
    analysis = await pipeline.analyze(
        description=scenario.description,
        architecture_style=scenario.architecture_style,
        language=scenario.language,
        paradigm=scenario.paradigm,
        focus=scenario.focus,
    )
    return analysis, recorder.snapshot()[offset:]


async def run_benchmark(options: RunOptions) -> dict[str, Any]:
    """Execute one arm and write its run directory; return the run id/paths for the CLI.

    A scenario failure aborts the run (live attribution must not average over failures), but
    the artifacts already on disk stay interpretable: an ``aborted.json`` marker records how far
    the arm got, and the original error is re-raised. ``compare`` refuses such directories.
    """
    try:
        return await _run_benchmark(options)
    except BaseException as exc:
        _write_abort_marker(options, exc)
        raise


def _write_abort_marker(options: RunOptions, exc: BaseException) -> None:
    """Best-effort ``aborted.json`` in the run directory (never masks the original error)."""
    marker = {
        "aborted": True,
        "mode": options.mode,
        "error": f"{type(exc).__name__}: {exc}",
        "reason": (
            "failures propagate (no silent degradation); discard this run directory — "
            "compare refuses it because manifest.json/summary.json are missing"
        ),
        "at": cfg.utc_now_iso(),
    }
    try:
        out_dir = options.out_dir or (cfg.DEFAULT_OUT_DIR / (options.run_id or cfg.default_run_id(options.mode)))
        cfg.write_json(cfg.RunPaths.for_run(out_dir).root / "aborted.json", marker)
    except OSError:
        pass


async def _run_benchmark(options: RunOptions) -> dict[str, Any]:
    """Execute one arm and write its run directory; return the run id/paths for the CLI."""
    config = cfg.load_server_config(options.mode)
    catalog = cfg.load_repo_catalog(config)
    corpus = cfg.load_scenarios(options.scenarios_path, catalog)
    selected = cfg.select_scenarios(corpus, split=options.split, limit=options.limit)
    if not selected:
        raise ValueError(f"no scenarios selected (split={options.split!r}, limit={options.limit})")
    known = {scenario.scenario_id for scenario in corpus}
    unknown_flips = sorted(set(options.flips) - known)
    if unknown_flips:
        raise ValueError(f"--flip ids not in {options.scenarios_path}: {unknown_flips}")
    if options.flips and options.mode != "offline":
        raise ValueError("--flip is an offline-only debug hook (compare dry run)")

    started_at = cfg.utc_now_iso()
    recorder = StageRecorder()
    log_capture.install()  # Step-0b: capture per-call LLM seam data from the src.* loggers
    executor: ScenarioExecutor
    if options.mode == "offline":
        executor = OfflineExecutor(config=config, catalog=catalog, recorder=recorder, flips=frozenset(options.flips))
    elif options.mode == "analyze":
        executor = await AnalyzeExecutor.create(config=config, catalog=catalog, recorder=recorder)
    else:
        executor = await LiveExecutor.create(config=config, catalog=catalog, recorder=recorder)

    run_id = options.run_id or cfg.default_run_id(options.mode)
    out_dir = options.out_dir or (cfg.DEFAULT_OUT_DIR / run_id)
    paths = cfg.RunPaths.for_run(out_dir)

    warmup_record: dict[str, Any] | None = None
    if options.warmup:
        warmup_record, _warmup_stages = await _execute(
            executor, selected[0], repeat=0, warmup=True, catalog=catalog, config=config
        )
        cfg.write_json(paths.root / "warmup.json", warmup_record)

    records: list[dict[str, Any]] = []
    stage_records: list[StageRecord] = []
    for repeat in range(1, options.repeats + 1):
        for scenario in selected:
            record, slice_records = await _execute(
                executor, scenario, repeat=repeat, warmup=False, catalog=catalog, config=config
            )
            records.append(record)
            stage_records.extend(slice_records)
            cfg.write_json(paths.scenarios_dir / f"{scenario.scenario_id}-r{repeat}.json", record)

    notes = _fixture_notes(options, records)
    notes.extend(_mode_notes(options))
    if not options.warmup:
        notes.append(
            "NO_WARMUP protocol deviation (2026-10-07 latency-reduction round): the warmup "
            "scenario is also a measured scenario, so it stays in the sample set; the reasoning "
            "LRU would otherwise hand it a ~0 ms reasoning.analyze cache hit (warmup-cache "
            "artifact understating the run mean). Warmup-on arms (e.g. the committed "
            "benchmarks/base, live-20261002-070756) are NOT mutually comparable with "
            "NO_WARMUP arms; base-fresh is the baseline for every comparison this round."
        )
    notes.append(
        "Step-0b instrumentation: per-call LLM seam data (prompt_chars / response_chars via the "
        "src.* log record extras; a repair re-call after an 'output failed validation' WARNING "
        "is flagged repair=True) attached to llm.* stage records and aggregated per stage as "
        "prompt_chars_mean / response_chars_mean / chars_per_s_mean / repair_calls; "
        "cached_tokens stays null at this seam (the llama-index LiteLLM wrapper does not expose "
        "prompt_tokens_details to the protocol caller)"
    )
    summary = summarise(
        records,
        stage_records=stage_records,
        warmup_record=warmup_record,
        options=options,
        config=config,
        catalog=catalog,
        executor=executor,
        paths=paths,
        run_id=run_id,
        started_at=started_at,
        notes=notes,
    )
    manifest = _manifest(
        options=options,
        config=config,
        catalog=catalog,
        executor=executor,
        records=records,
        warmup_record=warmup_record,
        run_id=run_id,
        started_at=started_at,
        notes=notes,
    )
    cfg.write_json(paths.manifest, manifest)
    cfg.write_json(paths.summary, summary)
    paths.report.write_text(render_report(summary, manifest), encoding="utf-8")
    log_capture.reset()  # Step-0b: the arm's capture window ends with the arm
    return {
        "run_id": run_id,
        "run_dir": str(paths.root),
        "records": len(records),
        "summary": summary,
        "manifest": manifest,
    }


async def _execute(
    executor: ScenarioExecutor,
    scenario: cfg.Scenario,
    *,
    repeat: int,
    warmup: bool,
    catalog: PatternLoader,
    config: ServerConfig,
) -> tuple[dict[str, Any], list[StageRecord]]:
    """Run one scenario, time it at the run boundary and assemble its record."""
    started_ns = executor.recorder.now_ns()
    analysis, result, slice_records, extra = await executor.run(scenario)
    e2e_ms = round((executor.recorder.now_ns() - started_ns) / 1_000_000.0, 3)
    table = stage_table(slice_records)
    final_pattern = (
        scoring.final_pattern_of(analysis, result)
        if result is not None
        else _analyze_dominant_pattern(analysis, catalog, config)
    )
    quality = _quality_block(analysis, result, scenario, final_pattern=final_pattern)
    stage_total = round(sum(float(entry["total_ms"]) for entry in table.values()), 3)
    residual = residual_ms(e2e_ms, table)
    record: dict[str, Any] = {
        "harness_version": HARNESS_VERSION,
        "mode": executor.name,
        "scenario": scenario.record_identity(),
        "repeat": repeat,
        "warmup": warmup,
        "e2e_ms": e2e_ms,
        "stages": table,
        "stage_total_ms": stage_total,
        "residual_ms": residual,
        "residual_note": _residual_note(residual),
        "quality": quality,
        "pipeline": _pipeline_block(result),
        "selection": _selection_block(analysis),
        "provenance": extra,
    }
    return record, slice_records


def _analyze_dominant_pattern(analysis: AnalysisResult, catalog: PatternLoader, config: ServerConfig) -> str:
    """§9 Stage-F election over an analyze-only payload.

    ``DesignPipeline._pipeline_result`` sets ``final_pattern`` to exactly this deterministic
    composition over the loop's selections (``src/pipeline.py:2064``). Replaying it for the
    analyze arm keeps its hit definition identical to the full arm's before any loop reselection;
    a payload that cannot compose raises, exactly as the full arm's GENERATE would.
    """
    return compose_selection(
        selections=analysis.selections,
        brief=analysis.design_brief,
        catalog=catalog,
        config=config,
    ).primary


def _residual_note(residual: float) -> str | None:
    """Explain a negative residual: overlapping stages break the sum-is-wall-clock identity.

    ``GENERATE`` dispatches up to ``generation_max_concurrency`` batches in flight
    (``src/pipeline.py:1473``), so per-call durations of parallel batches overlap; the stage *sum*
    then exceeds the measured wall clock. The residual stays a diagnostic of time spent outside
    the wrapped seams — never a hard accounting identity.
    """
    if residual >= 0.0:
        return None
    return (
        "negative: per-call stage durations overlap (concurrent GENERATE batches), so the stage "
        "sum exceeds the measured wall clock; use the per-stage table, not this residual"
    )


def _quality_block(
    analysis: AnalysisResult,
    result: PipelineResult | None,
    scenario: cfg.Scenario,
    *,
    final_pattern: str | None,
) -> dict[str, Any]:
    """Every quality metric for one scenario (plan §4).

    ``result`` is ``None`` on the analyze-only arm; its dominant pattern is the caller-supplied
    §9 composition of the analysis. Hit, recall, supporting PRF, veto conflicts and the
    calibration score (the elected primary's candidate ``fit_score``, not the evaluation score)
    are all observable from the analysis, so only the pipeline block is absent.
    """
    fallback = scoring.is_fallback(analysis, result)
    hit = (not fallback) and final_pattern is not None and final_pattern in set(scenario.acceptable_primary)
    selected = scoring.selected_ordered(analysis)
    score = scoring.primary_fit_score(analysis)
    conflicts = scoring.veto_conflicts(analysis)
    return {
        "hit": hit,
        "fallback": fallback,
        "final_pattern": final_pattern,
        "expected_primary": scenario.primary,
        "acceptable_primary": list(scenario.acceptable_primary),
        "acceptable_supporting": list(scenario.acceptable_supporting),
        "selected": selected,
        "selected_true_positives": sorted(set(selected) & set(scenario.acceptable)),
        "recall_at_3": round(scoring.recall_at_k(selected, scenario.acceptable, 3), 6),
        "recall_at_5": round(scoring.recall_at_k(selected, scenario.acceptable, 5), 6),
        "supporting": scoring.supporting_prf(analysis, scenario.acceptable_supporting),
        "acceptable_primary_f1": scoring.acceptable_primary_f1(hit, scenario.acceptable_primary),
        "veto_conflicts": conflicts,
        "veto_violations": len(conflicts),
        "score": score,
        "score_scaled": round(score * 100.0, 3),
        "reliability_bin": scoring.reliability_bin(score),
        "brier": round((score - (1.0 if hit else 0.0)) ** 2, 6),
        "calibration_eligible": scenario.split == "train",
    }


def _pipeline_block(result: PipelineResult | None) -> dict[str, Any] | None:
    """Run-level observables of the G–H loop (compact, replay-relevant); ``None`` when absent."""
    if result is None:
        return None
    return {
        "attempts": result.attempts,
        "final_pattern": result.final_pattern,
        "final_quality_score": result.final_quality_score,
        "evaluation_score": result.evaluation.score,
        "matched_categories": list(result.matched_categories),
        "is_fallback": result.is_fallback,
        "alternative_patterns": list(result.alternative_patterns),
    }


def _selection_block(analysis: AnalysisResult) -> list[dict[str, Any]]:
    """Per-part selection observables: the hand-audit view of one run."""
    block: list[dict[str, Any]] = []
    for part in analysis.decision_parts:
        selection = analysis.selections.get(part.id)
        if selection is None:
            block.append({"part_id": part.id, "statement": part.statement, "selection": None})
            continue
        block.append(
            {
                "part_id": part.id,
                "statement": part.statement,
                "scope": part.scope,
                "primary": selection.primary,
                "primary_category": selection.primary_category,
                "supporting": [choice.pattern for choice in selection.supporting],
                "alternatives": list(selection.alternatives),
                "near_tie": list(selection.near_tie),
                "ambiguous": selection.ambiguous,
                "is_fallback": selection.is_fallback,
                "fallback_reason": selection.fallback_reason,
                "veto_ledger": list(selection.veto_ledger),
                "weights_source": selection.weights_source,
                "weights": dict(sorted(selection.weights.items())),
                "candidates": [
                    {
                        "name": candidate.name,
                        "category": candidate.category,
                        "fit_score": round(candidate.fit_score, 4),
                        "analysis_score": round(candidate.analysis_score, 4),
                        "fusion_score_normalized": round(candidate.fusion_score_normalized, 4),
                        "retrieval_score": round(candidate.retrieval_score, 6),
                        "adjustments": [f"{entry.label}{entry.delta:+g}" for entry in candidate.adjustments],
                    }
                    for candidate in selection.candidates
                ],
            }
        )
    return block


def _fixture_notes(options: RunOptions, records: Sequence[dict[str, Any]]) -> list[str]:
    """Offline fixture integrity: a non-flipped scenario must hit, a flipped one must miss.

    Offline hits are scripted, so a deviation means the corpus script and the pipeline disagree
    (e.g. a label the §6 vetoes drop, or a decoy that shares the accepted category) — loud, not
    silently averaged away.
    """
    notes: list[str] = []
    if options.mode != "offline":
        return notes
    flips = set(options.flips)
    violations: list[str] = []
    for record in records:
        scenario_id = record["scenario"]["scenario_id"]
        hit = bool(record["quality"]["hit"])
        if scenario_id in flips and hit:
            violations.append(f"{scenario_id}: flipped scenario still hit ({record['quality']['final_pattern']})")
        if scenario_id not in flips and not hit:
            violations.append(
                f"{scenario_id}: scripted label pool missed (final={record['quality']['final_pattern']}, "
                f"fallback={record['quality']['fallback']}, pool={record['provenance'].get('script', {}).get('pool')})"
            )
    if violations:
        raise ValueError("offline fixture integrity failed:\n  " + "\n  ".join(violations))
    notes.append("offline fixture integrity verified: every non-flipped scenario hit, every flipped scenario missed")
    return notes


def _mode_notes(options: RunOptions) -> list[str]:
    """Mode-specific observability note: what this arm structurally cannot measure."""
    if options.mode != "analyze":
        return []
    return [
        (
            "analyze-only arm: hit = the §9 dominant election replayed over the ANALYZE selections "
            "(the same definition as the full arm's final_pattern before loop reselection), recall = "
            "the analysis selection; generate/evaluate are skipped, so the pipeline block is null "
            "and e2e_ms is a phase subset, not a full-run latency"
        )
    ]


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def summarise(
    records: Sequence[dict[str, Any]],
    *,
    stage_records: Sequence[StageRecord],
    warmup_record: dict[str, Any] | None,
    options: RunOptions,
    config: ServerConfig,
    catalog: PatternLoader,
    executor: ScenarioExecutor,
    paths: cfg.RunPaths,
    run_id: str,
    started_at: str,
    notes: Sequence[str],
) -> dict[str, Any]:
    """Aggregate the run: quality by split, stage latencies, calibration, risk-coverage."""
    train = [record for record in records if record["scenario"]["split"] == "train"]
    holdout = [record for record in records if record["scenario"]["split"] == "holdout"]
    warmup_e2e = warmup_record["e2e_ms"] if warmup_record is not None else None
    return {
        "harness_version": HARNESS_VERSION,
        "run_id": run_id,
        "mode": executor.name,
        "started_at": started_at,
        "finished_at": cfg.utc_now_iso(),
        "scenario_file": {
            "path": str(options.scenarios_path),
            "sha256": cfg.file_sha256(options.scenarios_path),
        },
        "selection": {
            "split": options.split,
            "limit": options.limit,
            "repeats": options.repeats,
            "warmup": options.warmup,
            "flips": sorted(options.flips),
            "scenario_ids": [record["scenario"]["scenario_id"] for record in records],
        },
        "run_dir": str(paths.root),
        "catalog": {"records": len(catalog.records), "ordered": len(catalog.order)},
        "quality": {
            "overall": _quality_aggregate(records),
            "train": _quality_aggregate(train),
            "holdout": _quality_aggregate(holdout),
        },
        "latency": {
            "e2e_ms": percentile_fields([float(record["e2e_ms"]) for record in records]),
            "warmup_e2e_ms": warmup_e2e,
            "stages": stage_table(stage_records),
            "negative_residual_scenarios": sum(1 for record in records if float(record["residual_ms"]) < 0.0),
            "residual_note": (
                "stage sums double-count wall clock when GENERATE batches run concurrently "
                "(generation_max_concurrency); a negative residual means overlap, not missing time"
                if any(float(record["residual_ms"]) < 0.0 for record in records)
                else None
            ),
            "per_scenario_e2e_ms": {
                f"{record['scenario']['scenario_id']}#{record['repeat']}": record["e2e_ms"] for record in records
            },
        },
        "calibration": scoring.calibration(
            [(record["quality"]["score"], record["quality"]["hit"]) for record in train]
        ),
        "risk_coverage": {
            "holdout": scoring.risk_coverage(
                [(record["quality"]["score"], record["quality"]["hit"]) for record in holdout]
            ),
            "train": scoring.risk_coverage(
                [(record["quality"]["score"], record["quality"]["hit"]) for record in train]
            ),
        },
        "per_scenario": [
            {
                "scenario_id": record["scenario"]["scenario_id"],
                "repeat": record["repeat"],
                "family": record["scenario"]["family"],
                "category": record["scenario"]["category"],
                "split": record["scenario"]["split"],
                "hit": record["quality"]["hit"],
                "fallback": record["quality"]["fallback"],
                "final_pattern": record["quality"]["final_pattern"],
                "expected_primary": record["quality"]["expected_primary"],
                "score": record["quality"]["score"],
                "recall_at_3": record["quality"]["recall_at_3"],
                "recall_at_5": record["quality"]["recall_at_5"],
                "supporting_f1": record["quality"]["supporting"]["f1"],
                "acceptable_primary_f1": record["quality"]["acceptable_primary_f1"]["f1"],
                "veto_violations": record["quality"]["veto_violations"],
                "e2e_ms": record["e2e_ms"],
                "residual_ms": record["residual_ms"],
                "effective_config": {
                    "primary_score_threshold": config.selection.primary_score_threshold,
                    "supporting_score_threshold": config.selection.supporting_score_threshold,
                },
            }
            for record in records
        ],
        "notes": list(notes),
    }


def _quality_aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Hit rate (+ Wilson CI) and mean/median metric summaries over ``records``."""
    return scoring.aggregate_quality(records)


# --------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------


def _manifest(
    *,
    options: RunOptions,
    config: ServerConfig,
    catalog: PatternLoader,
    executor: ScenarioExecutor,
    records: Sequence[dict[str, Any]],
    warmup_record: dict[str, Any] | None,
    run_id: str,
    started_at: str,
    notes: Sequence[str],
) -> dict[str, Any]:
    """Run provenance: git, environment, effective config, corpus hash, executor seams."""
    return {
        "harness_version": HARNESS_VERSION,
        "run_id": run_id,
        "mode": options.mode,
        "started_at": started_at,
        "finished_at": cfg.utc_now_iso(),
        "scenario_file": {
            "path": str(options.scenarios_path),
            "sha256": cfg.file_sha256(options.scenarios_path),
        },
        "selection": {
            "split": options.split,
            "limit": options.limit,
            "repeats": options.repeats,
            "warmup": options.warmup,
            "flips": sorted(options.flips),
        },
        "scenario_ids": [record["scenario"]["scenario_id"] for record in records],
        "warmup": {
            "enabled": options.warmup,
            "scenario_id": warmup_record["scenario"]["scenario_id"] if warmup_record else None,
            "e2e_ms": warmup_record["e2e_ms"] if warmup_record else None,
        },
        "git": cfg.git_snapshot(),
        "env": cfg.masked_env_snapshot(),
        "server_config": cfg.mask_secrets(config.model_dump()),
        "catalog": {"records": len(catalog.records), "ordered": len(catalog.order)},
        "executor": executor.describe(),
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
        "counts": {
            "records": len(records),
            "train": sum(1 for record in records if record["scenario"]["split"] == "train"),
            "holdout": sum(1 for record in records if record["scenario"]["split"] == "holdout"),
        },
        "notes": list(notes),
    }


def render_report(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Markdown report for one arm run."""
    quality = summary["quality"]
    latency = summary["latency"]
    lines: list[str] = [f"# Benchmark run {summary['run_id']}", ""]
    lines.append(
        f"- mode: `{summary['mode']}` · split `{summary['selection']['split']}` · "
        f"limit `{summary['selection']['limit']}` · repeats `{summary['selection']['repeats']}`"
    )
    lines.append(
        f"- scenarios: {manifest['counts']['records']} records "
        f"(train {manifest['counts']['train']} / holdout {manifest['counts']['holdout']})"
    )
    lines += [
        f"- corpus: `{summary['scenario_file']['path']}` sha256 `{summary['scenario_file']['sha256'][:16]}…`",
        f"- git: `{(manifest['git']['head'] or 'unknown')[:12]}` dirty={manifest['git']['dirty']}",
        f"- catalog: {summary['catalog']['records']} records",
        "",
        "## Quality (per split)",
        "",
        "| split | n | hit rate | Wilson 95% | fallbacks | recall@3 | recall@5 | supporting F1 | primary F1 | veto violations |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ("overall", "train", "holdout"):
        block = quality[name]
        wilson = block["hit_rate_wilson"]
        wilson_text = f"[{wilson['low']:.3f}, {wilson['high']:.3f}]" if wilson["low"] is not None else "null"
        lines.append(
            f"| {name} | {block['scenarios']} | {_pct(block['hit_rate'])} | {wilson_text} | {_int_or_null(block['fallbacks'])} | "
            f"{_num(block['recall_at_3']['mean'])} | {_num(block['recall_at_5']['mean'])} | "
            f"{_num(block['supporting_f1']['mean'])} | {_num(block['acceptable_primary_f1']['mean'])} | "
            f"{_int_or_null(block['veto_violations']['total'])} |"
        )
    lines += ["", "## Latency", ""]
    e2e = latency["e2e_ms"]
    lines.append(
        f"- e2e: n={e2e['samples']} p50={_ms(e2e['p50_ms'])} p95={_ms(e2e['p95_ms'])}"
        + (f" (warmup {_ms(latency['warmup_e2e_ms'])})" if latency["warmup_e2e_ms"] is not None else "")
        + (f" — {e2e['note']}" if e2e["note"] else "")
    )
    if latency.get("residual_note"):
        lines.append(f"- residual: {latency['residual_note']}")
    lines += [
        "",
        "| stage | calls | failures | total ms | p50 ms | p95 ms | ops |",
        "|---|---|---|---|---|---|---|",
    ]
    for stage, entry in sorted(latency["stages"].items(), key=lambda item: _stage_order(item[0], item[1])):
        lines.append(
            f"| `{stage}` | {entry['count']} | {entry['failures']} | {entry['total_ms']:.1f} | "
            f"{_ms(entry['p50_ms'])} | {_ms(entry['p95_ms'])} | "
            f"{', '.join(f'{op} x{count}' for op, count in sorted(entry['ops'].items()))} |"
        )
    lines += ["", "## Calibration (train split, monitoring only)", ""]
    calibration = summary["calibration"]
    lines.append(f"- Brier: {_num(calibration['brier'])} · ECE (10 bins): {_num(calibration['ece'])}")
    lines += ["", "| bin | range | n | avg score | accuracy | gap |", "|---|---|---|---|---|---|"]
    for entry in calibration["bins"]:
        if not entry["count"]:
            continue
        lines.append(
            f"| {entry['bin']} | [{entry['low']}, {entry['high']}) | {entry['count']} | "
            f"{_num(entry['avg_score'])} | {_num(entry['accuracy'])} | {_num(entry['gap'])} |"
        )
    lines += ["", "## Risk-coverage", ""]
    for split, block in summary["risk_coverage"].items():
        lines.append(
            f"- {split}: n={block['samples']} risk@80={_num(block['risk_at_80'])} "
            f"risk@90={_num(block['risk_at_90'])} risk@100={_num(block['risk_at_100'])} AURC={_num(block['aurc'])}"
        )
    lines += [
        "",
        "## Per scenario",
        "",
        "| scenario | split | category | hit | final | expected | score | e2e ms | residual ms | veto |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in summary["per_scenario"]:
        lines.append(
            f"| `{record['scenario_id']}` | {record['split']} | {record['category']} | "
            f"{'✔' if record['hit'] else '✘'} | `{record['final_pattern']}` | `{record['expected_primary']}` | "
            f"{_num(record['score'])} | {record['e2e_ms']:.1f} | {record['residual_ms']:.1f} | "
            f"{_int_or_null(record['veto_violations'])} |"
        )
    if summary["notes"]:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in summary["notes"]]
    lines.append("")
    return "\n".join(lines)


def _stage_order(stage: str, entry: dict[str, Any]) -> tuple[int, float]:
    """Report order: documented request buckets first (probes.REQUEST_STAGES), then by total ms."""
    if stage in REQUEST_STAGES:
        return REQUEST_STAGES.index(stage), 0.0
    if stage.startswith("reasoning."):
        return len(REQUEST_STAGES), -float(entry["total_ms"])
    return len(REQUEST_STAGES) + 1, -float(entry["total_ms"])


def _num(value: Any) -> str:
    """Format an optional float for the report."""
    return "null" if value is None else f"{float(value):.3f}"


def _ms(value: Any) -> str:
    """Format an optional millisecond value for the report."""
    return "null" if value is None else f"{float(value):.1f}"


def _pct(value: Any) -> str:
    """Format an optional rate as a percentage."""
    return "null" if value is None else f"{100.0 * float(value):.1f}%"


def _int_or_null(value: Any) -> str:
    """Format an optional count (``null`` when a mode cannot observe it)."""
    return "null" if value is None else str(int(value))
