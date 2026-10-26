"""Metric-math selfcheck: hand-computed anchors for every formula the harness reports (plan §5.1).

Run with ``make benchmark-selfcheck``. Each check asserts a value derived *independently* of the
implementation (standard textbook formulas recomputed by hand and cross-checked against an
independent in-kernel derivation), so a formula regression fails here instead of silently
changing a benchmark verdict. No network, no catalog, no LLM: pure functions plus synthetic
pydantic objects for the selection-shaped metrics.
"""

from __future__ import annotations

import asyncio
import math
import sys
from collections.abc import Sequence
from typing import Any

from src.schemas import (
    AnalysisResult,
    AnnotatedCandidate,
    DesignBrief,
    DesignEvaluation,
    DesignOverview,
    DesignResult,
    DecisionPart,
    PartSelection,
    PipelineResult,
    SupportingChoice,
)

from . import scoring
from .probes import (
    STAGE_LLM_ANALYZE,
    STAGE_LLM_WEIGHTS,
    STAGE_LLM_GENERATE,
    LLMTimingProxy,
    StageRecorder,
    percentile_fields,
    request_stage_total_ms,
    residual_ms,
    stage_table,
)


class CheckFailure(AssertionError):
    """A named selfcheck failure (kept distinct so the CLI can report all of them)."""


def _assert(condition: bool, message: str) -> None:
    """Raise :class:`CheckFailure` with ``message`` when ``condition`` is false."""
    if not condition:
        raise CheckFailure(message)


def _approx(actual: Any, expected: Any, *, label: str, tolerance: float = 1e-9) -> None:
    """Assert ``actual`` equals ``expected`` within ``tolerance`` (or both are ``None``)."""
    if expected is None:
        _assert(actual is None, f"{label}: expected null, got {actual!r}")
        return
    _assert(actual is not None, f"{label}: expected {expected!r}, got null")
    _assert(
        abs(float(actual) - float(expected)) <= tolerance,
        f"{label}: expected {expected!r}, got {actual!r}",
    )


# --------------------------------------------------------------------------------------
# Fixtures: the smallest pydantic objects the metrics read
# --------------------------------------------------------------------------------------


def _candidate(name: str, fit: float) -> AnnotatedCandidate:
    return AnnotatedCandidate(
        name=name,
        category="domain-modeling",
        retrieval_score=1.0,
        analysis_score=fit,
        fusion_score_normalized=1.0,
        blended=fit,
        fit_score=fit,
    )


def _selection(
    part_id: str,
    *,
    primary: str | None,
    supporting: Sequence[str] = (),
    alternatives: Sequence[str] = (),
    veto_ledger: Sequence[str] = (),
    candidates: Sequence[AnnotatedCandidate] = (),
    is_fallback: bool = False,
) -> PartSelection:
    return PartSelection(
        part=DecisionPart(id=part_id, statement=f"decide {part_id}", priority=1.0),
        primary=primary,
        primary_category="domain-modeling" if primary else None,
        supporting=[SupportingChoice(pattern=name, role="supports") for name in supporting],
        alternatives=list(alternatives),
        veto_ledger=list(veto_ledger),
        candidates=list(candidates),
        is_fallback=is_fallback,
    )


def _analysis(selections: Sequence[PartSelection]) -> AnalysisResult:
    return AnalysisResult(
        design_brief=DesignBrief(restated_goal="test"),
        decision_parts=[selection.part for selection in selections],
        selections={selection.part.id: selection for selection in selections},
        effective_language="python",
        is_fallback=any(selection.is_fallback for selection in selections),
    )


def _result(final_pattern: str | None, *, is_fallback: bool = False) -> PipelineResult:
    return PipelineResult(
        design=DesignResult(
            overview=DesignOverview(reasoning="r", primary_pattern=final_pattern or "none", category="domain-modeling"),
            applications=[],
        ),
        evaluation=DesignEvaluation(score=100.0),
        attempts=1,
        final_pattern=final_pattern,
        is_fallback=is_fallback,
    )


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


def check_percentiles() -> None:
    """Inclusive 20-quantile cut points; null p50/p95 below four samples (never NaN)."""
    sparse = percentile_fields([1.0, 2.0])
    _assert(sparse["p50_ms"] is None and sparse["p95_ms"] is None, f"2 samples must be null: {sparse}")
    _assert("at least 4" in str(sparse["note"]), f"missing explanatory note: {sparse}")
    empty = percentile_fields([])
    _assert(empty["samples"] == 0 and empty["p50_ms"] is None, f"empty input: {empty}")
    # [1, 2, 3, 4]: p50 = 2 + 0.5*(3-2) = 2.5; p95 = 3 + 0.85*(4-3) = 3.85
    dense = percentile_fields([1.0, 2.0, 3.0, 4.0])
    _approx(dense["p50_ms"], 2.5, label="p50 of [1,2,3,4]")
    _approx(dense["p95_ms"], 3.85, label="p95 of [1,2,3,4]")
    _assert(not math.isnan(float(dense["p50_ms"])), "p50 must never be NaN")


def check_stage_table() -> None:
    """Stage grouping: counts, totals, failures, residual and per-call percentiles."""
    recorder = StageRecorder()
    started = recorder.now_ns()
    recorder.mark(stage=STAGE_LLM_ANALYZE, op="extract_brief_and_parts", started_ns=started, ok=True)
    recorder.mark(
        stage=STAGE_LLM_ANALYZE, op="extract_brief_and_parts", started_ns=started, ok=False, meta={"error": "x"}
    )
    recorder.mark(stage=STAGE_LLM_WEIGHTS, op="extract_weights_batch", started_ns=started, ok=True)
    table = stage_table(recorder.snapshot())
    _assert(table[STAGE_LLM_ANALYZE]["count"] == 2, f"count: {table}")
    _assert(table[STAGE_LLM_ANALYZE]["failures"] == 1, f"failures: {table}")
    _assert(table[STAGE_LLM_ANALYZE]["ops"] == {"extract_brief_and_parts": 2}, f"ops: {table}")
    _assert(table[STAGE_LLM_ANALYZE]["p50_ms"] is None, f"2 samples must be null: {table}")
    _assert(table[STAGE_LLM_ANALYZE]["total_ms"] >= 0.0, "totals must be non-negative")
    request_total = request_stage_total_ms(table)
    _assert(request_total >= table[STAGE_LLM_ANALYZE]["total_ms"], "request total must include the stage")
    _approx(residual_ms(1000.0, table), round(1000.0 - request_total, 3), label="residual")
    _assert(residual_ms(1000.0, {}) == 1000.0, "empty table leaves the full residual")


def check_wilson() -> None:
    """Wilson intervals against the standard closed form (k=5,n=10 etc.)."""
    _approx(scoring.wilson_interval(5, 10)["low"], 0.236593, label="wilson(5,10).low")
    _approx(scoring.wilson_interval(5, 10)["high"], 0.763407, label="wilson(5,10).high")
    _approx(scoring.wilson_interval(0, 10)["low"], 0.0, label="wilson(0,10).low")
    _approx(scoring.wilson_interval(0, 10)["high"], 0.277533, label="wilson(0,10).high")
    _approx(scoring.wilson_interval(10, 10)["low"], 0.722467, label="wilson(10,10).low")
    _approx(scoring.wilson_interval(10, 10)["high"], 1.0, label="wilson(10,10).high")
    empty = scoring.wilson_interval(0, 0)
    _assert(empty["low"] is None and empty["high"] is None, f"no samples: {empty}")
    _assert(scoring.wilson_interval(5, 10)["p_hat"] == 0.5, "p_hat must equal k/n")


def check_sign_test() -> None:
    """Exact two-sided binomial sign test (zeros dropped, p capped at 1.0)."""
    _approx(scoring.sign_test([1.0] * 10)["p_two_sided"], 0.001953, label="10/10 positives")
    _approx(scoring.sign_test([1.0] * 9 + [-1.0])["p_two_sided"], 0.021484, label="9/10 positives")
    _approx(scoring.sign_test([1.0] * 8 + [-1.0] * 2)["p_two_sided"], 0.109375, label="8/10 positives")
    _approx(scoring.sign_test([1.0] * 5 + [-1.0] * 5)["p_two_sided"], 1.0, label="5/10 positives")
    zeros = scoring.sign_test([0.0, 0.0])
    _assert(zeros["n"] == 0 and zeros["p_two_sided"] == 1.0, f"all-zero deltas: {zeros}")
    _assert(zeros["n_dropped"] == 2, f"all-zero deltas drop every pair: {zeros}")
    mixed = scoring.sign_test([0.0, 2.0, 3.0])
    _assert(mixed["n"] == 2 and mixed["positive"] == 2, f"zeros dropped: {mixed}")
    _assert(mixed["n_dropped"] == 1, f"mixed deltas drop exactly the zero: {mixed}")


def check_newcombe() -> None:
    """Newcombe's Wilson-based difference CI (unrounded bounds inside the formula)."""
    unequal = scoring.newcombe_difference_ci(6, 10, 4, 10)
    _approx(unequal["difference"], 0.2, label="difference(6/10, 4/10)")
    _approx(unequal["low"], -0.206341, label="newcombe low")
    _approx(unequal["high"], 0.527843, label="newcombe high")
    equal = scoring.newcombe_difference_ci(5, 10, 5, 10)
    _approx(equal["difference"], 0.0, label="equal proportions difference")
    _approx(equal["low"], -0.372514, label="equal proportions low")
    _approx(equal["high"], 0.372514, label="equal proportions high")
    _assert(scoring.newcombe_difference_ci(0, 0, 1, 2)["difference"] is None, "no samples must be null")


def check_recall_and_f1() -> None:
    """recall@K, per-part supporting F1 (micro) and the acceptable-primary F1 direction."""
    _approx(scoring.recall_at_k(["a", "b", "z", "q"], ["a", "b", "c"], 3), 2 / 3, label="recall@3")
    _approx(scoring.recall_at_k(["a", "b", "z", "q"], ["a", "b", "c"], 4), 2 / 3, label="recall@4")
    _approx(scoring.recall_at_k(["a", "b", "c"], ["a", "b", "c"], 3), 1.0, label="full recall@3")
    _approx(scoring.recall_at_k(["a"], ["a", "b"], 5), 0.5, label="partial recall@5")
    _approx(scoring.recall_at_k(["a"], [], 5), 0.0, label="empty acceptable set")

    analysis = _analysis(
        [
            _selection("p1", primary="entity", supporting=["s1", "s2"]),
            _selection("p2", primary="aggregate", supporting=["s1"]),
        ]
    )
    block = scoring.supporting_prf(analysis, ["s1", "s3"])
    _assert(
        block["true_positive"] == 2 and block["false_positive"] == 1 and block["false_negative"] == 2,
        f"micro counts: {block}",
    )
    _approx(block["precision"], 2 / 3, label="supporting precision", tolerance=1e-6)
    _approx(block["recall"], 0.5, label="supporting recall", tolerance=1e-6)
    _approx(block["f1"], 0.571429, label="supporting F1", tolerance=1e-6)
    empty = scoring.supporting_prf(_analysis([_selection("p1", primary="entity")]), ["s1"])
    _assert(empty["precision"] == 0.0 and empty["f1"] == 0.0, f"empty selection vs expectation: {empty}")
    nothing_expected = scoring.supporting_prf(_analysis([_selection("p1", primary="entity")]), [])
    _assert(nothing_expected["f1"] == 1.0, f"no expectation and no selection: {nothing_expected}")

    hit = scoring.acceptable_primary_f1(True, ["a", "b", "c", "d"])
    _approx(hit["precision"], 1.0, label="primary F1 precision on a hit")
    _approx(hit["recall"], 0.25, label="primary F1 recall on a hit")
    _approx(hit["f1"], 0.4, label="primary F1 on a hit")
    miss = scoring.acceptable_primary_f1(False, ["a", "b"])
    _assert(miss["f1"] == 0.0, f"a miss scores 0: {miss}")


def check_hit_and_score() -> None:
    """Hit/fallback semantics and the calibrated score source (elected primary's fit_score)."""
    analysis = _analysis(
        [_selection("p1", primary="entity", candidates=[_candidate("aggregate", 80.0), _candidate("entity", 90.0)])]
    )
    _assert(scoring.is_fallback(analysis, _result("entity")) is False, "clean run is not a fallback")
    _approx(scoring.primary_fit_score(analysis), 0.9, label="primary fit score is scaled to [0, 1]")
    fallback_analysis = _analysis([_selection("p1", primary=None, is_fallback=True)])
    _assert(scoring.is_fallback(fallback_analysis, _result(None)) is True, "part fallback propagates")
    _assert(scoring.primary_fit_score(fallback_analysis) == 0.0, "a missing primary scores 0.0")
    _assert(scoring.final_pattern_of(fallback_analysis, _result(None)) is None, "no pattern in a fallback")

    selected = scoring.selected_ordered(
        _analysis(
            [
                _selection(
                    "p1",
                    primary="entity",
                    supporting=["repository"],
                    alternatives=[
                        "aggregate (fit 71.5)",
                        "money considered but discouraged in python; use value-object",
                    ],
                )
            ]
        )
    )
    _assert(selected == ["entity", "repository", "aggregate", "money"], f"ordered selection: {selected}")
    _approx(scoring.recall_at_k(selected, ["entity", "money"], 5), 1.0, label="recall over seeds")
    _assert(scoring.parse_seed_name("") is None, "empty seed has no name")


def check_analyze_mode() -> None:
    """Analyze-only arm: mode/config plumbing and the no-result quality record.

    The arm differs from ``live`` only by phase scope, so these anchors pin the two things a
    refactor can silently break: the CLI/mode registry forwarding and the record contract
    (hit/recall/score from the analysis, pipeline block null, never a fabricated zero).
    """
    from . import config as cfg
    from . import runner
    from .main import build_parser

    _assert("analyze" in cfg.MODES, f"analyze must be a registered harness mode: {cfg.MODES}")
    parsed = build_parser().parse_args(["--mode", "analyze"])
    _assert(parsed.mode == "analyze", "the harness CLI must accept --mode analyze")
    _assert(
        cfg.load_server_config("analyze") == cfg.load_server_config("live"),
        "analyze must run the shipped configuration; only offline may override it",
    )

    scenario = cfg.Scenario(
        scenario_id="selfcheck-analyze",
        family="selfcheck",
        category="domain-modeling",
        split="train",
        description="selfcheck",
        decision_statement="pick a primary pattern",
        acceptable_primary=("entity", "aggregate"),
        acceptable_supporting=("repository",),
    )
    analysis = _analysis(
        [
            _selection(
                "p1",
                primary="entity",
                supporting=["repository"],
                candidates=[_candidate("entity", 90.0)],
            )
        ]
    )
    hit = runner._quality_block(analysis, None, scenario, final_pattern="entity")
    _assert(hit["hit"] is True, "analyze record: an acceptable composed primary hits")
    _approx(hit["score"], 0.9, label="analyze record: score comes from the elected primary fit")
    _approx(hit["recall_at_5"], 2 / 3, label="analyze record: recall over the acceptable set", tolerance=1e-6)
    miss = runner._quality_block(analysis, None, scenario, final_pattern="repository")
    _assert(miss["hit"] is False, "analyze record: an acceptable supporting pattern is not a hit")
    _assert(runner._pipeline_block(None) is None, "analyze record: the pipeline block is null")


def check_veto_conflicts() -> None:
    """§8.3/§6-step-5 violations: elected/supported vetoed pattern + primary inside supporting."""
    analysis = _analysis(
        [
            _selection(
                "p1",
                primary="entity",
                supporting=["entity", "money", "repository"],
                veto_ledger=[
                    "entity considered but discouraged in python; use value-object",
                    "money considered but discouraged in python; use value-object",
                ],
            )
        ]
    )
    conflicts = scoring.veto_conflicts(analysis)
    kinds = sorted(conflict["type"] for conflict in conflicts)
    _assert(
        kinds
        == [
            "primary_in_supporting",
            "vetoed_pattern_elected",
            "vetoed_pattern_supported",
            "vetoed_pattern_supported",
        ],
        f"conflict kinds (entity is both elected and supported, money only supported): {kinds}",
    )
    _assert(all(conflict["part"] == "p1" for conflict in conflicts), "conflicts name the part")
    clean = scoring.veto_conflicts(
        _analysis(
            [
                _selection(
                    "p1",
                    primary="entity",
                    supporting=["repository"],
                    veto_ledger=["money considered but discouraged in python"],
                )
            ]
        )
    )
    _assert(clean == [], f"a clean selection has no conflicts: {clean}")


def check_calibration() -> None:
    """Brier, 10-bin equal-width ECE and the reliability table."""
    entries = [(0.8, True), (0.2, False)]
    block = scoring.calibration(entries)
    _approx(block["brier"], 0.04, label="Brier of [(0.8, hit), (0.2, miss)]")
    _approx(block["ece"], 0.2, label="ECE of [(0.8, hit), (0.2, miss)]")
    bins = {entry["bin"]: entry for entry in block["bins"]}
    _assert(bins[8]["count"] == 1 and bins[2]["count"] == 1, f"bin assignment: {bins}")
    _approx(bins[8]["gap"], 0.2, label="gap of bin 8")
    _approx(bins[2]["gap"], 0.2, label="gap of bin 2")
    _approx(scoring.brier([(1.0, True), (0.0, False)]), 0.0, label="perfect calibration Brier")
    _approx(scoring.brier([(1.0, False), (0.0, True)]), 1.0, label="worst calibration Brier")
    _assert(scoring.reliability_bin(1.0) == 9 and scoring.reliability_bin(0.0) == 0, "bin edges")
    _assert(scoring.reliability_bin(0.85) == 8, "0.85 belongs to bin 8")
    _assert(scoring.calibration([])["brier"] is None, "no entries: null Brier")


def check_risk_coverage() -> None:
    """Risk at 80/90/100% coverage and the trapezoidal risk-coverage AUC."""
    block = scoring.risk_coverage([(0.9, True), (0.8, False), (0.7, True), (0.6, False), (0.5, True)])
    _assert(block["samples"] == 5, f"samples: {block}")
    _approx(block["risk_at_80"], 0.5, label="risk at 80% coverage (k=4)")
    _approx(block["risk_at_90"], 0.4, label="risk at 90% coverage (k=5)")
    _approx(block["risk_at_100"], 0.4, label="risk at 100% coverage")
    # trapezoid: 0.2*0 + 0.2*(0.5)/2 + 0.2*(0.5+1/3)/2 + 0.2*(1/3+0.5)/2 + 0.2*(0.5+0.4)/2
    _approx(block["aurc"], 0.306667, label="AURC")
    ordered = scoring.risk_coverage([(0.1, True), (0.9, False)])
    _assert(ordered["risk_at_100"] == 0.5, f"ordering by score desc: {ordered}")
    _assert(scoring.risk_coverage([])["aurc"] is None, "no samples: null AUC")


class _RaisingLLM:
    """Minimal failing LLM double for the proxy failure-path check."""

    async def generate(self, seed: Any, *, reasoning_context: str = "") -> Any:
        raise RuntimeError("provider exploded")


def check_proxies() -> None:
    """Proxies record stage, op, outcome and re-raise; failures keep their attribution."""

    class _OkLLM:
        async def generate(self, seed: Any, *, reasoning_context: str = "") -> Any:
            await asyncio.sleep(0)
            return {"ok": True}

    async def _exercise() -> None:
        recorder = StageRecorder()
        ok = LLMTimingProxy(_OkLLM(), recorder)
        result = await ok.generate({"applications": [1, 2]})
        _assert(result == {"ok": True}, "proxy must return the wrapped result")
        _assert(len(recorder.records) == 1, f"one record per call: {recorder.records}")
        record = recorder.records[0]
        _assert(record.stage == STAGE_LLM_GENERATE and record.ok, f"stage/ok: {record}")
        _assert(record.meta["applications"] == 2, f"meta applications: {record.meta}")
        _assert(record.meta["reasoning_context_chars"] == 0, f"reasoning_context_chars: {record.meta}")
        _assert(record.duration_ms >= 0.0, "duration must be non-negative")
        failing = LLMTimingProxy(_RaisingLLM(), recorder)
        raised = False
        try:
            await failing.generate(None)
        except RuntimeError:
            raised = True
        _assert(raised, "proxy must re-raise the seam failure")
        failed = recorder.records[-1]
        _assert(not failed.ok and "provider exploded" in failed.meta["error"], f"failure attribution: {failed}")
        # Step-0b: stage_table exposes null data fields for stages without capture.
        table = stage_table(recorder.snapshot())
        entry = table[STAGE_LLM_GENERATE]
        _assert(
            entry["prompt_chars_mean"] is None and entry["response_chars_mean"] is None,
            f"no capture → null chars: {entry}",
        )
        _assert(
            entry["chars_per_s_mean"] is None and entry["cached_tokens_mean"] is None,
            f"no capture → null derived data: {entry}",
        )
        _assert(entry["repair_calls"] == 0, f"no capture → zero repairs: {entry}")
        # Step-0b log_capture: the validation WARNING flags the NEXT transport success
        # for the same operation as repair=True (schema→operation mapping).
        import logging as _logging

        import tests.benchmark.harness.log_capture as _lc

        _lc.reset()
        _lc.install()
        try:
            _logging.getLogger("src.validation").warning(
                "LLM %s output failed validation (attempt %d/%d): %s", "RawGeneration", 1, 1, "missing id"
            )
            _assert("generate" in _lc._PENDING_REPAIR, f"validation WARNING must arm repair: {_lc._PENDING_REPAIR}")
        finally:
            _lc.reset()

    asyncio.run(_exercise())


CHECKS: tuple[tuple[str, Any], ...] = (
    ("percentiles", check_percentiles),
    ("stage table + residual", check_stage_table),
    ("Wilson intervals", check_wilson),
    ("sign test", check_sign_test),
    ("Newcombe difference CI", check_newcombe),
    ("recall@K + F1s", check_recall_and_f1),
    ("hit/fallback + score source", check_hit_and_score),
    ("analyze-mode record contract", check_analyze_mode),
    ("veto-conflict counting", check_veto_conflicts),
    ("Brier + ECE", check_calibration),
    ("risk-coverage", check_risk_coverage),
    ("timing proxies", check_proxies),
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every check; print one PASS line per check; exit 1 on the first failure."""
    del argv
    failures: list[str] = []
    for name, check in CHECKS:
        try:
            check()
        except CheckFailure as exc:
            failures.append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}", file=sys.stderr)
        else:
            print(f"PASS {name}")
    if failures:
        print(f"\n{len(failures)} of {len(CHECKS)} checks failed", file=sys.stderr)
        return 1
    print(f"\nall {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
