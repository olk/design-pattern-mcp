#!/usr/bin/env python
"""Calibrate the Ch2 style seam's confidence floor against held-out style rows.

Replays the holdout-split rows of ``qa/typed/style.jsonl`` through a running
``clm-serve`` (``POST /v1/systemone`` with the served question verbatim) and
reports, per candidate floor: acted rate, acted accuracy, ECE and Brier score —
plus the confidence distribution. The operator picks the floor from that table
and sets ``decisions.style.confidence_floor`` (env
``DECISIONS_STYLE_CONFIDENCE_FLOOR``), per
``docs/JEV_DECISION_MODEL_ANALYSIS_PLAN.md`` §7 ("ECE/Brier on held-out
families before trusting any threshold").

``confidence`` here is TypeSafe-style (top prob − mean of rest,
``src/clm/schema.py::confidence`` at the pinned upstream revision), NOT the top
probability — the ``--metric`` flag replays the table under either definition so
the choice is explicit. Holdout rows only: ``--split holdout`` is the default
and training rows are refused unless ``--split`` says otherwise.

Acted accuracy is stratified, because the seam acts on the two answer kinds
differently (``src/pipeline.py::_apply_decision_style``): a confident ``none``
is a no-op that keeps the LLM style, while a confident wrong non-``none`` style
is an INJECTED override error. ``acted_accuracy`` (the recommendation bar) is
over non-``none`` choices only; ``none_accuracy`` is reported beside it.

Each floor row also carries a 95% Wilson interval on its acted accuracy, so a
floor recommended on 30 rows is auditable as an estimate, not a point claim.

The ``--labels`` input (``derive_style_labels.py``) is the enable-path gate for
label provenance: ``main`` refuses to recommend a floor until
``qa/typed/review.json`` records human review of exactly the labels being
calibrated against (same file bytes, every labeled scenario's label+rationale
covered, reviewer named). Printing the table always works — only the
recommendation, and the exit code it drives, is gated.

With ``--min-acted-accuracy`` and ``--min-acted-rate`` the report also names the
lowest floor that clears both bars (``recommended``), which is the value
``decisions.style.confidence_floor`` should take; with no floor qualifying it exits 2
so the release step fails instead of enabling a seam the measurements do not support.

Run it against a serving container (stdlib only, no torch)::

    python scripts/decision_models/clm_calibrate.py \\
        --style qa/typed/style.jsonl --base-url http://127.0.0.1:18700/v1 \\
        --floors 0.3 0.4 0.5 0.6 --metric confidence \\
        --min-acted-accuracy 0.8 --min-acted-rate 0.3 --report models/clm-calibration.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: The abstention criterion: a confident ``none`` keeps the LLM style (a no-op),
#: so override accuracy is measured over non-``none`` choices only. Must match
#: ``src/decisions/seam_prompts.py::STYLE_NONE_KEY`` (pinned by unit test).
DEFAULT_QUESTION_ID = "architecture_style"

NONE_CHOICE = "none"

#: Wilson score interval z for the 95% acted-accuracy bands (two-sided).
_WILSON_Z = 1.96


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the calibration replay."""
    parser = argparse.ArgumentParser(description="Calibrate decisions.style.confidence_floor on held-out style rows.")
    parser.add_argument("--style", default="qa/typed/style.jsonl", help="qa/typed/style.jsonl")
    parser.add_argument("--split", default="holdout", help="qa/ source split to replay (default: holdout).")
    parser.add_argument("--base-url", required=True, help="clm-serve /v1 root (e.g. http://127.0.0.1:18700/v1).")
    parser.add_argument("--model", default="clm-latest", help="Served model name.")
    parser.add_argument("--question-id", default=DEFAULT_QUESTION_ID, help="Question id to ask and score.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--floors",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        help="Candidate floors to tabulate.",
    )
    parser.add_argument(
        "--metric",
        choices=["confidence", "top_prob"],
        default="confidence",
        help="Which answer number the floor applies to (the seam gates on confidence).",
    )
    parser.add_argument("--bins", type=int, default=10, help="ECE bins.")
    parser.add_argument(
        "--min-acted-accuracy",
        type=float,
        default=None,
        help="Recommend the lowest floor whose acted answers are at least this accurate (with --min-acted-rate).",
    )
    parser.add_argument(
        "--min-acted-rate",
        type=float,
        default=None,
        help="Recommend the lowest floor that also acts on at least this share of held-out rows.",
    )
    parser.add_argument(
        "--min-acted",
        type=int,
        default=10,
        help="Recommend only a floor with at least this many acted overrides (small-n guard).",
    )
    parser.add_argument(
        "--report", default="", help="Write the full report JSON here (e.g. models/clm-calibration.json)."
    )
    parser.add_argument(
        "--labels",
        default="scripts/decision_models/derive_style_labels.py",
        help="Style label source; --review must cover this file's STYLE_LABELS to recommend.",
    )
    parser.add_argument(
        "--review",
        default="qa/typed/review.json",
        help="Human review ledger covering the style labels (enable-path gate).",
    )
    return parser.parse_args(argv)


def load_rows(path: Path, split: str) -> list[dict[str, Any]]:
    """Read the style rows whose source split matches; refuse an empty result."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            source: dict[str, Any] = row.get("source") or {}
            if source.get("split") != split:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"no rows with source.split={split!r} in {path}")
    return rows


def ask(base_url: str, model: str, row: dict[str, Any], question_id: str, timeout: float) -> dict[str, Any]:
    """One served-shape /v1/systemone call; returns the raw answer object."""
    body = {
        "model": model,
        "state": row["state"],
        "questions": {
            question_id: {
                "type": "choice",
                "instructions": row["instructions"],
                "criteria": row["criteria"],
            }
        },
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/systemone",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload: dict[str, Any] = json.load(response)
    except OSError as exc:
        raise SystemExit(f"clm-serve request failed for row {row.get('id')!r}: {exc}") from exc
    answers: dict[str, Any] = payload.get("answers") or {}
    answer = answers.get(question_id)
    if not isinstance(answer, dict):
        raise SystemExit(f"clm-serve answer for row {row.get('id')!r} carries no {question_id!r} entry: {payload!r}")
    return answer


def metric_of(answer: dict[str, Any], metric: str) -> float:
    """The floored number: served confidence, or the top probability for comparison."""
    if metric == "confidence":
        value = answer.get("confidence")
        if not isinstance(value, (int, float)):
            raise SystemExit(f"answer carries no numeric confidence: {answer!r}")
        return float(value)
    probabilities: dict[str, Any] = answer.get("probabilities") or {}
    if not probabilities:
        raise SystemExit(f"answer carries no probabilities: {answer!r}")
    return float(max(probabilities.values()))


def expected_calibration_error(pairs: Sequence[tuple[float, bool]], bins: int) -> float:
    """Mean |accuracy − confidence| over equal-width confidence bins."""
    buckets: list[list[bool]] = [[] for _ in range(bins)]
    confidences: list[list[float]] = [[] for _ in range(bins)]
    for confidence, correct in pairs:
        index = min(bins - 1, int(confidence * bins))
        buckets[index].append(correct)
        confidences[index].append(confidence)
    total = len(pairs)
    ece = 0.0
    for bucket, confs in zip(buckets, confidences, strict=True):
        if not bucket:
            continue
        accuracy = sum(bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(accuracy - sum(confs) / len(confs))
    return ece


def brier_score(pairs: Sequence[tuple[float, bool]]) -> float:
    """Mean squared error of the confidence against the 0/1 outcome."""
    return sum((confidence - (1.0 if correct else 0.0)) ** 2 for confidence, correct in pairs) / len(pairs)


def review_status(labels_path: Path, review_path: Path) -> tuple[bool, str]:
    """Whether the ledger certifies the labels file: (ok, reason).

    The gate compares the ledger's recorded sha256 of the label source plus its
    covered (scenario_id, label) pairs against the file's own ``STYLE_LABELS``
    (AST-read, never executed) — a label or rationale edit after review voids
    coverage for that scenario, and an empty or missing review never certifies.
    """
    try:
        source = labels_path.read_bytes()
    except OSError as exc:
        return False, f"cannot read labels file {labels_path}: {exc}"
    status = _ledger_status(labels_path, review_path, source)
    if status is None or status["wanted"] is None or status["digest"] is None:
        return False, _ledger_failure(labels_path, review_path, source)
    wanted: list[tuple[str, str]] = status["wanted"]
    digest: str = status["digest"]
    covered = _covered_pairs(review_path, digest)
    missing = sorted(set(wanted) - covered)
    if missing:
        preview = ", ".join(f"{scenario}:{label}" for scenario, label in missing[:3])
        return False, f"{len(missing)} style labels lack human review (e.g. {preview})"
    return True, f"review.json covers all {len(wanted)} style labels at {digest[:12]}"


def _ledger_failure(labels_path: Path, review_path: Path, source: bytes) -> str:
    """The fail-closed reason when the ledger cannot certify the labels file."""
    wanted = _style_labels(source)
    if wanted is None:
        return f"labels file {labels_path} carries no parseable STYLE_LABELS dict"
    if not wanted:
        return f"labels file {labels_path} carries an empty STYLE_LABELS dict"
    try:
        ledger: dict[str, Any] = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"cannot read review ledger {review_path}: {exc}"
    return f"review status is {ledger.get('status')!r}, not 'reviewed'"


def _ledger_status(labels_path: Path, review_path: Path, source: bytes) -> dict[str, Any] | None:
    """The ledger's certification inputs, or None when it cannot certify."""
    del labels_path
    wanted = _style_labels(source)
    if not wanted:
        return None
    try:
        ledger: dict[str, Any] = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if ledger.get("status") != "reviewed":
        return None
    return {"wanted": wanted, "digest": hashlib.sha256(source).hexdigest()}


def _covered_pairs(review_path: Path, digest: str) -> set[tuple[str, str]]:
    """Ledger (scenario_id, label) pairs whose review is current for these bytes."""
    try:
        ledger: dict[str, Any] = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    reviews = ledger.get("reviews")
    entries = reviews.get("style") if isinstance(reviews, dict) else None
    if not isinstance(entries, list):
        return set()
    return {
        (str(entry.get("scenario_id")), str(entry.get("label")))
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("labels_sha256") == digest
        and entry.get("reviewer")
        and entry.get("rationale_ok") is True
    }


def _style_labels(source: bytes) -> list[tuple[str, str]] | None:
    """The label source's (scenario_id, label) pairs, AST-read without executing."""
    try:
        module = ast.parse(source.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None
    pairs: list[tuple[str, str]] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.AnnAssign) or getattr(node.target, "id", "") != "STYLE_LABELS":
            continue
        if not isinstance(node.value, ast.Dict):
            return None
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if (
                not isinstance(key, ast.Constant)
                or not isinstance(value, ast.Tuple)
                or len(value.elts) != 2
                or not all(isinstance(elt, ast.Constant) for elt in value.elts)
            ):
                return None
            label = value.elts[0]
            assert isinstance(label, ast.Constant)
            pairs.append((str(key.value), str(label.value)))
        return pairs
    return None


def wilson_interval(hits: int, total: int, z: float = _WILSON_Z) -> tuple[float, float]:
    """95% Wilson score interval for a hit rate (empty sample → the unit interval)."""
    if total <= 0:
        return 0.0, 1.0
    center = (hits + z * z / 2) / (total + z * z)
    half = z * math.sqrt(hits * (total - hits) / total + z * z / 4) / (total + z * z)
    return max(0.0, center - half), min(1.0, center + half)


def recommend(
    table: Sequence[dict[str, Any]],
    min_acted_accuracy: float | None,
    min_acted_rate: float | None,
    min_acted: int = 10,
) -> dict[str, Any]:
    """Lowest floor whose acted answers clear all three bars — the value to put in config.

    ``None`` thresholds mean "no recommendation requested": the table is still printed,
    the floor stays unset. When both are given and nothing qualifies, the recommendation
    is ``None`` with the reason recorded, so a caller can fail loudly instead of
    shipping a floor the measurements do not support. ``min_acted`` is the small-n
    guard: a floor with fewer acted overrides never qualifies, however accurate.
    """
    if min_acted_accuracy is None or min_acted_rate is None:
        return {"floor": None, "reason": "no thresholds given — reporting only, nothing recommended"}
    for row in sorted(table, key=lambda entry: float(entry["floor"])):
        acted_accuracy = row["acted_accuracy"]
        if acted_accuracy is None:
            continue
        if int(row["acted"]) < min_acted:
            continue
        if float(acted_accuracy) >= min_acted_accuracy and float(row["acted_rate"]) >= min_acted_rate:
            interval = row.get("acted_accuracy_ci95") or [0.0, 1.0]
            return {
                "floor": float(row["floor"]),
                "reason": (
                    f"acted accuracy {float(acted_accuracy):.4f} (95% CI "
                    f"{float(interval[0]):.4f}-{float(interval[1]):.4f}) >= {min_acted_accuracy:.4f} and acted "
                    f"rate {float(row['acted_rate']):.4f} >= {min_acted_rate:.4f} on {int(row['acted'])} overrides"
                ),
                "min_acted_accuracy": min_acted_accuracy,
                "min_acted_rate": min_acted_rate,
                "min_acted": min_acted,
                "acted": int(row["acted"]),
                "acted_accuracy_ci95": [float(interval[0]), float(interval[1])],
            }
    return {
        "floor": None,
        "reason": (
            f"no floor yields acted accuracy >= {min_acted_accuracy:.4f} at acted rate >= "
            f"{min_acted_rate:.4f} on >= {min_acted} overrides — the head cannot be enabled safely on this corpus"
        ),
        "min_acted_accuracy": min_acted_accuracy,
        "min_acted_rate": min_acted_rate,
        "min_acted": min_acted,
    }


def summarize(
    pairs: Sequence[tuple[float, bool, str]],
    floors: Sequence[float],
    bins: int,
) -> dict[str, Any]:
    """Floor-candidate table plus distribution and calibration summaries.

    Each pair is ``(confidence, correct, choice)``. The seam only overrides on a
    confident non-``none`` choice, so per-floor ``acted_accuracy`` is over those
    override answers alone (with a 95% Wilson interval); confident ``none``
    answers are reported as ``none_accuracy`` instead of feeding the enable bar.
    """
    scored = [(confidence, correct) for confidence, correct, _ in pairs]
    table: list[dict[str, Any]] = []
    for floor in floors:
        acted = [(confidence, correct, choice) for confidence, correct, choice in pairs if confidence >= floor]
        overridden = [(correct, choice) for _, correct, choice in acted if choice != NONE_CHOICE]
        abstained = [correct for _, correct, choice in acted if choice == NONE_CHOICE]
        hits = sum(1 for correct, _ in overridden if correct)
        accuracy = (hits / len(overridden)) if overridden else None
        low, high = wilson_interval(hits, len(overridden))
        table.append(
            {
                "floor": floor,
                "acted": len(overridden),
                "acted_rate": (len(overridden) / len(pairs)) if pairs else 0.0,
                "acted_accuracy": accuracy,
                "acted_accuracy_ci95": [low, high],
                "none_acted": len(abstained),
                "none_accuracy": (sum(1 for correct in abstained if correct) / len(abstained)) if abstained else None,
            }
        )
    confidences = sorted(confidence for confidence, _, _ in pairs)
    return {
        "n": len(pairs),
        "accuracy": (sum(correct for _, correct, _ in pairs) / len(pairs)) if pairs else 0.0,
        "override_rate": (sum(1 for _, _, choice in pairs if choice != NONE_CHOICE) / len(pairs) if pairs else 0.0),
        "ece": expected_calibration_error(scored, bins),
        "brier": brier_score(scored) if scored else 0.0,
        "confidence": {
            "min": confidences[0] if confidences else None,
            "p50": confidences[len(confidences) // 2] if confidences else None,
            "max": confidences[-1] if confidences else None,
        },
        "floors": table,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Replay → score → print the table; recommend a floor when thresholds pass."""
    args = parse_args(argv)
    rows = load_rows(Path(args.style), args.split)
    pairs: list[tuple[float, bool, str]] = []
    for row in rows:
        answer = ask(args.base_url, args.model, row, args.question_id, args.timeout)
        choice = answer.get("choice")
        if not isinstance(choice, str):
            raise SystemExit(f"clm-serve answer for row {row.get('id')!r} carries no string choice: {answer!r}")
        pairs.append((metric_of(answer, args.metric), choice == row["label"], choice))
    summary = summarize(pairs, list(args.floors), args.bins)
    recommended = recommend(summary["floors"], args.min_acted_accuracy, args.min_acted_rate, args.min_acted)
    reviewed, review_reason = review_status(Path(args.labels), Path(args.review))
    if recommended["floor"] is not None and not reviewed:
        recommended = {
            "floor": None,
            "reason": f"refusing to recommend: {review_reason}",
            "min_acted_accuracy": args.min_acted_accuracy,
            "min_acted_rate": args.min_acted_rate,
            "min_acted": args.min_acted,
        }
    report = {
        "metric": args.metric,
        "split": args.split,
        "question_id": args.question_id,
        "label_review": {"certified": reviewed, "reason": review_reason},
        **summary,
        "recommended": recommended,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.min_acted_accuracy is not None and report["recommended"]["floor"] is None:
        print(f"clm-calibrate: FAIL: {report['recommended']['reason']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
