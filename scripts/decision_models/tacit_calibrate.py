#!/usr/bin/env python
"""Calibrate the Ch2 style seam's confidence floor against held-out style rows.

Replays the holdout-split rows of ``qa/typed/style.jsonl`` through a RUNNING AnyJev
gateway, using the served two-stage style procedure verbatim (stage 1 = the category
question on the capped description, stage 2 = the style question over that category's
shortlist), and reports per candidate floor: acted rate, acted accuracy, ECE and Brier
score — plus the confidence distribution. The operator picks the floor from that table
and sets ``decisions.style.confidence_floor`` (env
``DECISIONS_STYLE_CONFIDENCE_FLOOR``), per
``docs/JEV_DECISION_MODEL_ANALYSIS_PLAN.md`` §7 ("ECE/Brier on held-out
families before trusting any threshold").

``confidence`` here is the STAGE-2 top probability — the number the seam gates on. The
seam acts only when BOTH stages clear the floor (``src/decisions/tacit.py::choose_style``),
so a row is *acted* at floor ``f`` iff ``min(stage1, stage2) >= f`` and stage 2 answered a
real option: a wrong stage-1 category that excludes the gold style counts as a miss, not
as an abstention. ECE/Brier stay over the stage-2 probability among acted rows. Holdout
rows only: ``--split holdout`` is the default and training rows are refused unless
``--split`` says otherwise.

Acted accuracy is stratified, because the seam acts on the two answer kinds differently
(``src/pipeline.py::_apply_decision_style``): a confident ``none`` is a no-op that keeps
the LLM style, while a confident wrong non-``none`` style is an INJECTED override error.
``acted_accuracy`` (the recommendation bar) is over non-``none`` choices only;
``none_accuracy`` is reported beside it.

Each floor row also carries a 95% Wilson interval on its acted accuracy, so a floor
recommended on 30 rows is auditable as an estimate, not a point claim.

The ``--labels`` input (``derive_style_labels.py``) is the enable-path gate for label
provenance: ``main`` refuses to recommend a floor until ``qa/typed/review.json`` records
human review of exactly the labels being calibrated against (same file bytes, every
labeled scenario's label+rationale covered, reviewer named). Printing the table always
works — only the recommendation, and the exit code it drives, is gated.

The report also records what served the replay (``served``: container, image ID, image tag,
container start time, model repo, base URL) so a 30-row measurement is self-attributing and a
stale container cannot be mistaken for a fresh one. That identity is captured by the make rule
via ``docker inspect`` — ``anyjev.serve``'s ``GET /v1/stats`` exposes no identity over the wire,
and the flags are recorded verbatim rather than validated, so the script still runs against a
remote gateway on a host with no docker.

With ``--min-acted-accuracy`` and ``--min-acted-rate`` the report also names the lowest
floor that clears both bars (``recommended``), which is the value
``decisions.style.confidence_floor`` should take; with no floor qualifying it exits 2 so
the release step fails instead of enabling a seam the measurements do not support.

Run it against a serving container (stdlib only, no torch)::

    python scripts/decision_models/tacit_calibrate.py \\
        --style qa/typed/style.jsonl --base-url http://127.0.0.1:18100 \\
        --floors 0.3 0.4 0.5 0.6 \\
        --min-acted-accuracy 0.8 --min-acted-rate 0.3 --report models/tacit-calibration.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

#: The abstention criterion: a confident ``none`` keeps the LLM style (a no-op), so
#: override accuracy is measured over non-``none`` choices only. Must match
#: ``src/decisions/seam_prompts.py::STYLE_NONE_KEY`` (pinned by unit test).
NONE_CHOICE = "none"

#: Wilson score interval z for the 95% acted-accuracy bands (two-sided).
_WILSON_Z = 1.96

#: The acted rule, recorded in the report so a consumer never has to infer it.
ACTED_RULE = "min(stage1_category, stage2_style) >= floor"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the calibration replay."""
    parser = argparse.ArgumentParser(description="Calibrate decisions.style.confidence_floor on held-out style rows.")
    parser.add_argument("--style", default="qa/typed/style.jsonl", help="qa/typed/style.jsonl")
    parser.add_argument("--split", default="holdout", help="qa/ source split to replay (default: holdout).")
    parser.add_argument(
        "--base-url",
        required=True,
        help="AnyJev gateway ROOT (e.g. http://127.0.0.1:18100; the client appends /v1/decide).",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--seam-prompts",
        default="src/decisions/seam_prompts.py",
        help="The served prompt vocabulary, loaded by path (the trainer's rule).",
    )
    parser.add_argument(
        "--floors",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        help="Candidate floors to tabulate.",
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
        "--report", default="", help="Write the full report JSON here (e.g. models/tacit-calibration.json)."
    )
    parser.add_argument(
        "--served-container",
        default="",
        help="Name of the container that served the replay (the make rule fills this via docker inspect).",
    )
    parser.add_argument(
        "--served-image",
        default="",
        help="Image ID of the serving container (sha256:...), captured by the make rule — never hand-typed.",
    )
    parser.add_argument(
        "--served-image-tag",
        default="",
        help="Image tag the serving container requested (docker inspect .Config.Image).",
    )
    parser.add_argument(
        "--served-container-started-at",
        default="",
        help="docker inspect .State.StartedAt of the serving container — proves the container outlived a rebuild.",
    )
    parser.add_argument(
        "--served-model-repo",
        default="",
        help="HF repo of the served checkpoint (the make rule passes TACIT_MODEL_REPO).",
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


def load_seam_prompts(path: Path) -> ModuleType:
    """Import the served prompt vocabulary by path (stdlib-only, no package needed)."""
    spec = importlib.util.spec_from_file_location("seam_prompts", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load seam prompts from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decide(base_url: str, item: dict[str, Any], timeout: float) -> dict[str, Any]:
    """One ``POST /v1/decide``; returns the raw decision object."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/decide",
        data=json.dumps(item).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload: dict[str, Any] = json.load(response)
    except OSError as exc:
        raise SystemExit(f"anyjev gateway request failed: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("probs"), dict):
        raise SystemExit(f"anyjev gateway answer is not a decision: {payload!r}")
    return payload


def ask(base_url: str, row: dict[str, Any], seams: Any, timeout: float) -> dict[str, Any]:
    """Replay the served two-stage style choice; returns the replay's answer summary.

    ``confidence`` is the stage-2 top probability (the number the seam gates on);
    ``stage1`` rides along so the acted rule can require BOTH stages to clear the floor.
    A failed stage 1 (transport/shape/off-taxonomy) yields a never-acted answer instead of
    aborting the replay.
    """
    state = str(row["state"])[: seams.DECISION_STATE_MAX_CHARS]
    stage1 = _decide(
        base_url,
        {
            "state": state,
            "question": seams.category_question(),
            "options": list(seams.CATEGORY_SLUGS),
            "kind": "choice",
        },
        timeout,
    )
    answer1 = stage1.get("answer")
    probs1: dict[str, Any] = stage1.get("probs") or {}
    if not isinstance(answer1, str) or answer1 not in probs1 or answer1 not in seams.CATEGORY_SLUGS:
        return {"choice": None, "probabilities": {}, "confidence": 0.0, "stage1": 0.0}
    stage1_conf = float(probs1[answer1])
    styles = seams.styles_for_category(answer1)
    options = [*styles, seams.STYLE_NONE_KEY]
    stage2 = _decide(
        base_url,
        {"state": state, "question": seams.style_question(styles), "options": options, "kind": "choice"},
        timeout,
    )
    answer2 = stage2.get("answer")
    probs2_all: dict[str, Any] = stage2.get("probs") or {}
    probs2 = {
        key: float(value)
        for key, value in probs2_all.items()
        if key in options and isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not isinstance(answer2, str) or answer2 not in probs2:
        return {"choice": None, "probabilities": {}, "confidence": 0.0, "stage1": stage1_conf}
    return {
        "choice": answer2,
        "probabilities": probs2,
        "confidence": probs2[answer2],
        "stage1": stage1_conf,
    }


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
            f"{min_acted_rate:.4f} on >= {min_acted} overrides — the seam cannot be enabled safely on this corpus"
        ),
        "min_acted_accuracy": min_acted_accuracy,
        "min_acted_rate": min_acted_rate,
        "min_acted": min_acted,
    }


def summarize(
    pairs: Sequence[tuple[float, float, bool, str | None]],
    floors: Sequence[float],
    bins: int,
) -> dict[str, Any]:
    """Floor-candidate table plus distribution and calibration summaries.

    Each pair is ``(confidence, acted_confidence, correct, choice)``: ``confidence`` is the
    stage-2 probability the calibration curves are computed over, ``acted_confidence`` is
    ``min(stage1, stage2)`` — the number the seam actually compares with its floor — and
    ``choice`` is ``None`` when the replay produced no usable answer. Per-floor
    ``acted_accuracy`` is over non-``none`` overrides alone (with a 95% Wilson interval);
    confident ``none`` answers are reported as ``none_accuracy`` instead of feeding the
    enable bar. ``replay_failures`` is the number of rows the replay could not answer at
    all — floor-INVARIANT by construction (a failed row is never "acted", so a per-floor
    count would read 0 at every floor the operator would consider), which keeps a dead or
    broken sidecar from masquerading as abstention.
    """
    scored = [(confidence, correct) for confidence, _acted, correct, _choice in pairs]
    table: list[dict[str, Any]] = []
    for floor in floors:
        overridden = [
            correct
            for _confidence, acted_confidence, correct, choice in pairs
            if acted_confidence >= floor and choice is not None and choice != NONE_CHOICE
        ]
        abstained = [
            correct
            for _confidence, acted_confidence, correct, choice in pairs
            if acted_confidence >= floor and choice == NONE_CHOICE
        ]
        hits = sum(1 for correct in overridden if correct)
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
    confidences = sorted(confidence for confidence, _acted, _correct, _choice in pairs)
    return {
        "n": len(pairs),
        # Floor-invariant: failed replays are excluded from every acted bucket, so one
        # number here is the only honest way to report a broken gateway.
        "replay_failures": sum(1 for _c, _a, _r, choice in pairs if choice is None),
        "accuracy": (sum(correct for _c, _a, correct, _choice in pairs) / len(pairs)) if pairs else 0.0,
        "override_rate": (
            sum(1 for _c, _a, _r, choice in pairs if choice is not None and choice != NONE_CHOICE) / len(pairs)
            if pairs
            else 0.0
        ),
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
    seams = load_seam_prompts(Path(args.seam_prompts))
    pairs: list[tuple[float, float, bool, str | None]] = []
    for row in rows:
        answer = ask(args.base_url, row, seams, args.timeout)
        choice = answer["choice"]
        confidence = float(answer["confidence"])
        acted_confidence = min(float(answer["stage1"]), confidence) if choice is not None else 0.0
        pairs.append((confidence, acted_confidence, choice == row["label"], choice))
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
        "acted_rule": ACTED_RULE,
        "split": args.split,
        "served": {
            "base_url": args.base_url,
            "container": args.served_container or None,
            "image_id": args.served_image or None,
            "image_tag": args.served_image_tag or None,
            "container_started_at": args.served_container_started_at or None,
            "model_repo": args.served_model_repo or None,
        },
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
        print(f"tacit-calibrate: FAIL: {report['recommended']['reason']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
