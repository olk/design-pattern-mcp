#!/usr/bin/env python
"""Calibrate the Ch2 style seam's confidence floor against held-out style rows.

Replays the holdout-split rows of ``qa/typed/style.jsonl`` through a RUNNING AnyJev
gateway, using the served sharded style procedure verbatim (``style_shards``: the whole
35-style vocabulary in balanced questions of at most 25 styles plus ``none``, every shard
asked, the most confident named style winning), and reports per candidate floor: acted rate, acted accuracy, ECE and Brier
score — plus the confidence distribution. The operator picks the floor from that table
and sets ``decisions.style.confidence_floor`` (env
``DECISIONS_STYLE_CONFIDENCE_FLOOR``), per
``docs/JEV_DECISION_MODEL_ANALYSIS_PLAN.md`` §7 ("ECE/Brier on held-out
families before trusting any threshold").

``confidence`` here is the winning named style's probability across the style shards — the
number the seam gates on. Serving asks the 35-way style question in cap-sized shards
(``style_shards``; AnyJev takes 26 options at most) and keeps the most confident *named*
style; a shard's ``none`` means only "not from my list", so it never vetoes another
shard's answer (``src/decisions/tacit.py::choose_style``). So a row is *acted* at floor
``f`` iff that probability ``>= f`` and the winner is a real style: a confident wrong
style is an injected override error, not an abstention, and no shard narrows the answer
space. ECE/Brier are computed over every replayed row. Holdout rows only: ``--split
holdout`` is the default and training rows are refused unless ``--split`` says otherwise.

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

The report is per-row attributable: ``rows`` carries one record per replayed row (label,
gold shard and rank, the chosen style, every shard's answer and probabilities),
``diagnostics`` states whether the bars are reachable at all on these answers (the gate
ceiling a perfect re-ranking could reach, a dense threshold sweep, the cross-shard and
``none``-row breakdowns, AUROC and the information the floor carries), and ``consistency``
(with ``--repeats``) whether the served choice is repeat-stable. A recommendation needs the
bars, a stable choice, and — before serving — a certified label ledger.

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
ACTED_RULE = "max_over_style_shards(top) >= floor"


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
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
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
        "--repeats",
        type=int,
        default=1,
        help="Replays per row; >1 measures repeat consistency and gates the recommendation on it.",
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
    """Replay the served sharded style choice; returns the replay's answer summary.

    ``confidence`` is the winning named style's probability across the shards — the number
    the seam gates on (``ACTED_RULE``). Every shard is asked, so no style is unofferable; a
    shard's ``none`` means only "not from my list", so it never vetoes another shard's style
    and the row abstains only when no shard names one. A shard with an unusable reply is
    skipped, and no usable shard leaves the row never-acted instead of aborting the replay.
    """
    state = str(row["state"])[: seams.DECISION_STATE_MAX_CHARS]
    winner: str | None = None
    confidence = 0.0
    probabilities: dict[str, float] = {}
    abstained = False
    abstain_confidence = 0.0
    shard_records: list[dict[str, Any]] = []
    for styles in seams.style_shards():
        options = [*styles, seams.STYLE_NONE_KEY]
        reply = _decide(
            base_url,
            {"state": state, "question": seams.style_question(styles), "options": options, "kind": "choice"},
            timeout,
        )
        answer = reply.get("answer")
        raw_probs: dict[str, Any] = reply.get("probs") or {}
        probs = {
            key: float(value)
            for key, value in raw_probs.items()
            if key in options and isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        route = reply.get("route")
        route_name = route if isinstance(route, str) else None
        if not isinstance(answer, str) or answer not in probs:
            shard_records.append({"answer": None, "probs": probs, "route": route_name})
            continue
        shard_records.append({"answer": answer, "probs": probs, "route": route_name})
        if answer == seams.STYLE_NONE_KEY:
            abstained = True
            abstain_confidence = max(abstain_confidence, probs[answer])
            continue
        if probs[answer] > confidence:
            winner = answer
            confidence = probs[answer]
            probabilities = probs
    # ``none`` is an ABSTENTION, not a failed replay: the report scores it as ``none_accuracy``,
    # while ``None`` means no shard produced a usable answer at all.
    if winner is None and abstained:
        return {
            "choice": seams.STYLE_NONE_KEY,
            "probabilities": {},
            "confidence": abstain_confidence,
            "shards": shard_records,
        }
    return {"choice": winner, "probabilities": probabilities, "confidence": confidence, "shards": shard_records}


def _binary_entropy(p: float) -> float:
    """Shannon entropy of Bernoulli(p) in bits (0 at the degenerate ends)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def _entropy_of(counts: Sequence[int]) -> float:
    """Plug-in entropy of a count vector in bits."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts if count > 0)


def _counts(values: Sequence[Any]) -> dict[Any, int]:
    """Occurrence count per distinct value."""
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _auroc(scored: Sequence[tuple[float, bool]]) -> float | None:
    """Rank-based AUROC with ties shared; None when either outcome class is empty."""
    correct = [score for score, hit in scored if hit]
    wrong = [score for score, hit in scored if not hit]
    if not correct or not wrong:
        return None
    wins = sum(1.0 if c > w else 0.5 if c == w else 0.0 for c in correct for w in wrong)
    return wins / (len(correct) * len(wrong))


def row_record(row: dict[str, Any], answer: dict[str, Any], seams: Any) -> dict[str, Any]:
    """One replayed row's attribution: the label, the chosen style and every shard's answer.

    ``gold_shard``/``p_gold``/``gold_rank`` locate the label inside the served shards, so a
    miss separates "a wrong style won its own shard" from "the gold shard answered gold but
    the other shard's answer won" (``diagnostics.shards.cross_shard_losses``). ``shards`` is
    empty when ``ask`` was stubbed, which keeps the record valid for a fake replay.
    """
    label = str(row["label"])
    choice = answer["choice"]
    shard_list = list(seams.style_shards())
    gold_shard = next((index for index, styles in enumerate(shard_list) if label in styles), None)
    winner_shard = next(
        (
            index
            for index, entry in enumerate(answer.get("shards", []))
            if choice not in (None, NONE_CHOICE) and entry["answer"] == choice
        ),
        None,
    )
    records: list[dict[str, Any]] = []
    for index, entry in enumerate(answer.get("shards", [])):
        probs: dict[str, float] = entry.get("probs") or {}
        shard_answer = entry.get("answer")
        record: dict[str, Any] = {
            "shard": index,
            "answer": shard_answer,
            "p_answer": probs.get(shard_answer) if isinstance(shard_answer, str) else None,
            "p_none": probs.get(seams.STYLE_NONE_KEY),
            "p_gold": None,
            "gold_rank": None,
            "route": entry.get("route"),
        }
        if index == gold_shard:
            p_gold = probs.get(label)
            record["p_gold"] = p_gold
            if p_gold is not None:
                record["gold_rank"] = 1 + sum(1 for value in probs.values() if value > p_gold)
        records.append(record)
    return {
        "scenario_id": row["source"].get("scenario_id"),
        "label": label,
        "gold_shard": gold_shard,
        "choice": choice,
        "confidence": float(answer["confidence"]),
        "correct": choice == label,
        "winner_shard": winner_shard,
        "shards": records,
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

    Each pair is ``(confidence, acted_confidence, correct, choice)``: both numbers are the
    winning shard probability — ``confidence`` feeds the calibration curves and
    ``acted_confidence`` is the value the seam compares with its floor — and
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


def _gate_ceiling(
    named: Sequence[dict[str, Any]],
    named_hits: int,
    n: int,
    min_acted_accuracy: float | None,
    min_acted_rate: float | None,
    min_acted: int,
) -> tuple[dict[str, Any], int | None]:
    """The best accuracy a perfect re-ranking of these answers could gate on (necessary only).

    ``min_rows`` is the smallest acted count that can satisfy the rate bar and the small-n
    guard; ``required_hits`` the smallest hit count reaching the accuracy bar there. When
    that count exceeds the observed hits, no floor can pass however the scores are
    recalibrated — the answers themselves are the binding constraint.
    """
    if min_acted_accuracy is None or min_acted_rate is None or not n:
        return (
            {
                "min_rows": None,
                "required_hits": None,
                "attainable_with_perfect_ranking": False,
                "best_accuracy_with_perfect_ranking": None,
            },
            None,
        )
    k_rate = next((k for k in range(n + 1) if k / n >= min_acted_rate), None)
    if k_rate is None:
        return (
            {
                "min_rows": None,
                "required_hits": None,
                "attainable_with_perfect_ranking": False,
                "best_accuracy_with_perfect_ranking": None,
            },
            None,
        )
    min_rows = max(1, min_acted, k_rate)
    required_hits = next((h for h in range(min_rows + 1) if h / min_rows >= min_acted_accuracy), None)
    attainable = required_hits is not None and len(named) >= min_rows and named_hits >= required_hits
    best = (min(named_hits, min_rows) / min_rows) if len(named) >= min_rows else None
    return (
        {
            "min_rows": min_rows,
            "required_hits": required_hits,
            "attainable_with_perfect_ranking": attainable,
            "best_accuracy_with_perfect_ranking": best,
        },
        min_rows,
    )


def _dense_floor_sweep(
    named: Sequence[dict[str, Any]],
    n: int,
    min_rows: int | None,
    min_acted_accuracy: float | None,
    min_acted_rate: float | None,
    min_acted: int,
) -> dict[str, Any]:
    """Every distinct observed confidence as a candidate floor, not just the tabulated grid.

    A passing threshold can sit between two grid points, which would make a failing table
    an artifact of the grid. ``best_accuracy_at_min_rows`` is the accuracy of the best
    gate-sized cut (``min_rows`` = the smallest count both the rate bar and the small-n
    guard allow), so a near miss is visible as a number the operator can act on.
    """
    thresholds = sorted({float(record["confidence"]) for record in named})
    sweep: dict[str, Any] = {
        "candidates": len(thresholds),
        "lowest_passing_floor": None,
        "best_accuracy_at_min_rows": None,
    }
    if min_rows is None or min_acted_accuracy is None or min_acted_rate is None or not n:
        return sweep
    passing: list[float] = []
    best_at_min_rows: float | None = None
    for threshold in thresholds:
        acted = [record for record in named if float(record["confidence"]) >= threshold]
        if len(acted) < min_rows:
            continue
        accuracy = sum(1 for record in acted if record["correct"]) / len(acted)
        best_at_min_rows = accuracy if best_at_min_rows is None else max(best_at_min_rows, accuracy)
        if len(acted) >= min_acted and len(acted) / n >= min_acted_rate and accuracy >= min_acted_accuracy:
            passing.append(threshold)
    sweep["lowest_passing_floor"] = min(passing) if passing else None
    sweep["best_accuracy_at_min_rows"] = best_at_min_rows
    return sweep


def _shard_attribution(records: Sequence[dict[str, Any]], named_hits: int) -> dict[str, Any]:
    """Whether the loss is inside a shard or in the cross-shard comparison.

    ``gold_shard_hits`` counts rows whose own shard named the gold; when that exceeds the
    hit count the difference was answered correctly and then outvoted by the other shard —
    the combination rule's loss, not the model's.
    """
    gold_shard_hits = 0
    both_named = 0
    rank_counts: dict[str, int] = {"1": 0, "2": 0, "3": 0, ">3": 0}
    for record in records:
        entries = record["shards"]
        if sum(1 for entry in entries if entry["answer"] not in (None, NONE_CHOICE)) >= 2:
            both_named += 1
        gold_shard = record["gold_shard"]
        if gold_shard is None or gold_shard >= len(entries):
            continue
        if record["label"] != NONE_CHOICE and entries[gold_shard]["answer"] == record["label"]:
            gold_shard_hits += 1
        rank = entries[gold_shard]["gold_rank"]
        if rank is not None:
            rank_counts[">3" if int(rank) > 3 else str(int(rank))] += 1
    return {
        "gold_shard_hits": gold_shard_hits,
        "cross_shard_losses": gold_shard_hits - named_hits,
        "both_named_rows": both_named,
        "gold_rank": rank_counts,
    }


def _information_bits(
    records: Sequence[dict[str, Any]], named: Sequence[dict[str, Any]], floors: Sequence[float]
) -> dict[str, Any]:
    """How much the choice, and the floor itself, says about the outcome.

    ``floor_information`` is the outcome entropy a floor removes over the NAMED rows (the
    rows the floor decides). Near zero means the confidence carries almost no signal about
    correctness, so no threshold can select the right answers.
    """
    choice_counts = _counts([record["choice"] if record["choice"] is not None else "<failed>" for record in records])
    joint_counts = _counts(
        [(record["label"], record["choice"] if record["choice"] is not None else "<failed>") for record in records]
    )
    label_counts = _counts([record["label"] for record in records])
    label_entropy = _entropy_of(list(label_counts.values()))
    choice_entropy = _entropy_of(list(choice_counts.values()))
    joint_entropy = _entropy_of(list(joint_counts.values()))
    if not named:
        return {
            "label_entropy": label_entropy,
            "choice_entropy": choice_entropy,
            "choice_label_mutual_information": label_entropy + choice_entropy - joint_entropy,
            "override_outcome_entropy": None,
            "floor_information": None,
        }
    named_hits = sum(1 for record in named if record["correct"])
    prior_entropy = _binary_entropy(named_hits / len(named))
    floor_information: dict[str, float] = {}
    for floor in floors:
        above = [record for record in named if float(record["confidence"]) >= floor]
        below = [record for record in named if float(record["confidence"]) < floor]
        information = prior_entropy
        for side in (above, below):
            if side:
                hits = sum(1 for record in side if record["correct"])
                information -= (len(side) / len(named)) * _binary_entropy(hits / len(side))
        floor_information[f"{floor:g}"] = information
    return {
        "label_entropy": label_entropy,
        "choice_entropy": choice_entropy,
        "choice_label_mutual_information": label_entropy + choice_entropy - joint_entropy,
        "override_outcome_entropy": prior_entropy,
        "floor_information": floor_information,
    }


def diagnostics(
    records: Sequence[dict[str, Any]],
    floors: Sequence[float],
    min_acted_accuracy: float | None,
    min_acted_rate: float | None,
    min_acted: int,
) -> dict[str, Any]:
    """Attribution of the gate decision: are the bars reachable on these answers at all?

    ``gate_ceiling`` is a NECESSARY condition — it says whether enough of the recorded
    answers are right, not whether a threshold exists; ``dense_floor_sweep`` answers the
    complementary question over every distinct observed confidence. A report that fails
    with ``attainable_with_perfect_ranking: false`` cannot be fixed by a floor or a
    rank-preserving recalibration of the winner probability.
    """
    n = len(records)
    named = [record for record in records if record["choice"] not in (None, NONE_CHOICE)]
    named_hits = sum(1 for record in named if record["correct"])
    ceiling, min_rows = _gate_ceiling(named, named_hits, n, min_acted_accuracy, min_acted_rate, min_acted)
    margins: list[tuple[float, bool]] = []
    for record in named:
        winner_shard = record["winner_shard"]
        if winner_shard is None or winner_shard >= len(record["shards"]):
            continue
        p_none = record["shards"][winner_shard]["p_none"]
        margins.append(
            (float(record["confidence"]) - (float(p_none) if p_none is not None else 0.0), bool(record["correct"]))
        )
    none_records = [record for record in records if record["label"] == NONE_CHOICE]
    confusion = [
        {"label": label, "choice": choice, "count": count}
        for (label, choice), count in _counts([(record["label"], record["choice"]) for record in records]).items()
    ]
    confusion.sort(key=lambda entry: (-entry["count"], entry["label"], str(entry["choice"])))
    return {
        "n": n,
        "named": len(named),
        "named_hits": named_hits,
        "gate_ceiling": ceiling,
        "dense_floor_sweep": _dense_floor_sweep(named, n, min_rows, min_acted_accuracy, min_acted_rate, min_acted),
        "shards": _shard_attribution(records, named_hits),
        "none_rows": {
            "gold_none": len(none_records),
            "abstained": sum(1 for record in none_records if record["choice"] == NONE_CHOICE),
            "acted": sum(1 for record in none_records if record["choice"] not in (None, NONE_CHOICE)),
        },
        "auroc": _auroc([(float(record["confidence"]), bool(record["correct"])) for record in named]),
        "auroc_none_margin": _auroc(margins),
        "information_bits": _information_bits(records, named, floors),
        "confusion": confusion,
    }


def consistency(replays: Sequence[Sequence[dict[str, Any]]], floors: Sequence[float]) -> dict[str, Any] | None:
    """Repeat consistency of the served choice across replays of the same rows.

    Serving is one temperature-0 forward per shard, so the same request must return the same
    choice; a floor is recommended only when every row's choice is identical across replays
    and no row's acted status flips at that floor. ``None`` when fewer than 2 replays ran.
    """
    if len(replays) < 2:
        return None
    rows = len(replays[0])
    by_row: list[list[dict[str, Any]]] = [[replay[index] for replay in replays] for index in range(rows)]
    unstable = [group for group in by_row if len({entry["choice"] for entry in group}) > 1]
    acted_flips: dict[str, int] = {}
    for floor in floors:
        acted_flips[f"{floor:g}"] = sum(
            1
            for group in by_row
            if len(
                {entry["choice"] not in (None, NONE_CHOICE) and float(entry["confidence"]) >= floor for entry in group}
            )
            > 1
        )
    return {
        "repeats": len(replays),
        "rows": rows,
        "stable_rows": rows - len(unstable),
        "style_choice_repeat_agreement": ((rows - len(unstable)) / rows) if rows else None,
        "unstable_rows": [str(group[0]["scenario_id"]) for group in unstable],
        "acted_flips": acted_flips,
        "max_confidence_spread": max(
            (
                max(float(entry["confidence"]) for entry in group) - min(float(entry["confidence"]) for entry in group)
                for group in by_row
            ),
            default=0.0,
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Replay → score → print the table; recommend a floor when thresholds pass."""
    args = parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    rows = load_rows(Path(args.style), args.split)
    seams = load_seam_prompts(Path(args.seam_prompts))
    replays: list[list[dict[str, Any]]] = [[] for _ in range(args.repeats)]
    for row in rows:
        for replay in replays:
            replay.append(row_record(row, ask(args.base_url, row, seams, args.timeout), seams))
    pairs: list[tuple[float, float, bool, str | None]] = []
    for row, record in zip(rows, replays[0], strict=True):
        choice = record["choice"]
        confidence = float(record["confidence"])
        pairs.append((confidence, confidence, choice == row["label"], choice))
    summary = summarize(pairs, list(args.floors), args.bins)
    checked = diagnostics(replays[0], list(args.floors), args.min_acted_accuracy, args.min_acted_rate, args.min_acted)
    stable = consistency(replays, list(args.floors))
    recommended = recommend(summary["floors"], args.min_acted_accuracy, args.min_acted_rate, args.min_acted)
    if recommended["floor"] is not None and stable is not None:
        floor = float(recommended["floor"])
        flips = int(stable["acted_flips"].get(f"{floor:g}", 0))
        if float(stable["style_choice_repeat_agreement"]) < 1.0 or flips > 0:
            recommended = {
                "floor": None,
                "reason": (
                    "refusing to recommend: style choice is not repeat-stable "
                    f"({stable['stable_rows']}/{stable['rows']} rows identical across {stable['repeats']} replays, "
                    f"{flips} acted flips at floor {floor:g})"
                ),
                "min_acted_accuracy": args.min_acted_accuracy,
                "min_acted_rate": args.min_acted_rate,
                "min_acted": args.min_acted,
            }
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
        "rows": replays[0],
        "diagnostics": checked,
        "consistency": stable,
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
