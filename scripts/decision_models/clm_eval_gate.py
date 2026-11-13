#!/usr/bin/env python
"""Gate the CLM image build on the fine-tune's own evaluation (no retrain, no GPU).

``train/finetune.py --task choice`` at the pinned revision already evaluates: it
prints ``[choice] best epoch ...`` and writes ``finetune_summary.json`` with the
test accuracy of the trained head (``test.acc``) and of the released init
checkpoint (``init_test.acc``), each with a per-question map. This script is the
build gate over that summary plus the export manifest.

Rules (every rule prints PASS / FAIL / SKIP with its numbers):

  R0 manifest — the export manifest is this gate's second input: it supplies the
     per-question row count (one row = ``1/n`` of accuracy) and the gold-label
     histogram of the *same* shard the trainer scored. Missing manifest, unknown
     ``schema_version``, empty test shard, or a requested question absent from it
     fails closed: a silently absent input would turn R2/R5 into no-ops.
  R1 overall — ``test.acc >= init_test.acc``: the trained head is no worse than
     the released checkpoint on the family-disjoint test shard.
  R2 served questions — per-question ``test >= init_test - 1/n``: one flipped row
     is inside the measurement's granularity, two are a regression. Only questions
     a client actually asks belong here (``architecture_style``: the CLM client
     sends nothing else).
  R3 advisory questions — same arithmetic, printed, never blocking: the shared
     heads also answer the synthetic ``choice`` question behind ``/v1/rank``, and
     a category regression there is bounded by ``clm_rank_parity.py`` instead.
  R4 fresh head — ``best_epoch > 0``: ``best_head.pt`` is initialised to the
     released weights and only replaced on a val improvement, so epoch 0 means the
     "fine-tuned" head IS the released checkpoint and must not ship.
  R5 prior — per served question ``test >= majority_share``, the share of the most
     frequent gold label on that shard: a head that only learned the label prior
     cannot clear it. Skipped (and printed as such, never counted as a check) when
     the shard carries a single label, where the rule would be vacuous.
  R6 support — every gold label the test shard asks about for a served question
     must have at least one train-shard positive: a test label the head never saw
     as gold measures transfer, not learning, and R2's accuracy number would let
     the unsupported rows hide inside the average. Fails with the label names.
  R7 coverage — the train shard's ``uncovered`` criteria for a served question
     (never positive in training, still winnable at serve time) must be
     explicitly acknowledged in ``qa/typed/review.json``'s ``coverage_ack`` for
     the manifest's source bytes: unknown unknowns may not ship silently. An
     unacknowledged key fails with its name. Since the 2026-11 style-exemplar
     extension every served style criterion has a train positive, so the rule
     passes vacuously today — it stays as the guard for a future gold edit that
     uncovers a criterion again.
  R8 advisory — the category (choice) gold the heads also train on: prints
     whether ``review.json`` certifies the ``choice.jsonl`` bytes, never
     blocking. Category has no served CLM consumer (the seam is Tacit), so an
     unreviewed category set is a visible caveat on the shared weights, not a
     ship-blocker — served damage from it is already bounded by R1/R2/parity.

On success it prints the ``clm-eval-gate: PASS ...`` line the runbook greps for and
writes ``--report`` (default: beside the summary) so the cached-head marker can pin
the gate outcome to the head bytes.

Run it standalone (stdlib only)::

    python scripts/decision_models/clm_eval_gate.py \\
        --summary /clm/heads-trained/finetune_summary.json \\
        --manifest /clm/data/typed-choice/manifest.json \\
        --questions architecture_style --advisory-questions category
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: ``manifest.json`` layout this gate understands (``export_clm_dataset.py``).
SUPPORTED_MANIFEST_SCHEMA = 2

#: Report filename written beside the summary when ``--report`` is not given.
REPORT_NAME = "clm-eval-gate.json"


@dataclass
class GateResult:
    """One gate run: the failure list, the printed lines, and the JSON report."""

    failures: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def fail(self, rule: str, detail: str) -> None:
        self.failures.append(f"[{rule}] {detail}")
        self.lines.append(f"FAIL [{rule}] {detail}")
        self.checks.append({"rule": rule, "status": "fail", "detail": detail})

    def ok(self, rule: str, detail: str) -> None:
        self.lines.append(f"PASS [{rule}] {detail}")
        self.checks.append({"rule": rule, "status": "pass", "detail": detail})

    def note(self, rule: str, detail: str) -> None:
        self.lines.append(f"INFO [{rule}] {detail}")
        self.checks.append({"rule": rule, "status": "info", "detail": detail})

    def skip(self, rule: str, reason: str) -> None:
        self.lines.append(f"SKIP [{rule}] {reason}")
        self.skipped.append({"rule": rule, "reason": reason})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the image build."""
    parser = argparse.ArgumentParser(description="Fail the CLM build when the fine-tune regressed the head.")
    parser.add_argument("--summary", required=True, help="finetune.py --out-dir/finetune_summary.json")
    parser.add_argument(
        "--manifest",
        default="/clm/data/typed-choice/manifest.json",
        help="export_clm_dataset.py manifest (per-question counts + gold histograms per split)",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=["architecture_style"],
        help="Served question ids: per-question non-regression and prior rules block the build.",
    )
    parser.add_argument(
        "--advisory-questions",
        nargs="*",
        default=["category"],
        help="Question ids reported but never blocking (trained heads are shared with /v1/rank).",
    )
    parser.add_argument(
        "--choice",
        default="/build/qa/typed/choice.jsonl",
        help="qa/typed/choice.jsonl as exported (R8 certifies these exact bytes).",
    )
    parser.add_argument(
        "--style",
        default="/build/qa/typed/style.jsonl",
        help="qa/typed/style.jsonl as exported (R7 acks are recorded against these bytes).",
    )
    parser.add_argument(
        "--review",
        default="/build/qa/typed/review.json",
        help="Human review ledger (coverage_ack for R7, choice entries for R8).",
    )
    parser.add_argument(
        "--report",
        default="",
        help=f"JSON report path (default: {REPORT_NAME} beside --summary).",
    )
    return parser.parse_args(argv)


def load_summary(path: Path) -> dict[str, Any]:
    """Read the trainer's summary; a missing file is a gate failure, not a skip."""
    if not path.is_file():
        raise SystemExit(f"clm-eval-gate: FAIL: {path} does not exist — the fine-tune wrote no summary")
    try:
        summary: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SystemExit(f"clm-eval-gate: FAIL: {path} is not valid JSON: {exc}") from exc
    return summary


def load_manifest(path: Path) -> dict[str, Any] | None:
    """Read the export manifest; ``None`` when absent or of an unsupported layout."""
    if not path.is_file():
        return None
    try:
        manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if manifest.get("schema_version") != SUPPORTED_MANIFEST_SCHEMA:
        return None
    splits = manifest.get("splits")
    return manifest if isinstance(splits, dict) else None


def _block(manifest: dict[str, Any], split: str) -> dict[str, Any]:
    """The manifest's block for one shard split (empty when the key is absent)."""
    block: dict[str, Any] = (manifest.get("splits") or {}).get(split) or {}
    return block


def _test_block(manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest's test-split block (empty when the key is absent)."""
    return _block(manifest, "test")


def test_rows(manifest: dict[str, Any], question_id: str) -> int | None:
    """Rows the test shard carries for one question id (the accuracy denominator)."""
    counts: dict[str, Any] = _test_block(manifest).get("per_question") or {}
    count = counts.get(question_id)
    return count if isinstance(count, int) and count > 0 else None


def majority_share(manifest: dict[str, Any], question_id: str) -> tuple[float, int] | None:
    """``(share of the most frequent gold label, distinct label count)`` on the test shard."""
    coverage: dict[str, Any] = (_test_block(manifest).get("label_coverage") or {}).get(question_id) or {}
    labels: dict[str, Any] = coverage.get("labels") or {}
    counts = [count for count in labels.values() if isinstance(count, int)]
    total = sum(counts)
    if not counts or total <= 0:
        return None
    return max(counts) / total, len(counts)


def _accuracy(block: dict[str, Any], question_id: str) -> float | None:
    per_question: dict[str, Any] = block.get("per_question") or {}
    value = per_question.get(question_id)
    return float(value) if isinstance(value, (int, float)) else None


def _check_manifest(
    result: GateResult,
    manifest: dict[str, Any] | None,
    question_ids: Sequence[str],
) -> None:
    """R0: the manifest must exist and describe every question this gate reasons about."""
    if manifest is None:
        result.fail(
            "R0-manifest",
            "no usable export manifest (missing file or schema_version != "
            f"{SUPPORTED_MANIFEST_SCHEMA}) — per-question granularity and the prior baseline are "
            "unavailable, so R2/R5 cannot run",
        )
        return
    missing = [question_id for question_id in question_ids if test_rows(manifest, question_id) is None]
    if missing:
        result.fail("R0-manifest", f"manifest test shard carries no rows for question ids: {missing}")
        return
    counts = {question_id: test_rows(manifest, question_id) for question_id in question_ids}
    result.ok("R0-manifest", f"test shard rows per question: {counts}")


def _check_overall(result: GateResult, test: dict[str, Any], init_test: dict[str, Any]) -> None:
    """R1: the trained head is no worse than the released checkpoint over the whole shard."""
    test_acc = test.get("acc")
    init_acc = init_test.get("acc")
    if not isinstance(test_acc, (int, float)) or not isinstance(init_acc, (int, float)):
        result.fail("R1-accuracy", f"summary carries no test/init_test accuracy pair (test={test!r})")
        return
    if float(test_acc) < float(init_acc):
        result.fail("R1-accuracy", f"test acc {test_acc:.4f} regressed below init {init_acc:.4f}")
        return
    result.ok("R1-accuracy", f"test acc {test_acc:.4f} >= init {init_acc:.4f}")


def _check_fresh_head(result: GateResult, summary: dict[str, Any]) -> None:
    """R4: ``best_head.pt`` must not be the epoch-0 released checkpoint."""
    best_epoch = summary.get("best_epoch")
    if best_epoch == 0:
        result.fail(
            "R4-fresh-head",
            "best_epoch is 0: no val improvement — best_head.pt is the untrained released weights",
        )
    elif not isinstance(best_epoch, int):
        result.fail("R4-fresh-head", f"summary carries no integer best_epoch (got {best_epoch!r})")
    else:
        result.ok("R4-fresh-head", f"best epoch {best_epoch} (val-improved, not the released init)")


def _check_served(
    result: GateResult,
    test: dict[str, Any],
    init_test: dict[str, Any],
    manifest: dict[str, Any] | None,
    question_id: str,
) -> None:
    """R2 + R5 for one served question: one-row tolerance, then the label-prior floor."""
    trained = _accuracy(test, question_id)
    baseline = _accuracy(init_test, question_id)
    if trained is None:
        result.fail("R2-served", f"{question_id!r}: test carries no per-question accuracy")
        return
    if baseline is None:
        # Fail closed: a missing baseline is how a regression rule silently becomes a no-op.
        result.fail("R2-served", f"{question_id!r}: init_test carries no per-question baseline")
        return
    rows = test_rows(manifest, question_id) if manifest is not None else None
    tolerance = 1.0 / rows if rows else 0.0
    if trained < baseline - tolerance:
        result.fail(
            "R2-served",
            f"{question_id!r}: test {trained:.4f} below init {baseline:.4f} by more than one row "
            f"(1/{rows} = {tolerance:.4f})",
        )
    else:
        result.ok(
            "R2-served",
            f"{question_id!r}: test {trained:.4f} vs init {baseline:.4f} (tolerance 1/{rows} = {tolerance:.4f})",
        )
    if manifest is None:
        return
    prior = majority_share(manifest, question_id)
    if prior is None:
        result.fail("R5-prior", f"{question_id!r}: manifest carries no gold histogram for the test shard")
        return
    share, distinct = prior
    if distinct < 2:
        result.skip(
            "R5-prior", f"{question_id!r}: single gold label on the test shard (majority {share:.2f}) — vacuous"
        )
    elif trained < share:
        result.fail(
            "R5-prior",
            f"{question_id!r}: test {trained:.4f} below the majority-label share {share:.4f} — the head did "
            "not learn more than the prior",
        )
    else:
        result.ok("R5-prior", f"{question_id!r}: test {trained:.4f} >= majority share {share:.4f}")


def _check_support(result: GateResult, manifest: dict[str, Any], question_id: str) -> None:
    """R6: every test gold label for a served question has a train-shard positive."""
    train_cov: dict[str, Any] = (_block(manifest, "train").get("label_coverage") or {}).get(question_id) or {}
    test_cov: dict[str, Any] = (_test_block(manifest).get("label_coverage") or {}).get(question_id) or {}
    train_labels = set(train_cov.get("labels", {}))
    test_labels = set(test_cov.get("labels", {}))
    unsupported = sorted(test_labels - train_labels)
    if unsupported:
        result.fail(
            "R6-support",
            f"{question_id!r}: test labels {unsupported} have no train-shard positive — "
            "extend the gold or narrow the question before shipping this head",
        )
    else:
        result.ok("R6-support", f"{question_id!r}: all {len(test_labels)} test labels have train positives")


def _check_coverage(
    result: GateResult,
    manifest: dict[str, Any],
    ledger: dict[str, Any] | None,
    source_digests: dict[str, str | None] | None,
    question_id: str,
) -> None:
    """R7: train-uncovered criteria for a served question are explicitly acknowledged."""
    train_cov: dict[str, Any] = (_block(manifest, "train").get("label_coverage") or {}).get(question_id) or {}
    uncovered = train_cov.get("uncovered", [])
    if not uncovered:
        result.ok("R7-coverage", f"{question_id!r}: every criterion has a train positive")
        return
    style_digest = (source_digests or {}).get("style")
    acks = _coverage_acks(ledger, "architecture_style", style_digest)
    missing = sorted(set(uncovered) - acks)
    if missing:
        preview = ", ".join(missing[:5])
        stamp = style_digest[:12] if style_digest else "current"
        result.fail(
            "R7-coverage",
            f"{question_id!r}: {len(missing)} train-uncovered criteria lack review.json coverage_ack "
            f"(e.g. {preview}) — acknowledge the {stamp} bytes or extend the gold",
        )
    else:
        stamp = style_digest[:12] if style_digest else "unknown"
        result.ok(
            "R7-coverage",
            f"{question_id!r}: all {len(uncovered)} train-uncovered criteria acknowledged at {stamp}",
        )


def _coverage_acks(ledger: dict[str, Any] | None, question_id: str, digest: str | None) -> set[str]:
    """Acknowledged criterion keys for one question at one source digest."""
    if ledger is None or digest is None:
        return set()
    entries: Any = (ledger.get("coverage_ack") or {}).get(question_id) or []
    if not isinstance(entries, list):
        return set()
    return {
        str(entry.get("criterion"))
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("source_sha256") == digest
        and entry.get("reviewer")
        and entry.get("note")
    }


def _note_choice_review(
    result: GateResult,
    ledger: dict[str, Any] | None,
    source_digests: dict[str, str | None] | None,
) -> None:
    """R8: report whether the category gold is human-reviewed; never blocking."""
    digest = (source_digests or {}).get("choice")
    if ledger is None or digest is None:
        result.note("R8-choice-review", "no usable ledger or choice digest — category training stays unreviewed")
        return
    entries: Any = (ledger.get("reviews") or {}).get("choice") or []
    if not isinstance(entries, list):
        entries = []
    covered = sum(
        1
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("source_sha256") == digest
        and entry.get("reviewer")
        and entry.get("label_ok") is True
    )
    if covered >= 100:
        result.note("R8-choice-review", f"category gold reviewed: {covered} choice rows certified at {digest[:12]}")
    else:
        result.note(
            "R8-choice-review",
            f"category training stays unreviewed: {covered}/100 choice rows certified at {digest[:12]} "
            "(advisory only — no served CLM consumer asks the category question)",
        )


def _note_advisory(
    result: GateResult,
    test: dict[str, Any],
    init_test: dict[str, Any],
    question_id: str,
) -> None:
    """R3: report an unserved question's movement without gating on it."""
    trained = _accuracy(test, question_id)
    baseline = _accuracy(init_test, question_id)
    if trained is None or baseline is None:
        result.note("R3-advisory", f"{question_id!r}: no comparable pair (test={trained}, init={baseline})")
        return
    result.note(
        "R3-advisory",
        f"{question_id!r}: test {trained:.4f} vs init {baseline:.4f} — reported only, no CLM client asks it",
    )


def _load_ledger(path: Path) -> dict[str, Any] | None:
    """Read review.json; None when absent or invalid (R7 fails closed on None)."""
    if not path.is_file():
        return None
    try:
        ledger: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return ledger if isinstance(ledger, dict) else None


def _sha256_of(path: Path) -> str | None:
    """Hex digest of a qa/ input, or None when it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def check(
    summary: dict[str, Any],
    manifest: dict[str, Any] | None,
    questions: Sequence[str],
    advisory: Sequence[str] = (),
    ledger: dict[str, Any] | None = None,
    source_digests: dict[str, str | None] | None = None,
) -> GateResult:
    """Evaluate every rule; the result carries failures, printed lines and the report.

    ``ledger`` is the parsed ``review.json`` (or None when it cannot be read —
    R7 then fails closed, R8 reports the gap); ``source_digests`` maps
    ``{"choice": sha256, "style": sha256}`` of the qa/ inputs the export read.
    """
    result = GateResult()
    test: dict[str, Any] = summary.get("test") or {}
    init_test: dict[str, Any] = summary.get("init_test") or {}
    _check_manifest(result, manifest, [*questions, *advisory])
    _check_overall(result, test, init_test)
    _check_fresh_head(result, summary)
    for question_id in questions:
        _check_served(result, test, init_test, manifest, question_id)
    if manifest is not None:
        for question_id in questions:
            _check_support(result, manifest, question_id)
            _check_coverage(result, manifest, ledger, source_digests, question_id)
    for question_id in advisory:
        _note_advisory(result, test, init_test, question_id)
    _note_choice_review(result, ledger, source_digests)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Load both inputs, apply the gate, print PASS/FAIL, write the report."""
    args = parse_args(argv)
    summary_path = Path(args.summary)
    summary = load_summary(summary_path)
    manifest = load_manifest(Path(args.manifest))
    questions = list(args.questions)
    advisory = list(args.advisory_questions)
    ledger = _load_ledger(Path(args.review))
    source_digests: dict[str, str | None] = {
        "choice": _sha256_of(Path(args.choice)),
        "style": _sha256_of(Path(args.style)),
    }
    result = check(summary, manifest, questions, advisory, ledger, source_digests)
    for line in result.lines:
        print(f"clm-eval-gate: {line}")
    report_path = Path(args.report) if args.report else summary_path.with_name(REPORT_NAME)
    report = {
        "gate": "clm-eval-gate",
        "status": "pass" if result.passed else "fail",
        "manifest": args.manifest,
        "questions": questions,
        "advisory_questions": advisory,
        "checks": result.checks,
        "skipped": result.skipped,
        "failures": result.failures,
        "test_acc": (summary.get("test") or {}).get("acc"),
        "init_test_acc": (summary.get("init_test") or {}).get("acc"),
        "best_epoch": summary.get("best_epoch"),
        "train_questions": summary.get("train_questions"),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not result.passed:
        for failure in result.failures:
            print(f"clm-eval-gate: FAIL: {failure}", file=sys.stderr)
        return 1
    test: dict[str, Any] = summary.get("test") or {}
    init_test: dict[str, Any] = summary.get("init_test") or {}
    print(
        "clm-eval-gate: PASS: test acc {:.4f} vs init {:.4f} (best epoch {}; per_question {}; report {})".format(
            float(test.get("acc", 0.0)),
            float(init_test.get("acc", 0.0)),
            summary.get("best_epoch", "?"),
            json.dumps(test.get("per_question", {}), sort_keys=True),
            report_path,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
