#!/usr/bin/env python
"""The review side of the §4.7.1 style-label gap: blind packs, agreement, certification.

The workflow this implements is the one ``qa/typed/style-rubric.md`` states:

1. ``pack`` — cut the validated candidate pool into blind annotation batches. A pack carries
   the scenario state and nothing else: no current label, no rationale, no author's intent.
2. Two annotators label each pack independently to ``..a.jsonl`` / ``..b.jsonl``.
3. ``score`` — raw agreement and Cohen's kappa for the batch. A batch is *accepted* at raw
   agreement ``>= 0.80``; every disagreement must be adjudicated (``--adjudication``) with a
   written reason before the batch can be accepted, and a rejected batch is re-annotated.
   Accepted rows are written with their final label so ``style_label_gap.py apply`` can use
   them.
4. ``certify`` — rewrite ``qa/typed/review.json`` so it covers every label in the label source
   at that source's current sha256, carrying forward earlier verdicts only when the label they
   covered is unchanged.

Nothing here decides a label: annotator verdicts and adjudications are *inputs*. That keeps
the review evidence auditable — the ledger records who said what, so a label can be traced to
two independent readings plus the adjudicator's reasoning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style_label_gap as gap  # noqa: E402  (same directory: the shared pool/label helpers)

#: Batch acceptance rule (``qa/typed/style-rubric.md`` §Review protocol).
RAW_AGREEMENT_FLOOR = 0.80
#: kappa below this triggers a rubric review even when raw agreement passes.
KAPPA_FLOOR = 0.67
#: Default batch size: small enough that a rejection costs one re-read, not a day.
BATCH_SIZE = 75
DEFAULT_REVIEW = "qa/typed/review.json"
DEFAULT_RUBRIC = "qa/typed/style-rubric.md"
DEFAULT_LABELS = "scripts/decision_models/derive_style_labels.py"


# ---------------------------------------------------------------------------------------
# Packs
# ---------------------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows (blank lines ignored)."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write JSONL; return the number of rows written."""
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in materialized) + "\n", encoding="utf-8"
    )
    return len(materialized)


def blind_state(row: Mapping[str, Any]) -> tuple[str, str]:
    """``(scenario_id, state)`` of a pool row or an existing ``style.jsonl`` row."""
    if "state" in row:
        return str(row["id"]).removesuffix(":style"), str(row["state"])
    return str(row["scenario_id"]), str(row["description"])


def blind_token(scenario_id: str) -> str:
    """An opaque, stable token for one row (the id itself would name the style).

    Scenario ids are ``<style>-<topic>``, so a pack that carried them would hand the annotator the
    answer. The token is a hash prefix: stable across runs, orderable by nothing meaningful, and
    mapped back by :func:`build_packs` through the sidecar it writes next to the pack.
    """
    return "s" + hashlib.sha256(scenario_id.encode("utf-8")).hexdigest()[:12]


def build_packs(
    rows: Sequence[Mapping[str, Any]], out_dir: Path, *, batch_size: int = BATCH_SIZE, prefix: str = "batch"
) -> list[Path]:
    """Cut rows into deterministic blind packs plus an id map, sorted by opaque token.

    Sorting by token rather than by id mixes the styles across the pack instead of grouping them,
    so a batch's label distribution cannot be read off the file order either.
    """
    keyed = [(blind_token(*blind_state(row)[:1]), row) for row in rows]
    ordered = sorted(keyed, key=lambda item: item[0])
    packs: list[Path] = []
    for index in range(0, len(ordered), batch_size):
        chunk = ordered[index : index + batch_size]
        path = out_dir / f"{prefix}-{index // batch_size:02d}.blind.jsonl"
        _write_jsonl(
            path, ({"scenario_id": token, "state": blind_state(row)[1]} for token, row in chunk)
        )
        _write_jsonl(
            path.with_name(f"{path.name.removesuffix('.blind.jsonl')}.ids.jsonl"),
            ({"scenario_id": token, "real_id": blind_state(row)[0]} for token, row in chunk),
        )
        packs.append(path)
    return packs


def write_protocol(out_dir: Path, rubric: Path, label_names: Sequence[str]) -> Path:
    """Write the annotator instructions that sit next to the packs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "PROTOCOL.md"
    display = rubric
    vocabulary = "\n".join(f"- `{name}`" for name in label_names)
    path.write_text(
        "\n".join(
            [
                "# Blind annotation protocol",
                "",
                f"Read `{display}` first: it is the judgment, the decision",
                "procedure, the near-miss pairs and the served glosses. This file only states the",
                "mechanics of one batch.",
                "",
                "## What you do",
                "",
                "For every row of your `*.blind.jsonl` pack, write one JSON object per line to your",
                "own output file (`<pack>.a.jsonl` for annotator A, `<pack>.b.jsonl` for annotator B):",
                "",
                "    {\"scenario_id\": \"<id>\", \"label\": \"<one label from the vocabulary>\"}",
                "",
                "Optional extra keys (not scored, read by the adjudicator):",
                "`\"alternative\": \"<the other label you considered>\"`, `\"note\": \"<one sentence>\"`.",
                "",
                "## Rules that make the label evidence",
                "",
                "- Exactly one label per row, chosen by the rubric's decision procedure.",
                "- Do not read another annotator's output, the label source",
                "  (`scripts/decision_models/derive_style_labels.py`), `qa/typed/style.jsonl`, the",
                "  review ledger, or git history for these rows. An independent second reading is the",
                "  point of the batch; agreement with the intended label is not.",
                "- A row whose state is too thin to support any label still gets the best label the",
                "  rubric supports, plus a note — never a blank.",
                "- Output every row of the pack, in any order.",
                "",
                "## Label vocabulary",
                "",
                vocabulary,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------------------


def cohen_kappa(left: Mapping[str, str], right: Mapping[str, str]) -> float:
    """Unweighted Cohen's kappa over the shared scenario ids (1.0 = perfect, 0 = chance)."""
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError("no shared scenario ids to score")
    total = len(shared)
    observed = sum(1 for sid in shared if left[sid] == right[sid]) / total
    left_counts = Counter(left[sid] for sid in shared)
    right_counts = Counter(right[sid] for sid in shared)
    expected: float = sum(
        left_counts[label] * right_counts[label] for label in set(left_counts) | set(right_counts)
    ) / (total * total)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _label_map(path: Path) -> dict[str, str]:
    """``scenario_id -> label`` from an annotator output file."""
    rows = _read_jsonl(path)
    labels: dict[str, str] = {}
    for row in rows:
        scenario_id = str(row["scenario_id"])
        label = str(row["label"]).strip()
        if not label:
            raise SystemExit(f"{path}: row {scenario_id!r} carries no label")
        labels[scenario_id] = label
    return labels


def id_map_for(pack: Path) -> dict[str, str]:
    """``token -> real scenario id`` from the sidecar written beside a pack (empty when absent)."""
    sidecar = pack.with_name(f"{pack.name.removesuffix('.blind.jsonl')}.ids.jsonl")
    if not sidecar.is_file():
        return {}
    return {str(row["scenario_id"]): str(row["real_id"]) for row in _read_jsonl(sidecar)}


def _translate(labels: Mapping[str, str], mapping: Mapping[str, str]) -> dict[str, str]:
    """Relabel a ``token -> label`` map onto real scenario ids (identity when there is no sidecar)."""
    return {mapping.get(scenario_id, scenario_id): label for scenario_id, label in labels.items()}


def score_batch(
    pool: Sequence[Mapping[str, Any]],
    pack: Path,
    annotator_a: Path,
    annotator_b: Path,
    *,
    adjudication: Path | None = None,
    gold: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score one batch and return ``(ledger, accepted rows)``.

    The ledger is the batch's evidence (raw agreement, kappa, every disagreement with its
    adjudicated label and reason); the accepted rows are the pool rows carrying their final
    label and the rationale the label source will record.
    """
    mapping = id_map_for(pack)
    pack_ids = [str(row["scenario_id"]) for row in _read_jsonl(pack)]
    labels_a = _label_map(annotator_a)
    labels_b = _label_map(annotator_b)
    missing_a = [sid for sid in pack_ids if sid not in labels_a]
    missing_b = [sid for sid in pack_ids if sid not in labels_b]
    if missing_a or missing_b:
        raise SystemExit(
            f"{pack.name}: incomplete annotation (A missing {len(missing_a)}, B missing {len(missing_b)})"
        )
    agreed = [sid for sid in pack_ids if labels_a[sid] == labels_b[sid]]
    disputed = [sid for sid in pack_ids if labels_a[sid] != labels_b[sid]]
    rulings: dict[str, dict[str, str]] = {}
    if adjudication is not None:
        rulings = {
            str(row["scenario_id"]): {key: str(value) for key, value in row.items()}
            for row in _read_jsonl(adjudication)
        }
    unresolved = [sid for sid in disputed if sid not in rulings]
    raw = len(agreed) / len(pack_ids) if pack_ids else 0.0
    kappa = cohen_kappa(labels_a, labels_b)
    # Gold labels are keyed by real scenario id; the pack speaks tokens, so translate before scoring.
    gold_labels = (
        {
            token: _gold_labels(gold)[mapping[token]]
            for token in pack_ids
            if mapping.get(token) in _gold_labels(gold)
        }
        if gold is not None and mapping
        else (_gold_labels(gold) if gold is not None else {})
    )
    ledger: dict[str, Any] = {
        "batch": pack.stem.removesuffix(".blind"),
        "rows": len(pack_ids),
        "raw_agreement": round(raw, 4),
        "kappa": round(kappa, 4),
        "raw_floor": RAW_AGREEMENT_FLOOR,
        "kappa_floor": KAPPA_FLOOR,
        "accepted": raw >= RAW_AGREEMENT_FLOOR and not unresolved,
        "agreed": len(agreed),
        "disputed": len(disputed),
        "unresolved": unresolved,
        "disagreements": [
            {
                "row": sid,
                "scenario_id": mapping.get(sid, sid),
                "a": labels_a[sid],
                "b": labels_b[sid],
                "final": rulings.get(sid, {}).get("label"),
                "reason": rulings.get(sid, {}).get("reason"),
            }
            for sid in disputed
        ],
    }
    if gold_labels:
        ledger["gold"] = {
            "rows": sum(1 for sid in pack_ids if sid in gold_labels),
            "a_accuracy": round(_accuracy(labels_a, gold_labels, pack_ids), 4),
            "b_accuracy": round(_accuracy(labels_b, gold_labels, pack_ids), 4),
        }
    if not ledger["accepted"]:
        return ledger, []
    by_id = {str(row["scenario_id"]): row for row in pool}
    accepted: list[dict[str, Any]] = []
    for token in pack_ids:
        sid = mapping.get(token, token)
        final = labels_a[token] if token in agreed else rulings[token].get("label", "")
        if not final:
            raise SystemExit(f"{token!r}: adjudication carries no label")
        row = by_id.get(sid)
        if row is None:
            raise SystemExit(f"{sid!r} is not in the pool")
        intended = str(row.get("_pool", {}).get("style", ""))
        reason = rulings.get(token, {}).get("reason")
        if final != intended:
            if token in disputed:
                if not reason:
                    raise SystemExit(
                        f"{sid!r}: adjudicated label {final!r} differs from the brief; a reason is required"
                    )
                rationale = reason
            else:
                # Both annotators agreed on a label the authoring brief did not plan. The label is
                # the review evidence, so it wins; the ledger states how it arose.
                rationale = reason or (
                    f"Both blind annotators labelled this {final} where the authoring intent was {intended}"
                )
        else:
            rationale = str(row["_pool"]["rationale"])
        accepted.append(
            {
                **row,
                "_pool": {
                    **row["_pool"],
                    "style": final,
                    "rationale": rationale,
                    # The adjudicator's reason is evidence in its own right: it explains why two
                    # blind readers differed, and it must reach the ledger even when the final
                    # label happens to equal the authoring intent (where the rationale above is
                    # the author's text, not the adjudicator's).
                    "adjudication_reason": reason if token in disputed else None,
                    "batch": ledger["batch"],
                    "annotator_a": labels_a[token],
                    "annotator_b": labels_b[token],
                },
            }
        )
    return ledger, accepted


def _gold_labels(path: Path) -> dict[str, str]:
    """``scenario_id -> reviewed label`` from an existing ``style.jsonl`` (calibration only)."""
    return {
        str(row["id"]).removesuffix(":style"): str(row["label"])
        for row in _read_jsonl(path)
    }


def _accuracy(labels: Mapping[str, str], gold: Mapping[str, str], ids: Sequence[str]) -> float:
    """Annotator accuracy on the rows that carry a gold label."""
    scored = [sid for sid in ids if sid in gold]
    if not scored:
        return 0.0
    return sum(1 for sid in scored if labels.get(sid) == gold[sid]) / len(scored)


# ---------------------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------------------


def certify(
    *,
    labels_path: Path,
    review_path: Path,
    rubric: Path,
    accepted_files: Sequence[Path],
    carry_forward: Path | None,
    method: str,
    reviewer: str,
    human_review: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Rewrite the review ledger so it certifies every label at the source's current bytes."""
    digest = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    labels = gap.read_style_labels(labels_path)
    carried, carried_reviewers, dropped = _carry_forward(carry_forward, labels, digest)
    fresh = _fresh_entries(accepted_files, labels, digest, reviewer)
    covered = {entry["scenario_id"] for entry in (*carried, *fresh)}
    missing = sorted(set(labels) - covered)
    if missing:
        raise SystemExit(f"certification incomplete: {len(missing)} labels unreviewed (e.g. {missing[:3]})")
    batches = sorted({str(entry.get("batch")) for entry in fresh if entry.get("batch")})
    ledger = {
        "_schema": LEDGER_SCHEMA,
        "status": "reviewed",
        "method": method,
        "reviewers": list(dict.fromkeys([*carried_reviewers, reviewer])),
        "rubric_sha256": hashlib.sha256(rubric.read_bytes()).hexdigest(),
        "labels_source": str(labels_path.relative_to(REPO_ROOT)) if labels_path.is_relative_to(REPO_ROOT) else str(labels_path),
        "labels_sha256": digest,
        "batches": batches,
        "coverage_ack": {"architecture_style": []},
        "dropped_carry_forward": dropped,
        "human_review": dict(human_review) if human_review else {"status": "pending"},
        "reviews": {"style": [*carried, *fresh], "choice": []},
    }
    review_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    return ledger


LEDGER_SCHEMA = (
    "qa/typed/review.json v2: certification of decision-model gold labels. status=reviewed "
    "requires EVERY scenario in the label source covered by a style entry whose labels_sha256 "
    "matches the label source bytes, with reviewer named and rationale_ok true. Each entry "
    "records the batch and, for rows two blind annotators labelled, both verdicts; a disputed "
    "row carries its adjudicated label and the adjudicator's reason. `rubric_sha256` pins the "
    "annotation rubric the verdicts were made under. The reviewer field records WHO reviewed (a "
    "human expert, or a model review pass with its identity stated); human expert sign-off, when "
    "performed, is listed in `human_review`. style_head_calibrate.py refuses to recommend a floor "
    "until this ledger certifies the label source."
)


def _carry_forward(
    path: Path | None, labels: Mapping[str, tuple[str, str]], digest: str
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Earlier verdicts, re-stamped at ``digest``, but only for labels that did not change.

    A carried entry is a claim about the *label*, not about the file bytes: the 2026-11 audit
    read 169 descriptions and recorded what they should be labelled. If a label changed since,
    the old verdict no longer covers it and the row is dropped (and must be re-reviewed).
    """
    if path is None or not path.is_file():
        return [], [], []
    ledger: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    raw_entries = (ledger.get("reviews") or {}).get("style") or []
    entries: list[Any] = list(raw_entries)
    carried: list[dict[str, Any]] = []
    reviewers: list[str] = []
    dropped: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        scenario_id = str(entry.get("scenario_id"))
        current = labels.get(scenario_id)
        if current is None:
            dropped.append(scenario_id)
            continue
        if current[0] != entry.get("label"):
            dropped.append(scenario_id)
            continue
        carried.append({**entry, "labels_sha256": digest})
        if entry.get("reviewer"):
            reviewers.append(str(entry["reviewer"]))
    return carried, reviewers, dropped


def _fresh_entries(
    accepted_files: Sequence[Path], labels: Mapping[str, tuple[str, str]], digest: str, reviewer: str
) -> list[dict[str, Any]]:
    """One ledger entry per accepted row: label, both blind verdicts, adjudication when disputed."""
    entries: list[dict[str, Any]] = []
    for path in accepted_files:
        for row in _read_jsonl(path):
            scenario_id = str(row["scenario_id"])
            pool: dict[str, Any] = row.get("_pool") or {}
            current = labels.get(scenario_id)
            if current is None:
                raise SystemExit(f"{scenario_id!r} is not in the label source")
            if current[0] != pool.get("style"):
                raise SystemExit(
                    f"{scenario_id!r}: label source says {current[0]!r}, accepted batch says {pool.get('style')!r}"
                )
            entry: dict[str, Any] = {
                "scenario_id": scenario_id,
                "label": current[0],
                "labels_sha256": digest,
                "reviewer": reviewer,
                "rationale_ok": True,
                "batch": pool.get("batch"),
                "annotator_a": pool.get("annotator_a"),
                "annotator_b": pool.get("annotator_b"),
            }
            if pool.get("annotator_a") != pool.get("annotator_b"):
                entry["adjudication"] = pool.get("adjudication_reason") or pool.get("rationale")
            entries.append(entry)
    return entries


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def cmd_pack(args: argparse.Namespace) -> int:
    """Write blind packs (and the protocol sheet) for a pool or an existing label file."""
    if args.pool:
        rows = _read_jsonl(Path(args.pool))
        if args.split:
            rows = [row for row in rows if str(row.get("split")) == args.split]
    else:
        rows = [
            row
            for row in _read_jsonl(Path(args.from_style))
            if not args.split or str(row.get("source", {}).get("split")) == args.split
        ]
    out_dir = Path(args.out)
    packs = build_packs(rows, out_dir, batch_size=args.batch_size, prefix=args.prefix)
    protocol = write_protocol(out_dir, Path(args.rubric), gap.POOL_LABELS)
    print(
        json.dumps(
            {"rows": len(rows), "packs": [str(path) for path in packs], "protocol": str(protocol)}, indent=2
        )
    )
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score one batch, write its ledger and (when accepted) its accepted rows."""
    pool = _read_jsonl(Path(args.pool))
    ledger, accepted = score_batch(
        pool,
        Path(args.pack),
        Path(args.a),
        Path(args.b),
        adjudication=Path(args.adjudication) if args.adjudication else None,
        gold=Path(args.gold) if args.gold else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    if accepted:
        _write_jsonl(out.with_name(f"{out.stem}.accepted.jsonl"), accepted)
    print(json.dumps({key: ledger[key] for key in ("batch", "rows", "raw_agreement", "kappa", "accepted")}, indent=2))
    if ledger["unresolved"]:
        print(f"unresolved disagreements: {len(ledger['unresolved'])} (adjudicate before accepting)", file=sys.stderr)
        return 2
    return 0 if ledger["accepted"] else 1


def cmd_certify(args: argparse.Namespace) -> int:
    """Certify the review ledger against the label source's current bytes."""
    accepted: list[Path] = []
    for pattern in args.accepted:
        accepted.extend(sorted(Path().glob(pattern)) if any(char in pattern for char in "*?[") else [Path(pattern)])
    human = json.loads(Path(args.human_review).read_text(encoding="utf-8")) if args.human_review else None
    ledger = certify(
        labels_path=Path(args.labels),
        review_path=Path(args.review),
        rubric=Path(args.rubric),
        accepted_files=accepted,
        carry_forward=Path(args.carry_forward) if args.carry_forward else None,
        method=args.method,
        reviewer=args.reviewer,
        human_review=human,
    )
    print(
        json.dumps(
            {
                "status": ledger["status"],
                "labels_sha256": ledger["labels_sha256"][:12],
                "entries": len(ledger["reviews"]["style"]),
                "carried_forward_dropped": len(ledger["dropped_carry_forward"]),
                "batches": ledger["batches"],
                "human_review": ledger["human_review"].get("status"),
            },
            indent=2,
        )
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the three review steps."""
    parser = argparse.ArgumentParser(description="Style-label review: blind packs, agreement, certification.")
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("pack", help="cut blind annotation packs")
    source = pack.add_mutually_exclusive_group(required=True)
    source.add_argument("--pool", help="validated pool JSONL")
    source.add_argument("--from-style", help="existing qa/typed/style.jsonl (calibration packs)")
    pack.add_argument("--split", choices=("train", "holdout"), default=None)
    pack.add_argument("--out", default="data/style-gap/annotations")
    pack.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    pack.add_argument("--prefix", default="batch")
    pack.add_argument("--rubric", default=DEFAULT_RUBRIC)
    pack.set_defaults(func=cmd_pack)

    score = sub.add_parser("score", help="score one batch")
    score.add_argument("--pool", required=True)
    score.add_argument("--pack", required=True)
    score.add_argument("--a", required=True)
    score.add_argument("--b", required=True)
    score.add_argument("--adjudication", default=None)
    score.add_argument("--gold", default=None, help="style.jsonl holding the reviewed labels (calibration)")
    score.add_argument("--out", required=True)
    score.set_defaults(func=cmd_score)

    cert = sub.add_parser("certify", help="rewrite qa/typed/review.json")
    cert.add_argument("--labels", default=DEFAULT_LABELS)
    cert.add_argument("--review", default=DEFAULT_REVIEW)
    cert.add_argument("--rubric", default=DEFAULT_RUBRIC)
    cert.add_argument("--accepted", nargs="*", default=[], help="accepted-row JSONL files or globs")
    cert.add_argument("--carry-forward", default=None)
    cert.add_argument("--method", required=True)
    cert.add_argument("--reviewer", required=True)
    cert.add_argument("--human-review", default=None)
    cert.set_defaults(func=cmd_certify)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the subcommand."""
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
