#!/usr/bin/env python
"""Export this repo's typed labels as a CLM ``typed-decisions`` dataset.

CLM's choice fine-tune (``train/finetune.py --task choice``) reads parquet files laid
out as ``<root>/<workflow>/<split>*.parquet``, where each row is a System One request
plus its gold answer::

    {"id": ..., "state": ..., "questions": {"<qid>": {type, instructions, criteria}},
     "gold": {"<qid>": {"label": ..., "probabilities": {...}}}}

(taken from ``train/adapters.py::typed_decision_examples`` at the pinned revision —
it decodes JSON-encoded columns too, so this exporter writes the two object columns as
JSON strings and keeps the whole file schema-stable).

Two question sets are exported, one row per (scenario, question):

* the category question (``qa/typed/choice.jsonl``, qid ``category``) — the
  decision-taxonomy supervision;
* the style question (``qa/typed/style.jsonl``, qid ``architecture_style``) — the
  exact (instructions, criteria) the Ch2 seam serves, sourced from the shared
  ``src/decisions/seam_prompts.py`` vocabulary, with reviewed per-scenario style
  gold labels. The heads are contrastively trained on the task family "which
  criterion best matches this state", and this set is the family member the
  server actually asks.

Both inputs carry ``state`` as the scenario's raw description bytes — the same
bytes the Ch2 seam sends as ``state`` — so the head sees training states drawn
from the serving distribution.

The test shard is the ``holdout`` rows of both files, not a carve-out of the train rows:
the corpus split is family-level, so a family-disjoint test shard is the only slice that
measures generalisation to unseen families. ``finetune.py`` reads the test shard only
after checkpoint selection (``evaluate("test")`` over the val-selected head), so those
rows never reach the weights and ``init_test`` is scored on the same rows.

The manifest records, per split, the row counts per question id and the gold-label
histogram per question id. The build gate reads it for two things the trainer summary
cannot supply: the granularity of a per-question accuracy (one row = 1/n) and the
majority-class baseline a trained head must beat to prove it learned more than the prior.

Run it standalone (needs pyarrow, nothing else)::

    uv run --with pyarrow python scripts/decision_models/export_clm_dataset.py \\
        --choice qa/typed/choice.jsonl --style qa/typed/style.jsonl --out /clm/data/typed-choice
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("export-clm-dataset")

#: Question id under which every category row asks its question.
CATEGORY_QUESTION_ID = "category"

#: Question id under which every style row asks its question (the served Ch2 qid).
STYLE_QUESTION_ID = "architecture_style"

#: Layout version of ``manifest.json``; the build gate refuses any other value.
MANIFEST_SCHEMA_VERSION = 2

#: The shipped category instructions (identical to the qa/ export's instructions field).
DEFAULT_INSTRUCTIONS = "Which design-pattern category does this decision belong to?"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the image build."""
    parser = argparse.ArgumentParser(description="Export typed labels as a CLM dataset (category + style).")
    parser.add_argument("--choice", default="qa/typed/choice.jsonl", help="qa/typed/choice.jsonl")
    parser.add_argument("--style", default="qa/typed/style.jsonl", help="qa/typed/style.jsonl")
    parser.add_argument("--out", required=True, help="Dataset root written for CLM's --data.")
    parser.add_argument("--workflow", default="all", help="CLM workflow subdirectory name.")
    parser.add_argument(
        "--train-split",
        default="train",
        help="qa/ source split written to the train shard (holdout never decides weights).",
    )
    parser.add_argument(
        "--test-split",
        default="holdout",
        help="qa/ source split written to the test shard — family-disjoint by corpus construction.",
    )
    parser.add_argument(
        "--clm-pin",
        default="d5f9ef0",
        help="CLM git revision whose train/adapters.py typed_decision_examples defines the row layout.",
    )
    return parser.parse_args(argv)


def load_rows(path: Path, split: str) -> list[dict[str, Any]]:
    """Read the qa/ choice rows whose source split matches; refuse an empty result."""
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


def to_typed_row(row: dict[str, Any], question_id: str) -> dict[str, Any]:
    """One qa/ row → the typed-decisions wire row (objects JSON-encoded).

    ``question_id`` is the serving qid for this row's question set (``category``
    or ``architecture_style``); the label must be one of the row's own criteria keys.
    """
    criteria = row["criteria"]
    if not isinstance(criteria, dict) or not criteria:
        raise SystemExit(f"row {row.get('id')!r} has no criteria object")
    label = row["label"]
    if label not in criteria:
        raise SystemExit(f"row {row.get('id')!r}: label {label!r} is not among the criteria keys")
    questions = {
        question_id: {
            "type": "choice",
            "instructions": row.get("instructions") or DEFAULT_INSTRUCTIONS,
            "criteria": criteria,
        }
    }
    gold = {question_id: {"label": label}}
    return {
        "id": str(row["id"]),
        "state": row["state"],
        "questions": json.dumps(questions, ensure_ascii=False),
        "gold": json.dumps(gold, ensure_ascii=False),
    }


def question_id_of(row: dict[str, Any]) -> str:
    """The qid a typed row answers (each row carries exactly one question)."""
    gold = json.loads(str(row["gold"]))
    (question_id,) = gold
    return str(question_id)


def per_question_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Row count per question id — the denominator a per-question accuracy is over."""
    counts: dict[str, int] = {}
    for row in rows:
        question_id = question_id_of(row)
        counts[question_id] = counts.get(question_id, 0) + 1
    return dict(sorted(counts.items()))


def label_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-question gold histogram, with criteria keys that never appear as gold named.

    ``labels`` is the gold distribution the head is asked to learn; ``uncovered`` lists
    criteria keys that appear in some row's criteria but never as a gold label — the
    honest coverage record (today: 23 of the 36 style keys, and 2 category keys, have
    zero positive rows). A key that is never positive can only be learned as a negative.
    """
    histograms: dict[str, dict[str, int]] = {}
    criteria_keys: dict[str, set[str]] = {}
    for row in rows:
        question_id = question_id_of(row)
        gold = json.loads(str(row["gold"]))
        label = str(gold[question_id]["label"])
        histogram = histograms.setdefault(question_id, {})
        histogram[label] = histogram.get(label, 0) + 1
        questions = json.loads(str(row["questions"]))
        criteria_keys.setdefault(question_id, set()).update(questions[question_id]["criteria"])
    return {
        question_id: {
            "labels": dict(sorted(histograms.get(question_id, {}).items())),
            "uncovered": sorted(criteria_keys[question_id] - set(histograms.get(question_id, {}))),
        }
        for question_id in sorted(criteria_keys)
    }


def split_summary(rows: list[dict[str, Any]], source_split: str) -> dict[str, Any]:
    """One split's manifest block: provenance, row counts, coverage."""
    return {
        "source_split": source_split,
        "rows": len(rows),
        "per_question": per_question_counts(rows),
        "label_coverage": label_coverage(rows),
    }


def write_parquet(rows: list[dict[str, Any]], path: Path) -> str:
    """Write one parquet shard and return its sha256 (the manifest's integrity handle)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "id": pa.array([row["id"] for row in rows], type=pa.string()),
            "state": pa.array([row["state"] for row in rows], type=pa.string()),
            "questions": pa.array([row["questions"] for row in rows], type=pa.string()),
            "gold": pa.array([row["gold"] for row in rows], type=pa.string()),
        }
    )
    pq.write_table(table, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Export both splits → write parquet shards → write the manifest."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    inputs = {"category": (Path(args.choice), CATEGORY_QUESTION_ID), "style": (Path(args.style), STYLE_QUESTION_ID)}
    for path in (choice for choice, _ in inputs.values()):
        if not path.is_file():
            raise SystemExit(f"input {path} does not exist")
    exported: dict[str, list[dict[str, Any]]] = {}
    for split in (args.train_split, args.test_split):
        rows: list[dict[str, Any]] = []
        for name, (path, question_id) in inputs.items():
            logger.info("reading %s split rows from %s", split, path)
            rows.extend(to_typed_row(row, question_id) for row in load_rows(path, split))
        exported[split] = rows
    train, test = exported[args.train_split], exported[args.test_split]
    shared = {row["id"] for row in train} & {row["id"] for row in test}
    if shared:
        raise SystemExit(
            f"{len(shared)} ids appear in both splits (e.g. {sorted(shared)[:3]}) — splits must be disjoint"
        )
    root = Path(args.out) / args.workflow
    shards = {
        "train": write_parquet(train, root / "train-00000.parquet"),
        "test": write_parquet(test, root / "test-00000.parquet"),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "clm_pin": args.clm_pin,
        "sources": {name: str(path) for name, (path, _) in inputs.items()},
        "workflow": args.workflow,
        "question_ids": {name: question_id for name, (_, question_id) in inputs.items()},
        "splits": {
            "train": split_summary(train, args.train_split),
            "test": split_summary(test, args.test_split),
        },
        "shards": {
            name: {"path": f"{args.workflow}/{name}-00000.parquet", "sha256": digest} for name, digest in shards.items()
        },
    }
    (Path(args.out) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for split, block in manifest["splits"].items():
        logger.info(
            "exported %s shard: %d rows %s",
            split,
            block["rows"],
            ", ".join(f"{qid}: {count}" for qid, count in block["per_question"].items()),
        )
        for question_id, coverage in block["label_coverage"].items():
            logger.info(
                "%s %s: %d labels, %d criteria never positive",
                split,
                question_id,
                len(coverage["labels"]),
                len(coverage["uncovered"]),
            )
    logger.info("dataset root %s (test shard = source.split %r)", root, args.test_split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
