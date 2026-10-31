#!/usr/bin/env python
"""Export this repo's typed category labels as a CLM ``typed-decisions`` dataset.

CLM's choice fine-tune (``train/finetune.py --task choice``) reads parquet files laid
out as ``<root>/<workflow>/<split>*.parquet``, where each row is a System One request
plus its gold answer::

    {"id": ..., "state": ..., "questions": {"<qid>": {type, instructions, criteria}},
     "gold": {"<qid>": {"label": ..., "probabilities": {...}}}}

(taken from ``train/adapters.py::typed_decision_examples`` at the pinned revision —
it decodes JSON-encoded columns too, so this exporter writes the two object columns as
JSON strings and keeps the whole file schema-stable).

Why the *category* rows for a *style* head: the heads are contrastively trained on the
task family "which criterion best matches this state", not on a fixed label set. Our 70
train-split rows are exactly that family, drawn from this repo's own scenario corpus
with catalog-evidence criterion text, so a pass over them domain-adapts the encoder and
action head to the questions this server asks (style picks included). It is not a
category router: nothing here is wired to the category seam, which stays on Tacit.

The holdout families in ``qa/typed/choice.jsonl`` are excluded by construction
(``source.split != "train"``); the test split below is carved out of the train rows, so
no holdout scenario ever reaches the trained weights.

Run it standalone (needs pyarrow, nothing else)::

    uv run --with pyarrow python scripts/decision_models/export_clm_dataset.py \\
        --choice qa/typed/choice.jsonl --out /clm/data/typed-choice
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("export-clm-dataset")

#: Question id under which every exported row asks its category question.
QUESTION_ID = "category"

#: The shipped category instructions (identical to the qa/ export's instructions field).
DEFAULT_INSTRUCTIONS = "Which design-pattern category does this decision belong to?"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the image build."""
    parser = argparse.ArgumentParser(description="Export typed category labels as a CLM dataset.")
    parser.add_argument("--choice", default="qa/typed/choice.jsonl", help="qa/typed/choice.jsonl")
    parser.add_argument("--out", required=True, help="Dataset root written for CLM's --data.")
    parser.add_argument("--workflow", default="all", help="CLM workflow subdirectory name.")
    parser.add_argument("--split", default="train", help="qa/ source split to export.")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Fraction of exported ids held out.")
    parser.add_argument("--seed", type=int, default=13)
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


def to_typed_row(row: dict[str, Any]) -> dict[str, Any]:
    """One qa/ row → the typed-decisions wire row (objects JSON-encoded)."""
    criteria = row["criteria"]
    if not isinstance(criteria, dict) or not criteria:
        raise SystemExit(f"row {row.get('id')!r} has no criteria object")
    label = row["label"]
    if label not in criteria:
        raise SystemExit(f"row {row.get('id')!r}: label {label!r} is not among the criteria keys")
    questions = {
        QUESTION_ID: {
            "type": "choice",
            "instructions": row.get("instructions") or DEFAULT_INSTRUCTIONS,
            "criteria": criteria,
        }
    }
    gold = {QUESTION_ID: {"label": label}}
    return {
        "id": str(row["id"]),
        "state": row["state"],
        "questions": json.dumps(questions, ensure_ascii=False),
        "gold": json.dumps(gold, ensure_ascii=False),
    }


def split_rows(
    rows: list[dict[str, Any]], test_frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic id-hash split: the test slice is a subset of the exported train rows."""
    ids = sorted({row["id"] for row in rows})
    random.Random(seed).shuffle(ids)
    held_out = set(ids[: max(1, round(test_frac * len(ids)))])
    train = [row for row in rows if row["id"] not in held_out]
    test = [row for row in rows if row["id"] in held_out]
    return train, test


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
    """Export → split → write parquet shards → write the manifest."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    choice_path = Path(args.choice)
    if not choice_path.is_file():
        raise SystemExit(f"--choice {choice_path} does not exist")
    logger.info("reading %s split rows from %s", args.split, choice_path)
    rows = [to_typed_row(row) for row in load_rows(choice_path, args.split)]
    train, test = split_rows(rows, args.test_frac, args.seed)
    root = Path(args.out) / args.workflow
    shards = {
        "train": write_parquet(train, root / "train-00000.parquet"),
        "test": write_parquet(test, root / "test-00000.parquet"),
    }
    manifest = {
        "source": str(choice_path),
        "source_split": args.split,
        "workflow": args.workflow,
        "question_id": QUESTION_ID,
        "rows": {"train": len(train), "test": len(test)},
        "shards": {
            name: {"path": f"{args.workflow}/{name}-00000.parquet", "sha256": digest} for name, digest in shards.items()
        },
        "test_frac": args.test_frac,
        "seed": args.seed,
    }
    (Path(args.out) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("exported %d train / %d test rows to %s", len(train), len(test), root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
