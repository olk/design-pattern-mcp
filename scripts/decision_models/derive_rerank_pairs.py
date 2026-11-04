#!/usr/bin/env python
"""Derive the served rerank pairs the Tacit trainer and the eval replay use.

One row per (decision statement, candidate document): the statement is the corpus
scenario's ``decision_statement`` (the fixed proxy for the part statement the pipeline
extracts at serve time), the document is the candidate record's ``document_text``
(``src/patterns/loader.py`` — the §2.2 text the reranker actually receives), and the gain
is the candidate's reviewed ``label_tier`` (nDCG gain, holdout eval only).

Both splits are written: ``train`` rows feed ``tacit_self_distill.py --rerank-pairs``
(the teacher labels them; tiers are ignored by training), ``holdout`` rows are eval
material and never reach training weights.

The pair bytes ARE the serving bytes: stage 1 (the fused pool) already happened when the
corpus authors picked each scenario's candidates, so a pair is exactly one stage-2
``yes_no`` item (``seam_prompts.rerank_state`` + ``RERANK_RELEVANCE_QUESTION``).

Regenerate after any corpus, noul or catalog change: ``make tacit-payloads`` (and
``make tacit-payloads-check`` byte-compares in CI/the build). The Tacit extract marker
pins this artifact, so a stale file fails the cached image build.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.patterns.loader import load_catalog  # noqa: E402

#: Marker the noul export joins description and statement with.
STATEMENT_MARKER = "\n\nDecision: "

#: Relevance grade per noul label tier (TREC graded-relevance steps; eval only).
TIER_GAIN: dict[str, int] = {"primary": 3, "acceptable_primary": 2, "supporting": 1, "negative": 0}

#: Splits written in one run: train feeds the trainer, holdout feeds eval.
SPLITS: tuple[str, ...] = ("train", "holdout")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the pair derivation."""
    parser = argparse.ArgumentParser(description="Materialise the served rerank pairs.")
    parser.add_argument("--corpus", default="tests/benchmark/scenarios/full.json", help="scenario corpus")
    parser.add_argument("--noul", default="qa/typed/noul.jsonl", help="per-candidate relevance rows")
    parser.add_argument("--pattern-dir", default="pattern", help="catalog pattern directory")
    parser.add_argument("--docs-dir", default="docs", help="catalog docs directory")
    parser.add_argument("--out", default="qa/typed/rerank_pairs.jsonl", help="pair artifact path")
    return parser.parse_args(argv)


def load_scenarios(path: Path) -> dict[str, dict[str, Any]]:
    """Scenario id → scenario record."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("scenarios", payload)
    if not isinstance(items, list) or not items:
        raise SystemExit(f"{path} carries no scenarios")
    return {str(item["scenario_id"]): item for item in items}


def noul_rows(path: Path) -> list[dict[str, Any]]:
    """Every noul row, validated: tier known, label consistent with tier."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            tier = str(row.get("label_tier", ""))
            if tier not in TIER_GAIN:
                raise SystemExit(f"row {row.get('id')!r} carries unknown label_tier {tier!r}")
            if bool(row.get("label")) == (tier == "negative"):
                raise SystemExit(f"row {row.get('id')!r} mixes label {row.get('label')!r} with tier {tier!r}")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path} carries no rows")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Corpus + noul + catalog → pair artifact; refuses to guess a missing piece."""
    args = parse_args(argv)
    scenarios = load_scenarios(Path(args.corpus))
    catalog = load_catalog(Path(args.pattern_dir), Path(args.docs_dir))
    document_texts = {name: record.document_text for name, record in catalog.records.items()}
    pairs: list[dict[str, Any]] = []
    statements: dict[str, str] = {}
    for row in noul_rows(Path(args.noul)):
        source: dict[str, Any] = row.get("source") or {}
        split = str(source.get("split", ""))
        if split not in SPLITS:
            continue
        scenario_id = str(source.get("scenario_id", row.get("id")))
        scenario = scenarios.get(scenario_id)
        if scenario is None:
            raise SystemExit(f"noul scenario {scenario_id!r} is absent from {args.corpus}")
        if str(scenario.get("split")) != split:
            raise SystemExit(f"scenario {scenario_id!r} has corpus split {scenario.get('split')!r}, not {split!r}")
        state = str(row["state"])
        cut = state.rfind(STATEMENT_MARKER)
        if cut < 0:
            raise SystemExit(f"row {row.get('id')!r} carries no {STATEMENT_MARKER!r} statement marker")
        statement = state[cut + len(STATEMENT_MARKER) :].strip()
        if statement != str(scenario["decision_statement"]).strip():
            raise SystemExit(f"scenario {scenario_id!r}: noul statement does not match the corpus decision_statement")
        previous = statements.setdefault(scenario_id, statement)
        if previous != statement:
            raise SystemExit(f"scenario {scenario_id!r} mixes decision statements across its noul rows")
        name = str(row["candidate"])
        document = document_texts.get(name)
        if not document:
            raise SystemExit(f"scenario {scenario_id!r}: candidate {name!r} is not in the catalog")
        pairs.append(
            {
                "id": str(row.get("id")),
                "scenario_id": scenario_id,
                "split": split,
                "statement": statement,
                "name": name,
                "document": document,
                "gain": TIER_GAIN[str(row["label_tier"])],
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(pair, ensure_ascii=False) for pair in pairs) + "\n", encoding="utf-8")
    counts = {split: sum(1 for pair in pairs if pair["split"] == split) for split in SPLITS}
    print(f"wrote {len(pairs)} rerank pairs to {out} (train {counts['train']}, holdout {counts['holdout']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
