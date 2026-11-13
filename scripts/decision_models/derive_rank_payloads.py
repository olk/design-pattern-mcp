#!/usr/bin/env python
"""Derive the served ``/v1/rank`` payloads the build's rank-parity gate replays.

``CLMReranker._request`` (``src/patterns/clm_rerank.py:91``) sends
``{context: "", question: <decision-part statement>, answers: <pattern document texts>}``.
The noul corpus cannot stand in for that: its ``candidate_text`` is the short
``"name: intent"`` string, while the served answers are each record's
``document_text`` (name, aliases, intent, problem, context, use cases —
``src/patterns/loader.py:210``). A parity gate over the short strings bounds drift on a
payload the server never sends.

This script materialises the real payloads once, from the reviewed corpus and the
catalog, and commits them as ``qa/typed/rank_payloads.jsonl``. The CLM image copies
that file (it has no pattern catalog), so the gate replays served bytes without
dragging 486 records into the build context.

What is fixed and what is not: ``answers`` and ``context`` are exact. ``question`` is
the corpus's ``decision_statement`` — the same string the noul export composed its
states from (asserted here), and the fixed proxy for the statement the pipeline
extracts at serve time (LLM-derived, so no build artifact can reproduce it verbatim).

Run it after any corpus, noul or **catalog** change. Two guards exist and they cover
different halves: the Makefile pins this artifact plus the corpus and this script into
the CLM cache marker (so a changed payload or corpus invalidates a cached head build),
and `make clm-payloads-check` regenerates to a temp path and byte-compares, so a catalog
edit that was never re-derived fails the build instead of silently replaying
``document_text`` the server no longer sends. The catalog itself is not hashed (7290
files); the check is what closes that gap.

    uv run python scripts/decision_models/derive_rank_payloads.py
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

#: Relevance grade per noul label tier (nDCG gain + rank-parity reporting). The
#: order encodes the review judgment: primary > acceptable_primary > supporting
#: > negative, and the magnitudes (3/2/1/0) are the standard graded-relevance
#: steps from the TREC Web track — any rescaling needs a stated reason.
TIER_GAIN: dict[str, int] = {"primary": 3, "acceptable_primary": 2, "supporting": 1, "negative": 0}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the payload derivation."""
    parser = argparse.ArgumentParser(description="Materialise the served /v1/rank payloads.")
    parser.add_argument("--corpus", default="tests/benchmark/scenarios/full.json", help="scenario corpus")
    parser.add_argument("--noul", default="qa/typed/noul.jsonl", help="per-candidate relevance rows")
    parser.add_argument("--pattern-dir", default="pattern", help="catalog pattern directory")
    parser.add_argument("--docs-dir", default="docs", help="catalog docs directory")
    parser.add_argument("--split", default="holdout", help="qa/ source split to replay")
    parser.add_argument("--out", default="qa/typed/rank_payloads.jsonl", help="payload artifact path")
    return parser.parse_args(argv)


def load_scenarios(path: Path) -> dict[str, dict[str, Any]]:
    """Scenario id → scenario record."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("scenarios", payload)
    if not isinstance(items, list) or not items:
        raise SystemExit(f"{path} carries no scenarios")
    return {str(item["scenario_id"]): item for item in items}


def scenario_candidates(path: Path, split: str) -> dict[str, tuple[str, list[tuple[str, str, int]]]]:
    """Scenario id → (statement from the noul state, [(name, text, gain)] in file order).

    The statement is recovered from the export's own composition
    (``<description>\\n\\nDecision: <statement>``) and returns with the candidates so the
    caller can check it against the corpus before deriving anything. The gain is
    the candidate's ``TIER_GAIN`` from its reviewed ``label_tier`` — the rank
    gate's correctness signal travels with the payload so the CLM image (which
    has no qa/ corpus) can score nDCG without dragging noul into the build.
    """
    candidates: dict[str, tuple[str, list[tuple[str, str, int]]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            source: dict[str, Any] = row.get("source") or {}
            if source.get("split") != split:
                continue
            scenario_id = str(source.get("scenario_id", row.get("id")))
            tier = str(row.get("label_tier", ""))
            if tier not in TIER_GAIN:
                raise SystemExit(f"row {row.get('id')!r} carries unknown label_tier {tier!r}")
            if bool(row.get("label")) == (tier == "negative"):
                raise SystemExit(f"row {row.get('id')!r} mixes label {row.get('label')!r} with tier {tier!r}")
            state = str(row["state"])
            cut = state.rfind(STATEMENT_MARKER)
            if cut < 0:
                raise SystemExit(f"row {row.get('id')!r} carries no {STATEMENT_MARKER!r} statement marker")
            statement = state[cut + len(STATEMENT_MARKER) :].strip()
            entry = candidates.setdefault(scenario_id, (statement, []))
            if entry[0] != statement:
                raise SystemExit(f"scenario {scenario_id!r} mixes decision statements across its noul rows")
            entry[1].append((str(row["candidate"]), str(row["candidate_text"]), TIER_GAIN[tier]))
    if not candidates:
        raise SystemExit(f"no rows with source.split={split!r} in {path}")
    return candidates


def build_payload(
    scenario_id: str,
    scenario: dict[str, Any],
    statement: str,
    candidates: Sequence[tuple[str, str, int]],
    document_texts: dict[str, str],
) -> dict[str, Any]:
    """One ``/v1/rank`` payload in the served shape, with graded relevance."""
    if statement != str(scenario["decision_statement"]).strip():
        raise SystemExit(
            f"scenario {scenario_id!r}: noul statement does not match the corpus decision_statement — "
            "the corpus and the qa export disagree about what was asked"
        )
    answers: list[str] = []
    for name, _candidate_text, _gain in candidates:
        text = document_texts.get(name)
        if not text:
            raise SystemExit(f"scenario {scenario_id!r}: candidate {name!r} is not in the catalog")
        answers.append(text)
    return {
        "scenario_id": scenario_id,
        "family": scenario.get("family"),
        "context": "",
        "question": statement,
        "answers": answers,
        "answer_names": [name for name, _, _ in candidates],
        "relevance": [gain for _, _, gain in candidates],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Corpus + noul + catalog → payload artifact; refuses to guess a missing piece."""
    args = parse_args(argv)
    scenarios = load_scenarios(Path(args.corpus))
    candidates = scenario_candidates(Path(args.noul), args.split)
    catalog = load_catalog(Path(args.pattern_dir), Path(args.docs_dir))
    document_texts = {name: record.document_text for name, record in catalog.records.items()}
    payloads: list[dict[str, Any]] = []
    for scenario_id, (statement, names) in candidates.items():
        scenario = scenarios.get(scenario_id)
        if scenario is None:
            raise SystemExit(f"noul scenario {scenario_id!r} is absent from {args.corpus}")
        if scenario.get("split") != args.split:
            raise SystemExit(f"scenario {scenario_id!r} has corpus split {scenario.get('split')!r}, not {args.split!r}")
        payloads.append(build_payload(scenario_id, scenario, statement, names, document_texts))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(payload, ensure_ascii=False) for payload in payloads) + "\n", encoding="utf-8")
    widths = [len(payload["answers"]) for payload in payloads]
    chars = [len(answer) for payload in payloads for answer in payload["answers"]]
    print(
        f"wrote {len(payloads)} rank payloads to {out} "
        f"({sum(widths)} answers, {min(widths)}-{max(widths)} per payload, "
        f"answer chars {min(chars)}-{max(chars)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
