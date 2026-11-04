#!/usr/bin/env python
"""Wire-contract smoke for the running AnyJev gateway (``make decisions-smoke``).

Three probes, each with the SERVING bytes (``src/decisions/seam_prompts.py`` answers every
question, so a drift in the vocabulary fails here before it reaches a tool call):

1. J2 category ``choice`` — 9 options, a probabilities map comes back;
2. Ch2 stage 2 — the ``domain-modeling`` style shortlist plus ``none``, inside AnyJev's
   26-option ``choice`` cap;
3. J1 rerank — ONE batched request of ``yes_no`` items, one decision per item, each
   carrying ``p(Yes)``.

Stdlib only: it runs against a container with nothing but python3 installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.decisions import seam_prompts  # noqa: E402

#: AnyJev's ``choice``/``score`` option cap (``anyjev`` ``question.py`` ``MAX_OPTIONS``).
OPTION_CAP = 26


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the smoke probes."""
    parser = argparse.ArgumentParser(description="Smoke-test the AnyJev gateway's wire contract.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18100", help="Gateway ROOT (no /v1 suffix).")
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout in seconds.")
    return parser.parse_args(argv)


def post(base_url: str, body: dict[str, Any], timeout: float) -> Any:
    """One ``POST /v1/decide``; a non-200 or unparseable body aborts the smoke."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/decide",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except OSError as exc:
        raise SystemExit(f"gateway request failed ({base_url}): {exc}") from exc


def _require_decision(payload: Any, label: str) -> dict[str, Any]:
    """One decision object with a numeric-probabilities map, or abort."""
    if not isinstance(payload, dict) or not isinstance(payload.get("probs"), dict):
        raise SystemExit(f"{label}: answer is not a decision: {payload!r}")
    return payload


def probe_category(base_url: str, timeout: float) -> None:
    """J2: the 9-way category ``choice`` answers with a probabilities map."""
    options = list(seam_prompts.CATEGORY_SLUGS)
    payload = post(
        base_url,
        {
            "state": "System goal: serve order views\nDecision part: pick a category",
            "question": seam_prompts.category_question(),
            "options": options,
            "kind": "choice",
        },
        timeout,
    )
    decision = _require_decision(payload, "category")
    if not set(decision["probs"]) <= set(options):
        raise SystemExit(f"category: probabilities carry options that were never offered: {sorted(decision['probs'])}")


def probe_style_stage2(base_url: str, timeout: float) -> None:
    """Ch2 stage 2: one shortlist + ``none`` stays inside the option cap."""
    styles = seam_prompts.styles_for_category("domain-modeling")
    options = [*styles, seam_prompts.STYLE_NONE_KEY]
    if len(options) > OPTION_CAP:
        raise SystemExit(f"style shortlist carries {len(options)} options, over the {OPTION_CAP} cap")
    payload = post(
        base_url,
        {
            "state": "System goal: model a billing domain",
            "question": seam_prompts.style_question(styles),
            "options": options,
            "kind": "choice",
        },
        timeout,
    )
    decision = _require_decision(payload, "style-stage2")
    if not set(decision["probs"]) <= set(options):
        raise SystemExit(
            f"style-stage2: probabilities carry options that were never offered: {sorted(decision['probs'])}"
        )


def probe_rerank_batch(base_url: str, timeout: float) -> None:
    """J1: one batched request of ``yes_no`` items, one ``p(Yes)`` decision each."""
    items = [
        {
            "state": seam_prompts.rerank_state(
                "decide how to persist aggregates",
                text,
            ),
            "question": seam_prompts.RERANK_RELEVANCE_QUESTION,
            "options": ["Yes", "No"],
            "kind": "yes_no",
        }
        for text in (
            "Repository: mediates between the domain and data mapping layers",
            "Memento: captures and externalises an object's internal state",
        )
    ]
    payload = post(base_url, {"items": items}, timeout)
    decisions = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(decisions, list) or len(decisions) != len(items):
        raise SystemExit(f"rerank: expected {len(items)} decisions, got {payload!r}")
    for entry in decisions:
        probs = entry.get("probs") if isinstance(entry, dict) else None
        if not isinstance(probs, dict) or "Yes" not in probs:
            raise SystemExit(f"rerank: a decision carries no p(Yes): {entry!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the three probes in order; the first failure exits non-zero."""
    args = parse_args(argv)
    for label, probe in (
        ("Tacit category choice (J2)", probe_category),
        ("Tacit two-stage style (Ch2 stage 2, domain-modeling shortlist)", probe_style_stage2),
        ("Tacit rerank batch (J1, yes_no items)", probe_rerank_batch),
    ):
        probe(args.base_url, args.timeout)
        print(f"OK   {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
