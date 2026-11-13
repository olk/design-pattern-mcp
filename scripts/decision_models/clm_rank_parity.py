#!/usr/bin/env python
"""Bound the rank-wire movement of a choice fine-tune AND score its correctness (in-RUN check).

``/v1/rank`` is served through the SAME head pair ``--task choice`` trains (there are
no separate rank weights — ``src/clm/server.py::rank`` → ``engine.rank`` answers a
synthetic choice question with the loaded HeadPair), so the fine-tune moves the rank
scoring surface by construction. This script bounds that movement and scores its
direction: it loads the released and trained checkpoints side by side through the
pinned ``clm.Engine`` and compares their answers over the payloads in
``qa/typed/rank_payloads.jsonl`` — one payload per holdout scenario, each in the
exact ``CLMReranker._request`` shape (``context`` empty, the decision statement as
``question``, the candidates' served ``document_text`` as ``answers``), derived by
``derive_rank_payloads.py`` with the reviewed noul relevance grades carried in
``relevance`` (``TIER_GAIN``: primary 3 / acceptable_primary 2 / supporting 1 /
negative 0, index-aligned with ``answers``).

Three numbers per payload, all reported:

* top-1 flip — the budgeted quantity (``--max-flip-rate``, a policy budget passed
  explicitly by the build; this script has no default so the value is visible at the
  call site). The 0.25 the Dockerfile passes is ASSERTED, not data-derived: a
  principled budget needs the first head's own parity report (flip rate + nDCG
  deltas + top-5 overlap) plus a rank-seam A/B tying flip tolerance to served
  quality — both pending. Deriving it from the static payload nDCG distribution
  alone would be circular (payloads don't measure the head under test).
* nDCG@10 — graded correctness against the reviewed noul relevance: whether a flip
  moved the wire toward the labeled patterns or away from them. nDCG, not MRR:
  MRR scores only the first relevant hit and cannot see a flip that demotes a
  primary below a supporting answer while keeping some relevant answer on top —
  the exact regression a shared-head fine-tune can introduce.
* top-5 overlap — how much of the head of the ordering survived. Reported, not gated:
  it catches reordering the top-1 rule cannot see, and it is the number the runbook's
  seam A/B should use to set the budget from data rather than assertion.

It fails (exit 1) when the flip share exceeds the budget OR when the trained head's
mean nDCG@10 regresses below the released head's — a head that reorders the rank
wire past the budget, or ranks the labeled patterns worse than the baseline, never
bakes into the default target. The comparison is deterministic: ``clm.Engine.answer``
is ``argmax`` over ``scale*cos`` (no sampling), so repeated runs on the same bytes
give the same result.

Needs the pinned CLM package importable and a pooling embedder reachable (both true
inside the ``:finetune`` RUN, which runs this between the eval gate and the encoder
shutdown).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: How many candidates the reported top-5 overlap looks at.
TOP_N_OVERLAP = 5

#: nDCG cutoff: the fused pool is 20 deep and the rerank blend only sees the head,
#: so correctness is scored over the top 10 (half the pool, twice the overlap window).
NDCG_CUTOFF = 10

#: Report filename written beside the payload file when ``--report`` is not given.
REPORT_NAME = "clm-rank-parity.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the image build."""
    parser = argparse.ArgumentParser(description="Fail the CLM build when the fine-tune flips the rank wire.")
    parser.add_argument(
        "--payloads",
        required=True,
        help="qa/typed/rank_payloads.jsonl (served-shape payloads for the holdout split).",
    )
    parser.add_argument("--released", required=True, help="Released init checkpoint (the rank baseline).")
    parser.add_argument("--trained", required=True, help="Fine-tuned best_head.pt (the rank candidate).")
    parser.add_argument("--clm-src", required=True, help="Pinned CLM repo src/ dir (importable clm package).")
    parser.add_argument("--emb-url", required=True, help="Pooling embedder /v1/embeddings URL.")
    parser.add_argument("--emb-model", default="qwen3-8b", help="Model name the embedder serves.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Embedder truncation budget.")
    parser.add_argument(
        "--max-flip-rate",
        type=float,
        default=None,
        help="Required: fail when the top-1 flip share exceeds this policy budget.",
    )
    parser.add_argument(
        "--report",
        default="",
        help=f"JSON report path (default: {REPORT_NAME} beside --payloads).",
    )
    return parser.parse_args(argv)


def load_payloads(path: Path) -> list[dict[str, Any]]:
    """Read the served-shape payload artifact; refuse an empty or malformed file."""
    payloads: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            answers = payload.get("answers")
            if not isinstance(answers, list) or len(answers) < 2:
                raise SystemExit(f"payload {payload.get('scenario_id')!r} carries fewer than two answers")
            if not isinstance(payload.get("question"), str):
                raise SystemExit(f"payload {payload.get('scenario_id')!r} carries no question")
            relevance = payload.get("relevance")
            if (
                not isinstance(relevance, list)
                or len(relevance) != len(answers)
                or not all(isinstance(gain, (int, float)) and not isinstance(gain, bool) for gain in relevance)
            ):
                raise SystemExit(
                    f"payload {payload.get('scenario_id')!r} carries no relevance grades aligned with its answers — "
                    "regenerate with derive_rank_payloads.py (make clm-payloads)"
                )
            payloads.append(payload)
    if not payloads:
        raise SystemExit(f"no payloads in {path}")
    return payloads


def top_choices(engine: Any, checkpoint: str, payload: dict[str, Any], depth: int) -> list[str]:
    """Top-``depth`` candidate texts under one checkpoint (the served rank code path)."""
    ranked = engine.rank(
        payload.get("context", ""),
        list(payload["answers"]),
        payload["question"] or None,
        model=checkpoint,
    )
    return [str(entry["candidate"]) for entry in ranked[:depth]]


def candidate_names(payload: dict[str, Any]) -> dict[str, str]:
    """Answer text → pattern name, index-aligned from ``answer_names`` (empty when absent).

    The report records names, never the answer texts: each ``document_text`` runs to
    several kilobytes, so echoing the winners would duplicate the whole candidate pool
    into a JSON file that is sha256-pinned beside the checkpoint.
    """
    answers = payload.get("answers") or []
    names = payload.get("answer_names") or []
    if isinstance(names, list) and len(names) == len(answers):
        return {str(text): str(name) for text, name in zip(answers, names, strict=True)}
    return {}


def ndcg_at(ranked_texts: Sequence[str], answers: Sequence[str], relevance: Sequence[float], k: int) -> float:
    """nDCG@k of one ordering against graded relevance (texts → gains by index)."""
    gains = {str(text): float(gain) for text, gain in zip(answers, relevance, strict=True)}
    ordered = [gains.get(str(text), 0.0) for text in ranked_texts[:k]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ordered))
    ideal = sorted((float(gain) for gain in relevance), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 1.0


def top5_overlap(released: Sequence[str], trained: Sequence[str]) -> float:
    """Share of the released head that survives in the trained head (identity-safe)."""
    head = list(dict.fromkeys(released))[:TOP_N_OVERLAP]
    if not head:
        return 1.0
    return len(set(head) & set(trained[:TOP_N_OVERLAP])) / len(head)


def main(argv: Sequence[str] | None = None) -> int:
    """Compare released vs trained over the payload set; gate on the flip budget."""
    args = parse_args(argv)
    if args.max_flip_rate is None:
        raise SystemExit(
            "clm-rank-parity: FAIL: --max-flip-rate is required — the budget must be a reviewed constant at the "
            "call site, not a hidden default"
        )
    payloads = load_payloads(Path(args.payloads))
    sys.path.insert(0, args.clm_src)
    from clm.embedder import Embedder  # type: ignore[import-not-found]  # noqa: E402  (pinned CLM repo, build-only)
    from clm.engine import Engine  # type: ignore[import-not-found]  # noqa: E402  (pinned CLM repo, build-only)

    engine = Engine(
        Embedder(args.emb_url, args.emb_model, max_tokens=args.max_tokens),
        models={"released": args.released, "trained": args.trained},
        # No vector arena: the default budget reserves 2% of GPU memory while the
        # pooling encoder still holds the card — a 60-call gate must not OOM the build.
        action_cache="0",
    )
    flips: list[dict[str, Any]] = []
    overlaps: list[float] = []
    released_ndcg: list[float] = []
    trained_ndcg: list[float] = []
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        answers = [str(text) for text in payload["answers"]]
        gains = [float(gain) for gain in payload["relevance"]]
        released = top_choices(engine, "released", payload, NDCG_CUTOFF)
        trained = top_choices(engine, "trained", payload, NDCG_CUTOFF)
        overlap = top5_overlap(released, trained)
        overlaps.append(overlap)
        released_score = ndcg_at(released, answers, gains, NDCG_CUTOFF)
        trained_score = ndcg_at(trained, answers, gains, NDCG_CUTOFF)
        released_ndcg.append(released_score)
        trained_ndcg.append(trained_score)
        names = candidate_names(payload)
        released_top = names.get(released[0], released[0][:80])
        trained_top = names.get(trained[0], trained[0][:80])
        flipped = released_top != trained_top
        if flipped:
            flips.append({"scenario_id": payload["scenario_id"], "released": released_top, "trained": trained_top})
        rows.append(
            {
                "scenario_id": payload["scenario_id"],
                "family": payload.get("family"),
                "flipped": flipped,
                "top5_overlap": overlap,
                "released_top1": released_top,
                "trained_top1": trained_top,
                "released_ndcg10": released_score,
                "trained_ndcg10": trained_score,
            }
        )
    flip_rate = len(flips) / len(payloads)
    mean_overlap = sum(overlaps) / len(overlaps)
    mean_released = sum(released_ndcg) / len(released_ndcg)
    mean_trained = sum(trained_ndcg) / len(trained_ndcg)
    flips_ok = flip_rate <= args.max_flip_rate
    ndcg_ok = mean_trained >= mean_released
    status = "PASS" if (flips_ok and ndcg_ok) else "FAIL"
    print(
        "clm-rank-parity: {}: {}/{} top-1 flips (rate {:.3f}, budget {:.3f}), "
        "mean top-{} overlap {:.3f}, mean nDCG@{} trained {:.4f} vs released {:.4f}{}".format(
            status,
            len(flips),
            len(payloads),
            flip_rate,
            args.max_flip_rate,
            TOP_N_OVERLAP,
            mean_overlap,
            NDCG_CUTOFF,
            mean_trained,
            mean_released,
            (" — RANK REGRESSION" if not ndcg_ok else "")
            + (f" flipped={[flip['scenario_id'] for flip in flips]}" if flips else ""),
        )
    )
    report_path = Path(args.report) if args.report else Path(args.payloads).with_name(REPORT_NAME)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "gate": "clm-rank-parity",
                "status": status.lower(),
                "payloads": str(args.payloads),
                "n": len(payloads),
                "flip_rate": flip_rate,
                "max_flip_rate": args.max_flip_rate,
                "mean_top5_overlap": mean_overlap,
                "mean_released_ndcg10": mean_released,
                "mean_trained_ndcg10": mean_trained,
                "flips": flips,
                "per_payload": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0 if (flips_ok and ndcg_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
