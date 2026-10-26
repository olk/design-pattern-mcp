"""CLI of the benchmark harness: ``uv run python -m tests.benchmark.harness.main`` (plan §3).

Examples::

    uv run python -m tests.benchmark.harness.main --mode offline --limit 3 --out data/benchmark-runs/offline-smoke
    uv run python -m tests.benchmark.harness.main --mode offline --flip dm-order-lifecycle --out …/offline-flip
    uv run python -m tests.benchmark.harness.main --mode live --split holdout --repeat 2 --out …/live-holdout
    uv run python -m tests.benchmark.harness.main --mode analyze --out …/analyze-all

``--mode analyze`` runs the same live seams as ``--mode live`` but stops after ANALYZE:
selection hit/recall stay observable, generation work is not paid for.

``--mode e2e`` lives in :mod:`tests.benchmark.harness.e2e` (black-box, no stage attribution).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import config as cfg
from .runner import RunOptions, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    """The harness CLI surface."""
    parser = argparse.ArgumentParser(
        prog="tests.benchmark.harness.main",
        description="Run one benchmark arm with stage attribution and quality metrics.",
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "live", "analyze"),
        default="offline",
        help=(
            "offline = scripted seams (<60 s, no network); live = real in-process pipeline "
            "(analyze → generate → evaluate); analyze = live seams, ANALYZE phase only "
            "(selection hit/recall, no generation)"
        ),
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=cfg.DEFAULT_SCENARIOS,
        help="scenario corpus JSON (default: tests/benchmark/scenarios/seed.json)",
    )
    parser.add_argument(
        "--split", choices=(*cfg.SPLITS, "all"), default="all", help="which split to run (default: all)"
    )
    parser.add_argument("--limit", type=int, default=None, help="run only the first N scenarios of the split")
    parser.add_argument("--repeat", type=int, default=1, help="repeat count per scenario (default 1)")
    parser.add_argument(
        "--flip",
        action="append",
        default=[],
        metavar="SCENARIO_ID",
        help="offline-only debug hook: script decoys for this scenario (repeatable)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="run directory (default: data/benchmark-runs/<mode>-<timestamp>)"
    )
    parser.add_argument("--run-id", default=None, help="explicit run id (default: derived from the mode and time)")
    parser.add_argument("--no-warmup", action="store_true", help="skip the untimed warmup scenario")
    parser.add_argument("--force", action="store_true", help="write into a non-empty --out directory")
    parser.add_argument("--log-level", default="warning", help="harness log level (default: warning)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: run one arm and print its headline numbers as JSON."""
    args = build_parser().parse_args(argv)
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2
    if args.out is not None and args.out.exists() and any(args.out.iterdir()) and not args.force:
        print(f"--out {args.out} is not empty; pass --force to overwrite", file=sys.stderr)
        return 2
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    options = RunOptions(
        mode=args.mode,
        scenarios_path=args.scenarios,
        out_dir=args.out,
        split=args.split,
        limit=args.limit,
        repeats=args.repeat,
        flips=tuple(args.flip),
        warmup=not args.no_warmup,
        run_id=args.run_id,
    )
    try:
        result = asyncio.run(run_benchmark(options))
    except Exception as exc:
        print(f"run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("the run directory carries an aborted.json marker; discard it", file=sys.stderr)
        return 1
    summary = result["summary"]
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "run_dir": result["run_dir"],
                "records": result["records"],
                "mode": summary["mode"],
                "quality": summary["quality"]["overall"],
                "e2e_ms": summary["latency"]["e2e_ms"],
                "stages": sorted(summary["latency"]["stages"]),
                "notes": summary["notes"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
