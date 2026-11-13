#!/usr/bin/env python
"""Build-time preflight for the CLM fine-tune: prove the inputs resolve before training.

Both checks used to be inline ``python3 -c`` one-liners inside the fat ``RUN`` of
``docker/Dockerfile.clm``. A ``python3 -c "..."`` carrying its own quotes nests three
quoting layers (Dockerfile -> outer ``/bin/sh -c`` -> inner shell -> Python), which is
how a single stray ``'`` produced ``/bin/sh: 1: Syntax error: "(" unexpected`` at build
time. Keeping the logic in a file removes the quoting layers entirely, and makes the
checks testable off the builder.

Checks:

* ``--entity`` — the local encoder snapshot can supply a tokenizer. The pinned trainer
  (``train/embed_utils.py::Recipe``) hands ``--embed-model`` to
  ``AutoTokenizer.from_pretrained``, so that argument must be a directory or a Hub repo
  id, never a vLLM served alias (which 401s on ``tokenizer_config.json``).
* ``--manifest``/``--expect-pin`` — the exported dataset was produced for the CLM
  revision this image pins, so the trainer is reading the adapter layout it was
  verified against.

Runs stdlib-only for the pin check; the entity check imports ``transformers`` lazily.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PREFIX = "clm-preflight"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the build preflight."""
    parser = argparse.ArgumentParser(description="Verify the CLM trainer's inputs before the fine-tune runs.")
    parser.add_argument("--entity", default="", help="Local encoder directory that must supply a tokenizer.")
    parser.add_argument("--manifest", default="", help="Export manifest carrying the dataset's clm_pin.")
    parser.add_argument("--expect-pin", default="", help="CLM revision the build pins (matches ARG CLM_REPO_PIN).")
    args = parser.parse_args(argv)
    if not args.entity and not args.manifest:
        parser.error("nothing to check: pass --entity and/or --manifest with --expect-pin")
    if args.manifest and not args.expect_pin:
        parser.error("--manifest requires --expect-pin")
    if args.expect_pin and not args.manifest:
        parser.error("--expect-pin requires --manifest")
    return args


def check_entity(path: Path) -> str:
    """The local snapshot must be a directory that yields a tokenizer, offline.

    A Hub repo id would also satisfy ``from_pretrained``, but then the build depends on
    network access and a credential — the snapshot is already in the image, so require
    it. The failure this replaces (an alias resolving as a repo id) surfaced as a 401.
    """
    if not path.is_dir():
        raise SystemExit(
            f"{PREFIX}: FAIL: {path} is not a local directory — pass the baked snapshot, not a served alias"
        )
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]  # build-only dependency

        AutoTokenizer.from_pretrained(str(path))
    except Exception as exc:  # noqa: BLE001 - any load failure is fatal for the build
        raise SystemExit(f"{PREFIX}: FAIL: {path} cannot supply a tokenizer: {type(exc).__name__}: {exc}") from exc
    return f"tokenizer resolved locally from {path}"


def check_pin(manifest_path: Path, expect: str) -> str:
    """The export must record the same CLM revision the image pins."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"{PREFIX}: FAIL: cannot read {manifest_path}: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(f"{PREFIX}: FAIL: {manifest_path} is not valid JSON: {exc}") from exc
    recorded = manifest.get("clm_pin") if isinstance(manifest, dict) else None
    if recorded != expect:
        raise SystemExit(
            f"{PREFIX}: FAIL: dataset exported for CLM {recorded!r}, build pins {expect!r} — "
            "the adapter layout may have moved; re-verify train/adapters.py at the new revision"
        )
    return f"dataset pin matches the build ({expect})"


def main(argv: Sequence[str] | None = None) -> int:
    """Run every requested check; print one PASS line per check."""
    args = parse_args(argv)
    if args.entity:
        print(f"{PREFIX}: PASS: {check_entity(Path(args.entity))}", flush=True)
    if args.manifest:
        print(f"{PREFIX}: PASS: {check_pin(Path(args.manifest), args.expect_pin)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
