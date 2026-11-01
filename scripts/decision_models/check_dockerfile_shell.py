#!/usr/bin/env python
"""Validate the shell payloads of a Dockerfile before a build spends an hour failing.

``docker build --check`` does not evaluate shell syntax, and the shell a Dockerfile lands
in is dash (Debian) even when the developer's ``/bin/sh`` is bash. Two real defects got
through that gap in ``docker/Dockerfile.clm``:

* a stray ``'`` inside ``sh -c '...'`` terminated the quoted payload, so the rest was
  re-parsed unquoted and dash reported ``Syntax error: "(" unexpected``;
* a ``#`` inside the payload commented out every later command on the joined line,
  silently skipping a gate while the build still succeeded.

This checker extracts every shell-form ``RUN``/``CMD``/``ENTRYPOINT`` payload and reports,
per instruction: unbalanced/embedded quotes, a ``#`` outside quotes, and — when a dash
binary is available — the actual ``dash -n`` verdict. Exit 1 when any rule fails.

    python scripts/decision_models/check_dockerfile_shell.py docker/Dockerfile.clm
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

SHELL_INSTRUCTIONS = ("RUN", "CMD", "ENTRYPOINT")


@dataclass(frozen=True)
class Payload:
    """One shell-form instruction reduced to the program its shell will parse."""

    line: int
    instruction: str
    command: str
    payload: str


def joined_lines(text: str) -> list[tuple[int, str]]:
    """Logical Dockerfile lines with continuations joined, keeping the first line number."""
    out: list[tuple[int, str]] = []
    buffer = ""
    start = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not buffer:
            start = number
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        buffer += line
        out.append((start, buffer))
        buffer = ""
    if buffer:
        out.append((start, buffer))
    return out


def payloads(text: str) -> Iterator[Payload]:
    """Every shell-form RUN/CMD/ENTRYPOINT, with the ``sh -c '...'`` body when present."""
    for line, logical in joined_lines(text):
        stripped = logical.strip()
        if not stripped or stripped.startswith("#"):
            continue
        instruction = stripped.split(None, 1)[0]
        if instruction not in SHELL_INSTRUCTIONS:
            continue
        command = stripped[len(instruction) :].strip()
        while command.startswith("--"):  # Dockerfile flags such as --device=... / --mount=...
            command = command.split(None, 1)[1] if " " in command else ""
        payload = command
        marker = "sh -c '"
        if marker in command:
            head, _, tail = command.partition(marker)
            del head
            payload = tail[: tail.rfind("'")] if "'" in tail else tail
        yield Payload(line, instruction, command, payload)


def problems(payload: Payload) -> list[str]:
    """Rule violations for one payload (empty when it looks well formed)."""
    found: list[str] = []
    if payload.command.count("'") % 2 != 0:
        found.append("odd number of single quotes — a quoted span is unterminated")
    if "sh -c '" in payload.command and "'" in payload.payload:
        found.append("single quote inside the sh -c payload — it ends the quoted program early")
    if _unquoted_hash(payload.payload):
        found.append("'#' outside quotes — the shell will treat the rest of the line as a comment")
    return found


def _unquoted_hash(text: str) -> bool:
    """True when a '#' starts a comment, i.e. appears where the shell is not quoting."""
    single = False
    double = False
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not double:
            single = not single
            continue
        if char == '"' and not single:
            double = not double
            continue
        if char == "#" and not single and not double:
            return True
    return False


def dash_verdict(payload: str, dash: str) -> str | None:
    """``dash -n`` verdict for one payload, or None when it is not a complete program."""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as handle:
        handle.write(payload)
        path = handle.name
    try:
        result = subprocess.run([dash, "-n", path], capture_output=True, text=True, check=False)
    finally:
        Path(path).unlink(missing_ok=True)
    if result.returncode == 0:
        return None
    return (result.stderr or "dash -n failed").strip().splitlines()[0]


def check(path: Path, dash: str | None) -> int:
    """Validate one Dockerfile; returns a process exit code."""
    text = path.read_text(encoding="utf-8")
    failures = 0
    for payload in payloads(text):
        found = problems(payload)
        if dash is not None:
            verdict = dash_verdict(payload.payload, dash)
            if verdict:
                found.append(f"dash -n: {verdict}")
        label = f"{path.name}:{payload.line} ({payload.instruction})"
        if found:
            failures += 1
            print(f"FAIL {label}", file=sys.stderr)
            for problem in found:
                print(f"       {problem}", file=sys.stderr)
        else:
            print(f"ok   {label}")
    print(
        f"\n{path.name}: {failures} failing shell payload(s)"
        + ("" if dash else " — dash not found, quote/comment rules only")
    )
    return 1 if failures else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the checker."""
    parser = argparse.ArgumentParser(description="Validate Dockerfile shell payloads with dash.")
    parser.add_argument("dockerfiles", nargs="+", type=Path, help="Dockerfile(s) to validate")
    parser.add_argument("--dash", default="", help="dash binary (default: auto-detect dash or busybox)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate every given Dockerfile; non-zero when any payload fails."""
    args = parse_args(argv)
    dash = args.dash or shutil.which("dash")
    if not dash:
        busybox = shutil.which("busybox")
        dash = busybox if busybox else None
    worst = 0
    for path in args.dockerfiles:
        if not path.is_file():
            print(f"FAIL {path}: not a file", file=sys.stderr)
            return 2
        worst = max(worst, check(path, dash))
    return worst


if __name__ == "__main__":
    sys.exit(main())
