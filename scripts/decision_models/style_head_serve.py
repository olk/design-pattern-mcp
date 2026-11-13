#!/usr/bin/env python
"""Serve the Ch2 style seam's linear head over the TypeSafe ``/v1/systemone`` wire.

The wire is the one ``src/decisions/client.py`` already speaks, so this is a THIRD provider
behind the same seam (``decisions.style.provider = "linear"``), not a new client: the server
sends one ``choice`` question whose criterion keys are the 36-label answer space and reads
``answers[qid].choice`` / ``.confidence`` back.

Two routes, stdlib only:

* ``GET  /health``       → ``{"status": "ok", "classes": <n>, "model": <eval split>}``
* ``POST /v1/systemone`` → ``{"answers": {qid: {type, choice, probabilities, confidence}}}``

Design notes that are load-bearing:

* **Requirements are enforced, not guessed.** A request without a ``state`` string or with
  no questions is a 400 — the predecessor wire's callers all fail closed, and a 200 with an
  empty answer would let the seam act on nothing.
* **Criteria drift is a 422, not a soft answer.** The head is criteria-blind (it never sees
  the criterion texts), so a served criterion set that differs from the trained class set
  means train/serve drift — the server's prompt vocabulary moved and this image did not.
  Answering anyway would redistribute the score columns onto different questions; refusing
  makes the seam degrade to the LLM style with a visible cause.
* **``confidence`` is TypeSafe-style** (top probability minus the mean of the rest), which
  is what the seam floors on and what ``clm_calibrate.py --metric confidence`` reads.
  Reporting the top probability instead would silently change the meaning of every floor
  fitted on the CLM arms.
* **The bind address is the caller's choice.** The CMD passes ``--host 0.0.0.0`` so the
  compose bridge can forward; the loopback-only publish in
  ``docker/docker-compose.decision-models.yml`` is what confines it, exactly as
  ``docker/clm-entrypoint.sh`` does for clm-serve.

Delivery is one temperature-free ``predict_proba`` per request, so identical bytes in give
identical bytes out; the seam's own LRU cache sits in front of this.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger("style-head")

#: Artifact names written by ``train_style_head.py`` (the build puts both in ``/model``).
HEAD_FILENAME = "style_head.joblib"
REPORT_FILENAME = "style-head-report.json"

#: The wire's two routes.
HEALTH_PATH = "/health"
SYSTEMONE_PATH = "/v1/systemone"

#: Question id the seam asks under; the answer is echoed under whatever id it sent.
DEFAULT_QUESTION_ID = "architecture_style"

#: Decimal places kept in the reported probability map (the confidence stays exact).
PROBABILITY_PRECISION = 6


@dataclass(frozen=True)
class StyleHeadService:
    """The fitted head plus the label space it was trained on."""

    pipeline: Any
    classes: tuple[str, ...]
    eval_split: str

    def answer(self, state: str) -> dict[str, Any]:
        """One ``choice`` answer for ``state``: choice, per-class probabilities, confidence."""
        probability_row = self.pipeline.predict_proba([state])[0]
        model_classes = list(self.pipeline.classes_)
        # BY LABEL, never by column: LogisticRegression.classes_ is sorted, so index mapping
        # would permute the 36 answers into each other and still look plausible. A class the
        # head never trained on scores 0.0 — a legal answer no training row could teach.
        by_label = {str(label): float(prob) for label, prob in zip(model_classes, probability_row, strict=True)}
        scores = {label: by_label.get(label, 0.0) for label in self.classes}
        choice = max(self.classes, key=lambda label: scores[label])
        top = scores[choice]
        # TypeSafe-style confidence over the SERVED class set (see the module docstring).
        rest = [score for label, score in scores.items() if label != choice]
        confidence = top - (sum(rest) / len(rest)) if rest else top
        return {
            "type": "choice",
            "choice": choice,
            "probabilities": {label: round(score, PROBABILITY_PRECISION) for label, score in scores.items()},
            "confidence": confidence,
        }

    def criteria_drift(self, criteria: dict[str, Any]) -> str | None:
        """Why a served criterion set does not match this head, or None when it does."""
        served = set(criteria)
        trained = set(self.classes)
        if served == trained:
            return None
        missing = sorted(trained - served)
        extra = sorted(served - trained)
        return (
            f"criteria/class drift: {len(served)} served vs {len(trained)} trained; "
            f"missing={missing[:5]} unexpected={extra[:5]}"
        )


def answer_systemone(service: StyleHeadService, body: object) -> tuple[int, dict[str, Any]]:
    """The ``POST /v1/systemone`` decision: (status, response payload). Never raises."""
    if not isinstance(body, dict):
        return 400, {"error": "body must be a JSON object"}
    state = body.get("state")
    if not isinstance(state, str) or not state.strip():
        return 400, {"error": "state must be a non-empty string"}
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        return 400, {"error": "questions must be a non-empty object"}
    qid = next(iter(questions))
    question = questions[qid]
    if not isinstance(question, dict):
        return 400, {"error": f"questions[{qid!r}] must be an object"}
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        drift = service.criteria_drift(criteria)
        if drift is not None:
            return 422, {"error": drift}
    return 200, {"answers": {qid: service.answer(state)}}


def load_service(model_dir: Path) -> StyleHeadService:
    """Load the head plus its gate report; the report supplies the class order.

    ``classes`` comes from the artifact, not from a re-derived vocabulary: the runtime image
    carries no ``seam_prompts.py``, and a report and a checkpoint that disagree are exactly
    the drift this service refuses at request time.
    """
    import joblib  # image-only dependency, like scikit-learn behind it

    report_path = model_dir / REPORT_FILENAME
    if not report_path.is_file():
        raise SystemExit(f"style-head: {report_path} is missing (the image build gates on it)")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    classes = report.get("classes")
    if not isinstance(classes, list) or not classes or not all(isinstance(c, str) for c in classes):
        raise SystemExit(f"style-head: {report_path} carries no usable classes list")
    eval_split = report.get("eval_split")
    if not isinstance(eval_split, str) or not eval_split:
        raise SystemExit(f"style-head: {report_path} carries no eval_split")
    head_path = model_dir / HEAD_FILENAME
    if not head_path.is_file():
        raise SystemExit(f"style-head: {head_path} is missing")
    pipeline = joblib.load(head_path)
    trained = {str(label) for label in pipeline.classes_}
    if not trained.issubset(set(classes)):
        raise SystemExit(
            f"style-head: the head answers labels the report does not list: {sorted(trained - set(classes))[:5]}"
        )
    return StyleHeadService(pipeline=pipeline, classes=tuple(classes), eval_split=eval_split)


class _Handler(BaseHTTPRequestHandler):
    """Two routes over the pure functions above; every exception is a 500, never a crash."""

    service: StyleHeadService
    server_version = "style-head/1"

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, status: int, message: str) -> None:
        logger.error("%s %s -> %d: %s", self.command, self.path, status, message)
        self._respond(status, {"error": message})

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's dispatch name
        try:
            if self.path.split("?", 1)[0] != HEALTH_PATH:
                self._fail(404, f"unknown path {self.path!r}")
                return
            self._respond(
                200,
                {
                    "status": "ok",
                    "classes": len(self.service.classes),
                    "model": self.service.eval_split,
                },
            )
        except Exception as exc:  # noqa: BLE001 — the service must not die on one request
            self._fail(500, str(exc))

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's dispatch name
        try:
            if self.path.split("?", 1)[0] != SYSTEMONE_PATH:
                self._fail(404, f"unknown path {self.path!r}")
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw or b"null")
            except json.JSONDecodeError as exc:
                self._fail(400, f"invalid JSON body: {exc}")
                return
            status, payload = answer_systemone(self.service, body)
            if status != 200:
                self._fail(status, str(payload.get("error", "request rejected")))
                return
            self._respond(status, payload)
        except Exception as exc:  # noqa: BLE001 — the service must not die on one request
            self._fail(500, str(exc))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — signature is stdlib
        logger.info("%s - %s", self.address_string(), format % args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the linear Ch2 style head.")
    parser.add_argument("--model-dir", default="/model", help="Directory holding the head + report.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (compose passes 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8800, help="Bind port.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    service = load_service(Path(args.model_dir))
    _Handler.service = service
    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    logger.info(
        "style-head: serving %d classes (eval_split=%s, head=%s) on %s:%d",
        len(service.classes),
        service.eval_split,
        HEAD_FILENAME,
        args.host,
        args.port,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("style-head: interrupted, shutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
