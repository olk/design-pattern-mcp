#!/usr/bin/env python
"""Self-distil AnyJev Tacit-4B for this server's decision seams (runs in the image build).

The published Tacit recipe is self-distillation: the model writes decision problems,
answers them *with* its own thinking, and a student is trained to reproduce the
reasoning-conditioned label distribution in ONE forward. No human labels, no external
model, no teacher API — the checkpoint is its own teacher.

This script runs inside ``docker/Dockerfile.tacit`` on a GPU builder, but it is
standalone-runnable:

    python scripts/decision_models/tacit_self_distill.py \\
        --init-dir /models/tacit-4b --out /out/tacit-4b \\
        --seam-prompts src/decisions/seam_prompts.py

Prompts and vocabulary (the three question kinds this project actually serves) are
imported from ``seam_prompts.py`` — the server's own module, copied into the trainer
image — so the question bytes match production exactly.

The problems are the serving question set: one category ``choice`` plus the axis
``score`` questions per decision-part state, and one ``yes_no`` gate question per gate
state. Gate states are assembled from real serving material — the dataset's user
descriptions plus per-part aggregate payloads — in the exact ``tacit.gate_state`` shape
(phase line, then every declared payload key in order, capped identically), because the
server never sends any other shape.

Two more seams train from the same single checkpoint:

* **Ch2 style** (``build_style_problems``) — the SERVED two-stage choice over
  description-shaped states: stage 1 is the category question on the capped description,
  stage 2 the style question over one category's shortlist (the gateway caps ``choice`` at
  26 options, so the 36-way style question never exists as one item). The shortlist rotates
  over the nine categories so every option set trains. States are the corpus train hints
  first, then model-written descriptions (``generate_descriptions``) — the seam serves raw
  user prose.
* **J1 rerank** (``load_rerank_pairs`` / ``build_rerank_problems``) — one ``yes_no``
  relevance question per (statement, document) pair from ``qa/typed/rerank_pairs.jsonl``,
  whose bytes are exactly the serving items. Train-split rows only; the holdout pairs are
  eval material.

Retention is kind-aware. ``choice`` rows keep only teacher distributions whose max
clears the acting floor (``--answer-mass-floor``, defaulting to ``--confidence-floor``,
the category seam's ``confidence_floor``); ``score`` rows are kept unfiltered
(``--score-mass-floor``) because serving consumes the graded expected level; ``yes_no``
rows are always kept — the gate consumes ``p(Yes)`` as a graded probability and needs
calibration at its boundary, which any max-probability floor would remove. Below-floor
``choice`` rows are kept as a probe set: after training, one forward per probe row
measures the false-act rate (how often the student clears the acting floor where the
deliberating teacher did not) — the number retention-filtered eval cannot show. The
gate seam's own floor, for deployments that raise it alone, is
``--gate-confidence-floor``. A result that omits any option key is dropped instead of
rescaled: for the two threshold seams (category confidence, gate probability)
renormalizing a partial distribution fabricates the very number the server compares
against its floor.

Topic hints come from train-split rows only.

Sizing is max-effort by default: the fine-tune's aim is pattern-selection consistency and
quality, so the defaults spend the whole 4x3090 envelope on VOLUME and COVERAGE (all train
rerank pairs, ~1k style questions) rather than epochs — one epoch over self-distilled
targets is the recipe, and repeating it memorizes teacher noise. The teacher pass shards
across the visible GPUs (``teacher_answers``); retention and example construction stay
single-process, so sharding never changes the training set. An under-trained seam cannot
ship: the build fails when style or rerank problems are empty, when a serving kind keeps
fewer than ``--min-kind-rows`` rows, when any answer seam's student-vs-teacher agreement
drops below ``--min-seam-agreement``, or when the weights seam's expected-level MAE exceeds
``--max-score-mae``.

Output layout is what ``anyjev.serve`` expects: a HF causal-LM directory
(``config.json``, weights, tokenizer) that ``vllm serve`` can load.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger("tacit-self-distill")

#: Quality axes, in the order the server's weight vector uses (src/schemas/enums.py).
QUALITY_AXES: tuple[str, ...] = (
    "testability",
    "extensibility",
    "maintainability",
    "simplicity",
    "performance",
    "reliability",
    "security",
    "scalability",
)

#: One model-written design state, asked for in exactly this shape: the two header
#: lines are what ``src.pipeline._decision_part_state`` produces at serving time.
STATE_PROMPT = """You write short, concrete software design decision states.

Topic: {hint}

Write one state in exactly this shape, with no other text:

System goal: <one sentence about what the system must achieve>
Components: <two to four component names, comma separated>
Decision part: <one sentence naming one design decision that must be made>
Scope: <optional: one short sentence bounding the decision>
Constraints: <optional: zero to four constraints, semicolon separated>

State:"""

#: Description-shaped states for the Ch2 seam. Serving sends the RAW user description
#: (``_apply_decision_style(text, ...)``), so the style seam must not train only on the
#: corpus hint texts — the model writes more of them in the same shape.
DESCRIPTION_PROMPT = """You write short, concrete software system descriptions.

Topic: {hint}

Write one description in exactly this shape, with no other text: one paragraph of
2-5 sentences describing a software system a user wants design-pattern advice for —
what it must do, its main components, and one design concern. Plain prose, no lists,
no headings.

Description:"""

#: One gate question per this many teacher questions: the pipeline asks the gate once
#: per phase per run, the design questions once per decision part.
_GATE_SHARE = 10

#: Description-shaped states derived per this many teacher questions when
#: ``--style-states`` is left at its derived default (two style questions per state).
_STYLE_SHARE = 12

#: Mirrors ``DecisionSeamConfig.confidence_floor`` (the acting floor the seams gate on).
#: Pinned by ``tests/unit/test_decision_model_scripts.py``; override with
#: ``--confidence-floor`` (build arg ``TACIT_CONFIDENCE_FLOOR``) when a deployment
#: raises its floor, so training keeps exactly the rows the seam would act on.
_DEFAULT_CONFIDENCE_FLOOR = 0.5

#: The question kinds serving actually asks; every one must survive retention.
_SEAM_KINDS: tuple[str, ...] = ("choice", "score", "yes_no")


@dataclass(frozen=True)
class Example:
    """One supervised row: the serving prompt, its label token ids, the teacher target."""

    prompt: str
    kind: str
    label_ids: list[int]
    target: list[float]
    seam: str = ""  # category | weights | gate | style-stage1 | style-stage2 | rerank


class _Progress:
    """Count-based progress logs for a long build phase (one line per ~5%).

    Build logs are not TTYs, so this emits plain ``logger.info`` lines instead
    of a carriage-return bar — greppable and ``tee``-friendly.
    """

    def __init__(self, total: int, label: str, every: float = 0.05) -> None:
        self._total = max(1, total)
        self._label = label
        self._every = every
        self._next = every

    def advance(self, done: int) -> None:
        """Log when ``done`` crosses the next milestone (or finishes)."""
        frac = min(1.0, done / self._total)
        if frac >= self._next or done >= self._total:
            logger.info("%s %d/%d (%.0f%%)", self._label, min(done, self._total), self._total, frac * 100.0)
            while self._next <= frac:
                self._next += self._every


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line for the image build (all knobs are build args)."""
    parser = argparse.ArgumentParser(description="Self-distil Tacit-4B for the decision seams.")
    parser.add_argument("--init-dir", required=True, help="Downloaded Tacit-4B checkpoint directory.")
    parser.add_argument("--out", required=True, help="Output directory for the distilled checkpoint.")
    parser.add_argument("--seam-prompts", required=True, help="Path to the server's seam_prompts.py.")
    parser.add_argument("--seed-texts", default="", help="qa/qa.jsonl used as topic hints.")
    parser.add_argument("--repo", default="morriszjm/Tacit-4B", help="Checkpoint repo id (provenance).")
    parser.add_argument(
        "--problems",
        type=int,
        default=2000,
        help="Teacher question budget (one sequential CoT answer per question). State and "
        "question counts are derived from it; see --states.",
    )
    parser.add_argument(
        "--states",
        type=int,
        default=0,
        help="Design states to write (0 = derive from --problems). Raising it trades "
        "questions per state for state diversity at a fixed teacher budget; above "
        "--problems/2 the budget no longer holds and the run gets more expensive.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument(
        "--confidence-floor",
        type=float,
        default=_DEFAULT_CONFIDENCE_FLOOR,
        help="The seam config's confidence_floor: keep training rows the deployment would "
        "act on. Must match the served DESIGN_PATTERN_DECISIONS_*_CONFIDENCE_FLOOR.",
    )
    parser.add_argument(
        "--answer-mass-floor",
        type=float,
        default=None,
        help="min teacher max-prob for choice rows — defaults to --confidence-floor",
    )
    parser.add_argument(
        "--gate-confidence-floor",
        type=float,
        default=None,
        help="the reasoning-gate seam's own confidence_floor (yes_no act metrics) — defaults to "
        "--confidence-floor; set when the deployment raises only the gate's floor",
    )
    parser.add_argument(
        "--score-mass-floor",
        type=float,
        default=0.0,
        help="min teacher max-prob for score rows — graded targets must not be filtered",
    )
    parser.add_argument(
        "--min-kind-rows",
        type=int,
        default=1,
        help="Fail the build when a serving kind (choice/score/yes_no) keeps fewer rows "
        "than this: an under-trained seam must not ship.",
    )
    parser.add_argument("--eval-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cot-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--rerank-pairs",
        default="",
        help="qa/typed/rerank_pairs.jsonl — the served (statement, document) pairs. Required for "
        "the J1 seam: an under-trained seam must not ship (an empty value refuses the run "
        "unless --skip).",
    )
    parser.add_argument(
        "--rerank-problems",
        type=int,
        default=0,
        help="J1 teacher question budget; 0 = ALL train pairs (the max-effort default).",
    )
    parser.add_argument(
        "--style-states",
        type=int,
        default=0,
        help="Description-shaped states for the Ch2 seam; 0 = derive (problems // 12). Corpus "
        "train hints are always used first, generated descriptions fill the rest.",
    )
    parser.add_argument(
        "--teacher-shards",
        type=int,
        default=0,
        help="Teacher answer shards; 0 = auto (one per visible CUDA device), 1 = in-process.",
    )
    parser.add_argument(
        "--min-seam-agreement",
        type=float,
        default=0.6,
        help="Fail the build when an answer seam's (category/style-stage1/style-stage2/rerank/"
        "gate) student-vs-teacher eval agreement drops below this. Raise to the first "
        "max-effort run's achieved-minus-margin when it lands.",
    )
    parser.add_argument(
        "--max-score-mae",
        type=float,
        default=0.2,
        help="Fail the build when the weights seam's expected-level MAE exceeds this (the "
        "shipped artifact measured 0.110).",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Copy the released checkpoint to --out without training (out-of-band runs only; "
        "the image's no-GPU path is the runtime-released build target).",
    )
    # Hidden shard group: only teacher_answers' children pass these (see that function).
    parser.add_argument("--_answer-shard", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_shard-total", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_shard-problems", default="", help=argparse.SUPPRESS)
    parser.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    _validate_numeric_args(args)
    _validate_shard_args(args)
    return args


def _validate_numeric_args(args: argparse.Namespace) -> None:
    """Range-check every numeric knob and fill the two floor defaults."""
    if not 0.0 <= args.eval_frac < 1.0:
        raise SystemExit(f"--eval-frac {args.eval_frac} out of range [0.0, 1.0)")
    if not 0.0 <= args.confidence_floor <= 1.0:
        raise SystemExit(f"--confidence-floor {args.confidence_floor} out of range [0.0, 1.0]")
    if args.answer_mass_floor is None:
        args.answer_mass_floor = args.confidence_floor
    if args.gate_confidence_floor is None:
        args.gate_confidence_floor = args.confidence_floor
    if not 0.0 <= args.gate_confidence_floor <= 1.0:
        raise SystemExit(f"--gate-confidence-floor {args.gate_confidence_floor} out of range [0.0, 1.0]")
    for name in ("states", "min_kind_rows", "style_states", "teacher_shards"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} {getattr(args, name)} must be >= 0")
    if not 0.0 <= args.min_seam_agreement <= 1.0:
        raise SystemExit(f"--min-seam-agreement {args.min_seam_agreement} out of range [0.0, 1.0]")
    if args.max_score_mae < 0.0:
        raise SystemExit(f"--max-score-mae {args.max_score_mae} must be >= 0")


def _validate_shard_args(args: argparse.Namespace) -> None:
    """Validate the hidden shard group (only a teacher shard child passes it)."""
    if args._answer_shard is None:
        return
    if args._shard_total is None or args._shard_total < 1:
        raise SystemExit("--_answer-shard needs a positive --_shard-total")
    if not 0 <= args._answer_shard < args._shard_total:
        raise SystemExit("--_answer-shard must be within [0, --_shard-total)")
    if not args._shard_problems or not args._shard_out:
        raise SystemExit("--_answer-shard needs --_shard-problems and --_shard-out")


def _seed_everything(seed: int) -> None:
    """Seed the RNGs the build actually uses, so ``--seed`` means reproducible weights.

    ``random.Random(seed)`` alone only covers the state sampling and the training shuffle;
    generation and the QLoRA init draw from torch/CUDA (and ``transformers.set_seed`` for
    its own sampling helpers). A torch-less environment (unit tests, the ``--skip`` path on a
    CPU host) seeds what it can and returns instead of failing.
    """
    random.seed(seed)
    try:
        import torch
    except ImportError:  # host/stub runs without torch: only the RNG-free stages follow
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:  # pragma: no cover - transformers is present in the image
        pass


def _require_cuda() -> str:
    """Fail fast when the trainer cannot see a GPU, before loading any weights.

    The teacher is the dominant cost of this build (the state-writing pass plus one
    sequential CoT answer per question). Without a CUDA context it would grind
    through all of that on CPU for hours while nvtop stays empty — abort in
    seconds instead. Returns the device-0 name for the run banner.

    Unit tests never reach this: they stub ``anyjev`` and ``main`` short of the
    real ``Tacit.from_pretrained`` path (``main`` with fakes, or ``--skip``).
    """
    try:
        import torch
    except ImportError:
        raise SystemExit("torch is not installed; the trainer needs the GPU image stage")
    if not torch.cuda.is_available():
        raise SystemExit(
            "torch.cuda.is_available() is false: no GPU visible to the trainer "
            "(check RUN --device + docker build --allow on the trainer host)"
        )
    name = str(torch.cuda.get_device_name(0))
    logger.info("cuda preflight: %d device(s), device 0 = %s", torch.cuda.device_count(), name)
    return name


def load_seam_prompts(path: Path) -> ModuleType:
    """Import the server's ``seam_prompts.py`` by path (stdlib-only, no package needed)."""
    spec = importlib.util.spec_from_file_location("seam_prompts", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load seam prompts from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_topic_hints(seed_texts: Path, limit: int, max_chars: int = 1500) -> list[str]:
    """Topic hints from the repo's seed QA pairs (the catalog vocabulary, not labels).

    Holdout rows never become hints: the benchmark's holdout scenarios must not leak
    into the states the student is trained on. The full text is kept (bounded by the
    gate-state cap) because serving puts the WHOLE user description into the ``analyze``
    gate payload; callers that want a short topic line truncate at use.
    """
    hints: list[str] = []
    if seed_texts.is_file():
        with seed_texts.open(encoding="utf-8") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except ValueError:
                    continue
                text = row.get("question") or row.get("state")
                if not (isinstance(text, str) and text) or row.get("split") != "train":
                    continue
                hints.append(text[:max_chars])
                if len(hints) >= limit:
                    break
    return hints


def generate_states(
    tacit: Any,
    hints: Sequence[str],
    problems: int,
    seed: int,
    max_new_tokens: int,
    max_state_chars: int,
) -> list[str]:
    """Prompt the model for up to ``problems`` DISTINCT design states in the serving shape."""
    import torch

    rng = random.Random(seed)
    prompts = [
        STATE_PROMPT.format(hint=hints[i % len(hints)][:400] if hints else "a web service") for i in range(problems)
    ]
    model = tacit.engine.model
    tokenizer = tacit.tok
    states: list[str] = []
    batch_size = max(1, getattr(tacit, "batch_size", 8))
    logger.info("phase 1/3: writing %d design states", len(prompts))
    progress = _Progress(len(prompts), "state-writing")
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in chunk
            ]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
            flat = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
            for text in flat:
                state = _normalise_state(text, max_state_chars)
                if state is not None:
                    states.append(state)
            progress.advance(start + len(chunk))
    # Distinct states only: a repeated state would teach the same nine questions twice
    # while the build is paying for state diversity.
    unique = list(dict.fromkeys(states))
    if len(unique) < len(states):
        logger.info("dropped %d duplicate design states", len(states) - len(unique))
    rng.shuffle(unique)
    return unique[:problems]


def _normalise_state(text: str, max_chars: int) -> str | None:
    """Keep only model output that has the serving state's shape (header lines only).

    Serving always leads with ``System goal:`` (``src.pipeline._decision_part_state``)
    and truncates the joined lines at ``max_chars``; both are mirrored here so only
    bytes the server actually sends become training rows.
    """
    body = text.strip()
    if "Decision part:" not in body:
        return None
    kept: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            if kept:
                break
            continue
        if stripped.startswith(("System goal:", "Components:", "Decision part:", "Scope:", "Constraints:")):
            kept.append(stripped)
    state = "\n".join(kept)[:max_chars]
    if not state.startswith("System goal:"):
        return None
    return state if "Decision part:" in state and len(state) >= 40 else None


def generate_descriptions(
    tacit: Any,
    hints: Sequence[str],
    count: int,
    max_new_tokens: int,
    max_state_chars: int,
) -> list[str]:
    """Prompt the model for ``count`` DISTINCT description-shaped states (serving bytes).

    Mirrors :func:`generate_states` (same batching, temperature 0.9, dedupe, cap) but for
    free prose: the Ch2 seam serves the raw user description capped at
    ``DECISION_STATE_MAX_CHARS``, so its training states must be raw prose in that shape,
    not header-line part states. Topic hints come from train-split rows only (the
    :func:`load_topic_hints` rule).
    """
    import torch

    prompts = [
        DESCRIPTION_PROMPT.format(hint=hints[i % len(hints)][:400] if hints else "a web service") for i in range(count)
    ]
    model = tacit.engine.model
    tokenizer = tacit.tok
    batch_size = max(1, getattr(tacit, "batch_size", 8))
    descriptions: list[str] = []
    progress = _Progress(len(prompts), "description-writing")
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in chunk
            ]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
            flat = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
            for text in flat:
                description = text.strip()[:max_state_chars]
                if len(description) >= 80:
                    descriptions.append(description)
            progress.advance(start + len(chunk))
    unique = list(dict.fromkeys(descriptions))
    if len(unique) < len(descriptions):
        logger.info("dropped %d duplicate descriptions", len(descriptions) - len(unique))
    return unique[:count]


def _state_fields(state: str) -> dict[str, str]:
    """One design state's header lines as ``{key: value}`` (first occurrence wins)."""
    fields: dict[str, str] = {}
    for line in state.splitlines():
        key, separator, value = line.partition(":")
        if separator and value.strip():
            fields.setdefault(key.strip(), value.strip())
    return fields


def _gate_payload(
    phase: str, *, description: str, goal: str, parts: Sequence[str], applications: Sequence[str], primary: str
) -> dict[str, str]:
    """The payload the pipeline hands ``gate_state`` for ``phase`` (src/pipeline.py:1680-1808).

    Key sets and order mirror the inline pipeline payloads; ``seams.GATE_PHASE_PAYLOAD_KEYS``
    is the declared version of the same table and the sync test pins the two together.
    """
    if phase == "analyze":
        return {"description": description}
    if phase == "weights":
        return {"description": description, "restated_goal": goal, "decision_parts": "; ".join(parts)}
    if phase == "generate":
        return {"restated_goal": goal, "applications": "; ".join(applications)}
    return {
        "primary_pattern": primary,
        "applications": "; ".join(applications),
        "parts": "; ".join(parts),
    }


def _slug(text: str, words: int = 2) -> str:
    """A serving-shaped id slug (``p-delivery``-style) from a statement or name."""
    keep: list[str] = []
    for word in text.lower().split():
        cleaned = "".join(character for character in word if character.isalnum())
        if cleaned:
            keep.append(cleaned)
        if len(keep) == words:
            break
    return "-".join(keep) or "part"


def build_gate_states(
    states: Sequence[str],
    hints: Sequence[str],
    seams: ModuleType,
    count: int,
) -> list[str]:
    """Assemble ``count`` gate states from serving material, not model invention.

    The server never sends a model-written gate state: it sends a phase plus the payload
    the pipeline already computed, so ``analyze`` carries the WHOLE user description and
    the other phases carry joined per-part aggregates. A trainer that asks the model to
    write "one short state" trains the gate on inputs the server never produces, so these
    are composed instead: the description from the dataset's own descriptions, and the
    aggregates from the generated decision parts.

    Composition mirrors ``tacit.gate_state`` exactly, cap included: a payload whose joined
    text exceeds ``GATE_STATE_MAX_CHARS`` is truncated the way serving truncates it, which
    legitimately drops trailing keys. That is the shape production sends for a long user
    description, so it is the shape the gate must learn — not a budgeted variant. Phase
    and part-count vary independently so each phase sees the whole 1-4 part range.
    """
    kept: list[str] = []
    dropped = 0
    phases = seams.GATE_PHASES
    for index in range(count):
        phase = phases[index % len(phases)]
        # Decoupled from the phase cycle: binding them would give ``weights`` 2 parts,
        # ``generate`` 3 and ``evaluate`` 4 forever.
        spread = 1 + ((index // len(phases)) % 4)  # a decomposition yields one to four parts
        group = [states[(index + offset) % len(states)] for offset in range(spread)] if states else []
        goal = ""
        statements: list[str] = []
        component_names: list[str] = []
        for state in group:
            fields = _state_fields(state)
            goal = goal or fields.get("System goal", "")
            statement = fields.get("Decision part")
            if statement:
                statements.append(statement)
            components = fields.get("Components")
            if components:
                first = components.split(",")[0].strip()
                if first:
                    component_names.append(first)
        used: set[str] = set()
        slugs: list[str] = []
        parts: list[str] = []
        for statement in statements:
            slug = _slug(statement)
            unique = slug
            suffix = 2
            while unique in used:
                unique = f"{slug}{suffix}"
                suffix += 1
            used.add(unique)
            slugs.append(unique)
            parts.append(f"p-{unique}: {statement}")
        parts = parts or ["p-design: design"]
        slugs = slugs or ["design"]
        fallback = " ".join(statements[0].split()[:3]) if statements else "service"
        names = [component_names[offset] if offset < len(component_names) else fallback for offset in range(len(parts))]
        # Serving part/application ids are slug-shaped ("p-delivery",
        # "p-delivery-outbox" — the pipeline gate payloads), never enumerated "p1"/"a1".
        applications = [f"p-{slug}-{_slug(name)}: {name} for p-{slug}" for slug, name in zip(slugs, names, strict=True)]
        description = hints[index % len(hints)] if hints else (goal or "a web service")
        payload = _gate_payload(
            phase,
            description=description,
            goal=goal or "the system works",
            parts=parts,
            applications=applications,
            primary=names[0],
        )
        raw = "\n".join([f"phase: {phase}", *(f"{key}: {value}" for key, value in payload.items())])
        gate_text = _normalise_gate_state(raw[: seams.GATE_STATE_MAX_CHARS], seams)
        if gate_text is None:
            dropped += 1
            continue
        kept.append(gate_text)
    if dropped:
        logger.warning("dropped %d gate states that were not serving-shaped", dropped)
    return kept


def _gate_payload_prefix(payload_text: str, required: Sequence[str]) -> dict[str, str]:
    """Split a gate payload into its declared fields, in order.

    ``tacit.gate_state`` writes ``key: value`` with the value's own newlines intact, and a
    real user description contains newlines — so a field ends at the next declared key
    marker, never at the next newline. Returns the leading run of recognised fields; an
    absent or misordered key stops the prefix (the caller decides whether a short prefix
    is legitimate).
    """
    values: dict[str, str] = {}
    cursor = 0
    for position, key in enumerate(required):
        marker = f"{key}: "
        if not payload_text.startswith(marker, cursor):
            break
        start = cursor + len(marker)
        end = len(payload_text)
        if position + 1 < len(required):
            found = payload_text.find(f"\n{required[position + 1]}: ", start)
            if found != -1:
                end = found
        values[key] = payload_text[start:end]
        cursor = end + 1
    return values


def _normalise_gate_state(text: str, seams: ModuleType) -> str | None:
    """Keep only text with a serving ``gate_state`` shape, or ``None``.

    Serving joins the phase line and every scalar payload key of that phase, in pipeline
    order, and THEN truncates the whole string at ``GATE_STATE_MAX_CHARS``. A real state
    therefore always carries a PREFIX of the declared keys: all of them when the payload
    fits, fewer when the cap cut a trailing key away — a long ``description`` legitimately
    ends the ``weights`` state after its first line. Rejecting short payloads would reject
    exactly the states the server sends for long user descriptions.

    So: the present keys must be that prefix, in order, with no gaps and no undeclared
    key; and a short prefix is only accepted when the text sits exactly at the cap, which
    is the proof that truncation (not a malformed payload) dropped the rest.
    """
    cap = seams.GATE_STATE_MAX_CHARS
    body = text.strip()
    header, _, payload_text = body.partition("\n")
    if not header.startswith("phase:") or not payload_text:
        return None
    phase = header[len("phase:") :].strip()
    required = seams.GATE_PHASE_PAYLOAD_KEYS.get(phase)
    if required is None:
        return None
    values = _gate_payload_prefix(payload_text, required)
    if not values:
        return None
    rebuilt = "\n".join([f"phase: {phase}", *(f"{key}: {value}" for key, value in values.items())])
    if len(rebuilt) > cap:
        return None
    # A COMPLETE payload must reproduce the text exactly; an incomplete one is only a
    # serving shape when the cap truncated it, never when a key is simply missing.
    if tuple(values) == tuple(required) and rebuilt == body:
        return rebuilt
    tail = body[len(rebuilt) :]
    capped_tail = tail.lstrip("\n")
    return body if len(body) == cap and body.startswith(rebuilt) and "\n" not in capped_tail else None


def build_problems(
    states: Sequence[str],
    seams: ModuleType,
    gate_states: Sequence[str],
    axes_per_state: int = len(QUALITY_AXES),
) -> list[dict[str, Any]]:
    """The serving question set: choice + an axis window per state, gate per gate state.

    ``axes_per_state`` below eight trades per-state breadth for state diversity at a fixed
    teacher budget. The window rotates with the state index, so a budget of three axes per
    state still covers all eight axes evenly across the corpus instead of teaching the same
    three and ignoring the rest.
    """
    problems: list[dict[str, Any]] = []
    category_question = seams.category_question()
    window = max(1, min(axes_per_state, len(QUALITY_AXES)))
    for index, state in enumerate(states):
        problems.append(
            {
                "state": state,
                "question": category_question,
                "options": list(seams.CATEGORY_SLUGS),
                "kind": "choice",
                "seam": "category",
            }
        )
        for offset in range(window):
            axis = QUALITY_AXES[(index + offset) % len(QUALITY_AXES)]
            problems.append(
                {
                    "state": state,
                    "question": seams.axis_question(axis),
                    "options": list(seams.WEIGHT_LEVELS),
                    "kind": "score",
                    "seam": "weights",
                }
            )
    gate_question = seams.gate_question()
    for state in gate_states:
        problems.append(
            {"state": state, "question": gate_question, "options": ["Yes", "No"], "kind": "yes_no", "seam": "gate"}
        )
    return problems


def build_style_problems(states: Sequence[str], seams: ModuleType) -> list[dict[str, Any]]:
    """Ch2 two-stage style questions over description-shaped states.

    Stage 1 is the served category question on the capped description; stage 2 is the
    served style question over one category's shortlist. The shortlist rotates
    deterministically over the nine categories so every stage-2 option set is trained,
    independent of the teacher's own stage-1 answers (which would couple training to
    teacher errors and cost a sequential second pass).
    """
    problems: list[dict[str, Any]] = []
    category_question = seams.category_question()
    slugs = seams.CATEGORY_SLUGS
    for index, raw_state in enumerate(states):
        state = raw_state[: seams.DECISION_STATE_MAX_CHARS]
        problems.append(
            {
                "state": state,
                "question": category_question,
                "options": list(slugs),
                "kind": "choice",
                "seam": "style-stage1",
            }
        )
        category = slugs[index % len(slugs)]
        styles = seams.styles_for_category(category)
        problems.append(
            {
                "state": state,
                "question": seams.style_question(styles),
                "options": [*styles, seams.STYLE_NONE_KEY],
                "kind": "choice",
                "seam": "style-stage2",
            }
        )
    return problems


def load_rerank_pairs(path: Path) -> list[dict[str, Any]]:
    """Train-split (statement, document) pairs in serving bytes; holdout never trains."""
    pairs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except ValueError:
                continue
            if row.get("split") != "train":
                continue
            statement, document = row.get("statement"), row.get("document")
            if isinstance(statement, str) and isinstance(document, str) and statement and document:
                pairs.append({"statement": statement, "document": document})
    return pairs


def build_rerank_problems(pairs: Sequence[dict[str, Any]], seams: ModuleType, budget: int) -> list[dict[str, Any]]:
    """J1 relevance questions: one ``yes_no`` per (statement, document) pair.

    ``budget <= 0`` keeps every train pair (the max-effort default); a positive budget
    takes a deterministic stride over file order (scenario-major) so a reduced run still
    spreads across scenarios instead of exhausting the first ten.
    """
    selected = pairs if budget <= 0 else pairs[:: max(1, len(pairs) // budget)][:budget]
    return [
        {
            "state": seams.rerank_state(pair["statement"], pair["document"]),
            "question": seams.RERANK_RELEVANCE_QUESTION,
            "options": ["Yes", "No"],
            "kind": "yes_no",
            "seam": "rerank",
        }
        for pair in selected
    ]


def retention_floor(kind: str, answer_floor: float, score_floor: float) -> float:
    """Score rows keep graded mass (serving consumes the expected level); ``choice``
    rows keep only what clears the acting floor the category seam gates on.

    ``yes_no`` rows are never filtered: the gate consumes ``p(Yes)`` as a graded
    probability and needs calibration at its boundary, and a max-probability floor —
    which a raised ``--confidence-floor`` would otherwise impose — drops exactly the
    near-boundary rows the gate must learn. That is also why the drop telemetry is
    per kind.
    """
    if kind == "score":
        return score_floor
    if kind == "choice":
        return answer_floor
    return 0.0


def answer_problems(tacit: Any, problems: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Answer phase: raw ``decide_batch`` results, in problem order (no retention).

    Only the four wire keys are sent — ``seam`` is trainer bookkeeping and the gateway
    validates its items. Split from example construction so the pass can be sharded
    across GPUs (see :func:`teacher_answers`) without changing the training set.
    """
    results: list[dict[str, Any]] = []
    batch_size = max(1, getattr(tacit, "batch_size", 8))
    progress = _Progress(len(problems), "teacher")
    for start in range(0, len(problems), batch_size):
        chunk = problems[start : start + batch_size]
        wire = [{key: item[key] for key in ("state", "question", "options", "kind")} for item in chunk]
        answers = tacit.decide_batch(wire)
        if len(answers) != len(wire):
            raise SystemExit(f"teacher returned {len(answers)} decisions for {len(wire)} problems")
        results.extend(dict(result) for result in answers)
        progress.advance(start + len(chunk))
    return results


def examples_from_results(
    tacit: Any,
    problems: Sequence[dict[str, Any]],
    results: Sequence[dict[str, Any]],
    answer_mass_floor: float,
    score_mass_floor: float,
) -> tuple[list[Example], dict[str, dict[str, int]], list[Example]]:
    """Turn answered problems into supervised rows, keeping the sharp distributions.

    A result that omits any option is dropped (``shape``), never padded with zeros and
    rescaled: the renormalized target would silently fabricate the label — certainty on
    ``yes_no``, inflated confidence on ``choice`` — and the row would still be counted as
    kept. Serving tolerates the same omission (``_expected_level`` renormalizes by the
    mass it saw, ``gate`` returns ``None``), which is why the trainer must not convert it
    into a supervision signal.

    The third return value is the false-act probe: below-floor ``choice`` rows (route and
    shape intact) carried as full Examples, so one forward per row can later measure how
    often the student clears the acting floor where the deliberating teacher did not.
    """
    from anyjev.readout import DEFAULT_SYSTEM, build_prompt, label_ids_for_perm, render_chat, resolve_labels
    from anyjev.tacit import make_question

    examples: list[Example] = []
    probe: list[Example] = []
    dropped: dict[str, dict[str, int]] = {kind: {"unrouted": 0, "shape": 0, "unsharp": 0} for kind in _SEAM_KINDS}
    for item, result in zip(problems, results, strict=True):
        kind = item["kind"]
        if result.get("route") != "cot":
            dropped[kind]["unrouted"] += 1
            continue
        probabilities: dict[str, float] = result.get("probs") or {}
        if any(option not in probabilities for option in item["options"]):
            dropped[kind]["shape"] += 1
            continue
        target = [float(probabilities[option]) for option in item["options"]]
        total = sum(target)
        if total <= 0.0:
            dropped[kind]["unrouted"] += 1
            continue
        target = [value / total for value in target]
        question = make_question(item["question"], item["options"], item["kind"])
        labels, ids = resolve_labels(tacit.tok, question)
        permutation = list(range(question.k))
        spec = build_prompt(item["state"], question, permutation, DEFAULT_SYSTEM, labels)
        example = Example(
            prompt=render_chat(tacit.tok, spec),
            kind=kind,
            label_ids=[int(token) for token in label_ids_for_perm(question, ids, permutation)],
            target=target,
            seam=str(item.get("seam", "")),
        )
        if max(target) < retention_floor(kind, answer_mass_floor, score_mass_floor):
            dropped[kind]["unsharp"] += 1
            if kind == "choice":
                # Below-floor choice territory is where serving falls back to the
                # LLM; keep the row to measure the student's false-act rate there.
                probe.append(example)
            continue
        examples.append(example)
    logger.info(
        "teacher kept %d of %d problems %s (%d below-floor choice probes)",
        len(examples),
        len(problems),
        dropped,
        len(probe),
    )
    return examples, dropped, probe


def _load_teacher(args: argparse.Namespace) -> Any:
    """The deliberating teacher: every decision escalates to the model's own thinking.

    ``Tacit(adaptive=True, tau=<huge>, max_cot_share=None)`` makes each ``probs`` the
    reasoning-conditioned distribution over the option labels — the teacher target for the
    one-forward student. Loaded by the parent process and, with sharding, by each child.
    """
    from anyjev import Tacit

    return Tacit.from_pretrained(
        str(args.init_dir),
        engine="transformers",
        adaptive=True,
        tau=float("inf"),  # every decision escalates: the teacher must think
        max_cot_share=None,
        cot_window=None,
        cot_max_tokens=args.cot_max_tokens,
        batch_size=max(1, args.batch),
        device_map={"": 0},
    )


def _shard_command(
    args: argparse.Namespace, shard: int, total_shards: int, problems_path: Path, out_path: Path
) -> list[str]:
    """The child argv for one teacher shard.

    Carries EVERY required flag (``--init-dir``/``--out``/``--seam-prompts``) plus the
    teacher's generation knobs, because the child re-parses this argv with the parent's own
    ``parse_args``: a required flag missing here aborts the child in argparse (exit 2) and
    the parent refuses the partial answer set. ``--seam-prompts`` is unused by the child
    (it answers problems, it does not build them) and is passed only to satisfy that
    contract. Pinned by a test that runs the real ``parse_args`` over this argv.
    """
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_answer-shard",
        str(shard),
        "--_shard-total",
        str(total_shards),
        "--_shard-problems",
        str(problems_path),
        "--_shard-out",
        str(out_path),
        "--init-dir",
        str(args.init_dir),
        "--out",
        str(args.out),
        "--seam-prompts",
        str(args.seam_prompts),
        "--repo",
        str(args.repo),
        "--cot-max-tokens",
        str(args.cot_max_tokens),
        "--batch",
        str(args.batch),
    ]


def teacher_answers(
    tacit: Any, args: argparse.Namespace, problems: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Answer every problem, sharded across GPUs when more than one is visible.

    Each shard re-invokes this script with ``--_answer-shard``: the child pins
    ``CUDA_VISIBLE_DEVICES`` to its device, loads its own teacher replica, answers the
    strided slice ``problems[i::n]``, and writes ``{"index": <global>, "result": {...}}``
    JSONL. The parent merges by index and verifies full coverage — retention and example
    construction stay single-process, so sharding changes only wall time, never the
    training set. ``--teacher-shards 1`` (or one visible device) keeps the in-process pass,
    which reuses the teacher already resident for state writing. A child exiting non-zero
    aborts the build (no partial answer set).
    """
    import torch

    if not torch.cuda.is_available():
        total_shards = 1
    else:
        total_shards = args.teacher_shards or min(torch.cuda.device_count(), 8)
    if total_shards <= 1:
        return answer_problems(tacit, problems), 1
    logger.info("answering %d problems with %d teacher shards", len(problems), total_shards)
    workdir = Path(tempfile.mkdtemp(prefix="tacit-shards-"))
    problems_path = workdir / "problems.json"
    problems_path.write_text(json.dumps(list(problems)), encoding="utf-8")
    children: list[subprocess.Popen[bytes]] = []
    try:
        for shard in range(total_shards):
            out_path = workdir / f"shard-{shard}.jsonl"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(shard)}
            children.append(subprocess.Popen(_shard_command(args, shard, total_shards, problems_path, out_path), env=env))
        for child in children:
            if child.wait() != 0:
                raise SystemExit("a teacher shard failed — refusing to train on a partial answer set")
        merged: dict[int, dict[str, Any]] = {}
        for shard in range(total_shards):
            shard_path = workdir / f"shard-{shard}.jsonl"
            for raw_line in shard_path.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip():
                    continue
                entry = json.loads(raw_line)
                merged[int(entry["index"])] = dict(entry["result"])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if sorted(merged) != list(range(len(problems))):
        raise SystemExit(f"teacher shards covered {len(merged)}/{len(problems)} problems — aborting")
    return [merged[index] for index in range(len(problems))], total_shards


def _answer_shard(args: argparse.Namespace) -> int:
    """Child path: answer ``problems[shard::total]`` and write index-tagged JSONL."""
    problems: list[dict[str, Any]] = json.loads(Path(args._shard_problems).read_text(encoding="utf-8"))
    slice_ = problems[args._answer_shard :: args._shard_total]
    logger.info(
        "teacher shard %d/%d: answering %d of %d problems",
        args._answer_shard + 1,
        args._shard_total,
        len(slice_),
        len(problems),
    )
    results = answer_problems(_load_teacher(args), slice_)
    if len(results) != len(slice_):
        raise SystemExit(f"shard answered {len(results)} of {len(slice_)} problems")
    out = Path(args._shard_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for offset, result in enumerate(results):
            index = args._answer_shard + offset * args._shard_total
            handle.write(json.dumps({"index": index, "result": result}) + "\n")
    return 0


def train_student(init_dir: Path, examples: Sequence[Example], args: argparse.Namespace) -> Any:
    """QLoRA the 4-bit base so one forward reproduces the teacher distribution."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(str(init_dir))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantisation = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(init_dir), quantization_config=quantisation, device_map={"": 0}, torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    device = model.device
    rng = random.Random(args.seed)
    model.train()
    accum = max(1, args.accum)
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        total_steps = max(1, (len(order) + args.batch - 1) // args.batch)
        logger.info("phase 3/3: QLoRA epoch %d/%d (%d steps)", epoch + 1, args.epochs, total_steps)
        progress = _Progress(total_steps, f"epoch {epoch + 1}/{args.epochs}")
        pending = 0
        for step, start in enumerate(range(0, len(order), args.batch), start=1):
            rows = [examples[index] for index in order[start : start + args.batch]]
            # ``_batch_loss`` is already the mean over rows, so only the accumulation
            # divisor belongs here; dividing by len(rows) again would scale the whole
            # epoch's step size by 1/batch.
            loss = _batch_loss(model, tokenizer, rows, device) / accum
            loss.backward()
            pending += 1
            if pending == accum or step == total_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            if step % 50 == 0 or step == total_steps:
                logger.info("epoch %d step %d/%d loss %.4f", epoch, step, total_steps, float(loss.detach()) * accum)
            progress.advance(step)
    return model


def _batch_loss(model: Any, tokenizer: Any, rows: Sequence[Example], device: Any) -> Any:
    """Mean KL-style soft cross-entropy between the student and the teacher labels."""
    import torch

    encoded = tokenizer([row.prompt for row in rows], return_tensors="pt", padding=True, add_special_tokens=False).to(
        device
    )
    logits = model(**encoded).logits[:, -1, :].float()
    losses = []
    for index, row in enumerate(rows):
        label_logits = logits[index, torch.tensor(row.label_ids, device=logits.device)]
        log_probabilities = torch.log_softmax(label_logits, dim=-1)
        target = torch.tensor(row.target, dtype=log_probabilities.dtype, device=log_probabilities.device)
        losses.append(-(target * log_probabilities).sum())
    return torch.stack(losses).mean()


def _expected_level(probs: Sequence[float]) -> float:
    """Σ i·p_i / (len(levels) - 1) over WEIGHT_LEVELS positions, as serving computes it."""
    top = len(probs) - 1
    return sum(index * value for index, value in enumerate(probs)) / top


#: Seams whose student-vs-teacher agreement the build gates on (every ANSWER seam; the
#: weights seam is gated on its expected-level MAE instead).
_GATED_SEAMS: tuple[str, ...] = ("category", "style-stage1", "style-stage2", "rerank", "gate")


def _fidelity_metrics(
    samples: Sequence[tuple[list[float], list[float], str, str]], choice_floor: float, gate_floor: float
) -> dict[str, dict[str, Any]]:
    """Distillation fidelity per SEAM: agreement, Brier, and the seam's own decision metric.

    ``choice`` rows (category and both style stages) also report ``floor_accuracy``: the
    student agrees AND clears the acting floor, which is what those seams act on. The
    ``gate`` seam reports agreement on the ACT decision at ``gate_floor`` (options[0] is
    ``"Yes"`` on both sides) — a student that fires where the teacher would not is a false
    act even when the labels agree. The ``weights`` seam reports ``expected_level_mae``;
    ``rerank`` reports agreement alone, because serving consumes the whole ``p(Yes)``
    distribution rather than a thresholded answer.
    """
    by_seam: dict[str, list[tuple[list[float], list[float], str]]] = {}
    for student_probs, teacher_target, kind, seam in samples:
        by_seam.setdefault(seam, []).append((student_probs, teacher_target, kind))
    metrics: dict[str, dict[str, Any]] = {}
    for seam, rows in by_seam.items():
        n = len(rows)
        if n == 0:
            metrics[seam] = {"n": 0}
            continue
        agreements = sum(
            1 for student, teacher, _kind in rows if student.index(max(student)) == teacher.index(max(teacher))
        )
        brier = (
            sum(sum((s - t) ** 2 for s, t in zip(student, teacher, strict=True)) for student, teacher, _kind in rows)
            / n
        )
        entry: dict[str, Any] = {"n": n, "agreement": agreements / n, "brier": brier}
        kinds = {kind for _student, _teacher, kind in rows}
        if kinds == {"choice"}:
            floored = sum(
                1
                for student, teacher, _kind in rows
                if student.index(max(student)) == teacher.index(max(teacher)) and max(student) >= choice_floor
            )
            entry["floor_accuracy"] = floored / n
        elif kinds == {"yes_no"} and seam == "gate":
            acted = sum(
                1 for student, teacher, _kind in rows if (student[0] >= gate_floor) == (teacher[0] >= gate_floor)
            )
            entry["floor_accuracy"] = acted / n
        elif kinds == {"score"}:
            mae = sum(abs(_expected_level(student) - _expected_level(teacher)) for student, teacher, _kind in rows) / n
            entry["expected_level_mae"] = mae
        metrics[seam] = entry
    return metrics


def seam_gate_failures(
    metrics: dict[str, dict[str, Any]], min_agreement: float, max_score_mae: float
) -> list[str]:
    """Reasons the trained student must not ship; empty when every seam clears its bar.

    Every answer seam must be present in the eval split and clear
    ``--min-seam-agreement``; the weights seam must also stay under ``--max-score-mae``.
    A missing seam is a failure, not a skip: an eval split that lost a seam cannot certify
    it.
    """
    failures: list[str] = []
    for seam in _GATED_SEAMS:
        entry = metrics.get(seam)
        if entry is None or not entry.get("n"):
            failures.append(f"seam {seam!r} has no eval rows — the eval split lost a seam")
            continue
        agreement = float(entry["agreement"])
        if agreement < min_agreement:
            failures.append(
                f"seam {seam!r} student-teacher agreement {agreement:.3f} < --min-seam-agreement {min_agreement}"
            )
    weights = metrics.get("weights")
    if weights is None or not weights.get("n"):
        failures.append("seam 'weights' has no eval rows — the eval split lost a seam")
    else:
        mae = weights.get("expected_level_mae")
        if mae is None:
            failures.append("seam 'weights' eval carries no expected_level_mae")
        elif float(mae) > max_score_mae:
            failures.append(f"weights seam expected-level MAE {float(mae):.3f} > --max-score-mae {max_score_mae}")
    return failures


def _eval_tokenizer(init_dir: Path) -> Any:
    """Left-padded tokenizer for one-forward reads (the serving readout shape)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(init_dir))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _forward_probs(model: Any, tokenizer: Any, rows: Sequence[Example], batch_size: int) -> list[list[float]]:
    """One forward per row: softmax over the row's label tokens, as the gateway reads."""
    import torch

    device = next(model.parameters()).device
    model.eval()
    probs: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(rows), max(1, batch_size)):
            chunk = rows[start : start + max(1, batch_size)]
            encoded = tokenizer(
                [row.prompt for row in chunk], return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
            logits = model(**encoded).logits[:, -1, :].float()
            for index, row in enumerate(chunk):
                label_logits = logits[index, torch.tensor(row.label_ids, device=logits.device)]
                probs.append(torch.softmax(label_logits, dim=-1).tolist())
    return probs


def evaluate_student(
    model: Any,
    init_dir: Path,
    rows: Sequence[Example],
    batch_size: int,
    choice_floor: float,
    gate_floor: float,
) -> dict[str, dict[str, Any]]:
    """One forward per eval row; student-vs-teacher fidelity per seam, at the served floors."""
    probs = _forward_probs(model, _eval_tokenizer(init_dir), rows, batch_size)
    samples = [(prob, list(row.target), row.kind, row.seam) for prob, row in zip(probs, rows, strict=True)]
    return _fidelity_metrics(samples, choice_floor, gate_floor)


def false_act_rate(probabilities: Sequence[Sequence[float]], floor: float) -> float | None:
    """P(max(student) >= floor) over below-floor probe rows; ``None`` when there are none."""
    if not probabilities:
        return None
    return sum(1 for probs in probabilities if max(probs) >= floor) / len(probabilities)


def probe_false_act_rate(
    model: Any, init_dir: Path, rows: Sequence[Example], batch_size: int, floor: float
) -> float | None:
    """Measure the false-act rate on the below-floor choice probe rows (one forward each)."""
    if not rows:
        return None
    return false_act_rate(_forward_probs(model, _eval_tokenizer(init_dir), rows, batch_size), floor)


def save_checkpoint(model: Any, init_dir: Path, out_dir: Path) -> Any:
    """Merge the adapters, write a plain HF checkpoint, and return the merged model.

    The caller evaluates the RETURNED model: it is the exact weight set just serialized
    to ``out_dir`` — the QLoRA-wrapped student and the merged weights are different
    numbers, and only the merged ones are what the gateway serves.
    """
    merged = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer_source = init_dir / "tokenizer.json"
    if tokenizer_source.exists():
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
            source = init_dir / name
            if source.exists():
                shutil.copy2(source, out_dir / name)
    for extra in ("tacit_decision.py", "generation_config.json"):
        source = init_dir / extra
        if source.exists():
            shutil.copy2(source, out_dir / extra)
    return merged


def skip_training(init_dir: Path, out_dir: Path) -> None:
    """Copy the released checkpoint verbatim (out-of-band runs; not the image's no-GPU path)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(init_dir, out_dir)
    logger.info("skipped self-distillation; copied %s -> %s", init_dir, out_dir)


def _assemble_problems(
    teacher: Any,
    hints: Sequence[str],
    seams: ModuleType,
    args: argparse.Namespace,
    states: Sequence[str],
    gate_states: Sequence[str],
    axes_per_state: int,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """The full serving question set: category/weights/gate, both style stages, rerank.

    Ch2 style states are the corpus train hints first (real serving bytes), then generated
    descriptions until the budget is met — the seam serves raw user prose, not header-line
    part states, so a few dozen hints cannot train it. A seam that would train on nothing
    aborts the run: an under-trained seam must not ship. (``--skip`` needs no clause here —
    ``main`` returns from ``skip_training`` long before this runs.)
    """
    problems = build_problems(states, seams, gate_states, axes_per_state)
    style_state_count = args.style_states or max(1, args.problems // _STYLE_SHARE)
    # Only generate what the hint corpus does not already cover: a generation pass whose
    # output would be truncated away is pure teacher cost (70 train hints at today's corpus).
    needed = max(0, style_state_count - len(hints))
    descriptions = (
        generate_descriptions(teacher, hints, needed, args.max_new_tokens, seams.DECISION_STATE_MAX_CHARS)
        if needed
        else []
    )
    style_states = [*hints, *descriptions][: max(style_state_count, len(hints))]
    style_problems = build_style_problems(style_states, seams)
    if not style_problems:
        raise SystemExit("the Ch2 style seam trained on 0 description-shaped states — check --seed-texts")
    if not args.rerank_pairs:
        raise SystemExit("the J1 rerank seam needs --rerank-pairs (make tacit-payloads)")
    rerank_pairs = load_rerank_pairs(Path(args.rerank_pairs)) if args.rerank_pairs else []
    rerank_problems = build_rerank_problems(rerank_pairs, seams, args.rerank_problems)
    if not rerank_problems:
        raise SystemExit("the J1 rerank seam trained on 0 pairs — check the rerank-pairs file")
    logger.info(
        "problems: category/score/gate %d, style %d (states %d), rerank %d",
        len(problems),
        len(style_problems),
        len(style_states),
        len(rerank_problems),
    )
    return [*problems, *style_problems, *rerank_problems], style_states, style_problems, rerank_problems


def _post_train_metrics(
    merged: Any,
    init_dir: Path,
    eval_rows: Sequence[Example],
    probe_rows: Sequence[Example],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], dict[str, float | int | None]]:
    """Fidelity on the shipped weights, plus the below-floor false-act rate.

    Both run on the MERGED checkpoint (the weights just serialized), not the
    QLoRA-wrapped student; the probe rows are the below-floor ``choice`` rows the
    retention cut removed from training and eval, measured for how often the
    student clears the acting floor anyway.
    """
    metrics = (
        evaluate_student(merged, init_dir, eval_rows, args.batch, args.confidence_floor, args.gate_confidence_floor)
        if eval_rows
        else {}
    )
    false_act: dict[str, float | int | None] = {"n": len(probe_rows), "rate": None}
    if probe_rows:
        false_act["rate"] = probe_false_act_rate(merged, init_dir, probe_rows, args.batch, args.confidence_floor)
    return metrics, false_act


def _split_examples(
    examples: Sequence[Example], eval_frac: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """Deterministic train/eval split; a corpus under 20 rows keeps every row for training."""
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_eval = int(len(examples) * eval_frac) if len(examples) >= 20 else 0
    return [examples[i] for i in indices[n_eval:]], [examples[i] for i in indices[:n_eval]]


def _seam_report(
    metrics: dict[str, dict[str, Any]],
    examples: Sequence[Example],
    eval_rows: Sequence[Example],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """Enforce the per-seam fidelity gates and build the manifest's per-seam block.

    A build with too few rows to hold out any eval row (a stub or quick-smoke run, never the
    image build) cannot certify a seam, so it skips the gates with a warning instead of
    claiming a pass; every real build fails here rather than shipping an under-trained seam.
    """
    if not eval_rows:
        logger.warning("no eval rows (fewer than 20 kept rows) — per-seam fidelity gates skipped")
    else:
        failures = seam_gate_failures(metrics, args.min_seam_agreement, args.max_score_mae)
        if failures:
            raise SystemExit("; ".join(failures) + " — an under-trained seam must not ship")
    kept_by_seam = Counter(example.seam for example in examples)
    block: dict[str, dict[str, Any]] = {}
    for seam, count in kept_by_seam.items():
        entry = {key: value for key, value in metrics.get(seam, {}).items() if key != "n"}
        entry["kept"] = count
        block[seam] = entry
    return block


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: generate → teach → distill → save (or copy with ``--skip``)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    init_dir = Path(args.init_dir)
    out_dir = Path(args.out)
    if args._answer_shard is not None:
        # Teacher shard child (spawned by teacher_answers): answer, write, exit.
        return _answer_shard(args)
    if not init_dir.is_dir():
        raise SystemExit(f"--init-dir {init_dir} does not exist; download the checkpoint first")
    if args.skip:
        skip_training(init_dir, out_dir)
        return 0

    _seed_everything(args.seed)
    device_name = _require_cuda()
    logger.info("loading teacher %s onto CUDA (this takes minutes, no progress yet)", args.repo)
    seams = load_seam_prompts(Path(args.seam_prompts))
    hints = load_topic_hints(Path(args.seed_texts), limit=max(8, args.problems // 4)) if args.seed_texts else []
    # Explicit cuda:0 placement: the default device_map="auto" only uses CUDA when
    # torch.cuda.is_available(), and "auto" may also shard. Verified against the
    # anyjev 0.3.0 wheel (tacit.py): device_map is forwarded to
    # AutoModelForCausalLM.from_pretrained, which accepts the {"": 0} dict form.
    teacher = _load_teacher(args)
    teacher_device = str(getattr(getattr(getattr(teacher, "engine", None), "model", None), "device", "unknown"))
    logger.info("teacher resident on %s", teacher_device)
    if "cuda" not in teacher_device:
        raise SystemExit(
            f"teacher landed on {teacher_device}, not CUDA: refusing an hours-long CPU run "
            "(check the trainer logs above for an accelerate/device_map fallback)"
        )
    logger.info(
        "self-distill start: problems=%d epochs=%d batch=%d accum=%d cot_max=%d (teacher: %s on %s)",
        args.problems,
        args.epochs,
        args.batch,
        args.accum,
        args.cot_max_tokens,
        args.repo,
        device_name,
    )
    gate_count = max(1, args.problems // _GATE_SHARE)
    design_budget = max(1, args.problems - gate_count)
    states_wanted = args.states or max(1, args.problems // _GATE_SHARE)
    # axes_per_state trades breadth per state for state diversity at a fixed budget; the
    # window rotates across states, so a narrow window still covers all eight axes.
    axes_per_state = max(1, min(len(QUALITY_AXES), design_budget // states_wanted - 1))
    states = generate_states(
        teacher, hints, states_wanted, args.seed, args.max_new_tokens, seams.DECISION_STATE_MAX_CHARS
    )
    logger.info(
        "generated %d design states (wanted %d); %d gate states; %d questions per state",
        len(states),
        states_wanted,
        gate_count,
        axes_per_state + 1,
    )
    gate_states = build_gate_states(states, hints, seams, gate_count)
    problems, style_states, style_problems, rerank_problems = _assemble_problems(
        teacher, hints, seams, args, states, gate_states, axes_per_state
    )
    results, teacher_shards = teacher_answers(teacher, args, problems)
    examples, dropped, probe = examples_from_results(
        teacher, problems, results, args.answer_mass_floor, args.score_mass_floor
    )
    if not examples:
        raise SystemExit(f"no usable teacher examples (dropped {dropped}); lower --answer-mass-floor")
    kept = Counter(example.kind for example in examples)
    short = [kind for kind in _SEAM_KINDS if kept[kind] < args.min_kind_rows]
    if short:
        raise SystemExit(
            f"too few usable teacher rows for {short} (kept {dict(kept)} of {len(problems)} problems, "
            f"dropped {dropped}); lower --answer-mass-floor / --confidence-floor or raise --problems"
        )
    logger.info("retained rows per kind: %s", dict(kept))
    train_rows, eval_rows = _split_examples(examples, args.eval_frac, args.seed)
    student = train_student(init_dir, train_rows, args)
    logger.info("saving merged checkpoint to %s", out_dir)
    merged = save_checkpoint(student, init_dir, out_dir)
    # Fidelity runs on the merged checkpoint that ships — the QLoRA-wrapped student and
    # the merged weights are different numbers — and the below-floor probe rows quantify
    # the false-act rate that retention-filtered eval cannot show.
    metrics, false_act = _post_train_metrics(merged, init_dir, eval_rows, probe, args)
    seams_block = _seam_report(metrics, examples, eval_rows, args)
    # "eval" is distillation fidelity (student vs teacher), not task quality:
    # external quality stays with the serving-side benchmark.
    (out_dir / "self-distill.json").write_text(
        json.dumps(
            {
                "repo": args.repo,
                "init_dir": str(init_dir),
                "problems": len(problems),
                "examples": len(train_rows),
                "kept": dict(kept),
                "dropped": dropped,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch": args.batch,
                "accum": args.accum,
                "confidence_floor": args.confidence_floor,
                "gate_confidence_floor": args.gate_confidence_floor,
                "answer_mass_floor": args.answer_mass_floor,
                "score_mass_floor": args.score_mass_floor,
                "eval_frac": args.eval_frac,
                "seed": args.seed,
                "states": len(states),
                "gate_states": len(gate_states),
                "axes_per_state": axes_per_state,
                "style_states": len(style_states),
                "style_problems": len(style_problems),
                "rerank_problems": len(rerank_problems),
                "teacher_shards": teacher_shards,
                "min_seam_agreement": args.min_seam_agreement,
                "max_score_mae": args.max_score_mae,
                "seams": seams_block,
                "eval": metrics,
                "false_act": false_act,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote distilled checkpoint to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
