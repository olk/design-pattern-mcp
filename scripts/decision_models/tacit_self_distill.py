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

Output layout is what ``anyjev.serve`` expects: a HF causal-LM directory
(``config.json``, weights, tokenizer) that ``vllm serve`` can load.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import shutil
import sys
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

#: One gate question per this many teacher questions: the pipeline asks the gate once
#: per phase per run, the design questions once per decision part.
_GATE_SHARE = 10

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
        "--skip",
        action="store_true",
        help="Copy the released checkpoint to --out without training (out-of-band runs only; "
        "the image's no-GPU path is the runtime-released build target).",
    )
    args = parser.parse_args(argv)
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
    if args.states < 0:
        raise SystemExit(f"--states {args.states} must be >= 0")
    if args.min_kind_rows < 0:
        raise SystemExit(f"--min-kind-rows {args.min_kind_rows} must be >= 0")
    return args


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
            {"state": state, "question": category_question, "options": list(seams.CATEGORY_SLUGS), "kind": "choice"}
        )
        for offset in range(window):
            axis = QUALITY_AXES[(index + offset) % len(QUALITY_AXES)]
            problems.append(
                {
                    "state": state,
                    "question": seams.axis_question(axis),
                    "options": list(seams.WEIGHT_LEVELS),
                    "kind": "score",
                }
            )
    gate_question = seams.gate_question()
    for state in gate_states:
        problems.append({"state": state, "question": gate_question, "options": ["Yes", "No"], "kind": "yes_no"})
    return problems


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


def teacher_examples(
    tacit: Any,
    problems: Sequence[dict[str, Any]],
    answer_mass_floor: float,
    score_mass_floor: float,
) -> tuple[list[Example], dict[str, dict[str, int]], list[Example]]:
    """Answer every problem with thinking on; keep the sharp label distributions.

    ``Tacit(adaptive=True, tau=<huge>, max_cot_share=None)`` escalates every decision to
    the model's own reasoning, so each ``probs`` is the reasoning-conditioned distribution
    over the option labels — the teacher target for the one-forward student.

    A result that omits any option is dropped (``shape``), never padded with zeros and
    rescaled: the renormalized target would silently fabricate the label — certainty on
    ``yes_no``, inflated confidence on ``choice`` — and the row would still be counted as
    kept. Serving tolerates the same omission (``_expected_level`` renormalizes by the mass
    it saw, ``gate`` returns ``None``), which is why the trainer must not convert it into
    a supervision signal.

    The third return value is the false-act probe: below-floor ``choice`` rows (route and
    shape intact) carried as full Examples, so one forward per row can later measure how
    often the student clears the acting floor where the deliberating teacher did not.
    """
    from anyjev.readout import DEFAULT_SYSTEM, build_prompt, label_ids_for_perm, render_chat, resolve_labels
    from anyjev.tacit import make_question

    examples: list[Example] = []
    probe: list[Example] = []
    dropped: dict[str, dict[str, int]] = {kind: {"unrouted": 0, "shape": 0, "unsharp": 0} for kind in _SEAM_KINDS}
    batch_size = max(1, getattr(tacit, "batch_size", 8))
    logger.info("phase 2/3: teacher answering %d problems (sequential CoT)", len(problems))
    progress = _Progress(len(problems), "teacher")
    for start in range(0, len(problems), batch_size):
        chunk = problems[start : start + batch_size]
        results = tacit.decide_batch([dict(item) for item in chunk])
        for item, result in zip(chunk, results, strict=True):
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
            )
            if max(target) < retention_floor(kind, answer_mass_floor, score_mass_floor):
                dropped[kind]["unsharp"] += 1
                if kind == "choice":
                    # Below-floor choice territory is where serving falls back to the
                    # LLM; keep the row to measure the student's false-act rate there.
                    probe.append(example)
                continue
            examples.append(example)
        progress.advance(start + len(chunk))
    logger.info(
        "teacher kept %d of %d problems %s (%d below-floor choice probes)",
        len(examples),
        len(problems),
        dropped,
        len(probe),
    )
    return examples, dropped, probe


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


def _fidelity_metrics(
    samples: Sequence[tuple[list[float], list[float], str]], choice_floor: float, gate_floor: float
) -> dict[str, dict[str, float]]:
    """Distillation fidelity per kind: student-vs-teacher agreement, Brier, act metrics.

    ``choice`` ``floor_accuracy`` answers "would the seam act on the student's answer,
    and be right" — retained rows always clear the teacher floor, so the student must
    agree AND clear ``choice_floor``. ``yes_no`` ``floor_accuracy`` is agreement on the
    ACT decision, because the gate compares ``p(Yes)`` with its floor (options[0] is
    ``"Yes"`` on both sides): a student that fires where the teacher would not is a
    false act even when the labels agree.
    """
    by_kind: dict[str, list[tuple[list[float], list[float]]]] = {}
    for student_probs, teacher_target, kind in samples:
        by_kind.setdefault(kind, []).append((student_probs, teacher_target))
    metrics: dict[str, dict[str, float]] = {}
    for kind, rows in by_kind.items():
        n = len(rows)
        if n == 0:
            metrics[kind] = {"n": 0}
            continue
        agreements = sum(1 for student, teacher in rows if student.index(max(student)) == teacher.index(max(teacher)))
        brier = sum(sum((s - t) ** 2 for s, t in zip(student, teacher, strict=True)) for student, teacher in rows) / n
        entry: dict[str, float] = {"n": n, "agreement": agreements / n, "brier": brier}
        if kind == "choice":
            floored = sum(
                1
                for student, teacher in rows
                if student.index(max(student)) == teacher.index(max(teacher)) and max(student) >= choice_floor
            )
            entry["floor_accuracy"] = floored / n
        elif kind == "yes_no":
            acted = sum(1 for student, teacher in rows if (student[0] >= gate_floor) == (teacher[0] >= gate_floor))
            entry["floor_accuracy"] = acted / n
        elif kind == "score":
            mae = sum(abs(_expected_level(student) - _expected_level(teacher)) for student, teacher in rows) / n
            entry["expected_level_mae"] = mae
        metrics[kind] = entry
    return metrics


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
) -> dict[str, dict[str, float]]:
    """One forward per eval row; student-vs-teacher fidelity per kind, at the served floors."""
    probs = _forward_probs(model, _eval_tokenizer(init_dir), rows, batch_size)
    samples = [(prob, list(row.target), row.kind) for prob, row in zip(probs, rows, strict=True)]
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


def _post_train_metrics(
    merged: Any,
    init_dir: Path,
    eval_rows: Sequence[Example],
    probe_rows: Sequence[Example],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, float]], dict[str, float | int | None]]:
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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: generate → teach → distill → save (or copy with ``--skip``)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    init_dir = Path(args.init_dir)
    out_dir = Path(args.out)
    if not init_dir.is_dir():
        raise SystemExit(f"--init-dir {init_dir} does not exist; download the checkpoint first")
    if args.skip:
        skip_training(init_dir, out_dir)
        return 0

    from anyjev import Tacit

    _seed_everything(args.seed)
    device_name = _require_cuda()
    logger.info("loading teacher %s onto CUDA (this takes minutes, no progress yet)", args.repo)
    seams = load_seam_prompts(Path(args.seam_prompts))
    hints = load_topic_hints(Path(args.seed_texts), limit=max(8, args.problems // 4)) if args.seed_texts else []
    # Explicit cuda:0 placement: the default device_map="auto" only uses CUDA when
    # torch.cuda.is_available(), and "auto" may also shard. Verified against the
    # anyjev 0.3.0 wheel (tacit.py): device_map is forwarded to
    # AutoModelForCausalLM.from_pretrained, which accepts the {"": 0} dict form.
    teacher = Tacit.from_pretrained(
        str(init_dir),
        engine="transformers",
        adaptive=True,
        tau=float("inf"),  # every decision escalates: the teacher must think
        max_cot_share=None,
        cot_window=None,
        cot_max_tokens=args.cot_max_tokens,
        batch_size=max(1, args.batch),
        device_map={"": 0},
    )
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
    problems = build_problems(states, seams, gate_states, axes_per_state)
    examples, dropped, probe = teacher_examples(teacher, problems, args.answer_mass_floor, args.score_mass_floor)
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
    rng = random.Random(args.seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_eval = int(len(examples) * args.eval_frac) if len(examples) >= 20 else 0
    eval_rows = [examples[i] for i in indices[:n_eval]]
    train_rows = [examples[i] for i in indices[n_eval:]]
    student = train_student(init_dir, train_rows, args)
    logger.info("saving merged checkpoint to %s", out_dir)
    merged = save_checkpoint(student, init_dir, out_dir)
    # Fidelity runs on the merged checkpoint that ships — the QLoRA-wrapped student and
    # the merged weights are different numbers — and the below-floor probe rows quantify
    # the false-act rate that retention-filtered eval cannot show.
    metrics, false_act = _post_train_metrics(merged, init_dir, eval_rows, probe, args)
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
