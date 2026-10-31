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
image — so the student is trained on exactly the bytes it will answer in production.

The problems are the three seams, asked over model-written design states:
``choice`` over the nine pattern categories, ``score`` over the five weight levels for
each of the eight quality axes, and the ``yes_no`` reasoning gate. Items whose teacher
distribution is not sharp (max probability below ``--answer-mass-floor``) are dropped:
a soft label is only a label when the teacher is decided.

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

State:"""


@dataclass(frozen=True)
class Example:
    """One supervised row: the serving prompt, its label token ids, the teacher target."""

    prompt: str
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
    parser.add_argument("--problems", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--answer-mass-floor", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cot-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Copy the released checkpoint to --out without training (out-of-band runs only; "
        "the image's no-GPU path is the runtime-released build target).",
    )
    return parser.parse_args(argv)


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

    The teacher is the dominant cost of this build (2000 state writes plus one
    sequential CoT answer per problem). Without a CUDA context it would grind
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


def load_topic_hints(seed_texts: Path, limit: int) -> list[str]:
    """Topic hints from the repo's seed QA pairs (the catalog vocabulary, not labels)."""
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
                if isinstance(text, str) and text:
                    hints.append(text[:400])
                if len(hints) >= limit:
                    break
    return hints


def generate_states(
    tacit: Any,
    hints: Sequence[str],
    problems: int,
    seed: int,
    max_new_tokens: int,
) -> list[str]:
    """Prompt the model for ``problems`` design states in the serving state's shape."""
    import torch

    rng = random.Random(seed)
    prompts = [STATE_PROMPT.format(hint=hints[i % len(hints)] if hints else "a web service") for i in range(problems)]
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
                state = _normalise_state(text)
                if state is not None:
                    states.append(state)
            progress.advance(start + len(chunk))
    rng.shuffle(states)
    return states[:problems]


def _normalise_state(text: str) -> str | None:
    """Keep only model output that has the serving state's shape (header lines only)."""
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
        if stripped.startswith(("System goal:", "Components:", "Decision part:")):
            kept.append(stripped)
    state = "\n".join(kept)
    return state if "Decision part:" in state and len(state) >= 40 else None


def build_problems(
    states: Sequence[str],
    seams: ModuleType,
) -> list[dict[str, Any]]:
    """Three kinds per state, rotated: category choice, weight score, reasoning gate."""
    problems: list[dict[str, Any]] = []
    category_question = seams.category_question()
    gate_question = seams.gate_question()
    categories = list(seams.CATEGORY_SLUGS)
    levels = list(seams.WEIGHT_LEVELS)
    for index, state in enumerate(states):
        kind = index % 3
        if kind == 0:
            problems.append({"state": state, "question": category_question, "options": categories, "kind": "choice"})
        elif kind == 1:
            axis = QUALITY_AXES[(index // 3) % len(QUALITY_AXES)]
            problems.append({"state": state, "question": seams.axis_question(axis), "options": levels, "kind": "score"})
        else:
            problems.append({"state": state, "question": gate_question, "options": ["Yes", "No"], "kind": "yes_no"})
    return problems


def teacher_examples(
    tacit: Any,
    problems: Sequence[dict[str, Any]],
    answer_mass_floor: float,
) -> tuple[list[Example], dict[str, int]]:
    """Answer every problem with thinking on; keep the sharp label distributions.

    ``Tacit(adaptive=True, tau=<huge>, max_cot_share=None)`` escalates every decision to
    the model's own reasoning, so each ``probs`` is the reasoning-conditioned distribution
    over the option labels — the teacher target for the one-forward student.
    """
    from anyjev.readout import DEFAULT_SYSTEM, build_prompt, label_ids_for_perm, render_chat, resolve_labels
    from anyjev.tacit import make_question

    examples: list[Example] = []
    dropped = {"unsharp": 0, "unrouted": 0}
    batch_size = max(1, getattr(tacit, "batch_size", 8))
    logger.info("phase 2/3: teacher answering %d problems (sequential CoT)", len(problems))
    progress = _Progress(len(problems), "teacher")
    for start in range(0, len(problems), batch_size):
        chunk = problems[start : start + batch_size]
        results = tacit.decide_batch([dict(item) for item in chunk])
        for item, result in zip(chunk, results, strict=True):
            if result.get("route") != "cot":
                dropped["unrouted"] += 1
                continue
            probabilities: dict[str, float] = result.get("probs") or {}
            target = [float(probabilities.get(option, 0.0)) for option in item["options"]]
            total = sum(target)
            if total <= 0.0:
                dropped["unrouted"] += 1
                continue
            target = [value / total for value in target]
            if max(target) < answer_mass_floor:
                dropped["unsharp"] += 1
                continue
            question = make_question(item["question"], item["options"], item["kind"])
            labels, ids = resolve_labels(tacit.tok, question)
            permutation = list(range(question.k))
            spec = build_prompt(item["state"], question, permutation, DEFAULT_SYSTEM, labels)
            examples.append(
                Example(
                    prompt=render_chat(tacit.tok, spec),
                    label_ids=[int(token) for token in label_ids_for_perm(question, ids, permutation)],
                    target=target,
                )
            )
        progress.advance(start + len(chunk))
    logger.info("teacher kept %d of %d problems %s", len(examples), len(problems), dropped)
    return examples, dropped


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
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        total_steps = max(1, (len(order) + args.batch - 1) // args.batch)
        logger.info("phase 3/3: QLoRA epoch %d/%d (%d steps)", epoch + 1, args.epochs, total_steps)
        progress = _Progress(total_steps, f"epoch {epoch + 1}/{args.epochs}")
        for step, start in enumerate(range(0, len(order), args.batch), start=1):
            rows = [examples[index] for index in order[start : start + args.batch]]
            loss = _batch_loss(model, tokenizer, rows, device) / (len(rows) * args.accum)
            loss.backward()
            if step % args.accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0 or step == total_steps:
                current = float(loss.detach()) * len(rows) * args.accum
                logger.info("epoch %d step %d/%d loss %.4f", epoch, step, total_steps, current)
            progress.advance(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
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


def save_checkpoint(model: Any, init_dir: Path, out_dir: Path) -> None:
    """Merge the adapters and write a plain HF checkpoint the gateway can serve."""
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


def skip_training(init_dir: Path, out_dir: Path) -> None:
    """Copy the released checkpoint verbatim (out-of-band runs; not the image's no-GPU path)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(init_dir, out_dir)
    logger.info("skipped self-distillation; copied %s -> %s", init_dir, out_dir)


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
    states = generate_states(teacher, hints, args.problems, args.seed, args.max_new_tokens)
    logger.info("generated %d design states", len(states))
    problems = build_problems(states, seams)
    examples, dropped = teacher_examples(teacher, problems, args.answer_mass_floor)
    if not examples:
        raise SystemExit(f"no usable teacher examples (dropped {dropped}); lower --answer-mass-floor")
    student = train_student(init_dir, examples, args)
    logger.info("saving merged checkpoint to %s", out_dir)
    save_checkpoint(student, init_dir, out_dir)
    (out_dir / "self-distill.json").write_text(
        json.dumps(
            {
                "repo": args.repo,
                "init_dir": str(init_dir),
                "problems": len(problems),
                "examples": len(examples),
                "dropped": dropped,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch": args.batch,
                "accum": args.accum,
                "answer_mass_floor": args.answer_mass_floor,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote distilled checkpoint to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
