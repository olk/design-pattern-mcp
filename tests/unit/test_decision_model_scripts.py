"""Build-time decision-model scripts: train/serve prompt parity and the export contract.

These scripts run inside the sidecar image builds (``docker/Dockerfile.tacit``,
``docker/Dockerfile.clm``), where a wiring bug only surfaces after an hour of state
generation or a 16 GB encoder download. Everything expensive is stubbed here; what is
asserted is the *wiring*: the trainer asks exactly the questions the server asks
(``src/decisions/seam_prompts.py``), its ``main()`` walks generate → problems → teacher →
train → save, and the CLM exporter emits the row shape the pinned ``train/adapters.py``
reads.

The scripts are loaded by path (neither is an importable package), the same way
``test_client_examples.py`` loads the example clients.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from src.config import DecisionSeamConfig
from src.decisions import seam_prompts
from src.decisions.tacit import category_options, gate_state
from src.pipeline import (
    _gate_payload_analyze,
    _gate_payload_evaluate,
    _gate_payload_generate,
    _gate_payload_weights,
)
from src.reasoning.config import ReasoningConfig
from src.schemas.enums import QUALITY_AXES
from tests.unit.conftest import REPO_ROOT

SCRIPTS = REPO_ROOT / "scripts" / "decision_models"
SEAM_PROMPTS = REPO_ROOT / "src" / "decisions" / "seam_prompts.py"


def _load(name: str, filename: str) -> types.ModuleType:
    """Import one script by path (registered in sys.modules so dataclasses resolve)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer() -> Iterator[types.ModuleType]:
    """The Tacit trainer, with stub ``anyjev``/``torch`` so importing it needs no GPU."""
    stub = types.ModuleType("anyjev")

    class _StubTacit:
        batch_size = 2

        def __init__(self) -> None:
            engine = types.SimpleNamespace(model=types.SimpleNamespace(device="cuda:0"))
            self.engine = engine  # noqa: F841 - read by main()'s teacher-device check

        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> _StubTacit:
            return cls()

    stub.Tacit = _StubTacit  # type: ignore[attr-defined]

    # ``teacher_examples`` imports these at call time, exactly as in the image.
    readout = types.ModuleType("anyjev.readout")
    readout.DEFAULT_SYSTEM = "system"  # type: ignore[attr-defined]
    readout.build_prompt = lambda state, question, permutation, system, labels: (  # type: ignore[attr-defined]
        state,
        question,
        tuple(permutation),
        system,
        tuple(labels),
    )
    readout.label_ids_for_perm = lambda question, ids, permutation: list(ids)  # type: ignore[attr-defined]
    readout.render_chat = lambda tok, spec: f"{spec[0]} || {spec[1].text}"  # type: ignore[attr-defined]
    readout.resolve_labels = lambda tok, question: (["label"], [7])  # type: ignore[attr-defined]
    tacit_module = types.ModuleType("anyjev.tacit")
    tacit_module.make_question = lambda text, options, kind: types.SimpleNamespace(  # type: ignore[attr-defined]
        text=text, options=list(options), kind=kind, k=len(options)
    )
    stub.readout = readout  # type: ignore[attr-defined]
    stub.tacit = tacit_module  # type: ignore[attr-defined]

    torch_stub = types.ModuleType("torch")
    cuda_stub = types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda _i: "stub-cuda:0"
    )
    torch_stub.cuda = cuda_stub  # type: ignore[attr-defined]
    torch_stub.manual_seed = lambda _seed: None  # type: ignore[attr-defined]
    torch_stub.cuda.manual_seed_all = lambda _seed: None  # type: ignore[attr-defined]
    installed = ("anyjev", "anyjev.readout", "anyjev.tacit", "torch")
    previous = {name: sys.modules.get(name) for name in installed}
    sys.modules["anyjev"] = stub
    sys.modules["anyjev.readout"] = readout
    sys.modules["anyjev.tacit"] = tacit_module
    sys.modules["torch"] = torch_stub
    yield _load("tacit_self_distill", "tacit_self_distill.py")
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_trainer_asks_the_questions_the_server_asks(trainer: types.ModuleType) -> None:
    """The student is trained on the serving bytes: same question text, same options."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    state = "System goal: goal\nDecision part: part"
    gate_state = "phase: analyze\ndescription: build a checkout"
    problems = trainer.build_problems([state], seams, [gate_state])
    assert len(problems) == 10
    assert [problem["kind"] for problem in problems] == ["choice"] + ["score"] * 8 + ["yes_no"]
    assert problems[0]["question"] == seam_prompts.category_question()
    assert problems[0]["options"] == category_options()
    for index, axis in enumerate(trainer.QUALITY_AXES):
        score = problems[1 + index]
        assert score["question"] == seam_prompts.axis_question(axis)
        assert score["options"] == list(seam_prompts.WEIGHT_LEVELS)
    assert problems[9]["question"] == seam_prompts.gate_question()
    assert problems[9]["options"] == ["Yes", "No"]
    assert problems[9]["state"] == gate_state


def test_load_topic_hints_filters_holdout(trainer: types.ModuleType, tmp_path: Path) -> None:
    """Holdout rows must never become topic hints (qa/README train-split-only)."""
    seed = tmp_path / "qa.jsonl"
    seed.write_text(
        "\n".join(
            [
                json.dumps({"split": "train", "question": "train q"}),
                json.dumps({"split": "holdout", "question": "hold q"}),
                json.dumps({"question": "no split"}),
            ]
        ),
        encoding="utf-8",
    )
    assert trainer.load_topic_hints(seed, 10) == ["train q"]


def test_normalise_state_keeps_serving_shape(trainer: types.ModuleType) -> None:
    """Scope/Constraints kept, junk dropped, goal line required, cap mirrored exactly."""
    cap = seam_prompts.DECISION_STATE_MAX_CHARS
    text = "\n".join(
        [
            "System goal: achieve checkout",
            "Components: cart, payment",
            "Decision part: choose the payment flow",
            "Scope: online orders only",
            "Constraints: pci-dss; idempotent",
            "Extra: junk line",
        ]
    )
    assert trainer._normalise_state(text, cap) == "\n".join(text.splitlines()[:5])
    assert trainer._normalise_state("System goal: g\nComponents: a", cap) is None
    # Serving always leads with the goal line (pipeline._decision_part_state); a state
    # without it never ships, so it must never become a training row.
    assert trainer._normalise_state("Decision part: choose a payment flow " + "x" * 20, cap) is None
    # Over-cap states truncate exactly where serving truncates (mid-line included).
    long_state = "System goal: g\nDecision part: p\nConstraints: " + "c" * 2000
    assert trainer._normalise_state(long_state, cap) == long_state[:cap]


def test_normalise_gate_state(trainer: types.ModuleType) -> None:
    """Gate states must be a serving shape: the declared key PREFIX, in order, capped."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    cap = seam_prompts.GATE_STATE_MAX_CHARS
    assert (
        trainer._normalise_gate_state("phase: analyze\ndescription: Build a checkout", seams)
        == "phase: analyze\ndescription: Build a checkout"
    )
    assert trainer._normalise_gate_state("phase: unknown\ndescription: d", seams) is None
    # ``analyze`` has ONE free-form field, so a multi-line description (real user text
    # contains newlines) is legal and must round-trip untouched.
    multiline = "phase: analyze\ndescription: reconcile the ledger\nunder retry: partially"
    assert trainer._normalise_gate_state(multiline, seams) == multiline
    # Undeclared keys, gaps and reordering are shapes the server never sends.
    assert trainer._normalise_gate_state("phase: analyze", seams) is None
    assert trainer._normalise_gate_state("phase: weights\ndescription: d\nfoo: bar", seams) is None
    assert trainer._normalise_gate_state("phase: weights\nrestated_goal: g", seams) is None
    assert (
        trainer._normalise_gate_state("phase: weights\nrestated_goal: g\ndescription: d\ndecision_parts: p", seams)
        is None
    )
    # A SHORT prefix is legal only when the text sits exactly at the cap: that is how a
    # long description truncates trailing keys, and it is what serving really sends.
    prefix = "phase: weights\ndescription: "
    assert trainer._normalise_gate_state(prefix + "x" * 200, seams) is None
    truncated = prefix + "x" * (cap - len(prefix))
    assert trainer._normalise_gate_state(truncated, seams) == truncated
    # A complete payload that also hits the cap keeps every key.
    full = "phase: analyze\ndescription: " + "x" * (cap - len("phase: analyze\ndescription: "))
    assert trainer._normalise_gate_state(full, seams) == full


def test_gate_states_are_the_bytes_serving_sends(trainer: types.ModuleType) -> None:
    """Built gate states round-trip through the server's own gate_state, cap included."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    state = "System goal: check out a cart\nComponents: Cart, Pricing\nDecision part: choose a pricing strategy"
    hints = ["A commerce platform recalculates cart totals per request. " * 40]  # over the cap on purpose
    gate_states = trainer.build_gate_states([state], hints, seams, len(seam_prompts.GATE_PHASES) * 2)
    assert len(gate_states) == len(seam_prompts.GATE_PHASES) * 2
    assert {text.splitlines()[0].removeprefix("phase: ") for text in gate_states} == set(seam_prompts.GATE_PHASES)
    for text in gate_states:
        phase = text.splitlines()[0].removeprefix("phase: ")
        required = seam_prompts.GATE_PHASE_PAYLOAD_KEYS[phase]
        payload: dict[str, str] = {}
        for line in text.splitlines()[1:]:
            key, separator, value = line.partition(": ")
            if separator and value:
                payload[key] = value
        # A prefix: all keys when the payload fits, fewer when the cap truncated a tail.
        assert tuple(payload) == required[: len(payload)]
        if "decision_parts" in payload:
            # Serving part ids are slug-shaped ("p-delivery"), never enumerated ("p1").
            assert all(segment.startswith("p-") for segment in payload["decision_parts"].split("; "))
        if len(payload) < len(required):
            assert len(text) == seam_prompts.GATE_STATE_MAX_CHARS
        # The server composes this string from the same payload (never a longer one).
        assert text.startswith(gate_state(phase, payload))
        assert trainer._normalise_gate_state(text, seams) == text


def test_gate_part_ids_are_serving_shaped(trainer: types.ModuleType) -> None:
    """Composed parts/applications use serving id shapes (p-<slug>), never p1/a1."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    state = "System goal: serve ads\nComponents: AdServer, Bidder\nDecision part: choose the ad rotation strategy"
    gate_states = trainer.build_gate_states([state], [], seams, len(seam_prompts.GATE_PHASES) * 2)

    def field(text: str, key: str) -> str:
        payload = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)
        return payload[key]

    weights = next(text for text in gate_states if text.startswith("phase: weights"))
    assert all(segment.startswith("p-") for segment in field(weights, "decision_parts").split("; "))
    generate = next(text for text in gate_states if text.startswith("phase: generate"))
    assert all(segment.split(":")[0].startswith("p-") for segment in field(generate, "applications").split("; "))
    evaluate = next(text for text in gate_states if text.startswith("phase: evaluate"))
    assert all(segment.startswith("p-") for segment in field(evaluate, "parts").split("; "))


def test_long_aggregate_payloads_truncate_like_serving(trainer: types.ModuleType) -> None:
    """Phases with no description (generate/evaluate) overflow too: truncate, never crash."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    statement = "choose the reconciliation strategy under retry and partial failure " * 10
    component = "ReconciliationCoordinator" * 8
    states = [
        f"System goal: reconcile ledgers\nComponents: {component}, Ledger\nDecision part: {statement}{i}"
        for i in range(4)
    ]
    # 16 states cycle every phase through all four part spreads, so the biggest aggregates
    # (generate/evaluate) are built from four long parts and overflow the cap.
    gate_states = trainer.build_gate_states(states, [], seams, 16)
    assert len(gate_states) == 16  # nothing is dropped for being long
    capped = [text for text in gate_states if len(text) == seam_prompts.GATE_STATE_MAX_CHARS]
    assert capped, "the long aggregates must actually reach the cap"
    assert any(not text.startswith("phase: analyze") for text in capped)
    for text in gate_states:
        assert len(text) <= seam_prompts.GATE_STATE_MAX_CHARS
        assert trainer._normalise_gate_state(text, seams) == text


def test_gate_phases_match_the_pipeline_phases() -> None:
    """The trainer's phase list is the pipeline's, so no phase is trained that never runs."""
    assert tuple(ReasoningConfig().per_phase) == seam_prompts.GATE_PHASES


def test_gate_payload_keys_match_the_pipeline_builders() -> None:
    """GATE_PHASE_PAYLOAD_KEYS is the pipeline's payload table, not a parallel copy."""
    assert tuple(_gate_payload_analyze("d")) == seam_prompts.GATE_PHASE_PAYLOAD_KEYS["analyze"]
    assert tuple(_gate_payload_weights("d", "g", "p")) == seam_prompts.GATE_PHASE_PAYLOAD_KEYS["weights"]
    assert tuple(_gate_payload_generate("g", "a")) == seam_prompts.GATE_PHASE_PAYLOAD_KEYS["generate"]
    assert tuple(_gate_payload_evaluate("p", "a", "p")) == seam_prompts.GATE_PHASE_PAYLOAD_KEYS["evaluate"]
    assert set(seam_prompts.GATE_PHASE_PAYLOAD_KEYS) == set(seam_prompts.GATE_PHASES)


def test_confidence_floor_matches_the_server_default(trainer: types.ModuleType) -> None:
    """Training keeps rows the seam would act on: its floor defaults to the server's."""
    assert DecisionSeamConfig().confidence_floor == trainer._DEFAULT_CONFIDENCE_FLOOR


def test_quality_axes_match_the_enum(trainer: types.ModuleType) -> None:
    """The trainer's axis list is the canonical tuple, in order (same drift class as above)."""
    assert QUALITY_AXES == trainer.QUALITY_AXES


def test_retention_floor_is_kind_aware(trainer: types.ModuleType) -> None:
    """Score rows keep graded mass; choice rows clear the category seam's floor."""
    assert trainer.retention_floor("score", 0.5, 0.0) == 0.0
    assert trainer.retention_floor("choice", 0.5, 0.0) == 0.5
    assert trainer.retention_floor("choice", 0.9, 0.0) == 0.9
    # The gate needs its boundary rows: no max floor binds yes_no at any setting (a
    # raised --confidence-floor must not drop near-boundary calibration rows).
    assert trainer.retention_floor("yes_no", 0.9, 0.0) == 0.0


def test_fidelity_metrics_hand_computed(trainer: types.ModuleType) -> None:
    """Agreement, Brier, floor accuracy, and expected-level MAE follow the serving math."""
    choice = trainer._fidelity_metrics(
        [([0.9, 0.1], [1.0, 0.0], "choice"), ([0.4, 0.6], [0.5, 0.5], "choice")], 0.5, 0.5
    )["choice"]
    assert choice["n"] == 2
    assert choice["agreement"] == 0.5
    assert abs(choice["brier"] - 0.02) < 1e-9
    assert choice["floor_accuracy"] == 0.5
    teacher = [0.1, 0.3, 0.4, 0.15, 0.05]
    student = [0.0, 0.0, 1.0, 0.0, 0.0]
    expected_teacher = sum(index * value for index, value in enumerate(teacher)) / 4
    expected_student = sum(index * value for index, value in enumerate(student)) / 4
    score = trainer._fidelity_metrics([(student, teacher, "score")], 0.5, 0.5)["score"]
    assert abs(score["expected_level_mae"] - abs(expected_student - expected_teacher)) < 1e-9
    # yes_no floor accuracy is ACT-decision agreement (the gate compares p(Yes) with its
    # floor): a student that would not act where the teacher would is a miss.
    yes_no = trainer._fidelity_metrics(
        [([0.9, 0.1], [1.0, 0.0], "yes_no"), ([0.4, 0.6], [0.55, 0.45], "yes_no")], 0.5, 0.5
    )["yes_no"]
    assert yes_no["floor_accuracy"] == 0.5
    # At a raised gate floor the act rule bites: argmax agreement (both "Yes") must NOT
    # count as correct when the student fires (0.8 >= 0.7) and the teacher would not.
    raised = trainer._fidelity_metrics([([0.8, 0.2], [0.6, 0.4], "yes_no")], 0.5, 0.7)["yes_no"]
    assert raised["floor_accuracy"] == 0.0
    # The acting floor is the served one, not a literal: raising it drops the acted cases.
    assert trainer._fidelity_metrics([([0.9, 0.1], [1.0, 0.0], "choice")], 0.95, 0.5)["choice"]["floor_accuracy"] == 0.0
    assert trainer._fidelity_metrics([], 0.5, 0.5) == {}


def test_false_act_rate_counts_clearing_the_floor(trainer: types.ModuleType) -> None:
    """The probe metric is P(max(student) >= floor) on below-floor rows, else None."""
    assert trainer.false_act_rate([[0.9, 0.05, 0.05], [0.4, 0.3, 0.3]], 0.5) == 0.5
    assert trainer.false_act_rate([[0.51, 0.49]], 0.5) == 1.0
    assert trainer.false_act_rate([], 0.5) is None


def test_build_problems_windows_eight_axes_across_states(trainer: types.ModuleType) -> None:
    """A narrow axis window still covers every axis across the corpus, not just the first."""
    seams = trainer.load_seam_prompts(SEAM_PROMPTS)
    states = [f"System goal: g{i}\nDecision part: p{i}" for i in range(len(trainer.QUALITY_AXES))]
    problems = trainer.build_problems(states, seams, [], axes_per_state=3)
    score_axes = [problem["question"] for problem in problems if problem["kind"] == "score"]
    assert len(score_axes) == 3 * len(states)
    assert set(score_axes) == {seams.axis_question(axis) for axis in trainer.QUALITY_AXES}
    assert len(trainer.build_problems(states, seams, [])) == len(states) * (len(trainer.QUALITY_AXES) + 1)


class _FakeTeacher:
    """Stands in for anyjev's Tacit: ``decide_batch`` keyed by the problem's state."""

    batch_size = 2
    tok = object()

    def __init__(self, results: dict[str, dict[str, Any]]) -> None:
        self._results = results

    def decide_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._results[item["state"]] for item in items]


def test_teacher_examples_keeps_graded_targets_and_rejects_partial_labels(trainer: types.ModuleType) -> None:
    """Retention is per kind, and a result missing an option is dropped, never rescaled."""
    labels = list(seam_prompts.CATEGORY_SLUGS)
    unsure = dict.fromkeys(labels, 0.1)
    unsure[labels[0]] = 0.2  # renormalized max 0.2: below the acting floor
    sure = dict.fromkeys(labels, 0.05)
    sure[labels[1]] = 0.6  # clears the acting floor
    results = {
        "diffuse": {
            "route": "cot",
            "probs": {"none": 0.1, "low": 0.3, "moderate": 0.4, "high": 0.15, "critical": 0.05},
        },
        "unsure": {"route": "cot", "probs": unsure},
        "sure": {"route": "cot", "probs": sure},
        "partial": {"route": "cot", "probs": {"Yes": 1.0}},  # "No" missing
        "unrouted": {"route": "one_forward", "probs": {"Yes": 0.9, "No": 0.1}},
    }
    fake = _FakeTeacher(results)
    problems = [
        {"state": "diffuse", "question": "q", "options": list(seam_prompts.WEIGHT_LEVELS), "kind": "score"},
        {"state": "unsure", "question": "q", "options": labels, "kind": "choice"},
        {"state": "sure", "question": "q", "options": labels, "kind": "choice"},
        {"state": "partial", "question": "q", "options": ["Yes", "No"], "kind": "yes_no"},
        {"state": "unrouted", "question": "q", "options": ["Yes", "No"], "kind": "yes_no"},
    ]
    examples, dropped, probe = trainer.teacher_examples(fake, problems, 0.5, 0.0)
    assert [example.kind for example in examples] == ["score", "choice"]
    # Diffuse score rows survive: serving consumes the graded expected level.
    assert abs(sum(examples[0].target) - 1.0) < 1e-9
    assert abs(examples[0].target[2] - 0.4) < 1e-9
    # The unsure choice is filtered; the partial yes_no is rejected, not renormalized to [1, 0].
    assert dropped == {
        "choice": {"unrouted": 0, "shape": 0, "unsharp": 1},
        "score": {"unrouted": 0, "shape": 0, "unsharp": 0},
        "yes_no": {"unrouted": 1, "shape": 1, "unsharp": 0},
    }
    assert examples[0].label_ids == [7]
    assert examples[1].prompt.startswith("sure || ")  # the rebuilt serving prompt, not the raw state
    # The below-floor choice row survives as the false-act probe, on the same prompt bytes.
    assert [example.kind for example in probe] == ["choice"]
    assert probe[0].prompt.startswith("unsure || ")
    assert abs(max(probe[0].target) - 0.2) < 1e-9


def _main_fakes(
    trainer: types.ModuleType,
    seen: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    kinds: tuple[str, ...] = ("choice", "score", "yes_no"),
) -> None:
    """Patch the expensive stages of ``main()`` so the wiring is what gets asserted."""

    def fake_states(
        _tacit: Any, _hints: Any, problems: int, _seed: int, _max_new: int, max_state_chars: int
    ) -> list[str]:
        seen["state_cap"] = max_state_chars
        return [f"System goal: g{i}\nDecision part: p{i}" for i in range(problems)]

    def fake_gate_states(states: list[str], hints: list[str], _seams: Any, count: int) -> list[str]:
        seen["gate_sources"] = len(states)
        seen["gate_hints"] = list(hints)
        return [f"phase: analyze\ndescription: g{i}" for i in range(count)]

    def fake_teacher(
        _tacit: Any, problems: list[dict[str, Any]], floor: float, score_floor: float
    ) -> tuple[list[Any], dict[str, dict[str, int]], list[Any]]:
        seen["kinds"] = [problem["kind"] for problem in problems]
        seen["floors"] = (floor, score_floor)
        rows = [
            trainer.Example(prompt=f"prompt-{kind}-{repeat}", kind=kind, label_ids=[1, 2], target=[0.6, 0.4])
            for repeat in range(7)  # 21 rows: past the >=20 bar, so the eval split is real
            for kind in kinds
        ]
        probe = [trainer.Example(prompt="probe-choice", kind="choice", label_ids=[1, 2], target=[0.55, 0.45])]
        dropped_counts = {kind: {"unrouted": 0, "shape": 0, "unsharp": 0} for kind in ("choice", "score", "yes_no")}
        return rows, dropped_counts, probe

    def fake_train(_init: Path, examples: list[Any], args: Any) -> object:
        seen["examples"] = len(examples)
        seen["epochs"] = args.epochs
        return object()

    def fake_save(_model: Any, _init: Path, out_dir: Path) -> object:
        out_dir.mkdir(parents=True, exist_ok=True)
        seen["saved_model"] = object()  # sentinel: the merged artifact main() must evaluate
        return seen["saved_model"]

    def fake_eval(
        _model: Any, _init: Path, _rows: Any, _batch: int, choice_floor: float, gate_floor: float
    ) -> dict[str, Any]:
        seen["eval_model"] = _model
        seen["eval_floors"] = (choice_floor, gate_floor)
        return {"choice": {"n": 0}}

    def fake_probe(_model: Any, _init: Path, rows: Any, _batch: int, floor: float) -> float:
        seen["probe_model"] = _model
        seen["probe_floor"] = floor
        seen["probe_rows"] = len(rows)
        return 0.25

    monkeypatch.setattr(trainer, "generate_states", fake_states)
    monkeypatch.setattr(trainer, "build_gate_states", fake_gate_states)
    monkeypatch.setattr(trainer, "teacher_examples", fake_teacher)
    monkeypatch.setattr(trainer, "train_student", fake_train)
    monkeypatch.setattr(trainer, "save_checkpoint", fake_save)
    monkeypatch.setattr(trainer, "evaluate_student", fake_eval)
    monkeypatch.setattr(trainer, "probe_false_act_rate", fake_probe)


def test_trainer_main_walks_generate_to_save(
    trainer: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main()`` must thread states → problems → teacher → train → save, not unpack them."""
    seen: dict[str, Any] = {}
    _main_fakes(trainer, seen, monkeypatch)

    init_dir = tmp_path / "init"
    init_dir.mkdir()
    (init_dir / "config.json").write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "out"
    code = trainer.main(
        [
            "--init-dir",
            str(init_dir),
            "--out",
            str(out_dir),
            "--seam-prompts",
            str(SEAM_PROMPTS),
            "--problems",
            "10",
        ]
    )
    assert code == 0
    assert seen["kinds"] == ["choice"] + ["score"] * 8 + ["yes_no"]
    assert seen["floors"] == (0.5, 0.0)  # the answer floor defaults to the confidence floor
    assert seen["examples"] == 19  # 21 kept rows minus the 10% eval carve-out (2)
    assert seen["epochs"] == 1
    assert seen["gate_sources"] == 1  # gate states are built FROM the design states
    assert seen["state_cap"] == seam_prompts.DECISION_STATE_MAX_CHARS
    assert seen["eval_floors"] == (0.5, 0.5)  # the gate floor defaults to the confidence floor
    assert seen["probe_floor"] == 0.5
    assert seen["probe_rows"] == 1
    assert seen["eval_model"] is seen["saved_model"]  # fidelity scores the MERGED artifact
    assert seen["probe_model"] is seen["saved_model"]  # and so does the false-act probe
    manifest = json.loads((out_dir / "self-distill.json").read_text(encoding="utf-8"))
    assert manifest["problems"] == 10
    assert manifest["examples"] == 19
    assert manifest["kept"] == {"choice": 7, "score": 7, "yes_no": 7}
    assert manifest["repo"] == "morriszjm/Tacit-4B"
    assert manifest["confidence_floor"] == 0.5
    assert manifest["gate_confidence_floor"] == 0.5
    assert manifest["answer_mass_floor"] == 0.5
    assert manifest["score_mass_floor"] == 0.0
    assert manifest["eval_frac"] == 0.1
    assert manifest["states"] == 1
    assert manifest["gate_states"] == 1
    assert manifest["axes_per_state"] == 8
    assert manifest["eval"] == {"choice": {"n": 0}}
    assert manifest["false_act"] == {"n": 1, "rate": 0.25}


def test_trainer_main_honours_a_raised_confidence_floor(
    trainer: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Floors flow from one knob: answer, gate and eval floors follow --confidence-floor."""
    seen: dict[str, Any] = {}
    _main_fakes(trainer, seen, monkeypatch)
    init_dir = tmp_path / "init"
    init_dir.mkdir()
    (init_dir / "config.json").write_text("{}", encoding="utf-8")
    code = trainer.main(
        [
            "--init-dir",
            str(init_dir),
            "--out",
            str(tmp_path / "out"),
            "--seam-prompts",
            str(SEAM_PROMPTS),
            "--problems",
            "10",
            "--confidence-floor",
            "0.7",
        ]
    )
    assert code == 0
    assert seen["floors"] == (0.7, 0.0)
    assert seen["eval_floors"] == (0.7, 0.7)
    assert seen["probe_floor"] == 0.7
    manifest = json.loads((tmp_path / "out" / "self-distill.json").read_text(encoding="utf-8"))
    assert manifest["confidence_floor"] == 0.7
    assert manifest["gate_confidence_floor"] == 0.7
    assert manifest["answer_mass_floor"] == 0.7


def test_trainer_main_honours_a_distinct_gate_floor(
    trainer: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate seam's floor is its own knob: choice retention stays, act metrics follow it."""
    seen: dict[str, Any] = {}
    _main_fakes(trainer, seen, monkeypatch)
    init_dir = tmp_path / "init"
    init_dir.mkdir()
    (init_dir / "config.json").write_text("{}", encoding="utf-8")
    code = trainer.main(
        [
            "--init-dir",
            str(init_dir),
            "--out",
            str(tmp_path / "out"),
            "--seam-prompts",
            str(SEAM_PROMPTS),
            "--problems",
            "10",
            "--confidence-floor",
            "0.5",
            "--gate-confidence-floor",
            "0.8",
        ]
    )
    assert code == 0
    assert seen["floors"] == (0.5, 0.0)  # choice retention is unchanged
    assert seen["eval_floors"] == (0.5, 0.8)  # yes_no act metrics use the gate floor
    assert seen["probe_floor"] == 0.5
    manifest = json.loads((tmp_path / "out" / "self-distill.json").read_text(encoding="utf-8"))
    assert manifest["gate_confidence_floor"] == 0.8


def test_trainer_main_refuses_a_kind_with_no_rows(
    trainer: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An under-trained seam must fail the build, not ship: every kind needs rows."""
    seen: dict[str, Any] = {}
    _main_fakes(trainer, seen, monkeypatch, kinds=("choice", "score"))
    init_dir = tmp_path / "init"
    init_dir.mkdir()
    (init_dir / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="too few usable teacher rows"):
        trainer.main(
            [
                "--init-dir",
                str(init_dir),
                "--out",
                str(tmp_path / "out"),
                "--seam-prompts",
                str(SEAM_PROMPTS),
                "--problems",
                "10",
            ]
        )


def test_trainer_skip_copies_the_released_checkpoint(trainer: types.ModuleType, tmp_path: Path) -> None:
    init_dir = tmp_path / "init"
    init_dir.mkdir()
    (init_dir / "config.json").write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "copy"
    code = trainer.main(
        ["--init-dir", str(init_dir), "--out", str(out_dir), "--seam-prompts", str(SEAM_PROMPTS), "--skip"]
    )
    assert code == 0
    assert (out_dir / "config.json").is_file()


@pytest.fixture(scope="module")
def exporter() -> types.ModuleType:
    return _load("export_clm_dataset", "export_clm_dataset.py")


def test_export_row_matches_the_pinned_clm_adapter_contract(exporter: types.ModuleType) -> None:
    """``typed_decision_examples`` reads {id, state, questions, gold}; so must we write it.

    The row layout was verified against the upstream adapter at one CLM revision:
    the exporter's default pin, the Dockerfile's ``CLM_REPO_PIN`` default, and
    this test's constant must all name that same revision, so bumping the pin in
    either place re-opens the contract check instead of silently invalidating it.
    """
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.clm").read_text(encoding="utf-8")
    assert "\nARG CLM_REPO_PIN=d5f9ef0\n" in dockerfile
    assert exporter.parse_args(["--out", "/tmp/x"]).clm_pin == "d5f9ef0"
    row = exporter.to_typed_row(
        {
            "id": "scenario:category",
            "state": "System goal: x\nDecision part: y",
            "instructions": "Which category?",
            "criteria": {"domain-modeling": "entities and invariants", "behavior-selection": "dispatch"},
            "label": "behavior-selection",
            "source": {"split": "train"},
        },
        exporter.CATEGORY_QUESTION_ID,
    )
    assert row["id"] == "scenario:category"
    questions = json.loads(row["questions"])
    assert questions["category"]["type"] == "choice"
    assert questions["category"]["instructions"] == "Which category?"
    assert set(questions["category"]["criteria"]) == {"domain-modeling", "behavior-selection"}
    assert json.loads(row["gold"]) == {"category": {"label": "behavior-selection"}}


def test_export_refuses_a_label_outside_the_criteria(exporter: types.ModuleType) -> None:
    with pytest.raises(SystemExit):
        exporter.to_typed_row(
            {
                "id": "bad",
                "state": "s",
                "criteria": {"domain-modeling": "x"},
                "label": "security-verification",
                "source": {"split": "train"},
            },
            exporter.CATEGORY_QUESTION_ID,
        )


def test_export_filters_by_source_split(exporter: types.ModuleType, tmp_path: Path) -> None:
    """Holdout families must never reach the weights."""
    path = tmp_path / "choice.jsonl"
    rows = [
        {
            "id": "a",
            "state": "s",
            "criteria": {"domain-modeling": "x"},
            "label": "domain-modeling",
            "source": {"split": "train"},
        },
        {
            "id": "b",
            "state": "s",
            "criteria": {"domain-modeling": "x"},
            "label": "domain-modeling",
            "source": {"split": "holdout"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    exported = exporter.load_rows(path, "train")
    assert [row["id"] for row in exported] == ["a"]


def test_style_row_asks_the_served_question(exporter: types.ModuleType) -> None:
    """The style export must use the Ch2 qid, not the category one."""
    row = exporter.to_typed_row(
        {
            "id": "scenario:style",
            "state": "A checkout service with flash sales.",
            "instructions": "Which style?",
            "criteria": {"event-driven": "events", "none": "nothing applies"},
            "label": "event-driven",
            "source": {"split": "train"},
        },
        exporter.STYLE_QUESTION_ID,
    )
    questions = json.loads(row["questions"])
    assert list(questions) == ["architecture_style"]
    assert questions["architecture_style"]["type"] == "choice"
    assert questions["architecture_style"]["instructions"] == "Which style?"
    assert json.loads(row["gold"]) == {"architecture_style": {"label": "event-driven"}}


def test_style_corpus_matches_the_served_prompt_bytes() -> None:
    """Trainer asks exactly what the server asks: qid, instructions, criteria, state."""
    from src.decisions.client import STYLE_QUESTION_ID, build_style_criteria

    style_path = REPO_ROOT / "qa" / "typed" / "style.jsonl"
    criteria = build_style_criteria()
    rows = [json.loads(line) for line in style_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "qa/typed/style.jsonl must not be empty"
    corpus = json.loads((REPO_ROOT / "tests" / "benchmark" / "scenarios" / "full.json").read_text(encoding="utf-8"))
    scenarios = {item["scenario_id"]: item for item in corpus.get("scenarios", corpus)}
    for row in rows:
        scenario_id = row["source"]["scenario_id"]
        assert row["id"] == f"{scenario_id}:style"
        assert row["state"] == scenarios[scenario_id]["description"]
        assert row["instructions"] == seam_prompts.STYLE_INSTRUCTIONS
        assert row["criteria"] == criteria
        assert row["label"] in criteria
        assert row["source"]["split"] == scenarios[scenario_id]["split"]
    assert {row["source"]["scenario_id"] for row in rows} == set(scenarios)
    assert STYLE_QUESTION_ID == "architecture_style"


def test_style_criteria_matches_the_shared_vocabulary() -> None:
    """The served criteria are the shared glosses in affinity order + ``none`` last."""
    from src.decisions.client import build_style_criteria
    from src.style_affinity import STYLE_AFFINITY

    criteria = build_style_criteria()
    assert list(criteria) == [*list(STYLE_AFFINITY), seam_prompts.STYLE_NONE_KEY]
    glosses = {key: value for key, value in criteria.items() if key != seam_prompts.STYLE_NONE_KEY}
    assert glosses == seam_prompts.STYLE_GLOSSES


@pytest.fixture(scope="module")
def eval_gate() -> types.ModuleType:
    return _load("clm_eval_gate", "clm_eval_gate.py")


def _gate_manifest(**overrides: Any) -> dict[str, Any]:
    """A manifest shaped like ``export_clm_dataset.py`` v2: 15 holdout rows per question."""
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "splits": {
            "train": {
                "source_split": "train",
                "rows": 70,
                "per_question": {"architecture_style": 70, "category": 70},
                "label_coverage": {
                    # R6-neutral default: train supports every test label, so tests that do
                    # not target R6 keep their expectations (R6 tests mutate this block).
                    "architecture_style": {
                        "labels": {
                            "monolithic": 20,
                            "none": 11,
                            "broker": 2,
                            "saga": 1,
                            "api-gateway": 2,
                            "sse": 1,
                            "x": 1,
                        },
                        "uncovered": ["hexagonal"],
                    },
                    "category": {
                        "labels": {"domain-modeling": 12, "behavior-selection": 2, "persistence-state": 14},
                        "uncovered": [],
                    },
                },
            },
            "test": {
                "source_split": "holdout",
                "rows": 30,
                "per_question": {"architecture_style": 15, "category": 15},
                "label_coverage": {
                    "architecture_style": {
                        "labels": {
                            "monolithic": 4,
                            "none": 3,
                            "broker": 2,
                            "saga": 2,
                            "api-gateway": 2,
                            "sse": 1,
                            "x": 1,
                        },
                        "uncovered": ["saga"],
                    },
                    "category": {
                        "labels": {"domain-modeling": 5, "behavior-selection": 4, "persistence-state": 3},
                        "uncovered": [],
                    },
                },
            },
        },
    }
    manifest.update(overrides)
    return manifest


def _gate_summary(**overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "best_epoch": 3,
        "val_acc": 0.8,
        "test": {"acc": 0.75, "soft_ce": 0.5, "per_question": {"architecture_style": 0.7, "category": 0.8}},
        "init_test": {
            "acc": 0.6,
            "soft_ce": 0.7,
            "per_question": {"architecture_style": 0.4, "category": 0.5},
        },
        "majority_test": 0.3,
        "train_questions": 70,
    }
    summary.update(overrides)
    return summary


@lru_cache(maxsize=1)
def _gate_module() -> types.ModuleType:
    return _load("clm_eval_gate", "clm_eval_gate.py")


def _served_gate(
    summary: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    questions: tuple[str, ...] = ("architecture_style",),
    advisory: tuple[str, ...] = ("category",),
    ledger: dict[str, Any] | None = None,
    source_digests: dict[str, str | None] | None = None,
) -> Any:
    """Run the gate with the served question binding and a v2 manifest.

    R7-neutral by default: the fixture's ``uncovered`` criteria arrive
    acknowledged at the fixture digest, so tests that do not target R7 keep
    their expectations (pass ``ledger={}`` to exercise the unacknowledged path).
    """
    resolved = _gate_manifest() if manifest is None else manifest
    digests: dict[str, str | None] = (
        {"choice": "fixture-choice", "style": "fixture-style"} if source_digests is None else source_digests
    )
    active: dict[str, Any] | None
    if ledger is None:
        train_cov: dict[str, Any] = ((resolved.get("splits") or {}).get("train") or {}).get("label_coverage") or {}
        style_cov: dict[str, Any] = train_cov.get("architecture_style") or {}
        uncovered: list[str] = style_cov.get("uncovered", [])
        active = {
            "coverage_ack": {
                "architecture_style": [
                    {
                        "criterion": key,
                        "reviewer": "fixture",
                        "note": "R7-neutral default",
                        "source_sha256": digests["style"],
                    }
                    for key in uncovered
                ]
            },
            "reviews": {"choice": [], "style": []},
        }
    else:
        active = ledger
    return _gate_module().check(
        summary, resolved, list(questions), list(advisory), active, digests if active is not None else None
    )


def test_eval_gate_passes_an_improved_head() -> None:
    assert _served_gate(_gate_summary()).failures == []


def test_eval_gate_passes_an_equal_head() -> None:
    """No regression = PASS once some epoch improved val (equal test, best_epoch > 0)."""
    summary = _gate_summary()
    summary["test"]["acc"] = summary["init_test"]["acc"]
    summary["test"]["per_question"] = dict(summary["init_test"]["per_question"])
    assert _served_gate(summary).failures == []


def test_eval_gate_fails_a_no_improvement_run() -> None:
    """best_epoch 0 ships the released weights — never as ``runtime-finetuned``."""
    summary = _gate_summary()
    summary["best_epoch"] = 0
    summary["test"]["acc"] = summary["init_test"]["acc"]
    summary["test"]["per_question"] = dict(summary["init_test"]["per_question"])
    failures = _served_gate(summary).failures
    assert len(failures) == 1
    assert "best_epoch is 0" in failures[0]


def test_eval_gate_fails_a_regressed_head() -> None:
    summary = _gate_summary()
    summary["test"]["acc"] = 0.55
    failures = _served_gate(summary).failures
    assert any("regressed below init" in failure for failure in failures)


def test_eval_gate_tolerates_exactly_one_row_of_regression() -> None:
    """15 rows → one flipped row (1/15) is inside the measurement's granularity."""
    summary = _gate_summary()
    summary["test"]["per_question"]["architecture_style"] = 7 / 15
    summary["init_test"]["per_question"]["architecture_style"] = 8 / 15
    assert _served_gate(summary).failures == []
    summary["test"]["per_question"]["architecture_style"] = 6 / 15
    assert any("more than one row" in failure for failure in _served_gate(summary).failures)


def test_eval_gate_fails_closed_without_a_per_question_baseline() -> None:
    """A missing init baseline must fail, never silently disable the regression rule."""
    summary = _gate_summary()
    summary["init_test"] = {"acc": 0.6, "soft_ce": 0.7}
    assert any("no per-question baseline" in failure for failure in _served_gate(summary).failures)


def test_eval_gate_fails_when_the_head_only_learned_the_prior() -> None:
    """14/15 labels monolithic on the test shard → the head must beat 0.9333."""
    manifest = _gate_manifest()
    manifest["splits"]["test"]["label_coverage"]["architecture_style"]["labels"] = {"monolithic": 14, "none": 1}
    failures = _served_gate(_gate_summary(), manifest).failures
    assert len(failures) == 1
    assert "majority-label share" in failures[0]


def test_eval_gate_skips_the_prior_rule_on_a_single_label_shard() -> None:
    """A one-label shard makes the prior rule vacuous: SKIP, and never a quiet pass."""
    manifest = _gate_manifest()
    manifest["splits"]["test"]["label_coverage"]["architecture_style"]["labels"] = {"monolithic": 15}
    result = _served_gate(_gate_summary(), manifest)
    assert result.failures == []
    assert [entry["rule"] for entry in result.skipped] == ["R5-prior"]


def test_eval_gate_fails_closed_without_a_manifest() -> None:
    """No manifest (absent file, or a layout the gate does not know) blocks the build."""
    module = _gate_module()
    assert any(
        "schema_version" in failure
        for failure in module.check(_gate_summary(), None, ["architecture_style"], ["category"]).failures
    )
    assert any(
        "no rows for question ids" in failure
        for failure in module.check(_gate_summary(), {}, ["architecture_style"], ["category"]).failures
    )


def test_eval_gate_load_manifest_rejects_an_unknown_schema(tmp_path: Path) -> None:
    module = _gate_module()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "splits": {}}), encoding="utf-8")
    assert module.load_manifest(path) is None
    path.write_text(json.dumps(_gate_manifest()), encoding="utf-8")
    assert module.load_manifest(path) is not None


def test_eval_gate_reports_the_unserved_question_without_gating_it() -> None:
    """No CLM client sends the category question: a regression there is advisory."""
    summary = _gate_summary()
    summary["test"]["per_question"]["category"] = 0.1
    result = _served_gate(summary)
    assert result.failures == []
    assert any(check["rule"] == "R3-advisory" and "test 0.1000" in check["detail"] for check in result.checks)


def test_eval_gate_fails_a_missing_question() -> None:
    """Every manifest question the gate reasons about must have test rows."""
    manifest = _gate_manifest()
    del manifest["splits"]["test"]["per_question"]["category"]
    failures = _served_gate(_gate_summary(), manifest).failures
    assert any("no rows for question ids" in failure for failure in failures)


def test_eval_gate_fails_without_accuracies() -> None:
    assert _served_gate({}).failures != []


def test_eval_gate_fails_a_test_label_without_train_support() -> None:
    """R6: the fixture's train block lacks saga — a saga test label blocks the build."""
    manifest = _gate_manifest()
    manifest["splits"]["train"]["label_coverage"]["architecture_style"]["labels"] = {
        "monolithic": 20,
        "none": 11,
    }
    result = _served_gate(_gate_summary(), manifest)
    assert any("R6-support" in failure and "saga" in failure for failure in result.failures)


def test_eval_gate_passes_r6_when_train_supports_every_test_label() -> None:
    """R6 is silent when the train histogram covers every test label."""
    manifest = _gate_manifest()
    test_labels = manifest["splits"]["test"]["label_coverage"]["architecture_style"]["labels"]
    manifest["splits"]["train"]["label_coverage"]["architecture_style"]["labels"] = dict(test_labels)
    result = _served_gate(_gate_summary(), manifest)
    assert result.failures == []
    assert any(check["rule"] == "R6-support" and check["status"] == "pass" for check in result.checks)


def test_eval_gate_fails_unacknowledged_train_uncovered_criteria() -> None:
    """R7: the fixture's uncovered ``hexagonal`` blocks the build with an empty ledger."""
    result = _served_gate(_gate_summary(), ledger={})
    assert any("R7-coverage" in failure and "hexagonal" in failure for failure in result.failures)


def test_eval_gate_passes_r7_when_coverage_is_acknowledged() -> None:
    """R7 passes when the ledger acks every uncovered key at the source digest."""
    result = _served_gate(_gate_summary())
    assert result.failures == []
    assert any(check["rule"] == "R7-coverage" and check["status"] == "pass" for check in result.checks)


def test_eval_gate_voids_coverage_ack_on_stale_digest() -> None:
    """An ack recorded against other bytes does not certify the current export."""
    ledger: dict[str, Any] = {
        "coverage_ack": {
            "architecture_style": [
                {
                    "criterion": "hexagonal",
                    "reviewer": "ada",
                    "note": "reviewed last quarter",
                    "source_sha256": "stale-digest",
                }
            ]
        },
        "reviews": {"choice": [], "style": []},
    }
    result = _served_gate(_gate_summary(), ledger=ledger)
    assert any("R7-coverage" in failure for failure in result.failures)


def test_eval_gate_reports_unreviewed_category_without_blocking() -> None:
    """R8: unreviewed category gold is a visible caveat, never a failure."""
    result = _served_gate(_gate_summary())
    assert result.failures == []
    notes = [check for check in result.checks if check["rule"] == "R8-choice-review"]
    assert len(notes) == 1
    assert "stays unreviewed" in notes[0]["detail"]


def test_eval_gate_reports_reviewed_category_when_certified() -> None:
    """R8 flips to reviewed once 100 choice rows are certified at the digest."""
    ledger: dict[str, Any] = {
        "coverage_ack": {
            "architecture_style": [
                {
                    "criterion": "hexagonal",
                    "reviewer": "ada",
                    "note": "ok",
                    "source_sha256": "fixture-style",
                }
            ]
        },
        "reviews": {
            "choice": [
                {
                    "scenario_id": f"s{index}",
                    "label": "domain-modeling",
                    "reviewer": "ada",
                    "label_ok": True,
                    "source_sha256": "fixture-choice",
                }
                for index in range(100)
            ],
            "style": [],
        },
    }
    result = _served_gate(_gate_summary(), ledger=ledger)
    assert result.failures == []
    notes = [check for check in result.checks if check["rule"] == "R8-choice-review"]
    assert "category gold reviewed" in notes[0]["detail"]


def test_eval_gate_main_writes_the_runbook_line_and_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    eval_gate = _load("clm_eval_gate", "clm_eval_gate.py")
    summary_path = tmp_path / "finetune_summary.json"
    summary_path.write_text(json.dumps(_gate_summary()), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_gate_manifest()), encoding="utf-8")
    choice_path = tmp_path / "choice.jsonl"
    choice_path.write_text("{}\n", encoding="utf-8")
    style_path = tmp_path / "style.jsonl"
    style_bytes = b"{}\n"
    style_path.write_bytes(style_bytes)
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "coverage_ack": {
                    "architecture_style": [
                        {
                            "criterion": "hexagonal",
                            "reviewer": "fixture",
                            "note": "R7-neutral default",
                            "source_sha256": hashlib.sha256(style_bytes).hexdigest(),
                        }
                    ]
                },
                "reviews": {"choice": [], "style": []},
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "gate.json"
    args = [
        "--summary",
        str(summary_path),
        "--manifest",
        str(manifest_path),
        "--questions",
        "architecture_style",
        "--advisory-questions",
        "category",
        "--choice",
        str(choice_path),
        "--style",
        str(style_path),
        "--review",
        str(review_path),
        "--report",
        str(report_path),
    ]
    assert eval_gate.main(args) == 0
    assert "clm-eval-gate: PASS" in capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["questions"] == ["architecture_style"]
    assert report["advisory_questions"] == ["category"]
    with pytest.raises(SystemExit, match="does not exist"):
        eval_gate.main(["--summary", str(tmp_path / "absent.json")])


def test_eval_gate_main_fails_and_still_writes_a_report(tmp_path: Path) -> None:
    eval_gate = _load("clm_eval_gate", "clm_eval_gate.py")
    summary = _gate_summary()
    summary["best_epoch"] = 0
    summary_path = tmp_path / "finetune_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_gate_manifest()), encoding="utf-8")
    report_path = tmp_path / "gate.json"
    code = eval_gate.main(
        ["--summary", str(summary_path), "--manifest", str(manifest_path), "--report", str(report_path)]
    )
    assert code == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "fail"


@pytest.fixture(scope="module")
def rank_parity() -> types.ModuleType:
    return _load("clm_rank_parity", "clm_rank_parity.py")


def _payload_row(
    scenario_id: str, answers: list[str], question: str | None = None, relevance: list[int] | None = None
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "family": "family-a",
        "context": "",
        "question": question or f"decide {scenario_id}",
        "answers": answers,
        "answer_names": [f"pattern-{index}" for index in range(len(answers))],
        "relevance": relevance if relevance is not None else [3] + [0] * (len(answers) - 1),
    }


def _write_payloads(path: Path, payloads: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(payload) for payload in payloads), encoding="utf-8")
    return path


def test_rank_payloads_load_in_order(rank_parity: types.ModuleType, tmp_path: Path) -> None:
    """The artifact is replayed as written: one served-shape payload per scenario."""
    path = _write_payloads(
        tmp_path / "payloads.jsonl",
        [_payload_row("s2", ["a", "b"]), _payload_row("s1", ["a", "b"])],
    )
    payloads = rank_parity.load_payloads(path)
    assert [payload["scenario_id"] for payload in payloads] == ["s2", "s1"]
    assert payloads[0]["answers"] == ["a", "b"]


def test_rank_payloads_refuse_a_single_answer_payload(rank_parity: types.ModuleType, tmp_path: Path) -> None:
    """A one-answer payload has no ranking to measure — refuse rather than score it."""
    path = _write_payloads(tmp_path / "payloads.jsonl", [_payload_row("s1", ["only"])])
    with pytest.raises(SystemExit, match="fewer than two answers"):
        rank_parity.load_payloads(path)


def test_rank_payloads_refuse_an_empty_file(rank_parity: types.ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "payloads.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="no payloads"):
        rank_parity.load_payloads(path)


def test_rank_payloads_refuse_a_payload_without_relevance(rank_parity: types.ModuleType, tmp_path: Path) -> None:
    """nDCG is a gate now: a payload without aligned grades stops the build."""
    row = _payload_row("s1", ["a", "b"])
    del row["relevance"]
    path = _write_payloads(tmp_path / "payloads.jsonl", [row])
    with pytest.raises(SystemExit, match="no relevance grades"):
        rank_parity.load_payloads(path)


def test_ndcg_scores_graded_rankings(rank_parity: types.ModuleType) -> None:
    """Ideal ordering is 1.0; burying the primary below a negative costs DCG."""
    answers = ["best", "mid", "junk"]
    relevance = [3.0, 1.0, 0.0]
    assert rank_parity.ndcg_at(["best", "mid", "junk"], answers, relevance, 10) == pytest.approx(1.0)
    buried = 1.0 + 3.0 / 1.584962500721156  # gain-1 at rank 1, gain-3 demoted to rank 2 (log2(3))
    ideal = 3.0 + 1.0 / 1.584962500721156
    assert rank_parity.ndcg_at(["mid", "best", "junk"], answers, relevance, 10) == pytest.approx(buried / ideal)
    assert rank_parity.ndcg_at(["junk", "junk"], answers, relevance, 10) == pytest.approx(0.0)
    # All-negative payload: there is nothing to rank, so any ordering is perfect.
    assert rank_parity.ndcg_at(["junk"], ["junk"], [0.0], 10) == pytest.approx(1.0)


def test_top5_overlap_measures_head_survival(rank_parity: types.ModuleType) -> None:
    """Identity is 1.0, a disjoint head is 0.0, and only the first five count."""
    head = ["a", "b", "c", "d", "e"]
    assert rank_parity.top5_overlap(head, head) == pytest.approx(1.0)
    assert rank_parity.top5_overlap(head, ["f", "g"]) == pytest.approx(0.0)
    assert rank_parity.top5_overlap(head, ["b", "a"]) == pytest.approx(0.4)
    assert rank_parity.top5_overlap(["a"], ["b"]) == pytest.approx(0.0)


def test_rank_gate_requires_an_explicit_budget(rank_parity: types.ModuleType, tmp_path: Path) -> None:
    """The budget is a reviewed constant at the call site, never a hidden default."""
    path = _write_payloads(tmp_path / "payloads.jsonl", [_payload_row("s1", ["a", "b"])])
    with pytest.raises(SystemExit, match="--max-flip-rate is required"):
        rank_parity.main(
            [
                "--payloads",
                str(path),
                "--released",
                "r.pt",
                "--trained",
                "t.pt",
                "--clm-src",
                str(tmp_path),
                "--emb-url",
                "http://127.0.0.1:8090/v1/embeddings",
            ]
        )


def test_rank_gate_counts_flips_and_writes_the_report(
    rank_parity: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate compares released vs trained through the served rank call and reports."""
    path = _write_payloads(
        tmp_path / "payloads.jsonl",
        # Uniform gains: nDCG is 1.0 under any ordering, so this test isolates the flip budget.
        [
            _payload_row(f"s{index}", ["p0", "p1", "p2", "p3", "p4", "p5"], relevance=[1, 1, 1, 1, 1, 1])
            for index in range(4)
        ],
    )

    class _StubEngine:
        def rank(self, context: str, candidates: list[str], instructions: Any, model: str) -> list[dict[str, Any]]:
            assert context == ""
            order = candidates if model == "released" or instructions != "decide s0" else candidates[::-1]
            return [{"rank": index + 1, "candidate": text, "prob": 0.9} for index, text in enumerate(order)]

    class _StubEmbedder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    stub_engine = types.ModuleType("clm.engine")
    stub_engine.Engine = lambda *args, **kwargs: _StubEngine()  # type: ignore[attr-defined]
    stub_embedder = types.ModuleType("clm.embedder")
    stub_embedder.Embedder = _StubEmbedder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "clm.engine", stub_engine)
    monkeypatch.setitem(sys.modules, "clm.embedder", stub_embedder)
    report = tmp_path / "parity.json"
    base = [
        "--payloads",
        str(path),
        "--released",
        "released.pt",
        "--trained",
        "trained.pt",
        "--clm-src",
        str(tmp_path),
        "--emb-url",
        "http://127.0.0.1:8090/v1/embeddings",
        "--report",
        str(report),
    ]
    assert rank_parity.main([*base, "--max-flip-rate", "0.25"]) == 0
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["status"] == "pass"
    assert written["flip_rate"] == pytest.approx(0.25)
    assert [flip["scenario_id"] for flip in written["flips"]] == ["s0"]
    # Names, not answer texts: document_text runs to kilobytes and the report is sha-pinned.
    assert written["flips"][0] == {"scenario_id": "s0", "released": "pattern-0", "trained": "pattern-5"}
    # Reversing a six-answer list keeps four of the five released head entries.
    assert written["mean_top5_overlap"] == pytest.approx((3 * 1.0 + 0.8) / 4)
    assert written["mean_trained_ndcg10"] == pytest.approx(1.0)
    assert written["mean_released_ndcg10"] == pytest.approx(1.0)
    assert rank_parity.main([*base, "--max-flip-rate", "0.24"]) == 1


def test_rank_gate_fails_a_rank_regression_within_flip_budget(
    rank_parity: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flip toward a worse ranking fails even when the flip count is inside budget."""
    path = _write_payloads(
        tmp_path / "payloads.jsonl",
        [_payload_row("s0", ["best", "junk"], relevance=[3, 0])],
    )

    class _StubEngine:
        def rank(self, context: str, candidates: list[str], instructions: Any, model: str) -> list[dict[str, Any]]:
            del context, instructions
            order = candidates if model == "released" else candidates[::-1]
            return [{"rank": index + 1, "candidate": text, "prob": 0.9} for index, text in enumerate(order)]

    class _StubEmbedder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    stub_engine = types.ModuleType("clm.engine")
    stub_engine.Engine = lambda *args, **kwargs: _StubEngine()  # type: ignore[attr-defined]
    stub_embedder = types.ModuleType("clm.embedder")
    stub_embedder.Embedder = _StubEmbedder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "clm.engine", stub_engine)
    monkeypatch.setitem(sys.modules, "clm.embedder", stub_embedder)
    report = tmp_path / "parity.json"
    code = rank_parity.main(
        [
            "--payloads",
            str(path),
            "--released",
            "released.pt",
            "--trained",
            "trained.pt",
            "--clm-src",
            str(tmp_path),
            "--emb-url",
            "http://127.0.0.1:8090/v1/embeddings",
            "--max-flip-rate",
            "1.0",
            "--report",
            str(report),
        ]
    )
    assert code == 1
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["status"] == "fail"
    assert written["mean_released_ndcg10"] == pytest.approx(1.0)
    # Reversed: the gain-3 answer sits at rank 2, so nDCG = (3/log2(3)) / 3.
    assert written["mean_trained_ndcg10"] == pytest.approx(1 / 1.584962500721156)


@pytest.fixture(scope="module")
def calibrator() -> types.ModuleType:
    return _load("clm_calibrate", "clm_calibrate.py")


def test_calibration_metrics_match_definitions(calibrator: types.ModuleType) -> None:
    """ECE = binned |acc − conf|, Brier = mean (conf − outcome)²."""
    pairs = [(0.9, True), (0.9, False), (0.1, False), (0.1, False)]
    assert calibrator.brier_score(pairs) == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)
    ece = calibrator.expected_calibration_error(pairs, 10)
    assert ece == pytest.approx(0.5 * abs(0.5 - 0.9) + 0.5 * abs(0.0 - 0.1))


def test_calibration_table_reports_override_accuracy(calibrator: types.ModuleType) -> None:
    """Acted accuracy is over overrides only; confident ``none`` answers are stratified out."""
    pairs = [
        (0.9, True, "hexagonal"),
        (0.8, True, "hexagonal"),
        (0.4, False, "hexagonal"),
        (0.2, False, "hexagonal"),
    ]
    report = calibrator.summarize(pairs, [0.5, 0.85], 10)
    assert report["n"] == 4
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["override_rate"] == pytest.approx(1.0)
    by_floor = {row["floor"]: row for row in report["floors"]}
    assert by_floor[0.5]["acted"] == 2
    assert by_floor[0.5]["acted_accuracy"] == pytest.approx(1.0)
    assert by_floor[0.85]["acted"] == 1
    assert report["confidence"]["min"] == pytest.approx(0.2)
    assert report["confidence"]["max"] == pytest.approx(0.9)


def test_calibration_table_stratifies_abstentions(calibrator: types.ModuleType) -> None:
    """A confident ``none`` is a seam no-op: counted separately, never in the enable bar."""
    pairs = [(0.9, True, "hexagonal"), (0.9, True, "none"), (0.9, False, "none")]
    by_floor = {row["floor"]: row for row in calibrator.summarize(pairs, [0.5], 10)["floors"]}
    row = by_floor[0.5]
    assert row["acted"] == 1
    assert row["acted_rate"] == pytest.approx(1 / 3)
    assert row["acted_accuracy"] == pytest.approx(1.0)
    assert row["none_acted"] == 2
    assert row["none_accuracy"] == pytest.approx(0.5)
    low, high = row["acted_accuracy_ci95"]
    assert low <= 1.0 <= high


def test_calibration_table_bands_small_samples(calibrator: types.ModuleType) -> None:
    """Every floor row carries a 95% Wilson interval; an empty override set spans [0, 1]."""
    pairs = [(0.9, True, "none")]
    by_floor = {row["floor"]: row for row in calibrator.summarize(pairs, [0.5], 10)["floors"]}
    row = by_floor[0.5]
    assert row["acted"] == 0
    assert row["acted_accuracy"] is None
    assert row["acted_accuracy_ci95"] == [0.0, 1.0]
    low, high = calibrator.wilson_interval(24, 30)
    assert 0.0 < low < 0.8 < high < 1.0


def test_calibration_metric_selects_the_floored_number(calibrator: types.ModuleType) -> None:
    answer = {"choice": "x", "confidence": 0.3, "probabilities": {"x": 0.6, "y": 0.4}}
    assert calibrator.metric_of(answer, "confidence") == pytest.approx(0.3)
    assert calibrator.metric_of(answer, "top_prob") == pytest.approx(0.6)


def test_calibration_replays_holdout_only_by_default(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay asks the served question shape and scores choice vs label."""
    seen: list[dict[str, Any]] = []
    path = tmp_path / "style.jsonl"
    rows = [
        {
            "id": "s1:style",
            "state": "system one",
            "instructions": "Which style?",
            "criteria": {"a": "A", "b": "B"},
            "label": "a",
            "source": {"split": "holdout", "scenario_id": "s1"},
        },
        {
            "id": "s2:style",
            "state": "system two",
            "instructions": "Which style?",
            "criteria": {"a": "A", "b": "B"},
            "label": "b",
            "source": {"split": "train", "scenario_id": "s2"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def _fake_ask(base_url: str, model: str, row: dict[str, Any], question_id: str, timeout: float) -> dict[str, Any]:
        seen.append({"model": model, "state": row["state"], "qid": question_id})
        return {"choice": "a", "confidence": 0.7, "probabilities": {"a": 0.7, "b": 0.3}}

    monkeypatch.setattr(calibrator, "ask", _fake_ask)
    assert calibrator.main(["--style", str(path), "--base-url", "http://x/v1", "--floors", "0.5"]) == 0
    assert [entry["state"] for entry in seen] == ["system one"]
    assert seen[0]["qid"] == "architecture_style"


def test_recommend_picks_the_lowest_qualifying_floor(calibrator: types.ModuleType) -> None:
    """The recommendation is the lowest floor clearing both bars, not the most accurate one."""
    pairs = [
        (0.9, True, "hexagonal"),
        (0.8, True, "hexagonal"),
        (0.7, False, "hexagonal"),
        (0.6, False, "hexagonal"),
        (0.2, False, "hexagonal"),
    ]
    table = calibrator.summarize(pairs, [0.5, 0.65, 0.85], 10)["floors"]
    # 0.5 acts on four rows at 0.50 accuracy (fails the 0.6 bar); 0.65 acts on two of
    # three at 0.667 — so the recommendation is the lowest floor that clears both.
    # min_acted=1: this test targets floor selection, not the small-n guard.
    recommended = calibrator.recommend(table, 0.6, 0.4, 1)
    assert recommended["floor"] == pytest.approx(0.65)
    assert "acted accuracy" in recommended["reason"]


def test_recommend_refuses_a_floor_below_the_min_acted_count(calibrator: types.ModuleType) -> None:
    """A perfect floor on too few overrides never qualifies, however accurate."""
    pairs = [(0.9, True, "hexagonal"), (0.9, True, "hexagonal")]
    table = calibrator.summarize(pairs, [0.5], 10)["floors"]
    assert calibrator.recommend(table, 0.8, 0.1, 10)["floor"] is None
    assert calibrator.recommend(table, 0.8, 0.1, 2)["floor"] == pytest.approx(0.5)
    qualified = calibrator.recommend(table, 0.8, 0.1, 2)
    assert qualified["acted"] == 2
    assert qualified["acted_accuracy_ci95"][0] <= 1.0 <= qualified["acted_accuracy_ci95"][1]


def test_recommend_reports_nothing_when_no_floor_qualifies(calibrator: types.ModuleType) -> None:
    pairs = [(0.9, False, "hexagonal"), (0.8, False, "hexagonal"), (0.7, False, "hexagonal")]
    table = calibrator.summarize(pairs, [0.5, 0.85], 10)["floors"]
    recommended = calibrator.recommend(table, 0.9, 0.3)
    assert recommended["floor"] is None
    assert "cannot be enabled safely" in recommended["reason"]


def test_recommend_reports_only_without_thresholds(calibrator: types.ModuleType) -> None:
    table = calibrator.summarize([(0.9, True, "hexagonal")], [0.5], 10)["floors"]
    assert calibrator.recommend(table, None, None)["floor"] is None
    assert calibrator.recommend(table, 0.5, None)["floor"] is None


def test_calibration_main_fails_when_no_floor_qualifies(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corpus whose head is never right must not yield an enableable floor."""
    path = tmp_path / "style.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "s1:style",
                "state": "system one",
                "instructions": "Which style?",
                "criteria": {"a": "A", "b": "B"},
                "label": "a",
                "source": {"split": "holdout", "scenario_id": "s1"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "b", "confidence": 0.9, "probabilities": {"a": 0.1, "b": 0.9}},
    )
    report = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(path),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--report",
            str(report),
        ]
    )
    assert code == 2
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["recommended"]["floor"] is None
    assert written["floors"][0]["acted_accuracy"] == pytest.approx(0.0)


def _write_labels_source(path: Path) -> str:
    """A minimal STYLE_LABELS module; returns its sha256 for the test ledger."""
    source = 'STYLE_LABELS: dict[str, tuple[str, str]] = {\n    "s1": ("a", "rationale one"),\n}\n'
    path.write_text(source, encoding="utf-8")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _write_style_row(path: Path, rows: int = 1) -> None:
    """One acted override per row (all choice ``a``); ``rows`` clears the min-acted bar."""
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"s{index}:style",
                    "state": f"system {index}",
                    "instructions": "Which style?",
                    "criteria": {"a": "A", "b": "B"},
                    "label": "a",
                    "source": {"split": "holdout", "scenario_id": f"s{index}"},
                }
            )
            for index in range(1, rows + 1)
        ),
        encoding="utf-8",
    )


def test_calibration_main_refuses_to_recommend_without_human_review(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The table prints, but an unreviewed ledger blocks the floor and fails the run."""
    style = tmp_path / "style.jsonl"
    _write_style_row(style, rows=10)
    digest = _write_labels_source(tmp_path / "labels.py")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"status": "unreviewed", "reviews": {"style": []}}), encoding="utf-8")
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}},
    )
    report = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--labels",
            str(tmp_path / "labels.py"),
            "--review",
            str(review),
            "--report",
            str(report),
        ]
    )
    assert code == 2
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["label_review"]["certified"] is False
    assert written["recommended"]["floor"] is None
    assert "not 'reviewed'" in written["recommended"]["reason"]
    assert written["floors"][0]["acted_accuracy"] == pytest.approx(1.0)
    assert digest  # the ledger below pins these exact bytes


def test_calibration_main_recommends_once_the_ledger_certifies(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full coverage at the recorded sha unlocks the recommendation the table supports."""
    style = tmp_path / "style.jsonl"
    _write_style_row(style, rows=10)
    digest = _write_labels_source(tmp_path / "labels.py")
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "status": "reviewed",
                "reviewers": ["ada"],
                "reviews": {
                    "style": [
                        {
                            "scenario_id": "s1",
                            "label": "a",
                            "reviewer": "ada",
                            "rationale_ok": True,
                            "labels_sha256": digest,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}},
    )
    report = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--labels",
            str(tmp_path / "labels.py"),
            "--review",
            str(review),
            "--report",
            str(report),
        ]
    )
    assert code == 0
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["label_review"]["certified"] is True
    assert written["recommended"]["floor"] == pytest.approx(0.5)


def test_calibration_main_voids_coverage_when_labels_change(
    calibrator: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label edit after review changes the bytes, so the recorded sha no longer certifies."""
    style = tmp_path / "style.jsonl"
    _write_style_row(style, rows=10)
    digest = _write_labels_source(tmp_path / "labels.py")
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "status": "reviewed",
                "reviews": {
                    "style": [
                        {
                            "scenario_id": "s1",
                            "label": "a",
                            "reviewer": "ada",
                            "rationale_ok": True,
                            "labels_sha256": digest,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    with (tmp_path / "labels.py").open("a", encoding="utf-8") as handle:
        handle.write("# post-review tweak\n")
    monkeypatch.setattr(
        calibrator,
        "ask",
        lambda *args, **kwargs: {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}},
    )
    report = tmp_path / "calibration.json"
    code = calibrator.main(
        [
            "--style",
            str(style),
            "--base-url",
            "http://x/v1",
            "--floors",
            "0.5",
            "--min-acted-accuracy",
            "0.8",
            "--min-acted-rate",
            "0.1",
            "--labels",
            str(tmp_path / "labels.py"),
            "--review",
            str(review),
            "--report",
            str(report),
        ]
    )
    assert code == 2
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["recommended"]["floor"] is None


def test_label_coverage_names_histograms_and_gaps(exporter: types.ModuleType) -> None:
    """The manifest records per-QUESTION label counts and the criteria keys with zero rows."""
    rows = [
        {
            "id": "s1:category",
            "state": "s",
            "questions": json.dumps({"category": {"criteria": {"a": "A", "b": "B"}}}),
            "gold": json.dumps({"category": {"label": "a"}}),
        },
        {
            "id": "s2:category",
            "state": "s",
            "questions": json.dumps({"category": {"criteria": {"a": "A", "b": "B"}}}),
            "gold": json.dumps({"category": {"label": "a"}}),
        },
        {
            "id": "s1:style",
            "state": "s",
            "questions": json.dumps({"architecture_style": {"criteria": {"x": "X", "y": "Y"}}}),
            "gold": json.dumps({"architecture_style": {"label": "y"}}),
        },
    ]
    coverage = exporter.label_coverage(rows)
    assert coverage["category"] == {"labels": {"a": 2}, "uncovered": ["b"]}
    assert coverage["architecture_style"] == {"labels": {"y": 1}, "uncovered": ["x"]}
    assert exporter.per_question_counts(rows) == {"architecture_style": 1, "category": 2}


def test_manifest_schema_matches_the_gate(exporter: types.ModuleType) -> None:
    """Exporter and gate must agree on the manifest layout, or the gate fails closed."""
    assert exporter.MANIFEST_SCHEMA_VERSION == _gate_module().SUPPORTED_MANIFEST_SCHEMA


def test_manifest_records_the_clm_pin_the_exporter_assumes(exporter: types.ModuleType) -> None:
    """The pin selects which upstream adapter revision the row layout must match."""
    args = exporter.parse_args(["--out", "/tmp/x", "--clm-pin", "abc1234"])
    assert args.clm_pin == "abc1234"
    assert exporter.parse_args(["--out", "/tmp/x"]).clm_pin == "d5f9ef0"


def test_calibrator_none_choice_matches_the_served_vocabulary(calibrator: types.ModuleType) -> None:
    """The stratified ``none`` must be the seam's own abstention key, not a lookalike."""
    assert calibrator.NONE_CHOICE == seam_prompts.STYLE_NONE_KEY


def test_split_summary_carries_counts_and_coverage(exporter: types.ModuleType) -> None:
    rows = [
        {
            "id": "s1:style",
            "state": "s",
            "questions": json.dumps({"architecture_style": {"criteria": {"x": "X", "y": "Y"}}}),
            "gold": json.dumps({"architecture_style": {"label": "x"}}),
        }
    ]
    summary = exporter.split_summary(rows, "holdout")
    assert summary["source_split"] == "holdout"
    assert summary["rows"] == 1
    assert summary["per_question"] == {"architecture_style": 1}
    assert summary["label_coverage"]["architecture_style"]["uncovered"] == ["y"]


def _typed_source_row(row_id: str, split: str, label: str = "a") -> dict[str, Any]:
    return {
        "id": row_id,
        "state": f"state for {row_id}",
        "type": "choice",
        "criteria": {"a": "A", "b": "B"},
        "label": label,
        "source": {"split": split, "scenario_id": row_id},
    }


def _stub_parquet(monkeypatch: pytest.MonkeyPatch, exporter: types.ModuleType) -> list[list[str]]:
    """Record the shard each row lands in without needing pyarrow."""
    shards: list[list[str]] = []

    def _write(rows: list[dict[str, Any]], path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        shards.append([str(row["id"]) for row in rows])
        return f"sha-{path.name}-{len(rows)}"

    monkeypatch.setattr(exporter, "write_parquet", _write)
    return shards


def test_export_routes_the_holdout_split_to_the_test_shard(
    exporter: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate's evidence is the family-disjoint holdout shard, not a train carve-out."""
    choice = tmp_path / "choice.jsonl"
    choice.write_text(
        "\n".join(
            json.dumps(row) for row in [_typed_source_row("s1", "train"), _typed_source_row("s2", "holdout", label="b")]
        ),
        encoding="utf-8",
    )
    style = tmp_path / "style.jsonl"
    style.write_text(
        "\n".join(
            json.dumps(row) for row in [_typed_source_row("s3", "train", label="b"), _typed_source_row("s4", "holdout")]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "data"
    shards = _stub_parquet(monkeypatch, exporter)
    assert exporter.main(["--choice", str(choice), "--style", str(style), "--out", str(out), "--workflow", "all"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == exporter.MANIFEST_SCHEMA_VERSION
    assert manifest["splits"]["train"]["source_split"] == "train"
    assert manifest["splits"]["test"]["source_split"] == "holdout"
    assert manifest["splits"]["train"]["per_question"] == {"architecture_style": 1, "category": 1}
    assert manifest["splits"]["test"]["per_question"] == {"architecture_style": 1, "category": 1}
    assert shards == [["s1", "s3"], ["s2", "s4"]]


def test_export_refuses_the_same_id_in_both_shards(
    exporter: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train and eval must never share a row; a corpus bug must stop the build."""
    choice = tmp_path / "choice.jsonl"
    choice.write_text(
        "\n".join(json.dumps(row) for row in [_typed_source_row("dup", "train"), _typed_source_row("dup", "holdout")]),
        encoding="utf-8",
    )
    style = tmp_path / "style.jsonl"
    style.write_text(
        "\n".join(
            json.dumps(row) for row in [_typed_source_row("s1", "train"), _typed_source_row("s2", "holdout", label="b")]
        ),
        encoding="utf-8",
    )
    _stub_parquet(monkeypatch, exporter)
    with pytest.raises(SystemExit, match="both splits"):
        exporter.main(["--choice", str(choice), "--style", str(style), "--out", str(tmp_path / "data")])


def test_rank_payload_artifact_is_served_shaped() -> None:
    """The committed artifact carries served bytes: document_text answers, holdout only."""
    payloads = [
        json.loads(line)
        for line in (REPO_ROOT / "qa" / "typed" / "rank_payloads.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corpus = json.loads((REPO_ROOT / "tests" / "benchmark" / "scenarios" / "full.json").read_text(encoding="utf-8"))
    scenarios = {item["scenario_id"]: item for item in corpus["scenarios"]}
    holdout = {sid for sid, item in scenarios.items() if item["split"] == "holdout"}
    assert {payload["scenario_id"] for payload in payloads} == holdout
    for payload in payloads:
        assert payload["context"] == ""
        assert payload["question"] == scenarios[payload["scenario_id"]]["decision_statement"]
        assert len(payload["answers"]) == len(payload["answer_names"]) == 20
        # document_text, not the noul "name: intent" short string.
        assert all(len(answer) > 1000 and "\n" in answer for answer in payload["answers"])
        assert all(isinstance(name, str) and name for name in payload["answer_names"])
        # Graded relevance travels with the payload: aligned gains, at least one positive.
        assert len(payload["relevance"]) == 20
        assert all(gain in (0, 1, 2, 3) for gain in payload["relevance"])
        assert any(gain > 0 for gain in payload["relevance"])
