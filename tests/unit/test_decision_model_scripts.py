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

import importlib.util
import json
import sys
import types
from collections.abc import Iterator
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
    """``typed_decision_examples`` reads {id, state, questions, gold}; so must we write it."""
    row = exporter.to_typed_row(
        {
            "id": "scenario:category",
            "state": "System goal: x\nDecision part: y",
            "instructions": "Which category?",
            "criteria": {"domain-modeling": "entities and invariants", "behavior-selection": "dispatch"},
            "label": "behavior-selection",
            "source": {"split": "train"},
        }
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
            }
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
