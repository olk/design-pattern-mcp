"""Shared pytest fixtures: real-catalog access, config, synthetic layout-V8 record builders.

The catalog fixture is session-scoped: loading the 486-record catalog (join + integrity gate +
canonical-order parse) takes a few seconds and the data is read-only in tests.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from src.patterns.loader import PatternLoader, load_catalog
from src.config import ServerConfig, TasksConfig
from src.tools.jobs import JobsStore

REPO_ROOT = Path(__file__).resolve().parents[2]
PATTERN_DIR = REPO_ROOT / "pattern"
DOCS_DIR = REPO_ROOT / "docs"

LANGUAGES: tuple[str, ...] = (
    "generic",
    "c",
    "cplusplus",
    "csharp",
    "go",
    "java",
    "javascript",
    "kotlin",
    "python",
    "ruby",
    "rust",
    "scala",
    "swift",
    "typescript",
)

QUALITY_ATTRIBUTES: dict[str, float] = {
    "testability": 6,
    "extensibility": 5,
    "maintainability": 7,
    "simplicity": 6,
    "performance": 5,
    "reliability": 5,
    "security": 5,
    "scalability": 5,
}


@pytest.fixture(scope="session")
def catalog() -> PatternLoader:
    """The real catalog, loaded once per session through the production loader."""
    return load_catalog(PATTERN_DIR, DOCS_DIR)


@pytest.fixture
def config_path() -> str:
    """Path to the repository configuration file (the deployment unit for now)."""
    return str(REPO_ROOT / "config" / "config.json")


@pytest.fixture
def server_config() -> ServerConfig:
    """Default server configuration (guide §14 defaults)."""
    return ServerConfig.model_validate({})


@pytest.fixture
def tasks_config() -> TasksConfig:
    """Heartbeat configuration with a short interval, for tool tests."""
    return TasksConfig(heartbeat_enabled=False, heartbeat_interval_seconds=5)


@pytest.fixture(autouse=True)
def jobs_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate JobsStore to a per-test SQLite file (house invariant 10)."""
    monkeypatch.setenv("JOBS_DB", str(tmp_path / "jobs.db"))


@pytest.fixture(autouse=True)
async def jobs_store_reset() -> None:
    """Drop the JobsStore singleton around every test."""
    await JobsStore.reset_for_test()
    yield
    await JobsStore.reset_for_test()


def write_record(
    root: Path,
    name: str,
    *,
    category: str = "behavior-selection",
    intent: str = "a pattern intent",
    problem: str = "a recurring problem",
    context: str = "a context",
    aliases: Sequence[str] = (),
    quality_attributes: dict[str, float] | None = None,
    suitable_scopes: Sequence[str] = ("object", "module"),
    unsuitable_scopes: Sequence[str] = (),
    related_patterns: Sequence[str] = (),
    support: str = "idiomatic",
    support_map: dict[str, str] | None = None,
    variants_by_language: dict[str, dict[str, object]] | None = None,
    omit_language_keys: Sequence[str] = (),
    declared_without_file: Sequence[str] = (),
    wrong_pattern_language: str | None = None,
    extra_language_content: dict[str, dict[str, object]] | None = None,
    extra_files: Sequence[str] = (),
) -> Path:
    """Write a layout-V8 main record plus its colocated language files under ``root``.

    Every keyword exists to let a test build one specific integrity violation. Returns the main
    record path.
    """
    root.mkdir(parents=True, exist_ok=True)
    file_map = {language: f"{name}.{language}.json" for language in LANGUAGES}
    for language in omit_language_keys:
        file_map.pop(language, None)
    main: dict[str, object] = {
        "category": category,
        "name": name,
        "intent": intent,
        "problem": problem,
        "context": context,
        "benefits": ["a benefit"],
        "tradeoffs": ["a tradeoff"],
        "quality_attributes": quality_attributes or dict(QUALITY_ATTRIBUTES),
        "aliases": list(aliases),
        "suitable_scopes": list(suitable_scopes),
        "unsuitable_scopes": list(unsuitable_scopes),
        "related_patterns": list(related_patterns),
        "language_files": file_map,
        "version": "1.0",
    }
    main_path = root / f"{name}.json"
    main_path.write_text(json.dumps(main, indent=2), encoding="utf-8")
    for language, declared in file_map.items():
        if language in declared_without_file:
            continue
        payload: dict[str, object] = {
            "pattern": name if language != wrong_pattern_language else "some-other-pattern",
            "language": language,
            "support": (support_map or {}).get(language, support),
            "implementation_notes": f"{name} realized in {language}",
            "version": "1.0",
        }
        if payload["support"] == "discouraged":
            payload["alternative"] = "another pattern"
        variant_block = (variants_by_language or {}).get(language)
        if variant_block is not None:
            payload.update(variant_block)
        payload.update((extra_language_content or {}).get(language, {}))
        (root / declared).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for extra in extra_files:
        (root / extra).write_text(
            json.dumps(
                {"pattern": "orphan", "language": "python", "support": "manual",
                 "implementation_notes": "orphan file", "version": "1.0"},
                indent=2,
            ),
            encoding="utf-8",
        )
    return main_path


def write_catalog_doc(
    docs_dir: Path,
    rows: Sequence[tuple[str, Sequence[str]]],
    *,
    out_of_scope: str = "layered, hexagonal, microservices versus monolith, CQRS",
) -> Path:
    """Write a miniature catalog document with numbered sections for order/anti-scope tests."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Mini catalog", ""]
    for index, (section, names) in enumerate(rows, start=1):
        lines.append(f"## {index}. {section}")
        lines.append("")
        lines.append("### Subgroup")
        lines.append("")
        lines.append("| Pattern | Use when |")
        lines.append("|---|---|")
        for name in names:
            lines.append(f"| **{name}** | it applies |")
        lines.append("")
    lines.append("## Background, principles and excluded material")
    lines.append("")
    lines.append(f"- **Out of scope by design:** {out_of_scope}.")
    (docs_dir / "design-patterns.md").write_text("\n".join(lines), encoding="utf-8")
    return docs_dir / "design-patterns.md"


def build_mini_catalog(
    tmp_path: Path,
    records: Sequence[dict[str, object]],
    rows: Sequence[tuple[str, Sequence[str]]] | None = None,
):
    """Build a small catalog in ``tmp_path`` from record specs, validated by the real schemas.

    ``records`` items are the :func:`write_record` keyword arguments plus a mandatory ``name``.
    ``rows`` (section title, row names) writes a miniature catalog document so subgroup/subgroup
    limits and canonical order are exercised with controlled data.
    """
    import shutil

    from src.patterns.loader import load_catalog

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for schema in ("pattern-schema.json", "pattern-language-schema.json"):
        shutil.copy(DOCS_DIR / schema, docs_dir / schema)
    if rows is None:
        rows = [("Behaviour", [str(spec["name"]).replace("-", " ").title() for spec in records])]
    write_catalog_doc(docs_dir, rows)
    pattern_dir = tmp_path / "pattern"
    for spec in records:
        kwargs = {key: value for key, value in spec.items() if key != "name"}
        write_record(pattern_dir, str(spec["name"]), **kwargs)
    return load_catalog(pattern_dir, docs_dir)
