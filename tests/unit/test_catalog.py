"""§2.3 — catalog join, integrity gate and canonical order.

Positive path: the real 486-record catalog must load, join 14 entries per record (generic first,
then the concrete languages in enum order) and expose a canonical order derived from the catalog
document. Negative path: every layout-V8 violation must fail the gate with ``ERR_CATALOG_INVALID``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.patterns.loader import (
    CATALOG_DOC_FILE,
    parse_catalog_order,
    parse_out_of_scope_terms,
    load_catalog,
)
from src.errors import ERROR_CATALOG_INVALID, DesignPatternError
from tests.unit.conftest import DOCS_DIR, LANGUAGES, write_record


def _catalog_of(root: Path):
    """Load the catalog from ``root`` against the real schema documents."""
    return load_catalog(root, DOCS_DIR)


def test_join_order_and_identity(tmp_path: Path) -> None:
    """Joined entries are generic-first, then concrete languages in enum order."""
    write_record(tmp_path, "alpha-pattern")
    catalog = _catalog_of(tmp_path)

    record = catalog.get_by_name("alpha-pattern")
    assert record is not None
    assert list(record.language_implementations) == list(LANGUAGES)
    assert record.language_implementations["generic"].language == "generic"
    assert record.language_implementations["typescript"].support == "idiomatic"
    # file-level fields are stripped during the join (§2.3 rule 4)
    assert not hasattr(record.language_implementations["python"], "pattern")
    assert not hasattr(record.language_implementations["python"], "version")


def test_real_catalog_loads_and_is_ordered(catalog) -> None:
    """The shipped catalog passes the gate and nearly every record carries a catalog order index."""
    assert len(catalog.records) == 486
    assert len(catalog.order) >= 430
    assert catalog.enums.languages == LANGUAGES
    # section 1 row order vs section 8 row order (the §8.1 tie-break chain starts with catalog order)
    assert catalog.order_index("transaction-script") < catalog.order_index("guard-clause")
    assert catalog.order_index("timeout") < catalog.order_index("guard-clause")


def test_related_patterns_contract(catalog) -> None:
    """F5b contract: related_patterns carries only resolvable names; prose lives in notes.

    The §8.3 evidence_a membership test (``record.name in primary.related_patterns``) is an
    exact string match — prose entries silently no-op there, so every entry must resolve via
    :meth:`PatternLoader.resolve`. Distinction notes live in ``related_patterns_notes``
    (kebab keys per the schema propertyNames pattern).
    """
    key_pattern = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
    for name, record in catalog.records.items():
        assert len(set(record.related_patterns)) == len(record.related_patterns), name
        for related in record.related_patterns:
            resolved = catalog.resolve(related)
            assert resolved is not None, f"{name}: related_patterns entry {related!r} unresolvable"
            assert resolved != name, f"{name}: related_patterns entry is the record itself"
        for key, note in record.related_patterns_notes.items():
            assert key_pattern.match(key), f"{name}: notes key {key!r} not kebab-case"
            assert note.strip(), f"{name}: empty note for {key!r}"


def test_order_index_unknown_names_sort_after_every_catalog_record(catalog) -> None:
    """Records absent from the catalog document sort LAST, never first (§8.1 tie-break).

    Regression: the fallback once returned ``len(records) + _fallback_order(name)`` (~1.8k),
    which is below every real row (min 1_001_001) — healthy-object, identity-function and
    link-transfer therefore sorted to the very front of every ascending tie-break.
    """
    real_orders = [catalog.order_index(name) for name in catalog.order]
    assert real_orders, "the catalog document must define at least one order row"
    for name in ("healthy-object", "identity-function", "link-transfer"):
        assert catalog.order_index(name) > max(real_orders), name


def test_variant_ladder_data_is_joined(catalog) -> None:
    """A record with language variants exposes them on the joined entry (§8.4)."""
    record = catalog.get_by_name("timeout")
    assert record is not None
    entry = record.entry("typescript")
    assert entry is not None
    assert entry.variants, "timeout must carry TypeScript variants (ladder is load-bearing)"
    assert entry.default_variant in entry.variant_ids
    assert entry.variant(entry.default_variant or "") is not None


def test_alias_index_drops_ambiguous_aliases(catalog) -> None:
    """§2.1 worked case: Strategy's alias "Policy" collides with the catalog's Policy-based design."""
    assert catalog.get_by_name("policy-based-design") is not None
    assert "policy" not in catalog.alias_map
    assert catalog.resolve("strategy") == "strategy"
    assert catalog.ambiguous_aliases


def test_out_of_scope_terms_parsed_from_background() -> None:
    """§5 rule 5: the term set comes from the catalog document, not a literal."""
    terms = parse_out_of_scope_terms(DOCS_DIR / CATALOG_DOC_FILE)
    assert "microservices" in terms
    assert "monolith" in terms
    assert all(not term.startswith("**") for term in terms)


@pytest.mark.parametrize(
    ("kwargs", "expected_fragment"),
    [
        ({"omit_language_keys": ("generic",)}, "language_files"),
        ({"declared_without_file": ("python",)}, "declared but missing on disk"),
        ({"wrong_pattern_language": "python"}, "pattern field"),
        ({"extra_files": ("orphan.python.json",)}, "orphan language file"),
    ],
)
def test_integrity_gate_rejects_layout_violations(
    tmp_path: Path, kwargs: dict[str, object], expected_fragment: str
) -> None:
    """Each layout-V8 violation fails the gate with a targeted message."""
    write_record(tmp_path, "alpha-pattern", **kwargs)
    with pytest.raises(DesignPatternError) as excinfo:
        _catalog_of(tmp_path)
    assert excinfo.value.code == ERROR_CATALOG_INVALID
    assert expected_fragment in excinfo.value.detail


def test_integrity_gate_rejects_variant_flat_mismatch(tmp_path: Path) -> None:
    """§2.3 rule 5: the default variant's flat fields must equal the file's flat fields."""
    write_record(
        tmp_path,
        "alpha-pattern",
        variants_by_language={
            "python": {
                "variants": [
                    {
                        "variant_id": "frozen-dataclass",
                        "name": "frozen dataclass",
                        "paradigm": "oo",
                        "implementation_notes": "variant notes",
                    }
                ],
                "default_variant": "frozen-dataclass",
            }
        },
    )
    with pytest.raises(DesignPatternError) as excinfo:
        _catalog_of(tmp_path)
    assert excinfo.value.code == ERROR_CATALOG_INVALID
    assert "differs from the default variant" in excinfo.value.detail


def test_integrity_gate_accepts_consistent_variants(tmp_path: Path) -> None:
    """A consistent variant block joins with its ladder data intact."""
    write_record(
        tmp_path,
        "alpha-pattern",
        variants_by_language={
            "python": {
                "variants": [
                    {
                        "variant_id": "frozen-dataclass",
                        "name": "frozen dataclass",
                        "paradigm": "oo",
                        "implementation_notes": "alpha-pattern realized in python",
                        "min_standard": "Python3.10",
                        "requires_features": ["dataclasses"],
                    }
                ],
                "default_variant": "frozen-dataclass",
            }
        },
    )
    record = _catalog_of(tmp_path).get_by_name("alpha-pattern")
    assert record is not None
    entry = record.entry("python")
    assert entry is not None and entry.default_variant == "frozen-dataclass"
    assert entry.variant_ids == ("frozen-dataclass",)
    assert entry.variants[0].min_standard == "Python3.10"


def test_language_implementations_in_main_record_is_rejected(tmp_path: Path) -> None:
    """Layout V8: a re-inlined realization block fails the gate."""
    path = write_record(tmp_path, "alpha-pattern")
    payload = json.loads(path.read_text())
    payload["language_implementations"] = []
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(DesignPatternError) as excinfo:
        _catalog_of(tmp_path)
    assert excinfo.value.code == ERROR_CATALOG_INVALID
    assert "language_implementations" in excinfo.value.detail


def test_malformed_json_names_the_file_and_the_parse_error(tmp_path: Path) -> None:
    """A broken record file must say WHICH file and WHY — the loader's only diagnostic."""
    path = write_record(tmp_path, "alpha-pattern")
    path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(DesignPatternError) as excinfo:
        _catalog_of(tmp_path)
    assert excinfo.value.code == ERROR_CATALOG_INVALID
    assert str(path) in excinfo.value.detail
    assert "is not valid JSON" in excinfo.value.detail


def test_canonical_order_prefers_the_catalog_document(tmp_path: Path) -> None:
    """Row order inside a section defines the canonical index (§8.1 tie-break)."""
    write_record(tmp_path, "first-pattern")
    write_record(tmp_path, "second-pattern")
    catalog = _catalog_of(tmp_path)
    doc = tmp_path / "docs" / "design-patterns.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        "## 1. Model domain\n\n### Group\n\n| Pattern | Use when |\n|---|---|\n"
        "| **First Pattern** | x |\n| **Second Pattern** | y |\n",
        encoding="utf-8",
    )
    order, subgroups, unmatched = parse_catalog_order(doc, frozenset(catalog.records))
    assert order == {"first-pattern": 1_001_001, "second-pattern": 1_001_002}
    assert subgroups["first-pattern"] == "Group"
    assert unmatched == []
