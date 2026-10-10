"""§2.4 variant resource payload and §8.4 ladder / binding resolution.

The ladder is exercised on hand-built entries (so each layer is isolated) and on the real catalog
(the payload must expose exactly the species of data the resource promises, derived from catalog
text — never invented).
"""

from __future__ import annotations

import pytest

from src.patterns.loader import LanguageEntry, LanguageVariant
from src.errors import ERROR_CATALOG_INVALID, DesignPatternError
from src.schemas import VariantSummary
from src.variants import (
    NOTE_MAX_LENGTH,
    bind_entry,
    select_variant,
    variant_summaries,
    variants_for_pattern,
    variants_payload,
)

JAVA_VARIANTS = (
    LanguageVariant(
        variant_id="java8-streams",
        name="Streams and lambdas",
        paradigm="functional",
        implementation_notes="stream pipelines over the collection",
        min_standard="Java1.8",
    ),
    LanguageVariant(
        variant_id="java21-virtual-threads",
        name="Virtual threads",
        paradigm="concurrent",
        implementation_notes="one virtual thread per task",
        min_standard="Java21",
        requires_features=("virtual-threads",),
    ),
)


def _entry(variants: tuple[LanguageVariant, ...], default: str, *, language: str = "java") -> LanguageEntry:
    return LanguageEntry(
        language=language,
        support="idiomatic",
        implementation_notes="entry notes",
        variants=variants,
        default_variant=default,
    )


def test_payload_groups_by_language_and_pattern(catalog) -> None:
    """§2.4: language → pattern → choice surface only, deterministic order, no invented text."""
    payload = variants_payload(catalog)
    assert payload, "the corpus carries variant-carrying language files"
    for language, patterns in payload.items():
        assert language in catalog.enums.languages
        for pattern, block in patterns.items():
            record = catalog.get_by_name(pattern)
            assert record is not None
            entry = record.entry(language)
            assert entry is not None and entry.variants
            assert block["default_variant"] == entry.default_variant
            assert len(block["variants"]) == len(entry.variants)
            for variant, source in zip(block["variants"], entry.variants, strict=True):
                assert variant["variant_id"] == source.variant_id
                assert variant["paradigm"] == source.paradigm
                assert variant["min_standard"] == source.min_standard
                assert variant["requires_features"] == list(source.requires_features)
                assert variant["default"] == (source.variant_id == entry.default_variant)
                assert 0 < len(variant["note"]) <= NOTE_MAX_LENGTH


def test_pattern_slice_resolves_names_and_aliases(catalog) -> None:
    """The template slice accepts a canonical name and an alias, and rejects unknown names."""
    slice_ = variants_for_pattern(catalog, "timeout")
    assert slice_ is not None
    assert slice_["pattern"] == "timeout"
    assert "typescript" in slice_["languages"]
    assert variants_for_pattern(catalog, "no-such-pattern") is None


def test_bind_entry_falls_back_to_generic(catalog) -> None:
    """§8.4: the effective language's entry, else the record's generic entry (always present)."""
    record = catalog.get_by_name("timeout")
    assert record is not None
    assert bind_entry(record, "typescript").language == "typescript"
    assert bind_entry(record, "cobol").language == "generic"


def test_bind_entry_without_generic_is_a_catalog_error(catalog) -> None:
    """A record with no generic entry cannot be bound and fails loudly."""
    record = catalog.get_by_name("timeout")
    assert record is not None
    stripped = record.language_implementations
    without_generic = {key: value for key, value in stripped.items() if key != "generic"}
    from dataclasses import replace

    with pytest.raises(DesignPatternError) as excinfo:
        bind_entry(replace(record, language_implementations=without_generic), "cobol")
    assert excinfo.value.code == ERROR_CATALOG_INVALID


def test_layer_one_explicit_id_only() -> None:
    """Layer 1: a verbatim variant_id binds; a paraphrase falls through to the next layer."""
    entry = _entry(JAVA_VARIANTS, "java8-streams")
    chosen = select_variant(
        entry, focus="use the java21-virtual-threads variant", paradigm=None, technology_tokens=[]
    )
    assert chosen is not None and chosen.variant_id == "java21-virtual-threads"
    paraphrase = select_variant(
        entry, focus="use the virtual threads variant", paradigm=None, technology_tokens=[]
    )
    # "virtual-threads" is not quoted verbatim → layer 4 default, not layer 1
    assert paraphrase is not None and paraphrase.variant_id == "java8-streams"


def test_layer_two_paradigm_then_layer_three_floor() -> None:
    """Layer 2 narrows by paradigm; several variants sharing it fall through to the floor."""
    entry = _entry(JAVA_VARIANTS, "java8-streams")
    assert select_variant(
        entry, focus=None, paradigm="concurrent", technology_tokens=[]
    ) is not None
    assert select_variant(entry, focus=None, paradigm="concurrent", technology_tokens=[]).variant_id == (
        "java21-virtual-threads"
    )
    assert select_variant(entry, focus=None, paradigm="oo", technology_tokens=[]).variant_id == "java8-streams"


def test_layer_three_capability_floor_uses_the_compact_standard_form() -> None:
    """Layer 3: ``Java 8`` satisfies ``Java1.8`` (pre-Java-9 numbering) and the highest floor wins."""
    entry = _entry(JAVA_VARIANTS, "java8-streams")
    java8_only = select_variant(entry, focus=None, paradigm=None, technology_tokens=["Java 8"])
    assert java8_only is not None and java8_only.variant_id == "java8-streams"
    both = select_variant(entry, focus=None, paradigm=None, technology_tokens=["Java 8", "Java 21"])
    assert both is not None and both.variant_id == "java21-virtual-threads"


def test_cpp_floor_prefers_the_stated_standard() -> None:
    """A C++20 toolchain picks the C++20 realization over the C++17 one."""
    variants = (
        LanguageVariant(
            variant_id="cpp17-coroutines",
            name="C++17",
            paradigm="concurrent",
            implementation_notes="c++17 realization",
            min_standard="C++17",
        ),
        LanguageVariant(
            variant_id="cpp20-coroutines",
            name="C++20",
            paradigm="concurrent",
            implementation_notes="c++20 realization",
            min_standard="C++20",
        ),
    )
    entry = _entry(variants, "cpp17-coroutines", language="cplusplus")
    chosen = select_variant(entry, focus=None, paradigm=None, technology_tokens=["C++20"])
    assert chosen is not None and chosen.variant_id == "cpp20-coroutines"
    older = select_variant(entry, focus=None, paradigm=None, technology_tokens=["C++17"])
    assert older is not None and older.variant_id == "cpp17-coroutines"
    unknown_family = select_variant(entry, focus=None, paradigm=None, technology_tokens=["Python 3.12"])
    assert unknown_family is not None and unknown_family.variant_id == "cpp17-coroutines"


def test_no_variants_means_no_binding() -> None:
    """An entry without variants cannot bind one (§8.4: the ladder is a variant-carrying concern)."""
    entry = LanguageEntry(language="python", support="native", implementation_notes="notes")
    assert select_variant(entry, focus="anything", paradigm="oo", technology_tokens=["Python 3.12"]) is None


def test_variant_summaries_exclude_the_bound_variant_and_cap_at_five() -> None:
    """Sibling summaries carry the choice surface only: no notes, no sketches, max 5 entries."""
    variants = tuple(
        LanguageVariant(
            variant_id=f"variant-{index}",
            name=f"Variant {index}",
            paradigm="oo",
            implementation_notes=f"notes {index}",
            min_standard=f"Java{index}",
        )
        for index in range(1, 8)
    )
    entry = _entry(variants, "variant-1")
    summaries = variant_summaries(entry, "variant-2")
    assert len(summaries) == 5
    assert all(isinstance(summary, VariantSummary) for summary in summaries)
    assert "variant-2" not in {summary.variant_id for summary in summaries}
    assert summaries[0].variant_id == "variant-1"
