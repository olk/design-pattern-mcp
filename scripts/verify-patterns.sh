#!/usr/bin/env bash
# Copyright (c) 2026 Oliver Kowalke
# SPDX-License-Identifier: MIT
#
# Verify every pattern/*.json against its schema using ajv-cli:
#   pattern/<name>.json             -> docs/pattern-schema.json
#   pattern/<name>.<language>.json  -> docs/pattern-language-schema.json
# plus the ajv-inexpressible subset of the guide §2.3 rule 5 checks
# (duplicate variant_ids, default_variant resolution, flat-field equality,
# absent language_implementations, language_files resolving on disk).
# Map/filename agreement and orphans stay in the startup gate
# (src/patterns/loader.py::_load_language_entry/_check_orphans).
#
# Exit code 0 on clean, 1 on validation errors, 2 on environment/schema errors.
#
# Usage: scripts/verify-patterns.sh
# Requires: ajv (npm install -g ajv-cli), jq
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v ajv >/dev/null 2>&1; then
    echo "ERROR: ajv not found — install with: npm install -g ajv-cli" >&2
    exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq not found — install with: apt install jq" >&2
    exit 2
fi

for schema in docs/pattern-schema.json docs/pattern-language-schema.json; do
    if [ ! -f "$schema" ]; then
        echo "ERROR: schema not found: $schema" >&2
        exit 2
    fi
done
if [ ! -d pattern ]; then
    echo "ERROR: pattern/ directory not found" >&2
    exit 2
fi

# Concrete language suffixes of pattern/<name>.<lang>.json; "generic" is the
# layout-V8 language-neutral realization and validates as a language file too.
LANGS='generic c cplusplus csharp go java javascript kotlin python ruby rust scala swift typescript'

MAIN_FILES=()
LANG_FILES=()
SUFFIX_FAIL=0
shopt -s nullglob
for f in pattern/*.json; do
    base="${f##*/}"
    stem="${base%.json}"
    case "$stem" in
        *.*.*)
            echo "ERROR: unexpected file shape: $f" >&2
            SUFFIX_FAIL=1
            ;;
        *.*)
            lang="${stem##*.}"
            case " $LANGS " in
                *" $lang "*) LANG_FILES+=("$f") ;;
                *) echo "ERROR: unknown language suffix: $f" >&2; SUFFIX_FAIL=1 ;;
            esac
            ;;
        *) MAIN_FILES+=("$f") ;;
    esac
done
shopt -u nullglob
if [ "$SUFFIX_FAIL" -ne 0 ]; then
    exit 1
fi
if [ "${#MAIN_FILES[@]}" -eq 0 ]; then
    echo "ERROR: no main records found in pattern/" >&2
    exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# The language schema references pattern-schema.json#/definitions/... but both
# schemas carry opaque $ids, so ajv cannot resolve the relative $ref (the
# server loader pops the language $id for the same reason — see
# src/patterns/loader.py::_validators). Inline the three referenced definitions
# into a derived schema; docs/ stays the single source of truth.
jq --slurpfile ps docs/pattern-schema.json \
    '.properties.language.allOf = [$ps[0].definitions.language]
    | .properties.support = $ps[0].definitions.supportLevel
    | .properties.variants.items.properties.paradigm = $ps[0].definitions.paradigm' \
    docs/pattern-language-schema.json > "$TMPDIR/lang.resolved.json"

MAIN_ARGS=()
for f in "${MAIN_FILES[@]}"; do MAIN_ARGS+=(-d "$f"); done
LANG_ARGS=()
for f in "${LANG_FILES[@]}"; do LANG_ARGS+=(-d "$f"); done

FAIL=0
echo "=== Main records (pattern-schema.json) ==="
if ajv validate -s docs/pattern-schema.json "${MAIN_ARGS[@]}" --all-errors > "$TMPDIR/main.log" 2>&1; then
    echo "main records valid (${#MAIN_FILES[@]} files)"
else
    cat "$TMPDIR/main.log"
    FAIL=1
fi

echo
echo "=== Language files (pattern-language-schema.json) ==="
if ajv validate -s "$TMPDIR/lang.resolved.json" "${LANG_ARGS[@]}" --all-errors > "$TMPDIR/lang.log" 2>&1; then
    echo "language files valid (${#LANG_FILES[@]} files)"
else
    cat "$TMPDIR/lang.log"
    FAIL=1
fi

echo
echo "=== Structural checks (guide §2.3 rule 5) ==="
PROBLEMS="$TMPDIR/problems.txt"
: > "$PROBLEMS"

# Stored main records must not carry the inline realization field (layout V8).
ABSENT_MSG="language_implementations must be absent from stored records \
(layout V8; the generic realization lives in <name>.generic.json)"
jq -r --arg msg "$ABSENT_MSG" 'select(has("language_implementations"))
    | "\(input_filename): \($msg)"' \
    "${MAIN_FILES[@]}" >> "$PROBLEMS"

# Every language_files value must resolve to a file on disk (the loader joins
# through this map and fails hard on a dangling entry).
jq -r '.language_files | to_entries[] | "\(input_filename)\t\(.key)\t\(.value)"' \
    "${MAIN_FILES[@]}" \
    | while IFS=$'\t' read -r main lang file; do
        if [ ! -f "pattern/$file" ]; then
            echo "$main: language_files[$lang] -> $file missing" >> "$PROBLEMS"
        fi
    done

# Variant invariants: unique variant_ids, resolvable default_variant, flat
# fields equal to the default variant's (pitfalls/sources: null counts as []).
jq -r 'input_filename as $f | . as $doc | select((.variants // []) | type == "array" and length > 0) | ([$doc.variants[].variant_id]) as $ids | (($ids | group_by(.) | map(select(length > 1) | .[0]) | join(","))) as $dups | (if $dups != "" then "\($f): duplicate variant_id(s): [\($dups)]" else empty end), (if ($ids | index($doc.default_variant) | not) then "\($f): default_variant \($doc.default_variant | tojson) matches no variant_id \($ids | tojson)" else ($doc.variants[] | select(.variant_id == $doc.default_variant)) as $dv | ["idiom_name", "implementation_notes", "code_sketch", "pitfalls", "sources"][] as $k | (if $k == "pitfalls" or $k == "sources" then ($doc[$k] // []) else $doc[$k] end) as $flat | (if $k == "pitfalls" or $k == "sources" then ($dv[$k] // []) else $dv[$k] end) as $vv | (if $flat != $vv then "\($f): flat field \($k | tojson) differs from the default variant" else empty end) end)' "${LANG_FILES[@]}" >> "$PROBLEMS"

if [ -s "$PROBLEMS" ]; then
    cat "$PROBLEMS"
    FAIL=1
else
    echo "structural checks clean"
fi

echo
if [ "$FAIL" -eq 0 ]; then
    echo "ALL VALID (${#MAIN_FILES[@]} records + ${#LANG_FILES[@]} language files)"
else
    echo "VALIDATION FAILED"
    exit 1
fi
