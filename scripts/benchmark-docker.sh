#!/usr/bin/env bash
# Containerized benchmark orchestration — every `make benchmark-*` target routes through
# this script. The harness modules (tests/benchmark/harness/) are UNCHANGED and execute
# inside a container of the production server image with the repository bind-mounted, so
# workloads, corpora, metrics and verdicts stay identical to the host-native runs this
# replaces.
#
# Modes and stack needs (compose project = dev stack + benchmark override):
#   selfcheck offline draft compare   no stack (compose run --no-deps)
#   live analyze                      TEI sidecars up --wait, torn down afterwards
#   e2e                               full stack up --wait + POST /mcp initialize probe
#                                     (systemd/README.md form), torn down afterwards
#
# A mode that started the stack tears it down in an EXIT trap (compose down, NEVER down -v:
# the dense-cache volume survives). Nonzero exit on: missing images, stack startup failure,
# readiness timeout, benchmark failure (compose-run exit code propagated), missing result
# artifacts.
#
# Knobs (env, all optional): BENCH_SCENARIOS OUT RUN_ID LIMIT SPLIT REPEAT NO_WARMUP
# LOG_LEVEL FLIP CLIENT_URL CALL_TIMEOUT A B ALLOW_MISMATCH DRY_RUN PER_FAMILY FAMILIES
# DRAFT_OUT MCP_IMAGE TEI_IMAGE TEI_RERANK_IMAGE BENCH_STACK_TIMEOUT (2100) BENCH_READY_TIMEOUT
# (300) BENCH_PROBE_URL (http://127.0.0.1:8822/mcp).

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

[ $# -ge 1 ] || { echo "usage: scripts/benchmark-docker.sh {selfcheck|offline|live|analyze|e2e|draft|compare}" >&2; exit 2; }
MODE="$1"
case "$MODE" in
  selfcheck | offline | live | analyze | e2e | draft | compare) ;;
  *) echo "benchmark-docker: unknown mode: $MODE" >&2; exit 2 ;;
esac

compose() { docker compose -f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml "$@"; }
die() { echo "benchmark-docker: $*" >&2; exit 1; }

BENCH_SCENARIOS="${BENCH_SCENARIOS:-tests/benchmark/scenarios/seed.json}"
BENCH_RUNS="${BENCH_RUNS:-data/benchmark-runs}"
RUN_ID="${RUN_ID:-$MODE-$(date -u +%Y%m%d-%H%M%S)}"
MCP_IMAGE="${MCP_IMAGE:-design-pattern-mcp:latest}"
TEI_IMAGE="${TEI_IMAGE:-design-pattern-tei-embed:latest}"
TEI_RERANK_IMAGE="${TEI_RERANK_IMAGE:-design-pattern-tei-rerank:latest}"
# Optional knobs: default to empty so direct invocation (outside the Makefile's BENCH_ENV,
# which always exports them) is safe under `set -u`.
OUT="${OUT:-}" LIMIT="${LIMIT:-}" SPLIT="${SPLIT:-}" REPEAT="${REPEAT:-}"
NO_WARMUP="${NO_WARMUP:-}" LOG_LEVEL="${LOG_LEVEL:-}" FLIP="${FLIP:-}"
CALL_TIMEOUT="${CALL_TIMEOUT:-}" A="${A:-}" B="${B:-}"
ALLOW_MISMATCH="${ALLOW_MISMATCH:-}" DRY_RUN="${DRY_RUN:-}"
PER_FAMILY="${PER_FAMILY:-}" FAMILIES="${FAMILIES:-}"

case "$BENCH_SCENARIOS" in
  tests/*) ;;
  *) die "BENCH_SCENARIOS must be repo-relative under tests/ (got: $BENCH_SCENARIOS) — the benchmark container mounts only ../tests" ;;
esac
[ -f "$BENCH_SCENARIOS" ] || die "scenario corpus not found: $BENCH_SCENARIOS"
# Own these host dirs before Docker could auto-create them root-owned (uid-1000 writes).
mkdir -p data/dense-cache

stack="none"            # none | sidecars | full
out_dir=""              # repo-relative run dir (host artifacts land here)
draft_out=""            # repo-relative draft file (draft mode)
a_abs=""; b_abs=""      # resolved compare arms
args=()

add_out_arg() {  # shared run-dir flags for the run-producing module modes
  args+=(--out "/app/$out_dir" --run-id "$RUN_ID")
  if [ -n "$LIMIT" ]; then args+=(--limit "$LIMIT"); fi
  if [ -n "$SPLIT" ]; then args+=(--split "$SPLIT"); fi
  if [ -n "$REPEAT" ]; then args+=(--repeat "$REPEAT"); fi
  if [ -n "$NO_WARMUP" ]; then args+=(--no-warmup); fi
  if [ -n "$LOG_LEVEL" ]; then args+=(--log-level "$LOG_LEVEL"); fi
}

case "$MODE" in
  selfcheck)
    args=(-m tests.benchmark.harness.selfcheck)
    ;;
  offline | live | analyze)
    if [ -n "$OUT" ]; then out_dir="$OUT"; else out_dir="$BENCH_RUNS/$RUN_ID"; fi
    case "$out_dir" in
      tests/*) die "OUT must not live under tests/ (read-only mount): $out_dir" ;;
      /*) die "OUT must be repo-relative: $out_dir" ;;
    esac
    if [ -e "$out_dir" ] && [ -n "$(ls -A "$out_dir" 2>/dev/null)" ]; then
      die "run dir not empty (stale results would mix): $out_dir — choose a fresh OUT/RUN_ID"
    fi
    args=(-m tests.benchmark.harness.main --mode "$MODE" --scenarios "/app/$BENCH_SCENARIOS")
    if [ "$MODE" = offline ] && [ -n "$FLIP" ]; then args+=(--flip "$FLIP"); fi
    add_out_arg
    if [ "$MODE" != offline ]; then
      # Mirrors the former host recipes: warm disposable cache + dev-stack generator defaults.
      [ -f data/dense-cache/manifest.json ] && [ -f data/dense-cache/vectors.npy ] \
        || die "warm dense cache required for $MODE (data/dense-cache incomplete) — run: make dense-cache"
      export GENERATOR_PROVIDER="${GENERATOR_PROVIDER:-opencode-go}"
      export GENERATOR_MODEL="${GENERATOR_MODEL:-openai/muse-spark-1.3-contributor}"
      export GENERATOR_BASE_URL="${GENERATOR_BASE_URL:-https://opencode.ai/zen/go/v1}"
      export GENERATOR_API_KEY="${GENERATOR_API_KEY:-${OPENCODE_API_KEY:-}}"
      export GENERATOR_SESSION_ID="${GENERATOR_SESSION_ID:-design-pattern-mcp-benchmark}"
      export GENERATOR_TIMEOUT_SECONDS="${GENERATOR_TIMEOUT_SECONDS:-300.0}"
      # Generator and Jev share one Console key: point the decision client at the
      # already-forwarded GENERATOR_API_KEY so OPENCODE_* needs no forwarding.
      export DECISIONS_API_KEY_ENV="${DECISIONS_API_KEY_ENV:-GENERATOR_API_KEY}"
      stack=sidecars
    fi
    ;;
  e2e)
    if [ -n "$OUT" ]; then out_dir="$OUT"; else out_dir="$BENCH_RUNS/$RUN_ID"; fi
    case "$out_dir" in
      tests/*) die "OUT must not live under tests/ (read-only mount): $out_dir" ;;
      /*) die "OUT must be repo-relative: $out_dir" ;;
    esac
    if [ -e "$out_dir" ] && [ -n "$(ls -A "$out_dir" 2>/dev/null)" ]; then
      die "run dir not empty (stale results would mix): $out_dir — choose a fresh OUT/RUN_ID"
    fi
    args=(-m tests.benchmark.harness.e2e
      --url "${CLIENT_URL:-http://design-pattern-mcp:8050/mcp}"
      --scenarios "/app/$BENCH_SCENARIOS")
    add_out_arg
    if [ -n "$CALL_TIMEOUT" ]; then args+=(--call-timeout-seconds "$CALL_TIMEOUT"); fi
    stack=full
    ;;
  draft)
    draft_out="${DRAFT_OUT:-$BENCH_RUNS/drafts/full.draft.json}"
    case "$draft_out" in
      tests/*) die "DRAFT_OUT must not live under tests/ (read-only mount): $draft_out" ;;
      /*) die "DRAFT_OUT must be repo-relative: $draft_out" ;;
    esac
    args=(-m tests.benchmark.harness.draft --out "/app/$draft_out")
    if [ -n "$DRY_RUN" ]; then args+=(--dry-run); fi
    if [ -n "$PER_FAMILY" ]; then args+=(--per-family "$PER_FAMILY"); fi
    if [ -n "$FAMILIES" ]; then args+=(--families "$FAMILIES"); fi
    ;;
  compare)
    [ -n "$A" ] && [ -n "$B" ] || die "usage: A=<baseline-dir> B=<candidate-dir>"
    a_abs="$(realpath -m -- "$A")"
    b_abs="$(realpath -m -- "$B")"
    [ -f "$a_abs/manifest.json" ] || die "A is not a run directory (manifest.json missing): $A"
    [ -f "$b_abs/manifest.json" ] || die "B is not a run directory (manifest.json missing): $B"
    args=(-m tests.benchmark.harness.compare /mnt/a /mnt/b)
    if [ -n "$ALLOW_MISMATCH" ]; then args+=(--allow-mismatch); fi
    ;;
esac

# --- stack lifecycle --------------------------------------------------------
started=0
cleanup() {
  rc=$?
  if [ "$started" = 1 ]; then
    compose down --remove-orphans >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap cleanup EXIT

require_image() { docker image inspect "$1" >/dev/null 2>&1 || die "image missing: $1 — run: make docker-build-all"; }

COMPOSE_FILES=(-f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml)
case "$stack" in
  sidecars)
    require_image "$TEI_IMAGE"
    require_image "$TEI_RERANK_IMAGE"
    started=1 # set before `up`: a failed bring-up may leave partial services behind
    # `timeout` cannot invoke the `compose` shell function — spell out docker compose.
    timeout "${BENCH_STACK_TIMEOUT:-2100}" docker compose "${COMPOSE_FILES[@]}" up -d --wait design-pattern-tei-embed design-pattern-tei-rerank
    ;;
  full)
    require_image "$MCP_IMAGE"
    require_image "$TEI_IMAGE"
    require_image "$TEI_RERANK_IMAGE"
    started=1 # set before `up`: a failed bring-up may leave partial services behind
    timeout "${BENCH_STACK_TIMEOUT:-2100}" docker compose "${COMPOSE_FILES[@]}" up -d --wait design-pattern-mcp
    # MCP-level readiness: the documented systemd probe (POST /mcp initialize → 200).
    command -v curl >/dev/null 2>&1 || die "curl is required for the MCP readiness probe"
    probe_url="${BENCH_PROBE_URL:-http://127.0.0.1:8822/mcp}"
    probe_body='{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"probe","version":"0.0.0"}},"id":1}'
    deadline=$(( $(date +%s) + ${BENCH_READY_TIMEOUT:-300} ))
    while [ "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$probe_url" \
      -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
      -d "$probe_body")" != "200" ]; do
      if [ "$(date +%s)" -ge "$deadline" ]; then
        die "MCP readiness probe failed within ${BENCH_READY_TIMEOUT:-300}s: POST $probe_url — server logs: make docker-logs"
      fi
      sleep 5
    done
    echo ">> MCP server ready: $probe_url (initialize: 200)"
    ;;
esac

# --- per-run mounts and env forwarding --------------------------------------
run_mounts=()
case "$MODE" in
  offline | live | analyze | e2e)
    mkdir -p "$out_dir"
    run_mounts+=(-v "$REPO_ROOT/$out_dir:/app/$out_dir")
    ;;
  draft)
    parent="$(dirname "$draft_out")"
    mkdir -p "$parent"
    run_mounts+=(-v "$REPO_ROOT/$parent:/app/$parent")
    ;;
  compare)
    run_mounts+=(-v "$a_abs:/mnt/a:ro" -v "$b_abs:/mnt/b")
    ;;
esac

# Forward the deployment-relevant env the container cannot discover on its own:
# provider keys, generator/reasoning/decision namespaces, and the scalar config
# knobs (selection/loop/retrieval/scoring/rerank/validation/…) so host-side
# overrides like SELECTION_MAX_PARTS reach the in-container ConfigManager.
# Deliberately NOT forwarded (container values must win):
#   REASONING_*_CMD      image-embedded node entry points, absent on the host
#   EMBEDDER_*/RERANKER_* compose service-DNS endpoints; host loopback would break
#   DENSE_CACHE_PATH     container path (/app/…); a host path is meaningless inside
#   CONFIG_PATH, DOCS_/PATTERN_DIRECTORY, JOBS_DB  host paths, invalid in-container
#   TRANSPORT/HOST/PORT, CLIENT_URL  server bind surface; the script passes --url itself
# NOTE: RERANK_ matches RERANK_TEXT/RERANK_TOP_N only — RERANKER_* has E, not _,
# at position 7, so the reranker service-DNS endpoint stays pinned.
fwd=()
fwd_debug=()
while IFS= read -r name; do
  case "$name" in REASONING_*_CMD) continue ;; esac
  fwd+=(-e "$name=${!name}")
  case "$name" in *KEY* | *SECRET* | *TOKEN*) fwd_debug+=("$name=<redacted>") ;; *) fwd_debug+=("$name=${!name}") ;; esac
done < <(compgen -A export | grep -E '^(GENERATOR_|GENERATION_|REASONING_|DECISIONS_|SELECTION_|LOOP_|RETRIEVAL_|SCORING_|RERANK_|VALIDATION_|PROMPT_|LITELLM_|LOGGING_|TASKS_|DENSE_CACHE_(MODE|WRITE|PROBE_)|EMBED_(MODEL_REPO|ONNX_VARIANT))' || true)
if [ "${#fwd_debug[@]}" -gt 0 ]; then echo ">> container env: ${fwd_debug[*]}"; fi

# --- run --------------------------------------------------------------------
rc=0
compose run --rm -T --no-deps ${run_mounts+"${run_mounts[@]}"} ${fwd+"${fwd[@]}"} benchmark "${args[@]}" || rc=$?

# --- result collection ------------------------------------------------------
if [ "$rc" = 0 ]; then
  case "$MODE" in
    offline | live | analyze | e2e)
      for f in manifest.json summary.json report.md; do
        [ -f "$out_dir/$f" ] || { echo "benchmark-docker: result collection failed: $out_dir/$f missing" >&2; rc=1; }
      done
      ls "$out_dir"/scenarios/*.json >/dev/null 2>&1 \
        || { echo "benchmark-docker: result collection failed: no scenario records in $out_dir/scenarios/" >&2; rc=1; }
      ;;
    draft)
      # --dry-run prints the first prompt and writes nothing (draft.py returns 0
      # by design), so a dry run must exit 0 with no artifact check — same as
      # the old host-native recipe.
      if [ -z "$DRY_RUN" ]; then
        [ -f "$draft_out" ] || { echo "benchmark-docker: result collection failed: $draft_out missing" >&2; rc=1; }
      fi
      ;;
  esac
fi

if [ "$rc" = 0 ] && [ -n "$out_dir" ]; then echo ">> run directory: $out_dir"; fi
exit "$rc"
