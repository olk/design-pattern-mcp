#!/usr/bin/env bash
# TEI-only vs Tacit-4B-on-CPU e2e A/B (selection consistency + performance).
#
#   bash scripts/benchmark-tei-vs-tacit-cpu.sh
#
# Arm A (baseline): base stack only — TEI embed + TEI rerank, LLM style/weights,
#   no decision models. Arm B (candidate): base + benchmark override +
#   docker/docker-compose.benchmark-tacit-cpu.yml — TEI embed, TEI rerank (the CPU
#   lane keeps rerank on TEI), linear style head + Tacit-CPU weights/category/gate.
# Both arms: same corpus/split/limit/repeat, same generator, REASONING off.
#
# Knobs (env): STAMP BENCH_SCENARIOS SPLIT LIMIT REPEAT BENCH_RUNS MCP_IMAGE
#   TEI_IMAGE TEI_RERANK_IMAGE TACIT_CPU_IMAGE STYLE_HEAD_IMAGE
#   BENCH_STACK_TIMEOUT BENCH_READY_TIMEOUT.
#
# Each arm tears its compose stack down afterwards (benchmark-docker.sh trap);
# restart the dev stack when the driver finishes. Runtime: ~1 h (arm A) + ~2-3 h
# (arm B) at the default LIMIT=3/REPEAT=2. Launch detached, e.g.:
#   STAMP=$(date -u +%Y%m%d-%H%M%S); nohup bash scripts/benchmark-tei-vs-tacit-cpu.sh \
#     > data/benchmark-runs/$STAMP-driver.log 2>&1 & echo $! > data/benchmark-runs/$STAMP-driver.pid

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
SCENARIOS="${BENCH_SCENARIOS:-tests/benchmark/scenarios/seed.json}"
SPLIT="${SPLIT:-holdout}"
LIMIT="${LIMIT:-3}"
REPEAT="${REPEAT:-2}"
RUNS="${BENCH_RUNS:-data/benchmark-runs}"
A_ID="$STAMP-tei-only"
B_ID="$STAMP-tacit-cpu"
OVERLAY="docker/docker-compose.benchmark-tacit-cpu.yml"
STYLE_CALIBRATION="models/style-head-calibration.json"

# Pinned images: both arms run the same bytes (provenance, no :latest drift).
export MCP_IMAGE="${MCP_IMAGE:-design-pattern-mcp:1.0.1}"
export TEI_IMAGE="${TEI_IMAGE:-design-pattern-tei-embed:1.0.1}"
export TEI_RERANK_IMAGE="${TEI_RERANK_IMAGE:-design-pattern-tei-rerank:1.0.1}"
export TACIT_CPU_IMAGE="${TACIT_CPU_IMAGE:-design-pattern-tacit-cpu:1.0.1}"
export STYLE_HEAD_IMAGE="${STYLE_HEAD_IMAGE:-design-pattern-style-head:1.0.1}"
# Bring-up budget: Tacit-CPU model load (start_period 900 s) + MCP cold re-embed
# of the catalog (~20 min on CPU under DENSE_CACHE_MODE=off, which every e2e arm
# runs with) stack on first bring-up.
export BENCH_STACK_TIMEOUT="${BENCH_STACK_TIMEOUT:-3600}"
export BENCH_READY_TIMEOUT="${BENCH_READY_TIMEOUT:-600}"

die() { echo "tei-vs-tacit-cpu: $*" >&2; exit 1; }
log() { echo ">> $(date -u +%FT%TZ) $*"; }
# --- seam-usage watch -----------------------------------------------------------
# e2e's wire contract hides fallbacks, and a dead/slow Tacit seam degrades to the
# LLM path with only a log warning — an arm can "complete" while measuring nothing.
# A host-side poller therefore samples the sidecar's /v1/stats decisions counter
# (published on 127.0.0.1:18100 by the overlay) plus any "degraded" server-log lines
# once a minute per arm. Arm B must show decisions > 0 (else the arm is void and
# fails loud); arm A must show none (negative control: no Tacit served the baseline).
TACIT_STATS_URL="http://127.0.0.1:18100/v1/stats"
TACIT_WATCH_FILE="" # set per arm; the EXIT trap stops whatever is polling

stop_tacit_watch() {
  if [ -n "$TACIT_WATCH_FILE" ] && [ -f "$TACIT_WATCH_FILE.pid" ]; then
    kill "$(cat "$TACIT_WATCH_FILE.pid")" 2>/dev/null || true
    rm -f "$TACIT_WATCH_FILE.pid"
  fi
  TACIT_WATCH_FILE=""
}
trap stop_tacit_watch EXIT

start_tacit_watch() { # $1=outfile
  stop_tacit_watch
  TACIT_WATCH_FILE="$1"
  # Direct backgrounding: $! is only valid in the shell that spawned the job, so a
  # `( … & )` wrapper would capture the SUBSHELL's job id, unkillable from here.
  while true; do
    echo "=== $(date -u +%FT%TZ) ==="
    curl -fsS -m 10 "$TACIT_STATS_URL" 2>/dev/null || echo "STATS_UNREACHABLE"
    docker logs --tail 30 design-pattern-mcp-dev 2>&1 | grep -i 'degrad' || true
    sleep 60
  done > "$1" 2>&1 & echo $! > "$1.pid"
  log "tacit watch polling -> $1"
}

tacit_decisions_max() { # $1=watchfile; max decisions counter seen (empty if none)
  grep -o '"decisions": *[0-9]*' "$1" 2>/dev/null | grep -o '[0-9]*' | sort -n | tail -1 || true
}

# --- preflight (fail loud before warming anything) ---------------------------
[ -f "$SCENARIOS" ] || die "scenario corpus not found: $SCENARIOS"
[ -f "$OVERLAY" ] || die "overlay not found: $OVERLAY"
for img in "$MCP_IMAGE" "$TEI_IMAGE" "$TEI_RERANK_IMAGE" "$TACIT_CPU_IMAGE" "$STYLE_HEAD_IMAGE"; do
  docker image inspect "$img" >/dev/null 2>&1 || die "image missing: $img"
done
log "images pinned: $MCP_IMAGE $TEI_IMAGE $TEI_RERANK_IMAGE $TACIT_CPU_IMAGE $STYLE_HEAD_IMAGE"

# Ambient seam vars would leak through compose ${VAR:-default} interpolation and
# silently rewire an arm (e.g. RERANKER_PROVIDER=tacit in arm A). The driver sets
# every seam var per arm itself; refuse a polluted environment.
for leaked in RERANKER_PROVIDER DECISIONS_STYLE_ENABLED DECISIONS_WEIGHTS_ENABLED \
  DECISIONS_CATEGORY_ENABLED DECISIONS_REASONING_GATE_ENABLED TASKS_DEADLINE_SECONDS \
  REASONING_ENABLED BENCH_DECISION_OVERLAY; do
  [ -z "${!leaked:-}" ] || die "ambient $leaked=${!leaked} would leak into an arm — unset it first"
done

# The style floor must come from the calibration artifact, never hand-copied.
python3 - "$STYLE_CALIBRATION" <<'EOF' || die "floor 0.6 not certified by $STYLE_CALIBRATION"
import json, sys
floors = {r.get("floor") for r in json.load(open(sys.argv[1])).get("floors", [])}
sys.exit(0 if 0.6 in floors else 1)
EOF
log "style floor 0.6 certified by $STYLE_CALIBRATION"

# An arm whose generator key renders empty still "completes" down the degraded
# paths and measures noise — assert the rendered value in BOTH file sets.
for files in "-f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml" \
  "-f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml -f $OVERLAY"; do
  # shellcheck disable=SC2086
  rendered_key="$(docker compose $files config 2>/dev/null \
    | sed -n 's/^ *GENERATOR_API_KEY: *//p' | head -1)"
  case "$rendered_key" in
    "" | "''" | '""') die "GENERATOR_API_KEY renders EMPTY under [$files] — export OPENCODE_API_KEY" ;;
  esac
done
log "GENERATOR_API_KEY renders non-empty in both file sets"

# The candidate file set must actually carry the Tacit service; the baseline set
# must not (a silent no-op arm would compare TEI against TEI).
docker compose -f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml \
  -f "$OVERLAY" config --services 2>/dev/null | grep -qx "design-pattern-tacit" \
  || die "overlay does not render a design-pattern-tacit service"
if docker compose -f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml \
  config --services 2>/dev/null | grep -qx "design-pattern-tacit"; then
  die "baseline file set renders design-pattern-tacit — the arms would not differ"
fi
log "overlay renders design-pattern-tacit; baseline does not"

for dir in "$RUNS/$A_ID" "$RUNS/$B_ID"; do
  [ ! -e "$dir" ] || [ -z "$(ls -A "$dir" 2>/dev/null)" ] \
    || die "run dir not empty (stale results would mix): $dir"
done
if docker ps --format '{{.Names}}' | grep -qx "design-pattern-mcp-dev-benchmark-run-.*"; then
  die "another benchmark run container is active"
fi
log "preflight green: corpus=$SCENARIOS split=$SPLIT limit=$LIMIT repeat=$REPEAT"

# --- arm runner: clean slate + one retry --------------------------------------
# 2026-11-22: arm A died ~60 s into `up --wait` with "No such container" — a
# container replaced mid-wait (concurrent compose op or recreate race). Each arm
# therefore starts from `compose down` (create path, no Recreate, no orphans)
# and gets exactly one retry after a fresh `down`. A twice-failed arm fails
# loud: its run dir stays partial and no comparison runs against it.
BASE_FILES="-f docker/docker-compose.yml -f docker/docker-compose.benchmark.yml"
CAND_FILES="$BASE_FILES -f $OVERLAY"

arm_b_pre() {
  # The style head is NOT in MCP depends_on (its seam degrades silently), so
  # bring it up explicitly — otherwise arm B runs the LLM style and misattributes it.
  log "candidate style head up (explicit: not a compose dependency)"
  # shellcheck disable=SC2086
  docker compose $CAND_FILES up -d --wait design-pattern-style-head
}

run_arm() { # $1=tag $2=files $3=RUN_ID $4=pre-hook-fn $5...=extra make args
  local tag="$1" files="$2" run_id="$3" pre="$4"; shift 4
  local attempt
  for attempt in 1 2; do
    log "$tag: clean slate (attempt $attempt)"
    # shellcheck disable=SC2086
    docker compose $files down --remove-orphans >/dev/null 2>&1 || true
    if ! "$pre"; then log "$tag: pre-hook failed"; continue; fi
    if make benchmark-e2e BENCH_SCENARIOS="$SCENARIOS" SPLIT="$SPLIT" \
      LIMIT="$LIMIT" REPEAT="$REPEAT" RUN_ID="$run_id" "$@"; then
      log "$tag complete: $RUNS/$run_id"
      return 0
    fi
    log "$tag: attempt $attempt failed"
  done
  die "$tag failed twice — not comparing against a partial arm"
}

# --- arm A: TEI only ----------------------------------------------------------
# Deadline parity with arm B (3600): a whole-run budget difference would cancel
# long scenarios in one arm only — a non-seam difference the comparison must not carry.
# Pre-down BEFORE the watch starts, so the first sample already proves Tacit absent
# (the usage assert counts decisions; a pre-down sample could catch the old container).
# shellcheck disable=SC2086
docker compose $BASE_FILES down --remove-orphans >/dev/null 2>&1 || true
start_tacit_watch "$RUNS/$STAMP-tei-only-tacit-watch.log"
run_arm "arm 1/2 TEI-only" "$BASE_FILES" "$A_ID" true TASKS_DEADLINE_SECONDS=3600
stop_tacit_watch
A_DECISIONS=$(tacit_decisions_max "$RUNS/$STAMP-tei-only-tacit-watch.log"); A_DECISIONS=${A_DECISIONS:-0}
[ "$A_DECISIONS" = "0" ] || die "baseline shows $A_DECISIONS Tacit decisions — the arms would not differ"
log "arm A negative control: no Tacit decisions served the baseline"

# --- arm B: Tacit on CPU ------------------------------------------------------
# shellcheck disable=SC2086
docker compose $CAND_FILES down --remove-orphans >/dev/null 2>&1 || true
start_tacit_watch "$RUNS/$STAMP-tacit-cpu-tacit-watch.log"
run_arm "arm 2/2 Tacit-CPU" "$CAND_FILES" "$B_ID" arm_b_pre \
  BENCH_DECISION_OVERLAY="$OVERLAY" \
  DECISIONS_STYLE_ENABLED=true DECISIONS_STYLE_CONFIDENCE_FLOOR=0.6 \
  DECISIONS_STYLE_PROVIDER=linear \
  DECISIONS_STYLE_BASE_URL=http://design-pattern-style-head:8800/v1 \
  DECISIONS_STYLE_MODEL=style-linear \
  DECISIONS_WEIGHTS_ENABLED=true DECISIONS_CATEGORY_ENABLED=true \
  DECISIONS_REASONING_GATE_ENABLED=true TASKS_DEADLINE_SECONDS=3600
stop_tacit_watch
B_DECISIONS=$(tacit_decisions_max "$RUNS/$STAMP-tacit-cpu-tacit-watch.log"); B_DECISIONS=${B_DECISIONS:-0}
[ "$B_DECISIONS" -gt 0 ] || die "arm B shows ZERO Tacit decisions — it measured the fallback path, not the seam"
log "arm B seam usage: $B_DECISIONS Tacit decisions observed (window max)"
if grep -qi 'degrad' "$RUNS/$STAMP-tacit-cpu-tacit-watch.log"; then
  log "WARNING: degraded-seam lines in arm B watch log (the lane's as-deployed behavior — counted in the report, not hidden):"
  grep -i 'degrad' "$RUNS/$STAMP-tacit-cpu-tacit-watch.log" | sort | uniq -c | sort -rn | head -20 || true
fi

# --- compare ------------------------------------------------------------------
# Exit 1 on a candidate-loss verdict is the pre-registered outcome to REPORT
# (the rule asks "is the candidate faster without regression?"), never a crash.
log "compare (pre-registered rule + repeat-stability gate)"
set +e
make benchmark-compare A="$RUNS/$A_ID" B="$RUNS/$B_ID" REQUIRE_REPEAT_STABILITY=1
rc=$?
set -e
log "compare exit=$rc (nonzero = candidate-loss verdict — make wraps compare's exit 1 as its own exit 2; artifacts in $RUNS/$B_ID/comparison/)"
log "done: A=$RUNS/$A_ID B=$RUNS/$B_ID — restart the dev stack (e.g. make docker-up-decisions-cpu)"
exit "$rc"
