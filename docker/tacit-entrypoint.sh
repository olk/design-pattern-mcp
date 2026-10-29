#!/usr/bin/env bash
# Tacit sidecar entrypoint: vLLM serves the model, AnyJev's gateway fronts it.
#
# The gateway (``python -m anyjev.serve``) reads only the tokenizer and forwards
# decisions to the vLLM server, so both live in this container and the MCP server talks
# to exactly one port:
#
#   POST /v1/decide  {"state","question","options","kind"}  -> decision
#   POST /v1/decide  {"items":[...]}                        -> {"decisions":[...]}
#   GET  /v1/stats   escalation share (the healthcheck)
#
# ``TACIT_ADAPTIVE=false`` (the shipped default) keeps every seam deterministic: one
# forward per decision, no thinking, no sampling. Set it to true to let the gateway
# escalate low-margin decisions to the model's reasoning — useful for exploration, not
# for selection-affecting seams.
set -euo pipefail

MODEL_DIR="${TACIT_MODEL_DIR:-/models/tacit}"
SERVED_MODEL="${TACIT_SERVED_MODEL:-tacit-4b}"
UPSTREAM_PORT="${TACIT_UPSTREAM_PORT:-8000}"
GATEWAY_PORT="${TACIT_PORT:-8100}"
STARTUP_TIMEOUT="${TACIT_STARTUP_TIMEOUT:-900}"

if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "tacit-entrypoint: ${MODEL_DIR} does not look like an HF checkpoint" >&2
    exit 1
fi

vllm serve "${MODEL_DIR}" \
    --served-model-name "${SERVED_MODEL}" \
    --host 127.0.0.1 \
    --port "${UPSTREAM_PORT}" \
    --max-model-len 8192 &
VLLM_PID=$!

shutdown() {
    trap - TERM INT
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap shutdown TERM INT

waited=0
until python3 -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:'+sys.argv[1]+'/health', timeout=3)" "${UPSTREAM_PORT}" 2>/dev/null; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "tacit-entrypoint: vLLM exited before becoming healthy" >&2
        wait "${VLLM_PID}"
        exit 1
    fi
    waited=$((waited + 2))
    if [ "${waited}" -ge "${STARTUP_TIMEOUT}" ]; then
        echo "tacit-entrypoint: vLLM not healthy after ${STARTUP_TIMEOUT}s" >&2
        shutdown
        exit 1
    fi
    sleep 2
done

ARGS=(--model "${MODEL_DIR}" --upstream "http://127.0.0.1:${UPSTREAM_PORT}/v1" --served-model "${SERVED_MODEL}"
      --host 0.0.0.0 --port "${GATEWAY_PORT}")
if [ "${TACIT_ADAPTIVE:-false}" = "true" ]; then
    ARGS+=(--adaptive)
fi

exec python3 -m anyjev.serve "${ARGS[@]}"
