#!/usr/bin/env bash
# CLM sidecar entrypoint: vLLM serves the encoder, clm-serve puts the heads in front.
#
# The encoder is frozen Qwen3-8B in "pooling" runner mode (it answers /v1/embeddings);
# clm-serve reads those embeddings and answers the two wires this server uses:
#
#   POST /v1/systemone  typed questions -> {answers: {id: {choice, confidence, probabilities}}}
#   POST /v1/rank       {context, question, answers[]} -> {ranked: [{rank, candidate, prob}]}
#   GET  /health        readiness (the healthcheck)
#
# ``--no-download`` is deliberate: the heads are baked into the image, and a silent
# network fetch would make the container depend on the Hub at start.
set -euo pipefail

MODEL_DIR="${CLM_MODEL_DIR:-/models/qwen3-8b}"
SERVED_MODEL="${CLM_SERVED_MODEL:-qwen3-8b}"
UPSTREAM_PORT="${CLM_UPSTREAM_PORT:-8090}"
SERVE_PORT="${CLM_PORT:-8700}"
CKPT="${CLM_CKPT:-/clm/heads/best_head.pt}"
MAX_TOKENS="${CLM_EMB_MAX_TOKENS:-2048}"
STARTUP_TIMEOUT="${CLM_STARTUP_TIMEOUT:-900}"

if [ ! -f "${CKPT}" ]; then
    echo "clm-entrypoint: heads checkpoint ${CKPT} is missing" >&2
    exit 1
fi
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "clm-entrypoint: ${MODEL_DIR} does not look like an HF checkpoint" >&2
    exit 1
fi

vllm serve "${MODEL_DIR}" \
    --served-model-name "${SERVED_MODEL}" \
    --runner pooling \
    --max-model-len "${MAX_TOKENS}" \
    --host 127.0.0.1 \
    --port "${UPSTREAM_PORT}" &
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
        echo "clm-entrypoint: the encoder exited before becoming healthy" >&2
        wait "${VLLM_PID}"
        exit 1
    fi
    waited=$((waited + 2))
    if [ "${waited}" -ge "${STARTUP_TIMEOUT}" ]; then
        echo "clm-entrypoint: encoder not healthy after ${STARTUP_TIMEOUT}s" >&2
        shutdown
        exit 1
    fi
    sleep 2
done

exec clm-serve \
    --host 0.0.0.0 \
    --port "${SERVE_PORT}" \
    --emb-url "http://127.0.0.1:${UPSTREAM_PORT}/v1/embeddings" \
    --emb-model "${SERVED_MODEL}" \
    --max-tokens "${MAX_TOKENS}" \
    --ckpt "${CKPT}" \
    --no-download \
    --no-ui
