# syntax=docker/dockerfile:1.7

# Containerized deployment of design-pattern-mcp (house style, cf.
# ../architecture-pattern-mcp):
#   - dependencies install BEFORE src/ + pattern/ are copied (layer cache)
#   - the project itself is never installed; runtime uses python -m src.main
#   - non-root production stage, COPY --chown (no chown -R extra layer)
#   - ENTRYPOINT/CMD split: MCP Registry OCI launchers append server args after
#     the image reference (docker run -i <image> --transport stdio)
#
# Retrieval needs the two TEI sidecars (docker/docker-compose.yml); the
# server itself starts without them using --offline (deterministic lexical
# embedder/reranker).

# ---- Build stage ------------------------------------------------------------
FROM python:3.12-slim-bookworm AS build

# Copy uv binary from the official distroless image (~50 MB, no Python layer).
# Pin to a stable tag; replace 0.7.0 with the current stable at release time.
COPY --from=ghcr.io/astral-sh/uv:0.7.0 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
  UV_LINK_MODE=copy \
  UV_NO_CACHE=1 \
  UV_PYTHON_DOWNLOADS=0 \
  PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first — layer caching: a src/ or pattern/ edit does not
# re-run uv sync.  The project itself is never installed (--no-install-project),
# so hatchling does not need src/ yet.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --no-install-project

# Source + runtime assets — copied after deps so an edit here does not bust the
# dependency cache layer.  docs/ is a runtime asset (schemas + design-patterns.md
# catalog read by src/catalog.py); pattern/ is the 39 MB record corpus.
COPY src/ ./src/
COPY pattern/*.json ./pattern/
COPY docs/ ./docs/
COPY config/config.json ./config/config.json
# Dense-vector cache: present when `make embed-cache` ran before the build;
# the loader treats a missing/short manifest as a cache miss.
COPY data/dense-cache/ ./data/dense-cache/

# ---- Reasoning MCPs build stage ---------------------------------------------
# The shannonthinking and code-reasoning MCP servers are baked into the image so
# ReasoningClient invokes them directly via node (no runtime npx fetch). Version
# pins verified on npm; override via --build-arg. Entry-point existence is
# asserted here so a wrong dist path fails the BUILD loudly.
FROM node:20-bookworm-slim AS build-mcps

ARG SHANNON_NPM_PKG=server-shannon-thinking@0.1.1
ARG CODE_REASONING_NPM_PKG=@mettamatt/code-reasoning@0.8.1

RUN --mount=type=cache,target=/root/.npm \
  npm install -g --omit=dev "${SHANNON_NPM_PKG}" "${CODE_REASONING_NPM_PKG}" && \
  test -f /usr/local/lib/node_modules/server-shannon-thinking/dist/index.js && \
  test -f /usr/local/lib/node_modules/@mettamatt/code-reasoning/dist/index.js
# NOTE: the embedder's default api_key is "" (config.json) — fine with the
# --offline deterministic embedder (the HEALTHCHECK path), but litellm's OpenAI
# SDK route needs a non-empty credential string against the TEI sidecar; the
# compose stack sets EMBEDDER_API_KEY accordingly.

# ---- Production stage ------------------------------------------------------
FROM python:3.12-slim-bookworm AS production

RUN groupadd --gid 1000 mcpgroup && \
  useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home mcpuser

WORKDIR /app

# libstdc++6: faiss _swigfaiss links it (libgomp is bundled in faiss_cpu.libs);
# ca-certificates: httpx/litellm HTTPS calls.
RUN apt-get update && \
  apt-get install -y --no-install-recommends libstdc++6 ca-certificates && \
  rm -rf /var/lib/apt/lists/*

# --chown creates files owned by mcpuser in a single filesystem operation.
# Unlike a separate `RUN chown -R`, this does NOT duplicate the multi-GB venv
# into an extra image layer.
COPY --from=build --chown=1000:1000 /app/.venv /app/.venv
COPY --from=build --chown=1000:1000 /app/src ./src
COPY --from=build --chown=1000:1000 /app/pyproject.toml ./
COPY --from=build --chown=1000:1000 /app/pattern ./pattern
COPY --from=build --chown=1000:1000 /app/docs ./docs
COPY --from=build --chown=1000:1000 /app/config ./config
COPY --from=build --chown=1000:1000 /app/data ./data

# The dense-cache directory is the mount point of both compose stacks' named
# volume. Docker seeds a FRESH volume from this directory and copies its
# ownership, which is what makes write-back work without a chown step — so the
# --chown above is a runtime contract, not an image nicety. Assert it: a
# root-owned directory would make every cache write fail (logged as
# `Dense cache write failed`), costing a full re-embed on every restart.
RUN test "$(stat -c '%u:%g' /app/data)" = "1000:1000" && \
  test "$(stat -c '%u:%g' /app/data/dense-cache)" = "1000:1000" && \
  echo "Dense-cache mount point owned by 1000:1000"

# Embedded reasoning MCP packages: node binary from the same image the packages
# were installed under (no apt nodejs version mismatch), plus the global
# node_modules tree. Entry points are invoked by absolute path — no NODE_PATH.
COPY --from=build-mcps /usr/local/bin/node /usr/local/bin/node
COPY --from=build-mcps --chown=1000:1000 /usr/local/lib/node_modules /usr/local/lib/node_modules

# Console shim: preloaded via NODE_OPTIONS (src/reasoning/client.py) so the
# servers' console.log/info lines go to stderr instead of the JSON-RPC stdout.
# It ships inside src/, which the build stage already copied.
# Build-time contract: node runs, both reasoning entry points exist, and the
# console shim actually silences stdout under NODE_OPTIONS.
RUN node --version && \
  test -f /usr/local/lib/node_modules/server-shannon-thinking/dist/index.js && \
  test -f /usr/local/lib/node_modules/@mettamatt/code-reasoning/dist/index.js && \
  test "$(NODE_OPTIONS='--require /app/src/reasoning/quiet-console.cjs' \
  node -e 'console.log("x"); console.info("y")' 2>/dev/null | wc -c)" = "0" && \
  echo "Reasoning MCPs embedded: shannon + code-reasoning"

# NOTE: PYTHONPATH=/app (not /app/src as in the sibling) — this repo's imports
# are absolute (`from src.x import y`), so the PARENT of src/ must be importable.
# pyproject.toml is baked in as the --version fallback (the dist is not installed).
ENV PATH="/app/.venv/bin:$PATH" \
  PYTHONPATH="/app" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

USER 1000:1000

# Readiness = the MCP HTTP surface accepts connections. The server binds its
# port only AFTER lifespan warmup (catalog + retrieval index), so a successful
# TCP connect implies full readiness. The offline `--health` diagnostic stays
# available as a CLI flag but is NOT the container probe: it exercises the
# deterministic embedder, not the serving path.
HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s --retries=3 \
  CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8050), timeout=5).close()" || exit 1

ENTRYPOINT ["/app/.venv/bin/python", "-m", "src.main"]
CMD ["--transport", "streamable-http"]

LABEL org.opencontainers.image.title="design-pattern-mcp" \
  org.opencontainers.image.description="MCP server selecting design patterns for a described system" \
  org.opencontainers.image.version="1.0.1" \
  org.opencontainers.image.authors="Oliver Kowalke" \
  org.opencontainers.image.source="https://github.com/olk/design-pattern-mcp" \
  org.opencontainers.image.licenses="MIT" \
  io.design-pattern-mcp.reasoning-mcps="shannon,code-reasoning" \
  io.modelcontextprotocol.server.name="io.github.olk/design-pattern-mcp"
