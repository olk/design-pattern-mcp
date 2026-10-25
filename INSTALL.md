# INSTALL — deployment runbook for AI agents

Step-by-step instructions for an AI agent (Claude Code, Codex, OpenCode, …)
to install **design-pattern-mcp** on the Linux host the agent runs on.
All steps are non-interactive: files are written with heredocs, every step has
a verification command with its expected output. Run commands from the repo
root unless noted. Requires a clone of this repo (this file lives in it).

Human-oriented background: [README.md](README.md) (configuration reference,
MCP client setup) and [systemd/README.md](systemd/README.md) (systemd
deep-dive).

## What gets installed

| Component | Image (Docker Hub — public registry) | Role |
|---|---|---|
| MCP server | `olkowa/design-pattern-mcp` | Tool server, streamable-http (container port 8050 both stacks) |
| TEI embedder | `olkowa/pattern-tei-embed` | Qwen3-Embedding-0.6B sidecar; required for retrieval |
| TEI reranker | `olkowa/pattern-tei-rerank` | Reranker sidecar (`gte-reranker-modernbert-base`); required for retrieval |

Tags: `latest` plus a pinned release tag per image — TEI embed/rerank `1.1.1`
(the TEI images change rarely and are shared with the sibling
`architecture-pattern-mcp` server). Use Docker Hub. Images are
**linux/amd64 only**; on arm64 hosts build locally instead: `make docker-build-all`.

Pick exactly ONE path:

| | Path A — Compose stack | Path B — systemd stack |
|---|---|---|
| Use when | evaluating / developing; no sudo available | persistent host service, starts at boot |
| Needs root | no (Docker group membership only) | yes (`sudo`) |
| Host port | 8822 | 8812 |
| TEI sidecars | own copies inside the compose project | shared `pattern-tei-infra` stack |
| Survives reboot | no (manual `up -d`) | yes (systemd unit) |

Both paths use fixed host ports (8822 dev, 8812 systemd), so dev and
systemd stacks run concurrently. Container names
and compose projects are already distinct
(`design-pattern-mcp-dev` vs `design-pattern-mcp`,
`design-pattern-mcp-dev` vs `dpml-systemd`).

## 0. Preconditions (both paths)

```bash
test -f docker/docker-compose.yml && echo "repo clone: OK"
docker compose version                      # needs ≥ v2.20
id -nG | tr ' ' '\n' | grep -qx docker && echo "docker group: OK"
df -h /var/lib/docker                       # ≥ 8 GB free (MCP ≈ 1 GB with baked dense cache, TEI ≈ 6 GB)
```

For Path B additionally:

```bash
systemctl is-system-running                 # expect "running" or "degraded"
```

The deployed stack defaults to **MiniMax** as generator LLM, so obtain a
`MINIMAXAI_API_KEY` (`sk-...`). To use another provider (OpenAI, DeepSeek,
Ollama, …) see [Switching the generator LLM](#switching-the-generator-llm) —
the install steps below stay the same.

## Path A — Compose stack (repo clone, no build, no root)

`docker/docker-compose.yml` has no `build:` sections; images are parameterized
(`MCP_IMAGE`, `TEI_IMAGE`, `TEI_RERANK_IMAGE`, `MINIMAXAI_API_KEY`)
and resolved from an env file.

```bash
# A1. Resolve the MCP release version and pull the images. TEI pins differ
#     from the MCP version (shared TEI images release rarely); check the tags
#     table at the top of this file for the current TEI pin.
VERSION=$(grep -m 1 '^version' pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/')
TEI_VERSION=1.1.1
docker pull "olkowa/design-pattern-mcp:${VERSION}"
# If the MCP pull 404s (no published release yet — confirm at
# https://hub.docker.com/r/olkowa/design-pattern-mcp/tags), build it
# locally instead (the TEI pulls below still work; those images are published):
#   make docker-build-all
# and in step A2 use MCP_IMAGE=design-pattern-mcp:latest (the local build tag).
docker pull "olkowa/pattern-tei-embed:${TEI_VERSION}"
docker pull "olkowa/pattern-tei-rerank:${TEI_VERSION}"
# If a pull fails (tag not published), list available tags and use latest:
#   curl -fsS 'https://hub.docker.com/v2/repositories/olkowa/design-pattern-mcp/tags'

# A2. Write the env file next to the compose file (compose interpolation source).
cat > docker/.env <<EOF
MCP_IMAGE=olkowa/design-pattern-mcp:${VERSION}
TEI_IMAGE=olkowa/pattern-tei-embed:${TEI_VERSION}
TEI_RERANK_IMAGE=olkowa/pattern-tei-rerank:${TEI_VERSION}
MINIMAXAI_API_KEY=sk-REPLACE-ME
EOF

# A3. Sanity check: interpolation resolved to the Hub refs (each printed image:
#     olkowa/design-pattern-mcp:…, olkowa/pattern-tei-embed:…,
#     olkowa/pattern-tei-rerank:…). Empty MINIMAXAI_API_KEY warning is expected
#     only if you skipped the key.
docker compose --env-file docker/.env -f docker/docker-compose.yml config | grep 'image:'

# A4. Start. TEI sidecars cold-start model weights; first start takes ~2 min.
#     (Published server images bake in the dense embedding cache; a locally
#     built cache-less image instead embeds the whole catalog at first start,
#     ~19 min on CPU.)
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d

# A5. Wait until all three report "healthy" (poll if still "starting").
docker inspect -f '{{.Name}} {{.State.Health.Status}}' \
  design-pattern-tei-embed-dev design-pattern-tei-rerank-dev design-pattern-mcp-dev

# A6. Probe the MCP endpoint — must print 200.
#     (GET /health is 404 by design; the JSON-RPC initialize POST is the probe.)
curl -s -o /dev/null -w '%{http_code}\n' -X POST "http://localhost:8822/mcp" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"probe","version":"0.0.0"}},"id":1}'
```

Installed endpoint: `http://localhost:8822/mcp` (streamable-http).

Stop / remove the stack:

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml down
rm docker/.env   # contains the API key — remove or keep as you prefer
```

Config in Path A is the repo file `config/config.json` (mounted at
`/app/config`); after editing it, restart the server container to apply.

## Path B — systemd stack (persistent, sudo required)

Layout on the host:

| Path | Content |
|---|---|
| `/etc/pattern-tei-infra/` | TEI compose (from the `pattern-tei-infra` repo) |
| `/etc/systemd/system/pattern-tei-infra.service` | TEI stack unit |
| `/etc/design-pattern-mcp/` | MCP compose + `.env` |
| `/etc/design-pattern-mcp/config/config.json` | MCP configuration |
| `/etc/systemd/system/design-pattern-mcp.service` | MCP stack unit |

Run all steps as the user that will own the services (must be in the `docker`
group). The unit template hardcodes `User=graemer` (the author's username) —
the `sed` below substitutes the invoking user.

### B0 — shared TEI infra (prerequisite, once)

```bash
TEI_VERSION=1.1.1   # TEI pins differ from the MCP version; see the tags table
git clone https://github.com/olk/pattern-tei-infra /tmp/pattern-tei-infra

# Pull Hub images and retag to the local names the infra compose expects
# (pattern-tei-embed:latest / pattern-tei-rerank:latest).
docker pull "olkowa/pattern-tei-embed:${TEI_VERSION}"
docker pull "olkowa/pattern-tei-rerank:${TEI_VERSION}"
docker tag "olkowa/pattern-tei-embed:${TEI_VERSION}" pattern-tei-embed:latest
docker tag "olkowa/pattern-tei-rerank:${TEI_VERSION}" pattern-tei-rerank:latest

sudo install -d /etc/pattern-tei-infra
sudo install -m 644 /tmp/pattern-tei-infra/docker-compose.yml /etc/pattern-tei-infra/
sed "s/^User=graemer$/User=${USER}/" /tmp/pattern-tei-infra/pattern-tei-infra.service \
  | sudo tee /etc/systemd/system/pattern-tei-infra.service > /dev/null
sudo chmod 644 /etc/systemd/system/pattern-tei-infra.service
sudo systemctl daemon-reload
sudo systemctl enable --now pattern-tei-infra.service

# Verify: both containers must report "healthy" (cold start up to ~2 min).
docker inspect -f '{{.Name}} {{.State.Health.Status}}' pattern-tei-embed pattern-tei-rerank
```

Note: this stack is shared with the sibling `architecture-pattern-mcp`,
`agent-pattern-mcp`, and `blockchain-pattern-mcp` servers. If any of them
has it already installed and running on the host, skip B0 entirely.

### B1 — MCP image under the local name the unit checks

```bash
VERSION=$(grep -m 1 '^version' pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/')
docker pull "olkowa/design-pattern-mcp:${VERSION}"
docker tag "olkowa/design-pattern-mcp:${VERSION}" design-pattern-mcp:latest
# (If the pull 404s — no published MCP release yet — `make docker-build-all`
# locally and skip the retag: the unit inspects `design-pattern-mcp:latest`,
# which the local build already tags.)
```

(`design-pattern-mcp.service` runs `ExecStartPre=docker image inspect
design-pattern-mcp:latest` — the retag is what makes that pass.)

### B2 — configuration files under `/etc/design-pattern-mcp/`

```bash
sudo install -d /etc/design-pattern-mcp/config
sudo install -m 644 systemd/docker-compose.yml /etc/design-pattern-mcp/
# The repo config is the canonical template ({env:VAR:-default} expansion;
# compose-provided env vars override it).
sudo install -m 644 config/config.json /etc/design-pattern-mcp/config/

# .env — read by the unit (EnvironmentFile) and by compose interpolation.
# root:docker 640: the service runs as a non-root docker-group user, which
# must be able to read the file; the docker group is effectively privileged.
sudo tee /etc/design-pattern-mcp/.env > /dev/null <<EOF
MINIMAXAI_API_KEY=sk-REPLACE-ME
COMPOSE_PROJECT_NAME=dpml-systemd
EOF
sudo chown root:docker /etc/design-pattern-mcp/.env
sudo chmod 640 /etc/design-pattern-mcp/.env
```

### B3 — install and enable the unit

```bash
sed "s/^User=graemer$/User=${USER}/" systemd/design-pattern-mcp.service \
  | sudo tee /etc/systemd/system/design-pattern-mcp.service > /dev/null
sudo chmod 644 /etc/systemd/system/design-pattern-mcp.service
sudo systemctl daemon-reload
sudo systemctl enable --now design-pattern-mcp.service
```

### B4 — verify

```bash
systemctl is-active design-pattern-mcp                          # "active" (Type=oneshot → active (exited) is correct)
docker inspect -f '{{.State.Health.Status}}' design-pattern-mcp       # "healthy" (fast with the baked dense cache; ≤ ~19 min on a cache-less image)
curl -s -o /dev/null -w '%{http_code}\n' -X POST "http://localhost:8812/mcp" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"probe","version":"0.0.0"}},"id":1}'   # 200
docker exec design-pattern-mcp python -c "import urllib.request; print(urllib.request.urlopen('http://pattern-tei-embed:8080/health').status)"  # 200 = TEI reachable
```

Installed endpoint: `http://localhost:8812/mcp` (streamable-http).

### B5 — day-2 operations

```bash
sudo systemctl stop     design-pattern-mcp   # compose down
sudo systemctl start    design-pattern-mcp   # compose up -d
sudo systemctl reload   design-pattern-mcp   # recreate containers after editing
                                                   # /etc/design-pattern-mcp/ compose or .env
# Update to a newer release: re-run B1 pull+retag, then
sudo systemctl reload design-pattern-mcp
# Logs
journalctl -u design-pattern-mcp -n 50
docker compose -p dpml-systemd -f /etc/design-pattern-mcp/docker-compose.yml logs -f
```

## Dense-embedding cache (restart durability)

**On by default**: `dense_cache.mode=auto` with `write=true` (Pydantic defaults,
`config/config.json`, and both compose stacks) — the serving path always passes the
cache, so neither deployment path has to enable anything (host `uv run` needs the
`DESIGN_PATTERN_DENSE_CACHE_PATH` export shown in the README's Local Development block,
because the image default does not exist on a workstation). Off switches:
`DENSE_CACHE_MODE=off` in a stack's env, `DESIGN_PATTERN_DENSE_CACHE_MODE=off` on a host
run, `mode: off` in a config file.

The dense retrieval leg embeds one `document_text` per catalog record (486) at startup
(~19 min on the CPU TEI sidecar). That matrix is snapshotted to
`DESIGN_PATTERN_DENSE_CACHE_PATH` (`/app/data/dense-cache` in the image) as
`manifest.json` + `vectors.npy` and reloaded on the next start, so a restart costs one
probe embedding instead of a full corpus pass.

- **Every stack mounts a NAMED volume there** —
  `design-pattern-mcp-dev-dense-cache` (Path A) /
  `design-pattern-mcp-systemd-dense-cache` (Path B). The snapshot then survives image
  upgrades, container recreation and `docker compose down`; `make docker-down` keeps it.
  It is deliberately not inside the config bind mount, so a config re-install cannot
  take it with it.
- **A fresh volume is seeded from the image's baked snapshot** (Docker copies the image
  directory into an empty named volume once), so a *first* install of a re-pulled image
  starts warm. An **existing** volume keeps its own snapshot: after a rebuild that
  changes the catalogue or the model, delete it first
  (`docker volume rm design-pattern-mcp-systemd-dense-cache` resp.
  `design-pattern-mcp-dev-dense-cache`, or `docker compose down -v`) — otherwise the
  fingerprint guards force one re-embed (~19 min) and rewrite the volume. For a
  genuinely cold cache use a cache-less image (`make docker-build-nocache`) or switch
  the cache off: `DENSE_CACHE_MODE=off` in the stack's env (the compose interpolation
  overwrites `DESIGN_PATTERN_DENSE_CACHE_MODE`, so the long name only works on host
  runs / bare `docker run`). A bare `docker run` gets no volume; pass
  `-v <your-cache-volume>:/app/data/dense-cache` to keep the cache — a new volume name
  is a separate, cold cache (one re-embed), the two stack volumes are the shared ones.
- **Invalidated automatically** by a changed embedding function (model string, text
  instruction, declared `model_repo`/`onnx_variant`), a changed catalog (record count,
  name order, any `document_text`), an absent/shape-invalid/off-norm `vectors.npy`, or a
  probe-vector mismatch (a redeployed TEI sidecar serving a different model behind an
  unchanged config). Each change costs exactly one re-embed — the write-back stores a
  snapshot with the new identity, so the next restart hits (a snapshot baked by an older
  release misses once for the same reason). Every decision is logged with the directory
  it used: `Dense cache hit` (`path`, `records`, `mode`), `Dense cache miss` (`reason`,
  `path`), `Dense cache written` (`path`, `records`, `dimension`),
  `Dense cache disabled` (`mode`, `path`), `Dense cache write failed` (`path`,
  `reason`), plus the retriever's summary `Dense vectors resolved` (`source`,
  `documents`).
- **Writability**: `/app/data` is copied `1000:1000` in the image and the volume inherits
  that ownership — a fresh volume is writable without a chown step. A root-owned
  read-only directory logs `Dense cache write failed` and the server runs cacheless;
  write-back never fails the start.
- **Single writer**: no locking — run one server process per cache directory (dev and
  systemd use separate volumes).
- **Upgrading from a pre-volume deployment**: the older compose file has no mount — the
  cache then lives in the container layer and dies with the container. Re-install the
  compose file for your path — Path B:
  `sudo install -m 644 systemd/docker-compose.yml /etc/design-pattern-mcp/` (B2);
  Path A: `make docker-build` — then recreate the container
  (`docker compose up -d` / `sudo systemctl reload design-pattern-mcp`). Path B runs
  compose from `/etc`, so without that install the deployed stack keeps the old
  container-layer cache and gains nothing. The first start after that re-embeds once and
  writes into the volume; later restarts are warm.
- **Inspect**: the logged path is the *in-container* view. Use
  `docker exec <container> ls -l /app/data/dense-cache`, or
  `docker volume inspect --format '{{.Mountpoint}}' design-pattern-mcp-dev-dense-cache`
  (resp. `-systemd-`) for the host side, and
  `docker inspect -f '{{json .Mounts}}' <container>` to verify the mount.

## Switching the generator LLM

Both compose files hardcode the MiniMax generator env block. To use another
LiteLLM provider, edit the four `DESIGN_PATTERN_GENERATOR_*` lines in the
installed compose file — Path A: `docker/docker-compose.yml`; Path B:
`/etc/design-pattern-mcp/docker-compose.yml` — e.g. for OpenAI:

```yaml
      - DESIGN_PATTERN_GENERATOR_PROVIDER=openai
      - DESIGN_PATTERN_GENERATOR_MODEL=openai/gpt-4o-mini      # LiteLLM syntax: provider/model
      - DESIGN_PATTERN_GENERATOR_BASE_URL=https://api.openai.com/v1
      - DESIGN_PATTERN_GENERATOR_API_KEY=${OPENAI_API_KEY}
```

then put `OPENAI_API_KEY=sk-...` in the env file (`docker/.env` resp.
`/etc/design-pattern-mcp/.env`) and restart (A: `down` + `up -d`;
B: `sudo systemctl restart design-pattern-mcp`). Full provider list:
[LiteLLM providers](https://docs.litellm.ai/docs/providers).

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `GET /health` returns 404 | Expected — FastMCP streamable-http exposes no GET /health. Probe `POST /mcp` (A6/B4). |
| `docker image inspect design-pattern-mcp:latest` failed (unit start) | B1 retag step missed or used a different tag. |
| `network pattern-tei-shared not found` | TEI infra not running: `sudo systemctl start pattern-tei-infra.service`. |
| MCP container unhealthy; logs show TEI connection errors | TEI still cold-starting (start_period 120 s) — recheck health after ~2 min; if persistent, verify B0 (Path B) or the dev sidecars (Path A). |
| MCP container unhealthy for many minutes; logs show dense-cache re-embed | Cache-less image, empty volume, or an invalidated snapshot embeds the whole catalog at first start (~19 min on CPU; healthcheck start_period 1800 s). Wait, or rebuild with `make docker-build-all` (bakes the cache). A reset volume re-seeds from the image, so the re-embed happens once per new catalogue/model. |
| Dense cache seems missing on the host although the log shows a cache hit | Expected — the logged path is the *in-container* view of the named volume. Inspect `docker exec <container> ls -l /app/data/dense-cache`, or `docker volume inspect --format '{{.Mountpoint}}' design-pattern-mcp-dev-dense-cache` (resp. `-systemd-`); verify the mount with `docker inspect -f '{{json .Mounts}}' <container>`. |
| Log shows `Dense cache write failed` but startup continues | By design (best-effort write-back): the cache directory is read-only/root-owned or full. Fix ownership or point `DESIGN_PATTERN_DENSE_CACHE_PATH` at a writable directory; the run keeps the vectors it just embedded. |
| Container healthy but design tool calls fail with 401/502, or every LLM call returns `ERR_009` | Generator API key wrong/missing in the env file, or provider env block mismatched — see generator section. |
| Tool calls rejected for ~60 s after 3 transport failures | Circuit breaker open (`..._CIRCUIT_BREAKER_*` knobs); check the provider key/endpoint, then wait out the cooldown. |
| Unit fails reading environment file | `/etc/design-pattern-mcp/.env` missing or unreadable: recreate per B2 (`root:docker 640`, user in `docker` group). |
| `start request repeated too quickly` | Earlier start failed; inspect `journalctl -u design-pattern-mcp -n 50`, fix, then `systemctl reset-failed design-pattern-mcp` and start again. |
| Reranker HTTP 429 "Model is overloaded" | One full-text rerank request already saturates the CPU sidecar's token buffer — keep `DESIGN_PATTERN_RETRIEVAL_MAX_CONCURRENT_PARTS=1` (shipped default). |
| Image pull fails for `${VERSION}`/`${TEI_VERSION}` | Tag not published; check tags endpoints (A1 comment) and use `latest`. |

## Uninstall

```bash
# Path B
sudo systemctl disable --now design-pattern-mcp.service
sudo rm /etc/systemd/system/design-pattern-mcp.service
sudo rm -rf /etc/design-pattern-mcp
# TEI infra — ONLY if no other pattern MCP server on this host uses it
# (stop MCP stacks first: removing it breaks their TEI connectivity)
sudo systemctl disable --now pattern-tei-infra.service
sudo rm /etc/systemd/system/pattern-tei-infra.service
sudo rm -rf /etc/pattern-tei-infra
docker network rm pattern-tei-shared 2>/dev/null || true
sudo systemctl daemon-reload

# Path A
docker compose --env-file docker/.env -f docker/docker-compose.yml down
rm docker/.env

# Dense-cache volumes (both paths) — keeps ~2 MB of vectors per stack
docker volume rm design-pattern-mcp-dev-dense-cache design-pattern-mcp-systemd-dense-cache 2>/dev/null || true
```
