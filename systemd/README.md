# systemd integration

This directory contains the systemd unit that starts the
**design-pattern-mcp** Docker Compose stack on boot.

## File layout

| File | Purpose |
|---|---|
| `docker/docker-compose.yml` (repo root) | **Development**. Has `build:` sections, designed for `make docker-up`. |
| `systemd/docker-compose.yml` (this dir) | **Production / systemd**. No `build:` (images pre-built), absolute paths, designed for `systemctl enable`. |
| `systemd/docker-compose.cpu.yml` (this dir) | CPU lane overlay: `EMBEDDER_BATCH_SIZE=4` for the CPU TEI pair; enabled via `COMPOSE_FILE` in `.env` (see "CPU lane"). |

## Prerequisites

- systemd-based Linux host.
- Docker Engine with the Compose plugin.
- `graemer` is in the `docker` group.
- Images available locally (Step 0).
- **Shared TEI infra stack** — see [`pattern-tei-infra`](https://github.com/olk/pattern-tei-infra).

## Step 0 — Build or pull images

This host is GPU-less by design: production TEI runs in the external `pattern-tei-infra` stack, so this stack only runs the server and the style head (both CPU-only). `make docker-build-base` still builds the GPU-only TEI pair too, but those images are not started here.

```bash
# From the repo root (server + TEI pair + style head):
make docker-build-base
```

Or pull published images and retag to `:latest` so the `ExecStartPre` guards pass.

## Step 1 — Install

```bash
# 0. Install + enable shared TEI infra (once)
sudo install -m 644 ${HOME}/pattern-tei-infra/pattern-tei-infra.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pattern-tei-infra.service

# 1. Create deployment directory.
sudo install -d /etc/design-pattern-mcp/config

# 2. Copy the production compose file.
sudo install -m 644 systemd/docker-compose.yml /etc/design-pattern-mcp/
# 3. Copy application config.
sudo install -m 644 ~/.config/design-pattern-mcp/config.json \
                /etc/design-pattern-mcp/config/

# 4. Create the .env file.
sudo install -o root -g docker -m 640 /dev/null /etc/design-pattern-mcp/.env
sudo $EDITOR /etc/design-pattern-mcp/.env
# File contents:
#   GENERATOR_API_KEY=sk-...
#   COMPOSE_PROJECT_NAME=dpml-systemd

# 5. Install the unit file.
sudo install -m 644 systemd/design-pattern-mcp.service \
                /etc/systemd/system/

# 6. Reload systemd, enable, and start.
sudo systemctl daemon-reload
sudo systemctl enable --now design-pattern-mcp.service
```

## Verify

```bash
systemctl status design-pattern-mcp
journalctl -u design-pattern-mcp -n 50
curl -s -o /dev/null -w '%{http_code}\n' -X POST "http://localhost:8812/mcp" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"probe","version":"0.0.0"}},"id":1}'   # 200
```

## Sidecars: where each one lives

| Sidecar | Stack | Why there |
|---|---|---|
| TEI embed / rerank | external `pattern-tei-infra` (`pattern-tei-shared`) | shared GPU/CPU infra, one copy for all pattern servers |
| style head (Ch2) | in this stack (`design-pattern-style-head`) | CPU-only, no GPU, no port — the seam ships on and must resolve |

## Style head

`docker-compose.yml` runs MCP + style head (Ch2 on, floor 0.6) with TEI external; no overlay is needed. To run a published head build instead of the local one, set `STYLE_HEAD_IMAGE=olkowa/design-pattern-style-head:<tag>` in `/etc/design-pattern-mcp/.env`, then `sudo systemctl reload design-pattern-mcp`. To stand the seam down, set `DECISIONS_STYLE_ENABLED=false` in the same file.

## CPU lane

When the `pattern-tei-infra` host has no NVIDIA GPU, run the CPU TEI pair there and this overlay here. Infra side (once):

```bash
make docker-build-tei-cpu   # if the -cpu images are not local yet
docker tag design-pattern-tei-embed-cpu:latest pattern-tei-embed:latest
docker tag design-pattern-tei-rerank-cpu:latest pattern-tei-rerank:latest
sudo systemctl restart pattern-tei-infra
```

This side:

```bash
sudo install -m 644 systemd/docker-compose.cpu.yml /etc/design-pattern-mcp/
# append to /etc/design-pattern-mcp/.env (edit the line if one exists):
#   COMPOSE_FILE=docker-compose.yml:docker-compose.cpu.yml
sudo systemctl reload design-pattern-mcp
```

The overlay only lowers `EMBEDDER_BATCH_SIZE` to 4 (candle-on-CPU cap; the base default 8 assumes the GPU pair) — the server and style head are CPU-only in both lanes, and the dense cache is shared, so switching never re-embeds. An explicit `EMBED_CLIENT_BATCH_SIZE` in `.env` overrides the overlay default in both lanes. Switch back by deleting the `COMPOSE_FILE` line, retagging the `-gpu` images onto `pattern-tei-*:latest`, and reloading both units.

## Updating

```bash
make docker-build-base
sudo systemctl reload design-pattern-mcp
```

## Uninstall

```bash
sudo systemctl disable --now design-pattern-mcp.service
sudo rm /etc/systemd/system/design-pattern-mcp.service
sudo systemctl daemon-reload
sudo rm -rf /etc/design-pattern-mcp
```
