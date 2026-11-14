# systemd integration

This directory contains the systemd unit that starts the
**design-pattern-mcp** Docker Compose stack on boot.

## File layout

| File | Purpose |
|---|---|
| `docker/docker-compose.yml` (repo root) | **Development**. Has `build:` sections, designed for `make docker-up`. |
| `systemd/docker-compose.yml` (this dir) | **Production / systemd**. No `build:` (images pre-built), absolute paths, designed for `systemctl enable`. |

## Prerequisites

- systemd-based Linux host.
- Docker Engine with the Compose plugin.
- `graemer` is in the `docker` group.
- Images available locally (Step 0).
- **Shared TEI infra stack** — see [`pattern-tei-infra`](https://github.com/olk/pattern-tei-infra).

## Step 0 — Build or pull images

This host is GPU-less by design, so it builds the CPU images only — the full
`make docker-build-all` includes Tacit's GPU self-distill and aborts here on the
committed sentinels (weights untracked). Run Tacit on a GPU host instead
("Decision models" below).

```bash
# From the repo root (server + TEI pair + style head):
make docker-build-base
```

Or pull published images and retag to `:local` so the `ExecStartPre` guard passes.

## Step 1 — Install

```bash
# 0. Install + enable shared TEI infra (once)
sudo install -m 644 ${HOME}/pattern-tei-infra/pattern-tei-infra.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pattern-tei-infra.service

# 1. Create deployment directory.
sudo install -d /etc/design-pattern-mcp/config

# 2. Copy the production compose files (base + decision overlay).
sudo install -m 644 systemd/docker-compose.yml /etc/design-pattern-mcp/
sudo install -m 644 systemd/docker-compose.decision-models.yml /etc/design-pattern-mcp/
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
| Tacit gateway (J1/J2/J4/J5) | GPU host, plain HTTP (`http://tacit:8100` by default) | needs a card; `anyjev.serve` has no auth, so trusted networks only |

## Decision models

Two files, two stacks: `docker-compose.yml` runs MCP + style head (Ch2 on, floor 0.6)
with TEI external; `docker-compose.decision-models.yml` adds the env that points the
Tacit seams at the gateway. Enable the overlay persistently in
`/etc/design-pattern-mcp/.env`:

```
COMPOSE_FILE=docker-compose.yml:docker-compose.decision-models.yml
```

then `sudo systemctl reload design-pattern-mcp`. To run a published head build
instead of the local one, set `STYLE_HEAD_IMAGE=olkowa/design-pattern-style-head:<tag>`
in the same `.env`.

The Tacit seams (J1 rerank, J2 category, J4 weights, J5 gate) stay **off** until the
overlay is on AND the switches below are set. The gateway lives on a GPU host and is
reached over plain HTTP.

1. **GPU host** — run the gateway on a trusted-network address (it has no auth of its
   own — never a publicly reachable port):

   ```bash
   make docker-build-tacit            # tags :1.0.1 and :latest
   docker run -d --name design-pattern-tacit --restart unless-stopped \
     --device nvidia.com/gpu=0 -p 10.0.0.5:8100:8100 \
     design-pattern-tacit:latest
   ```

   (Replace `10.0.0.5` with the host's firewalled private address; the entrypoint
   listens on `0.0.0.0` inside the container, so the publish is what confines it.)

2. **`/etc/design-pattern-mcp/.env`** — gateway address plus the seam switches
   (all default off):

   ```
   COMPOSE_FILE=docker-compose.yml:docker-compose.decision-models.yml
   TACIT_GATEWAY_URL=http://10.0.0.5:8100
   DECISIONS_WEIGHTS_ENABLED=true
   DECISIONS_CATEGORY_ENABLED=true
   DECISIONS_REASONING_GATE_ENABLED=true
   RERANKER_PROVIDER=tacit
   ```

   `TACIT_GATEWAY_URL` feeds all four Tacit endpoints (each `*_BASE_URL` still wins
   individually when set). The overlay default assumes a host named `tacit`.

   ```bash
   sudo systemctl reload design-pattern-mcp
   ```

   Startup log shows `ok (tacit, model=tacit-4b)` per seam; any transport failure degrades silently to
   the non-model path, so check the logs for `degraded` warnings if a seam reports
   `ok` but behaves like the baseline.

Never by including the dev overlay `docker/docker-compose.decision-models.yml` (a
different file from the systemd overlay above): its
`depends_on: design-pattern-tacit: service_healthy` would make the production server
unstartable without a local healthy gateway.

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
