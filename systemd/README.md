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
- All three images available locally: MCP server, TEI pair, style head (Step 0).
- **Shared TEI infra stack** — see [`pattern-tei-infra`](https://github.com/olk/pattern-tei-infra).

## Step 0 — Build or pull images

```bash
# From the repo root (server + TEI pair, dense cache baked in):
make docker-build-all
# Plus the CPU style head (~20 s, no GPU):
make docker-build-style-head
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
| Tacit gateway (J1/J2/J4/J5) | a GPU host, reached over SSH (below) | needs a card; the gateway has no auth, so it never faces the network |

## Decision models

The Ch2 style seam ships **on** (linear head, floor 0.6), served by the in-stack
`design-pattern-style-head` service. Step 0 builds it locally
(`make docker-build-style-head`); to run a published build instead, set
`STYLE_HEAD_IMAGE=olkowa/design-pattern-style-head:<tag>` in
`/etc/design-pattern-mcp/.env`. Set `DECISIONS_STYLE_ENABLED=false` there to stand
the seam down to the LLM style.

The Tacit seams (J1 rerank, J2 category, J4 weights, J5 gate) are **off** and have no
service here. To enable them:

1. **GPU host** — run the gateway on loopback (it has no auth; compose confines it
   to `127.0.0.1` for the same reason):

   ```bash
   make docker-build-tacit            # tags :1.0.1 and :latest
   docker run -d --name design-pattern-tacit --restart unless-stopped \
     --device nvidia.com/gpu=0 -p 127.0.0.1:8100:8100 \
     design-pattern-tacit:latest
   ```

2. **This host** — persistent SSH tunnel as its own unit
   (`/etc/systemd/system/tacit-tunnel.service`; key-based auth, no password prompts):

   ```ini
   [Unit]
   Description=SSH tunnel to Tacit gateway
   After=network-online.target
   Wants=network-online.target

   [Service]
   ExecStart=/usr/bin/ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
     -L 8100:127.0.0.1:8100 gpu-host
   Restart=always
   RestartSec=10
   User=graemer

   [Install]
   WantedBy=multi-user.target
   ```

   ```bash
   sudo systemctl enable --now tacit-tunnel
   ```

3. **`/etc/design-pattern-mcp/.env`** — all four URLs at the tunnel endpoint, plus
   the switches (all default off):

   ```
   DECISIONS_WEIGHTS_BASE_URL=http://127.0.0.1:8100
   DECISIONS_CATEGORY_BASE_URL=http://127.0.0.1:8100
   DECISIONS_REASONING_GATE_BASE_URL=http://127.0.0.1:8100
   RERANKER_PROVIDER=tacit
   RERANKER_TACIT_BASE_URL=http://127.0.0.1:8100
   DECISIONS_WEIGHTS_ENABLED=true
   DECISIONS_CATEGORY_ENABLED=true
   DECISIONS_REASONING_GATE_ENABLED=true
   ```

   Then `sudo systemctl reload design-pattern-mcp`. Startup log shows
   `ok (tacit, model=tacit-4b)` per seam; any transport failure degrades silently to
   the non-model path, so check the logs for `degraded` warnings if a seam reports
   `ok` but behaves like the baseline.

Never by including the dev `docker-compose.decision-models.yml`: its
`depends_on: design-pattern-tacit: service_healthy` would make the production server
unstartable without a local healthy gateway. (Alternative to the tunnel: publish the
gateway on a firewalled private address — an unauthenticated gateway on the wire by
explicit operator choice, trusted nets only.)

## Updating

```bash
make docker-build-all
sudo systemctl reload design-pattern-mcp
```

## Uninstall

```bash
sudo systemctl disable --now design-pattern-mcp.service
sudo rm /etc/systemd/system/design-pattern-mcp.service
sudo systemctl daemon-reload
sudo rm -rf /etc/design-pattern-mcp
```
