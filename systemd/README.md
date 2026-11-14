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
| Tacit gateway (J1/J2/J4/J5) | GPU host via `tacit-tunnel` service (`--profile tacit`) | needs a card; gateway has no auth and container loopback is its own, so ssh runs in-stack |

## Decision models

The Ch2 style seam ships **on** (linear head, floor 0.6), served by the in-stack
`design-pattern-style-head` service. Step 0 builds it locally
(`make docker-build-style-head`); to run a published build instead, set
`STYLE_HEAD_IMAGE=olkowa/design-pattern-style-head:<tag>` in
`/etc/design-pattern-mcp/.env`. Set `DECISIONS_STYLE_ENABLED=false` there to stand
the seam down to the LLM style.

The Tacit seams (J1 rerank, J2 category, J4 weights, J5 gate) are **off** and their
gateway lives on a GPU host — reached through the opt-in `tacit-tunnel` service
(below), never directly: the gateway has no auth, so it binds loopback on its own
host, and the production container's `127.0.0.1` is itself, so a host-level tunnel
would be invisible to it. The sidecar runs the ssh client on `pattern-tei-shared`
and forwards the gateway as `tacit-tunnel:8100`.

1. **GPU host** — run the gateway on loopback:

   ```bash
   make docker-build-tacit            # tags :1.0.1 and :latest
   docker run -d --name design-pattern-tacit --restart unless-stopped \
     --device nvidia.com/gpu=0 -p 127.0.0.1:8100:8100 \
     design-pattern-tacit:latest
   ```

2. **This host** — deployment key + pinned host key (key auth only, no passwords,
   no trust-on-first-use — the entrypoint refuses to run without both files):

   ```bash
   sudo install -o root -g docker -m 750 -d /etc/design-pattern-mcp/ssh
   sudo install -o root -g docker -m 640 ~/.ssh/tacit_tunnel_ed25519 \
     /etc/design-pattern-mcp/ssh/id_ed25519
   ssh-keyscan -t ed25519 gpu-host | sudo tee /etc/design-pattern-mcp/ssh/known_hosts
   sudo chmod 640 /etc/design-pattern-mcp/ssh/known_hosts
   ```

   Verify the `known_hosts` line against the GPU host's key fingerprint before
   proceeding — `ssh-keyscan` output is unverified bytes until you do.

3. **`/etc/design-pattern-mcp/.env`** — tunnel target plus the seam switches
   (all default off):

   ```
   TACIT_TUNNEL_SSH_HOST=gpu-host
   TACIT_TUNNEL_SSH_USER=graemer
   DECISIONS_WEIGHTS_ENABLED=true
   DECISIONS_CATEGORY_ENABLED=true
   DECISIONS_REASONING_GATE_ENABLED=true
   RERANKER_PROVIDER=tacit
   ```

   The four `*_BASE_URL`s already default to `http://tacit-tunnel:8100` in the
   compose file — set them only to override. Then bring the tunnel up alongside
   the stack:

   ```bash
   sudo systemctl reload design-pattern-mcp   # compose up without the tunnel
   docker compose --profile tacit up -d       # from /etc/design-pattern-mcp
   ```

   (Or add `--profile tacit` to the unit's `ExecStart`/`ExecReload` lines if the
   tunnel should always run with the stack.) Startup log shows
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
