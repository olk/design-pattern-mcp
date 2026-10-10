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
- Both images pre-built locally (Step 0).
- **Shared TEI infra stack** — see [`pattern-tei-infra`](https://github.com/olk/pattern-tei-infra).

## Step 0 — Build or pull images

```bash
# From the repo root:
make docker-build-all
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
#   DESIGN_PATTERN_GENERATOR_API_KEY=sk-...
#   COMPOSE_PROJECT_NAME=dpml-systemd
#   MCP_HOST_PORT=8062

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
curl -fsS http://localhost:8062/health
```

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
