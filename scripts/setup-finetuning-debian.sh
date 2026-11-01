set -euxo pipefail
. /etc/os-release   # $ID/$VERSION_CODENAME: ubuntu/noble or debian/bookworm
sudo apt update && sudo apt install -y curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL "https://download.docker.com/linux/$ID/gpg" | sudo tee /etc/apt/keyrings/docker.asc >/dev/null
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$ID $VERSION_CODENAME stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#" \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit
sudo apt install -y git make curl ca-certificates   # Step 5 git-clone + Steps 7-8 make; curl/ca-certificates for key fetches
sudo usermod -aG docker "$USER" || true
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# Re-login (or: newgrp docker) BEFORE the lines below — the group change is not active in this shell:
newgrp docker <<'EOF'
command -v nvidia-ctk && docker buildx version   # nvidia-ctk ships INSIDE nvidia-container-toolkit
docker compose version   # from docker-compose-plugin; unknown without it
docker info | grep -i -E "cdi|runtime"   # expect cdi: nvidia.com/gpu=all AND runtimes include nvidia
docker version --format "{{.Server.Version}}"   # expect >= 25
