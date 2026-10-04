#!/bin/bash
# Install Docker, pull the official vLLM CPU image, and download the safetensors checkpoints vLLM
# needs. Containers are blocked from IMDS so code inside them cannot get the VM identity's token.
# Env: IMAGE (default vllm/vllm-openai-cpu:latest-x86_64), REPOS (HF repos to download).
set -euxo pipefail
IMAGE=${IMAGE:-vllm/vllm-openai-cpu:latest-x86_64}
REPOS=${REPOS:-"Qwen/Qwen3.8-27B Avesed/Qwen3.8-27B-INT8-W8A8 Avesed/Qwen3.8-27B-INT4-W4A16"}
if ! command -v docker >/dev/null; then
  apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q docker.io
fi
mkdir -p /etc/docker /mnt/data/docker
[ -f /etc/docker/daemon.json ] || echo '{"data-root": "/mnt/data/docker"}' > /etc/docker/daemon.json
systemctl restart docker
iptables -C DOCKER-USER -d 169.254.169.254 -j DROP 2>/dev/null || iptables -I DOCKER-USER -d 169.254.169.254 -j DROP
mkdir -p /mnt/data/results
docker pull -q "$IMAGE"
docker image inspect "$IMAGE" --format '{{.Id}} {{.Created}} {{.RepoDigests}}' | tee /mnt/data/results/vllm-image.txt
docker run --rm --entrypoint python3 "$IMAGE" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)' | tee -a /mnt/data/results/vllm-image.txt
export HF_XET_HIGH_PERFORMANCE=1
for r in $REPOS; do
  echo "$r $(curl -sS https://huggingface.co/api/models/$r | jq -r .sha)" >> /mnt/data/results/model-revisions.txt
  /opt/azslm/venv/bin/hf download "$r" --local-dir "/mnt/data/models/$r" --exclude '*.gguf' > /dev/null
  echo "$r: $(du -sh /mnt/data/models/$r | cut -f1)"
done
