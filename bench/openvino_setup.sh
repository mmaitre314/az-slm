#!/bin/bash
# E10/E11 setup on bench-ov: OpenVINO GenAI venv, Docker + OVMS image, pre-converted IRs from the
# OpenVINO Hugging Face org. Idempotent. Prints the versions and revisions it ends up with.
# Env: OV_VERSION, GENAI_VERSION (default 2026.4.1 / 2026.4.1.0), OVMS_IMAGE (default
# openvino/model_server:2026.4.0), REPOS (HF repos to download), SKIP_OPTIMUM=1 to skip optimum-intel.
set -euxo pipefail
OV_VERSION=${OV_VERSION:-2026.4.1}
GENAI_VERSION=${GENAI_VERSION:-2026.4.1.0}
OVMS_IMAGE=${OVMS_IMAGE:-openvino/model_server:2026.4.0}
REPOS=${REPOS:-"OpenVINO/Qwen3.8-27B-int4-ov OpenVINO/Qwen3.8-27B-int8-ov"}
VENV=/opt/ov/venv
mkdir -p /opt/ov /mnt/data/models /mnt/data/results

[ -x $VENV/bin/python ] || python3 -m venv $VENV
$VENV/bin/pip install -q -U pip wheel
$VENV/bin/pip install -q "openvino==$OV_VERSION" "openvino-genai==$GENAI_VERSION" \
  "openvino-tokenizers==$GENAI_VERSION" nncf huggingface_hub hf_xet numpy
if [ -z "${SKIP_OPTIMUM:-}" ]; then
  # CPU-only torch wheels (the default PyPI torch drags in ~3 GB of CUDA libraries)
  $VENV/bin/pip install -q "optimum-intel[openvino]" --extra-index-url https://download.pytorch.org/whl/cpu \
    || echo "WARNING: optimum-intel install failed"
  # optimum-intel may move openvino; put the pinned versions back
  $VENV/bin/pip install -q "openvino==$OV_VERSION" "openvino-genai==$GENAI_VERSION" "openvino-tokenizers==$GENAI_VERSION"
fi
$VENV/bin/pip freeze | grep -iE '^(openvino|optimum|nncf|transformers|torch|huggingface|hf_xet|numpy|tokenizers)' || true

# Docker for OVMS. Containers are blocked from IMDS so they can't get the VM identity's token.
if ! command -v docker >/dev/null; then
  apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q docker.io
fi
mkdir -p /etc/docker /mnt/data/docker
[ -f /etc/docker/daemon.json ] || echo '{"data-root": "/mnt/data/docker"}' > /etc/docker/daemon.json
systemctl restart docker
iptables -C DOCKER-USER -d 169.254.169.254 -j DROP 2>/dev/null || iptables -I DOCKER-USER -d 169.254.169.254 -j DROP
docker pull -q "$OVMS_IMAGE"
docker image inspect "$OVMS_IMAGE" --format '{{.Id}} {{.Created}} {{index .RepoDigests 0}}'

export HF_XET_HIGH_PERFORMANCE=1
for r in $REPOS; do
  $VENV/bin/hf download "$r" --local-dir "/mnt/data/models/$r" > /dev/null
  echo "$r: $(du -sh /mnt/data/models/$r | cut -f1) revision $(curl -s https://huggingface.co/api/models/$r | python3 -c 'import sys,json; print(json.load(sys.stdin)["sha"])')"
done
echo "setup done $(date -u +%T)"
