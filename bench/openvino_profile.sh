#!/bin/bash
# Per-node profile (bench/openvino_profile.py) of one IR. Env: MODEL, TAG, PROPS, TOKENS (512), ARGS.
set -uo pipefail
MODEL=${MODEL:-OpenVINO/Qwen3.8-27B-int4-ov}
TAG=${TAG:-profile}
echo "== profile $MODEL tag=$TAG props=${PROPS:-{\}} $(date -u +%T)"; top -bn1 | head -10 | tail -4 | cut -c1-110
/opt/ov/venv/bin/python /opt/azslm/bench/openvino_profile.py "/mnt/data/models/$MODEL" --tag "$TAG" \
  --tokens "${TOKENS:-512}" --props "${PROPS:-{\}}" ${ARGS:-} 2>&1 | cut -c1-200 | tail -n 45
