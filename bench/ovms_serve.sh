#!/bin/bash
# Start/stop an OpenVINO Model Server container (OpenAI-compatible /v3/chat/completions) on the VM.
#   bash ovms_serve.sh start   (env below)      bash ovms_serve.sh stop
# Env: MODEL (dir under /mnt/data/models, default OpenVINO/Qwen3.8-27B-int4-ov), NAME (served model
# name, default qwen), IMAGE, PORT (8000), PREFIX_CACHING (true|false), MAX_SEQS (32), CACHE_GB (24),
# MBT (max_num_batched_tokens, default 256), PLUGIN_CONFIG (JSON), EXTRA (more ovms args), TAG (log
# name). The container cannot reach IMDS (iptables rule from openvino_setup.sh). Port is bound to
# 127.0.0.1 only. graph.pbtxt is regenerated on every start so the options take effect.
set -uo pipefail
IMAGE=${IMAGE:-openvino/model_server:2026.4.0}
MODEL=${MODEL:-OpenVINO/Qwen3.8-27B-int4-ov}
NAME=${NAME:-qwen}
PORT=${PORT:-8000}
TAG=${TAG:-ovms}
case "${1:-start}" in
stop)
  docker logs ovms > "/mnt/data/results/ovms-$TAG.log" 2>&1 || true
  docker rm -f ovms >/dev/null 2>&1 || true
  echo "stopped"; exit 0;;
start)
  docker rm -f ovms >/dev/null 2>&1 || true
  rm -f "/mnt/data/models/$MODEL/graph.pbtxt"
  iptables -C DOCKER-USER -d 169.254.169.254 -j DROP 2>/dev/null || iptables -I DOCKER-USER -d 169.254.169.254 -j DROP
  PC=()
  [ -n "${PLUGIN_CONFIG:-}" ] && PC=(--plugin_config "$PLUGIN_CONFIG")
  docker run -d --name ovms --rm -u 0:0 -p 127.0.0.1:$PORT:$PORT -v /mnt/data/models:/models:rw "$IMAGE" \
    --rest_port $PORT --model_repository_path /models --source_model "$MODEL" --model_name "$NAME" \
    --task text_generation --target_device CPU --max_num_seqs "${MAX_SEQS:-32}" --cache_size "${CACHE_GB:-24}" \
    --max_num_batched_tokens "${MBT:-256}" --enable_prefix_caching "${PREFIX_CACHING:-false}" \
    --log_level INFO "${PC[@]}" ${EXTRA:-} > /tmp/ovms-cid.txt
  t0=$(date +%s)
  until curl -sf "http://127.0.0.1:$PORT/v3/models" >/dev/null 2>&1 && curl -sf "http://127.0.0.1:$PORT/v2/health/ready" >/dev/null 2>&1; do
    if ! docker ps -q -f name=ovms | grep -q .; then echo "container exited"; docker logs ovms 2>&1 | tail -n 30 | cut -c1-300; exit 1; fi
    if [ $(( $(date +%s) - t0 )) -gt 1500 ]; then echo "timeout waiting for readiness"; docker logs ovms 2>&1 | tail -n 30 | cut -c1-300; exit 1; fi
    sleep 5
  done
  echo "ready after $(( $(date +%s) - t0 )) s"
  curl -s "http://127.0.0.1:$PORT/v3/models" | head -c 400; echo;;
esac
