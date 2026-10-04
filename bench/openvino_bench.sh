#!/bin/bash
# OpenVINO GenAI continuous-batching throughput for one IR (bench/openvino_bench.py).
# Env: MODEL (dir under /mnt/data/models; default OpenVINO/Qwen3.8-27B-int4-ov), TAG, WORKLOADS,
# BATCHES, REPS, PROPS (JSON plugin properties), ARGS (extra args for openvino_bench.py), plus any
# OpenVINO/oneDNN env (ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 turns AMX off).
# Appends to /mnt/data/results/openvino.jsonl; log /mnt/data/results/openvino-bench-$TAG.log
set -uo pipefail
MODEL=${MODEL:-OpenVINO/Qwen3.8-27B-int4-ov}
TAG=${TAG:-$(echo "$MODEL" | tr '/' '_')}
OUT=/mnt/data/results
mkdir -p "$OUT"
LOG="$OUT/openvino-bench-$TAG.log"
echo "== $MODEL tag=$TAG props=${PROPS:-{\}} $(date -u +%T)"
echo "-- top before the run (nothing else should use CPU):"
top -bn1 | head -12 | tail -7 | cut -c1-110
/opt/ov/venv/bin/python /opt/azslm/bench/openvino_bench.py "/mnt/data/models/$MODEL" --tag "$TAG" \
  --workloads "${WORKLOADS:-prefill:512:1,decode:32:256,mixed:512:128}" --batches "${BATCHES:-1,4,16,32}" \
  --reps "${REPS:-2}" --props "${PROPS:-{\}}" ${ARGS:-} > "$LOG" 2>&1
rc=$?
echo "rc=$rc"
tail -n 40 "$LOG" | cut -c1-420
if [ $rc -ne 0 ]; then grep -iE "error|exception|Traceback|failed" "$LOG" | tail -n 10 | cut -c1-400; fi
echo "finished $(date -u +%T)"
