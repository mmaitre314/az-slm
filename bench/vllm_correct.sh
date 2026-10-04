#!/bin/bash
# Correctness run (3 fixed prompts, greedy, 64 tokens) of one model in the vLLM CPU container.
# Env: MODEL (path under /mnt/data/models), TAG, IMAGE, KV_GB, BIND (OpenMP CPU list; default one
# thread per physical core), ONEDNN_VERBOSE (set to 1 for the AMX check), ARGS (extra args for
# vllm_correct.py, e.g. --extra '{"quantization":"..."}'). Log: /mnt/data/results/vllm-correct-$TAG.log
set -uo pipefail
IMAGE=${IMAGE:-vllm/vllm-openai-cpu:latest-x86_64}
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
TAG=${TAG:-$(echo "$MODEL" | tr '/' '_')}
KV_GB=${KV_GB:-16}
BIND=${BIND:-$(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -u | cut -d, -f1 | paste -sd,)}
OUT=/mnt/data/results
mkdir -p "$OUT"
LOG="$OUT/vllm-correct-$TAG.log"
echo "== $MODEL tag=$TAG bind=$BIND $(date -u +%T)"
docker run --rm --privileged --shm-size 8g -v /mnt/data:/mnt/data -v /opt/azslm/bench:/bench:ro \
  -e VLLM_CPU_KVCACHE_SPACE="$KV_GB" -e VLLM_CPU_OMP_THREADS_BIND="$BIND" \
  ${ONEDNN_VERBOSE:+-e ONEDNN_VERBOSE=$ONEDNN_VERBOSE} \
  --entrypoint python3 "$IMAGE" /bench/vllm_correct.py "/mnt/data/models/$MODEL" --tag "$TAG" ${ARGS:-} > "$LOG" 2>&1
rc=$?
echo "rc=$rc"
grep -E '^\{|peak_rss' "$LOG" | cut -c1-600
if [ $rc -ne 0 ]; then grep -iE "error|exception|not supported|unsupported|Traceback" "$LOG" | tail -n 15 | cut -c1-400; fi
if [ -n "${ONEDNN_VERBOSE:-}" ]; then
  echo "-- oneDNN ISA lines:"; grep -iE "onednn_verbose.*(info|cpu)" "$LOG" | head -n 5 | cut -c1-300
  grep -ioE "avx512_core_amx[a-z_0-9]*|amx[a-z_0-9]*" "$LOG" | sort | uniq -c | sort -rn | head -n 8
  grep -c "^onednn_verbose,.*exec" "$LOG"
fi
