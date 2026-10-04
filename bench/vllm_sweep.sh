#!/bin/bash
# In-process vLLM CPU sweep (bench/vllm_sweep.py) of one model: one model load, then every
# (workload, batch size) combination, appending rows to /mnt/data/results/vllm-sweep.jsonl.
# Env: MODEL (path under /mnt/data/models), RUN (label, e.g. t8/t16), BATCHES, WORKLOADS, IMAGE,
# KV_GB, BIND (OpenMP CPU list; default one thread per physical core), ARGS (extra vllm_sweep.py
# args), DOCKER_ENV (extra docker args, e.g. "-e ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16" for an AMX-off
# control). A persistent compile cache (/mnt/data/vllm-cache) avoids recompiling on every launch.
set -uo pipefail
IMAGE=${IMAGE:-vllm/vllm-openai-cpu:latest-x86_64}
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
KV_GB=${KV_GB:-16}
BIND=${BIND:-$(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -u | cut -d, -f1 | paste -sd,)}
RUN=${RUN:-t$(echo "$BIND" | tr ',' '\n' | wc -l)}
BATCHES=${BATCHES:-1,4,16,32}
WORKLOADS=${WORKLOADS:-prefill:512:1,decode:32:256,mixed:512:128}
OUT=/mnt/data/results
mkdir -p "$OUT" /mnt/data/vllm-cache
tag=$(echo "$MODEL" | tr '/' '_')-$RUN${TAG_SUFFIX:-}
LOG="$OUT/vllm-sweep-$tag.log"
echo "== $MODEL run=$RUN bind=$BIND batches=$BATCHES workloads=$WORKLOADS $(date -u +%T)"
{ echo "== $tag $(date -u +%T)"; top -bn1 -o %CPU | sed -n 1,12p; } >> "$OUT/vllm-top.log"
docker run --rm --privileged --shm-size 8g -v /mnt/data:/mnt/data -v /opt/azslm/bench:/bench:ro \
  -v /mnt/data/vllm-cache:/root/.cache/vllm \
  -e VLLM_CPU_KVCACHE_SPACE="$KV_GB" -e VLLM_CPU_OMP_THREADS_BIND="$BIND" ${DOCKER_ENV:-} \
  --entrypoint python3 "$IMAGE" /bench/vllm_sweep.py "/mnt/data/models/$MODEL" --run "$RUN${TAG_SUFFIX:-}" \
  --batches "$BATCHES" --workloads "$WORKLOADS" ${ARGS:-} > "$LOG" 2>&1
rc=$?
echo "rc=$rc"
grep -E '^\{|^loaded|container_peak' "$LOG" | jq -R -r 'fromjson? // . | if type=="object" then "\(.workload) n=\(.num_prompts) \(.elapsed_time)s \(.tokens_per_second) tok/s total, ttft_max \(.ttft_max_s // "-")s, decode \(.decode_tok_per_s_after_prefill // "-") tok/s" else . end'
if [ $rc -ne 0 ]; then grep -iE "error|exception|not supported|unsupported|Traceback" "$LOG" | tail -n 15 | cut -c1-400; fi
echo "done $(date -u +%T)"
