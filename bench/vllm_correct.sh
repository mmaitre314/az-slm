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
mkdir -p "$OUT" /mnt/data/vllm-cache
LOG="$OUT/vllm-correct-$TAG.log"
echo "== $MODEL tag=$TAG bind=$BIND $(date -u +%T)"
# With ONEDNN_VERBOSE set, a verbose line is printed per primitive execution (millions of lines), so the
# stream is aggregated on the fly: non-oneDNN lines go to $LOG, oneDNN lines are counted per
# (kind, implementation, src dtype) into $LOG.onednn and the first few raw lines are kept.
docker run --rm --privileged --shm-size 8g -v /mnt/data:/mnt/data -v /opt/azslm/bench:/bench:ro \
  -v /mnt/data/vllm-cache:/root/.cache/vllm \
  -e VLLM_CPU_KVCACHE_SPACE="$KV_GB" -e VLLM_CPU_OMP_THREADS_BIND="$BIND" \
  ${ONEDNN_VERBOSE:+-e ONEDNN_VERBOSE=$ONEDNN_VERBOSE} \
  --entrypoint python3 "$IMAGE" /bench/vllm_correct.py "/mnt/data/models/$MODEL" --tag "$TAG" ${ARGS:-} 2>&1 \
  | awk -v agg="$LOG.onednn" -v raw="$LOG.onednn-head" '
      /^onednn_verbose/ { split($0, f, ","); split(f[9], d, ":"); k = f[3] "," f[4] "," f[5] "," f[6] "," f[7] "," d[1] ":" d[2]
                          n[k]++; if (++r <= 40) print > raw; next }
      { print; fflush() }
      END { for (k in n) print n[k], k > agg }' > "$LOG"
rc=${PIPESTATUS[0]}
echo "rc=$rc"
grep -E '^\{|peak_rss' "$LOG" | cut -c1-600
if [ $rc -ne 0 ]; then grep -iE "error|exception|not supported|unsupported|Traceback" "$LOG" | tail -n 15 | cut -c1-400; fi
if [ -n "${ONEDNN_VERBOSE:-}" ]; then
  echo "-- oneDNN primitive counts (count kind,impl,src dtype), top 12:"
  sort -rn "$LOG.onednn" 2>/dev/null | head -n 12 | cut -c1-200
  echo "-- distinct ISA tokens in implementation names:"
  grep -ioE "amx[a-z_0-9]*|avx512[a-z_0-9]*|avx2[a-z_0-9]*" "$LOG.onednn" 2>/dev/null | sort | uniq -c | sort -rn | head -n 8
  grep -iE "info,cpu|isa" "$LOG.onednn-head" 2>/dev/null | head -n 3 | cut -c1-300
fi
