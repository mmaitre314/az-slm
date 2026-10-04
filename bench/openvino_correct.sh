#!/bin/bash
# Correctness run of one OpenVINO IR (3 fixed prompts, greedy, 64 tokens; single vs batched).
# Env: MODEL (dir name under /mnt/data/models), TAG, PIPE (auto|llm|vlm|cb), ARGS (extra args for
# openvino_correct.py), ONEDNN_VERBOSE (1 = AMX check: counts ISA names in the log), plus any
# OpenVINO/oneDNN env (ONEDNN_MAX_CPU_ISA, OMP_NUM_THREADS...) passed through.
# Log: /mnt/data/results/openvino-correct-$TAG.log
set -uo pipefail
MODEL=${MODEL:-OpenVINO/Qwen3.8-27B-int4-ov}
TAG=${TAG:-$(echo "$MODEL" | tr '/' '_')}
OUT=/mnt/data/results
mkdir -p "$OUT"
LOG="$OUT/openvino-correct-$TAG.log"
echo "== $MODEL tag=$TAG pipe=${PIPE:-auto} $(date -u +%T)"
top -bn1 | head -12 | tail -6
/opt/ov/venv/bin/python /opt/azslm/bench/openvino_correct.py "/mnt/data/models/$MODEL" --tag "$TAG" \
  --pipe "${PIPE:-auto}" ${ARGS:-} > "$LOG" 2>&1
rc=$?
echo "rc=$rc"
grep -E '^(loaded|warmup|templated|\{|peak_rss)' "$LOG" | cut -c1-500
if [ $rc -ne 0 ]; then grep -iE "error|exception|not supported|unsupported|Traceback|failed" "$LOG" | tail -n 15 | cut -c1-500; fi
if [ -n "${ONEDNN_VERBOSE:-}" ]; then
  echo "-- oneDNN ISA / kernel names seen:"
  grep -ioE "avx512_core_amx[a-z_0-9]*|avx512_core_bf16|avx512_core|avx2|jit:[a-z0-9_:]*amx[a-z_0-9]*|brg[a-z_0-9:]*amx[a-z_0-9]*" "$LOG" | sort | uniq -c | sort -rn | head -n 12
  echo "exec lines: $(grep -c '^onednn_verbose,.*exec' "$LOG")"
fi
