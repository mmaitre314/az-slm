#!/bin/bash
# Batch throughput: N independent prompts processed together (llama-batched-bench), the setting
# that matters for offline batch processing. Appends JSON lines to
# /mnt/data/results/llama-batched.jsonl. Env: QUANTS, BUILDS, NPL (comma list), PP, TG, THREADS.
set -uo pipefail
MODELS=/mnt/data/models/bartowski/Qwen3.8-27B-GGUF
OUT=/mnt/data/results
mkdir -p "$OUT"
QUANTS=${QUANTS:-"Q4_0 IQ4_XS Q4_K_M Q5_K_M Q6_K Q8_0"}
BUILDS=${BUILDS:-"build"}
NPL=${NPL:-"1,4,8,16"}
PP=${PP:-512}
TG=${TG:-128}
THREADS=${THREADS:-8}
max_npl=$(tr ',' '\n' <<< "$NPL" | sort -n | tail -1)
ctx=$(( max_npl * (PP + TG) + 256 ))
for q in $QUANTS; do
  m=$MODELS/Qwen3.8-27B-$q.gguf
  [ "$q" = bf16 ] && m=$(ls "$MODELS"/Qwen3.8-27B-bf16/*-00001-of-*.gguf)
  for b in $BUILDS; do
    log=$OUT/llama-batched-$q-$b.log
    echo "== $q $b npl=$NPL ctx=$ctx $(date -u +%T)"
    /opt/llama.cpp/$b/bin/llama-batched-bench -m "$m" -c "$ctx" -b 2048 -ub 512 -npp "$PP" -ntg "$TG" -npl "$NPL" \
      -t "$THREADS" -tb "$THREADS" --output-format jsonl 2> "$log" \
      | grep '^{' | jq -c --arg q "$q" --arg b "$b" '. + {quant: $q, build: $b}' | tee -a "$OUT/llama-batched.jsonl" \
      | jq -r '"  npl=\(.pl): prefill \(.speed_pp * 10 | floor / 10) t/s, decode \(.speed_tg * 10 | floor / 10) t/s, total \(.speed * 10 | floor / 10) t/s"'
    [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "  FAILED (see $log)"; tail -n 5 "$log"; }
  done
done
echo "done $(date -u +%T)"
