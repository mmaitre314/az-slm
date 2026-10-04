#!/bin/bash
# Single-sequence prefill (pp) and decode (tg) throughput with llama-bench, per quant and per build
# (build = native/AMX, build-noamx = same commit with AMX compiled out). Appends JSON lines to
# /mnt/data/results/llama-bench.jsonl. Env: QUANTS, BUILDS, THREADS (list), REPS, PP, TG.
set -uo pipefail
MODELS=/mnt/data/models/bartowski/Qwen3.8-27B-GGUF
OUT=/mnt/data/results
mkdir -p "$OUT"
QUANTS=${QUANTS:-"Q4_0 IQ4_XS Q4_K_M Q5_K_M Q6_K Q8_0 bf16"}
BUILDS=${BUILDS:-"build build-noamx"}
THREADS=${THREADS:-8}
REPS=${REPS:-3}
PP=${PP:-512}
TG=${TG:-128}
model_path() {
  if [ "$1" = bf16 ]; then ls "$MODELS"/Qwen3.8-27B-bf16/*-00001-of-*.gguf; else echo "$MODELS/Qwen3.8-27B-$1.gguf"; fi
}
for q in $QUANTS; do
  m=$(model_path "$q")
  for b in $BUILDS; do
    for t in $THREADS; do
      log=$OUT/llama-bench-$q-$b-t$t.log
      echo "== $q $b t=$t $(date -u +%T)"
      /opt/llama.cpp/$b/bin/llama-bench -m "$m" -p "$PP" -n "$TG" -t "$t" -r "$REPS" -o jsonl -v 2> "$log" \
        | jq -c --arg q "$q" --arg b "$b" '. + {quant: $q, build: $b}' | tee -a "$OUT/llama-bench.jsonl" \
        | jq -r '"  pp\(.n_prompt) tg\(.n_gen): \(.avg_ts * 10 | floor / 10) t/s (sd \(.stddev_ts * 10 | floor / 10))"'
      grep -ioE '(AMX|CPU_REPACK|CPU)[A-Za-z_ ]*(model )?buffer size *= *[0-9.]+ MiB' "$log" | sort -u | sed 's/^/  /'
    done
  done
done
echo "done $(date -u +%T)"
