#!/bin/bash
# Cost of one forward pass over n tokens of a single sequence (what speculative verification pays),
# AMX build vs AMX-free control. llama-bench "-p n -n 0" = a batch of n prompt tokens; the time per
# pass is n / (tokens/s). Appends to /mnt/data/results/spec-stepcost.jsonl. Waits for phase 3.
# Env: QUANT (default Q4_K_M), NS (default list).
set -uo pipefail
while systemctl is-active --quiet azslm-job-phase3; do sleep 10; done
M=/mnt/data/models/bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-${QUANT:-Q4_K_M}.gguf
NS=${NS:-1,2,3,4,5,6,8,12,16,32}
for b in build build-native-noamx; do
  echo "== $b $(date -u +%T)"
  /opt/llama.cpp/$b/bin/llama-bench -m "$M" -p "$NS" -n 0 -r 3 -t 8 -o jsonl 2>/dev/null \
    | jq -c --arg b "$b" '. + {build: $b, secs_per_pass: (.n_prompt / .avg_ts)}' | tee -a /mnt/data/results/spec-stepcost.jsonl \
    | jq -r '"  n=\(.n_prompt): \(.avg_ts * 100 | floor / 100) tok/s, \(.secs_per_pass * 1000 | floor) ms per pass"'
done
echo "done $(date -u +%T)"
