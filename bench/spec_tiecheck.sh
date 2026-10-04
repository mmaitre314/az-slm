#!/bin/bash
# Run bench/spec_tiecheck.py against a plain (non-speculative) llama-server on the AMX build, to see
# whether greedy divergence between speculative and plain decoding is a floating-point near-tie.
# Env: FILE (results jsonl, default spec-single.jsonl), BASE, SPEC (tags), QUANT, BUILD, SET.
set -uo pipefail
cd "$(dirname "$0")"
Q=${QUANT:-Q4_K_M}; B=${BUILD:-build}
FILE=${FILE:-/mnt/data/results/spec-single.jsonl}
BASE=${BASE:-base-$Q-$B-np1}; SPEC=${SPEC:-mtp3-$Q-$B-np1}
mkdir -p /mnt/data/results/spec-logs
/opt/llama.cpp/$B/bin/llama-server -m /mnt/data/models/bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-$Q.gguf -c 2048 -np 1 \
  -t 8 -tb 8 --host 127.0.0.1 --port 8081 --no-webui > /mnt/data/results/spec-logs/tiecheck.log 2>&1 &
spid=$!
for i in $(seq 1 180); do curl -sf http://127.0.0.1:8081/health >/dev/null && break; sleep 2; done
python3 spec_tiecheck.py "$FILE" "$BASE" "$SPEC" --url http://127.0.0.1:8081 --set "${SET:-ten}" \
  --out /mnt/data/results/spec-tiecheck.jsonl
kill $spid; wait $spid 2>/dev/null
echo "tiecheck done $(date -u +%T)"
