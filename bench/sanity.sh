#!/bin/bash
# Output sanity check before trusting throughput numbers: greedy completion of a fixed prompt, one
# sequence and 4 parallel sequences, on the AMX and no-AMX builds (llama.cpp's AMX path has had
# wrong-output bugs with parallel sequences). Env: QUANT (default Q4_K_M), PROMPT.
set -uo pipefail
MODELS=/mnt/data/models/bartowski/Qwen3.8-27B-GGUF
m=$MODELS/Qwen3.8-27B-${QUANT:-Q4_K_M}.gguf
PROMPT=${PROMPT:-"The three largest cities in France, in order of population, are"}
for b in build build-noamx; do
  echo "== $b, 1 sequence"
  timeout 600 /opt/llama.cpp/$b/bin/llama-completion -m "$m" -p "$PROMPT" -n 40 --temp 0 -t 8 < /dev/null 2>/dev/null | tail -c 400
  echo
  # llama-batched shares one prompt across sequences, which recurrent/hybrid models (Qwen3.5+ Gated
  # DeltaNet) reject; llama-parallel decodes independent prompts together, like batch processing does.
  echo "== $b, 4 independent sequences decoded together"
  timeout 900 /opt/llama.cpp/$b/bin/llama-parallel -m "$m" -np 4 -ns 4 -n 48 --temp 0 -t 8 -c 4096 2>&1 \
    | grep -E '^(Input|Response):' | cut -c1-200
done
