#!/bin/bash
# Single-sequence speculative decoding on the AMX build: for each configuration start a server with
# one slot and run the prompt sets (std3 = correctness prompts, ten = 10 varied prompts).
# Env: CONFIGS (space-separated names from spec_configs.sh), SETS (default "std3 ten"),
#      QUANT (default Q4_K_M), BUILD (default build), OUT, plus anything spec_run.sh takes.
set -uo pipefail
cd "$(dirname "$0")"
source ./spec_configs.sh
CONFIGS=${CONFIGS:-"base mtp3"}
export SETS=${SETS:-"std3 ten"}
export BUILD=${BUILD:-build}
export MODEL=/mnt/data/models/bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-${QUANT:-Q4_K_M}.gguf
export NP=1 INFLIGHT=1
for c in $CONFIGS; do
  export TAG=$c-${QUANT:-Q4_K_M}-$BUILD-np1
  SPEC_ARGS=$(spec_args "$c") || continue
  export SPEC_ARGS
  timeout 3000 bash ./spec_run.sh || echo "CONFIG $c FAILED rc=$?"
done
echo "all done $(date -u +%T)"
