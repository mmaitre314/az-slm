#!/bin/bash
# Aggregate throughput with several sequences decoded together, with and without speculation, on the
# AMX-free control build (the AMX build corrupts multi-sequence output, see E04). For each parallelism
# NP and configuration: a llama-server with NP slots and a client keeping NP requests in flight; each
# request is max_tokens=MAXTOK (default 128), COUNT requests in total (continuous batching refills slots).
# Env: CONFIGS (names from spec_configs.sh), NPS (default "4 16"), COUNT4/COUNT16 (default 12/32),
#      MAXTOK (default 128), QUANT (default Q4_K_M), BUILD (default build-native-noamx), OUT.
set -uo pipefail
cd "$(dirname "$0")"
source ./spec_configs.sh
CONFIGS=${CONFIGS:-"base mtp3"}
NPS=${NPS:-"4 16"}
export SETS=sixteen
export BUILD=${BUILD:-build-native-noamx}
export MAXTOK=${MAXTOK:-128}
export MODEL=/mnt/data/models/bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-${QUANT:-Q4_K_M}.gguf
for np in $NPS; do
  for c in $CONFIGS; do
    export NP=$np INFLIGHT=$np TAG=$c-${QUANT:-Q4_K_M}-$BUILD-np$np
    case $np in 4) export COUNT=${COUNT4:-12} ;; 16) export COUNT=${COUNT16:-32} ;; *) export COUNT=$((np * 2)) ;; esac
    SPEC_ARGS=$(spec_args "$c") || continue
    export SPEC_ARGS
    timeout 3000 bash ./spec_run.sh || echo "CONFIG $c np=$np FAILED rc=$?"
  done
done
echo "all done $(date -u +%T)"
