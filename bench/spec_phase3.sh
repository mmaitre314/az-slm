#!/bin/bash
# E12 follow-up: is the greedy divergence of speculative decoding specific to the AMX kernels?
# Same single-sequence runs as spec_single.sh but on the AMX-free control build. Waits for phase 2.
set -uo pipefail
cd "$(dirname "$0")"
while systemctl is-active --quiet azslm-job-phase2; do sleep 10; done
CONFIGS=${CONFIGS:-"base mtp3"} SETS=ten BUILD=build-native-noamx OUT=/mnt/data/results/spec-noamx-single.jsonl bash spec_single.sh
