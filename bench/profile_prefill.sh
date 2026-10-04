#!/bin/bash
# Where does prefill time go? CPU profile of llama-bench pp512 by function (perf, cpu-clock sampling,
# works inside VMs without hardware PMU access). Env: QUANT (Q4_0), BUILD (build), THREADS (8).
set -uo pipefail
Q=${QUANT:-Q4_0}
B=${BUILD:-build}
T=${THREADS:-8}
if ! command -v perf >/dev/null || ! perf --version >/dev/null 2>&1; then
  apt-get install -y -q "linux-tools-$(uname -r)" linux-tools-common >/dev/null 2>&1 \
    || apt-get install -y -q linux-tools-azure linux-tools-common >/dev/null 2>&1
fi
m=/mnt/data/models/bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-$Q.gguf
data=/mnt/data/results/perf-$Q-$B.data
perf record -e cpu-clock -F 499 -g -o "$data" -- \
  /opt/llama.cpp/$B/bin/llama-bench -m "$m" -p 512 -n 0 -r 1 -t "$T" -o jsonl 2>/dev/null | jq -c '{n_prompt, avg_ts}'
echo "== top functions ($Q, $B): % of samples"
perf report -i "$data" --no-children --sort dso,symbol --stdio 2>/dev/null | grep -E '^ +[0-9.]+%' | head -25 | cut -c1-160
