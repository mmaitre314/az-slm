#!/bin/bash
# Threads / SMT / pinning for llama-bench pp512 + tg128 on one quant and build. Variants:
#   t8       8 threads, unpinned (default)
#   t8-pin   8 threads pinned to one hyperthread per physical core (--cpu-strict)
#   t16      16 threads (both hyperthreads of each core)
# Appends to /mnt/data/results/llama-threads.jsonl. Env: QUANT, BUILD, REPS.
set -uo pipefail
QUANT=${QUANT:-Q4_K_M}
BUILD=${BUILD:-build}
REPS=${REPS:-3}
OUT=/mnt/data/results
m=/mnt/data/models/bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-$QUANT.gguf
# Mask with the first hyperthread of each physical core (from lscpu).
mask=$(python3 -c "
import subprocess
cores = {}
for line in subprocess.run(['lscpu', '-p=CPU,CORE'], capture_output=True, text=True).stdout.splitlines():
    if line.startswith('#'): continue
    cpu, core = map(int, line.split(','))
    cores.setdefault(core, cpu)
print(hex(sum(1 << c for c in cores.values())))")
for variant in "t8:-t 8" "t8-pin:-t 8 -C $mask --cpu-strict 1" "t16:-t 16"; do
  name=${variant%%:*}
  opts=${variant#*:}
  echo "== $QUANT $BUILD $name ($opts) $(date -u +%T)"
  /opt/llama.cpp/$BUILD/bin/llama-bench -m "$m" -p 512 -n 128 -r "$REPS" -o jsonl $opts 2>/dev/null \
    | jq -c --arg q "$QUANT" --arg b "$BUILD" --arg v "$name" '. + {quant: $q, build: $b, variant: $v}' \
    | tee -a "$OUT/llama-threads.jsonl" | jq -r '"  pp\(.n_prompt) tg\(.n_gen): \(.avg_ts * 100 | floor / 100) t/s"'
done
