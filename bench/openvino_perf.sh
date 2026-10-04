#!/bin/bash
# Where does the continuous-batching prefill time go? perf (cpu-clock sampling, works in a VM) on
# openvino_bench.py running a 512-token prefill batch. Env: MODEL, N (prompts, default 2), EXTRA (args for openvino_bench.py).
set -uo pipefail
MODEL=${MODEL:-OpenVINO/Qwen3.8-27B-int4-ov}
D=/mnt/data/results/perf-ov.data
echo "== perf $MODEL $(date -u +%T)"
perf record -e cpu-clock -F 199 -g -o $D -- /opt/ov/venv/bin/python /opt/azslm/bench/openvino_bench.py \
  "/mnt/data/models/$MODEL" --tag perf --workloads prefill:512:1 --batches "${N:-2}" --reps 1 --warmup quick \
  ${EXTRA:-} 2>&1 | grep -E '^\{|warmup' | cut -c1-300
echo "== top symbols (self %), all threads"
perf report -i $D --no-children --sort dso,symbol --stdio 2>/dev/null | grep -E '^ +[0-9.]+%' | head -40 | cut -c1-170
echo "== by dso"
perf report -i $D --no-children --sort dso --stdio 2>/dev/null | grep -E '^ +[0-9.]+%' | head -12 | cut -c1-120
