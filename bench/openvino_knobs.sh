#!/bin/bash
# E10 knob sweep: short workloads (prefill 512/1 at N=1,2; decode 32/32 at N=1,8) under different
# OpenVINO CPU plugin settings, to pick the configuration for the full sweep. Rows go to
# /mnt/data/results/openvino.jsonl with tag knob-<name>.
# Env: MODEL, CONFIGS (lines of "name|PROPS json|ENV=V ENV2=V|extra args"; default list below),
# WORKLOADS, BATCHES.
set -uo pipefail
MODEL=${MODEL:-OpenVINO/Qwen3.8-27B-int4-ov}
DEFAULT_CONFIGS='base|{}||
thr8|{"INFERENCE_NUM_THREADS":8,"ENABLE_HYPER_THREADING":false}||
thr16|{"INFERENCE_NUM_THREADS":16}||
dq0|{"DYNAMIC_QUANTIZATION_GROUP_SIZE":0}||
f32dq32|{"INFERENCE_PRECISION_HINT":"f32","DYNAMIC_QUANTIZATION_GROUP_SIZE":32}||
f16|{"INFERENCE_PRECISION_HINT":"f16"}||
mbt1024|{}||--max-batched-tokens 1024
noamx|{}|ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16|'
CONFIGS=${CONFIGS:-$DEFAULT_CONFIGS}
LOGDIR=/mnt/data/results
echo "== knob sweep $MODEL $(date -u +%T)"; top -bn1 | head -10 | tail -4 | cut -c1-110
while IFS='|' read -r name props envs extra; do
  [ -z "$name" ] && continue
  echo "---- $name props=$props env=$envs extra=$extra $(date -u +%T)"
  env $envs /opt/ov/venv/bin/python /opt/azslm/bench/openvino_bench.py "/mnt/data/models/$MODEL" --tag "knob-$name" \
    --workloads "${WORKLOADS:-prefill:512:1,decode:32:32}" --batches "${BATCHES:-1,8}" --reps 1 --warmup quick \
    --props "$props" --note "knob sweep" $extra > "$LOGDIR/openvino-knob-$name.log" 2>&1
  rc=$?
  grep -E '^\{' "$LOGDIR/openvino-knob-$name.log" | python3 -c "
import sys, json
for l in sys.stdin:
    r = json.loads(l)
    print('  {workload} n={n} wall={wall_s}s total={total_tok_s} in={in_tok_s} est_decode={e}'.format(e=r.get('est_decode_tok_s'), **r))"
  [ $rc -ne 0 ] && { echo "  rc=$rc"; grep -iE "error|exception|unsupported|not found" "$LOGDIR/openvino-knob-$name.log" | tail -n 3 | cut -c1-300; }
done <<< "$CONFIGS"
echo "knob sweep done $(date -u +%T)"
