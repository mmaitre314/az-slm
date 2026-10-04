#!/bin/bash
# E09 queue: correctness (compiled mode, as benchmarked) then throughput for each model, in order.
# A model is benchmarked only if its correctness run exits 0 and produced the 3 answers.
# Env: QUEUE (space separated "tag:repo" pairs), VERBOSE (ONEDNN_VERBOSE value for the correctness run; default 1, set empty to disable: the W4A16 path issues millions of tiny ukernel calls and the verbose output throttles the run), BIND, RUN, plus anything vllm_bench.sh takes.
set -uo pipefail
B=/opt/azslm/bench
QUEUE=${QUEUE:-"bf16:Qwen/Qwen3.8-27B w8a8:Avesed/Qwen3.8-27B-INT8-W8A8 w4a16:Avesed/Qwen3.8-27B-INT4-W4A16"}
SKIP_BENCH=${SKIP_BENCH:-""}   # tags whose throughput run is skipped (e.g. already measured)
for item in $QUEUE; do
  tag=${item%%:*}; repo=${item#*:}
  echo "######## $tag $repo $(date -u +%T)"
  n0=$(wc -l < /mnt/data/results/vllm-correct.jsonl 2>/dev/null || echo 0)
  MODEL=$repo TAG="$tag-compiled" ONEDNN_VERBOSE="${VERBOSE-1}" ARGS="--no-enforce-eager" bash $B/vllm_correct.sh
  n1=$(wc -l < /mnt/data/results/vllm-correct.jsonl 2>/dev/null || echo 0)
  if [ $((n1 - n0)) -ne 3 ]; then echo "correctness run for $tag failed (rows: $((n1-n0))); not benchmarking"; continue; fi
  case " $SKIP_BENCH " in *" $tag "*) continue;; esac
  MODELS=$repo bash $B/vllm_bench.sh
done
echo "queue done $(date -u +%T)"
