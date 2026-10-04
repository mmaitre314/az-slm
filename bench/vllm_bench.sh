#!/bin/bash
# Offline batch throughput with vLLM's CPU backend (`vllm bench throughput`, random prompts).
# Per model, three workloads: prefill-only (output 1 token), decode-heavy, and mixed 512-in/128-out,
# all submitted as one batch of NUM_PROMPTS requests. Appends JSON lines to
# /mnt/data/results/vllm.jsonl. Env: MODELS (paths under /mnt/data/models), NUM_PROMPTS, IMAGE,
# KV_GB (KV cache size), BIND (OpenMP CPU list; default one thread per physical core), EXTRA.
set -uo pipefail
IMAGE=${IMAGE:-vllm/vllm-openai-cpu:latest-x86_64}
MODELS=${MODELS:-"Qwen/Qwen3.8-27B Avesed/Qwen3.8-27B-INT8-W8A8"}
NUM_PROMPTS=${NUM_PROMPTS:-16}
KV_GB=${KV_GB:-16}
BIND=${BIND:-$(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -u | cut -d, -f1 | paste -sd,)}
OUT=/mnt/data/results
mkdir -p "$OUT"
# name input_len output_len
WORKLOADS=${WORKLOADS:-"prefill 512 1
decode 32 256
mixed 512 128"}
for model in $MODELS; do
  while read -r wl in_len out_len; do
    [ -z "$wl" ] && continue
    tag=$(echo "$model" | tr '/' '_')-$wl
    echo "== $model $wl in=$in_len out=$out_len prompts=$NUM_PROMPTS bind=$BIND $(date -u +%T)"
    docker run --rm --privileged --shm-size 8g -v /mnt/data:/mnt/data \
      -e VLLM_CPU_KVCACHE_SPACE="$KV_GB" -e VLLM_CPU_OMP_THREADS_BIND="$BIND" \
      --entrypoint vllm "$IMAGE" bench throughput --model "/mnt/data/models/$model" \
      --dataset-name random --random-input-len "$in_len" --random-output-len "$out_len" \
      --num-prompts "$NUM_PROMPTS" --max-model-len 2048 --dtype bfloat16 \
      --limit-mm-per-prompt '{"image": 0, "video": 0}' \
      --output-json "$OUT/vllm-$tag.json" ${EXTRA:-} > "$OUT/vllm-$tag.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$OUT/vllm-$tag.json" ]; then
      jq -c --arg m "$model" --arg w "$wl" --argjson i "$in_len" --argjson o "$out_len" --argjson n "$NUM_PROMPTS" \
        '{model: $m, workload: $w, input_len: $i, output_len: $o, num_prompts: $n} + .' "$OUT/vllm-$tag.json" \
        | tee -a "$OUT/vllm.jsonl" | jq -r '"  \(.requests_per_second * 1000 | floor / 1000) req/s, \(.tokens_per_second * 10 | floor / 10) tok/s total, elapsed \(.elapsed_time * 10 | floor / 10)s"'
    else
      echo "  FAILED rc=$rc"; grep -iE "error|exception|not supported|unsupported" "$OUT/vllm-$tag.log" | tail -n 5 | cut -c1-300
    fi
  done <<< "$WORKLOADS"
done
grep -h -iE "amx|avx512|onednn|isa" "$OUT"/vllm-*.log 2>/dev/null | sort -u | head -5
echo "done $(date -u +%T)"
