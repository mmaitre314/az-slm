#!/bin/bash
# Offline batch throughput with vLLM's CPU backend (`vllm bench throughput`, random prompts).
# Per model, workloads such as prefill-only (output 1 token), decode-heavy and mixed 512-in/128-out,
# each submitted as one batch of N requests. Appends JSON lines to /mnt/data/results/vllm.jsonl.
#
# Env: MODELS (paths under /mnt/data/models), NUM_PROMPTS or PROMPT_SET (space-separated batch
# sizes to loop over), IMAGE, KV_GB (KV cache GiB), BIND (OpenMP CPU list; default one thread per
# physical core), RUN (label stored in each row, e.g. t8 or t16), WORKLOADS ("name in out" lines),
# EXTRA (extra `vllm bench throughput` args), MODEL_ARGS (extra args for quantized models).
# Before each run it logs `top` (to see mdatp or other CPU users) into vllm-top.log, and samples the
# largest host process RSS every 5 s (peak_rss_gb in the row; page cache excluded).
set -uo pipefail
IMAGE=${IMAGE:-vllm/vllm-openai-cpu:latest-x86_64}
MODELS=${MODELS:-"Qwen/Qwen3.8-27B Avesed/Qwen3.8-27B-INT8-W8A8"}
NUM_PROMPTS=${NUM_PROMPTS:-16}
PROMPT_SET=${PROMPT_SET:-$NUM_PROMPTS}
KV_GB=${KV_GB:-16}
BIND=${BIND:-$(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -u | cut -d, -f1 | paste -sd,)}
RUN=${RUN:-t$(echo "$BIND" | tr ',' '\n' | wc -l)}
OUT=/mnt/data/results
mkdir -p "$OUT" /mnt/data/vllm-cache
# name input_len output_len
WORKLOADS=${WORKLOADS:-"prefill 512 1
decode 32 256
mixed 512 128"}
for model in $MODELS; do
  for np in $PROMPT_SET; do
    while read -r wl in_len out_len; do
      [ -z "$wl" ] && continue
      tag=$(echo "$model" | tr '/' '_')-$wl-n$np-$RUN
      echo "== $model $wl in=$in_len out=$out_len prompts=$np run=$RUN bind=$BIND $(date -u +%T)"
      { echo "== $tag $(date -u +%T)"; top -bn1 -o %CPU | sed -n 1,12p; } >> "$OUT/vllm-top.log"
      ( peak=0; while sleep 5; do r=$(ps -eo rss= --sort=-rss | head -n 1 | tr -d ' '); [ "${r:-0}" -gt "$peak" ] && peak=$r && echo "$peak" > "$OUT/.rss-$tag"; done ) &
      sampler=$!
      docker run --rm --privileged --shm-size 8g -v /mnt/data:/mnt/data -v /mnt/data/vllm-cache:/root/.cache/vllm \
        -e VLLM_CPU_KVCACHE_SPACE="$KV_GB" -e VLLM_CPU_OMP_THREADS_BIND="$BIND" \
        --entrypoint vllm "$IMAGE" bench throughput --model "/mnt/data/models/$model" \
        --dataset-name random --random-input-len "$in_len" --random-output-len "$out_len" \
        --num-prompts "$np" --max-model-len 2048 --dtype bfloat16 \
        --limit-mm-per-prompt '{"image": 0, "video": 0}' \
        --output-json "$OUT/vllm-$tag.json" ${MODEL_ARGS:-} ${EXTRA:-} > "$OUT/vllm-$tag.log" 2>&1
      rc=$?
      kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
      rss=$(cat "$OUT/.rss-$tag" 2>/dev/null || echo 0); rm -f "$OUT/.rss-$tag"
      if [ $rc -eq 0 ] && [ -f "$OUT/vllm-$tag.json" ]; then
        jq -c --arg m "$model" --arg w "$wl" --arg run "$RUN" --argjson i "$in_len" --argjson o "$out_len" \
          --argjson n "$np" --argjson rss "$rss" --arg bind "$BIND" --arg img "$IMAGE" \
          '{model: $m, workload: $w, run: $run, input_len: $i, output_len: $o, num_prompts: $n, omp_bind: $bind,
            peak_rss_gb: ($rss / 1e6 * 10 | floor / 10), image: $img} + .' "$OUT/vllm-$tag.json" \
          | tee -a "$OUT/vllm.jsonl" | jq -r '"  \(.requests_per_second * 1000 | floor / 1000) req/s, \(.tokens_per_second * 10 | floor / 10) tok/s total, elapsed \(.elapsed_time * 10 | floor / 10)s, rss \(.peak_rss_gb) GB"'
      else
        echo "  FAILED rc=$rc"; grep -iE "error|exception|not supported|unsupported" "$OUT/vllm-$tag.log" | tail -n 5 | cut -c1-300
        jq -cn --arg m "$model" --arg w "$wl" --arg run "$RUN" --argjson n "$np" --argjson rc "$rc" \
          '{model: $m, workload: $w, run: $run, num_prompts: $n, failed: true, rc: $rc}' >> "$OUT/vllm.jsonl"
      fi
    done <<< "$WORKLOADS"
  done
done
echo "done $(date -u +%T)"
