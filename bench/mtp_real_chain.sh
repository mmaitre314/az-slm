#!/bin/bash
# E17 follow-up: vLLM MTP speculative decoding on real text. `vllm bench throughput --dataset-name
# random` gives the MTP head random-token prompts, which understates acceptance, so this runs the
# first 200 GSM8K test questions (E16's prompts, greedy, thinking off, up to 512 output tokens) with
# W8A8 and num_speculative_tokens = 0 (baseline), 1, 2 and 3, on one VM. Per run: accuracy, wall time,
# output tokens and vLLM's spec-decode counters (acceptance) in quality-summary.jsonl.
# Uses e16_chain.sh (setup, datasets, quality_tasks.py); waits for a running vllm_chain.sh to finish.
# Env: RUN, NS (spec token counts; 0 = baseline), KV_GB (default 24: MTP's extra KV cache), MAX_SEQS.
set -uo pipefail
B=/opt/azslm/bench
REPO=Avesed/Qwen3.8-27B-INT8-W8A8
RUN=${RUN:-mtpreal-$(date -u +%Y%m%dT%H%M)}
NS=${NS:-"0 1 2 3"}
export KV_GB=${KV_GB:-24} MAX_SEQS=${MAX_SEQS:-64}
# vllm_chain.sh holds this lock while it runs: wait for it (up to 6 h)
flock -w 21600 /var/lib/azslm/chain.lock true
for n in $NS; do
  if [ "$n" = 0 ]; then tag=w8a8-base; q='{}'
  else tag=w8a8-mtp$n; q="{\"speculative_config\":{\"method\":\"mtp\",\"num_speculative_tokens\":$n}}"; fi
  MODELS="$tag=$REPO" REPOS="$REPO" BENCHES=gsm8k LIMIT=0 RUN="$RUN" QEXTRA="$q" SKIP_CORRECT=1 bash "$B/e16_chain.sh"
done
echo "mtp_real_chain done $(date -u +%T)" | tee -a /mnt/data/results/chain.log
bash "$B/save_results.sh" chain.log quality-summary.jsonl quality-tasks.jsonl >/dev/null 2>&1
