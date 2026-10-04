#!/bin/bash
# E16 chain (task-level quality of the vLLM formats), self-contained and never aborting:
#   1. vllm_setup.sh (Docker, image pull, model downloads), retried up to 3 times
#   2. datasets (openai/gsm8k, cais/mmlu) downloaded with `hf`, converted to JSONL in the container
#   3. per model, in order BF16, W8A8, W4A16: 3-prompt correctness check (vllm_correct.sh), then
#      quality_tasks.py (GSM8K 200 + MMLU 400, greedy, thinking off); each step is retried once
# A failed step is logged and the chain continues with the next one. Progress goes to
# /mnt/data/results/chain.log; after EVERY step the results are saved with save_results.sh.
# Env: REPOS, IMAGE, KV_GB, MAX_SEQS, LIMIT (questions per benchmark; 0 = full), BENCHES, MODELS
# ("tag=repo ..." in run order), RUN (label stored in each result row). Smoke test: R=/mnt/data/results-smoke
# SAVE=0 LIMIT=5 MODELS=w8a8=Avesed/Qwen3.8-27B-INT8-W8A8 (separate output files, nothing saved to the OS disk).
set -uo pipefail
B=/opt/azslm/bench
R=${R:-/mnt/data/results}          # chain outputs (vllm_setup.sh and vllm_correct.sh logs stay in /mnt/data/results)
RS=/mnt/data/results
SAVE=${SAVE:-1}
IMAGE=${IMAGE:-vllm/vllm-openai-cpu:latest-x86_64}
REPOS=${REPOS:-"Qwen/Qwen3.8-27B Avesed/Qwen3.8-27B-INT8-W8A8 Avesed/Qwen3.8-27B-INT4-W4A16"}
MODELS=${MODELS:-"bf16=Qwen/Qwen3.8-27B w8a8=Avesed/Qwen3.8-27B-INT8-W8A8 w4a16=Avesed/Qwen3.8-27B-INT4-W4A16"}
KV_GB=${KV_GB:-16}
MAX_SEQS=${MAX_SEQS:-64}
LIMIT=${LIMIT:-0}
BENCHES=${BENCHES:-gsm8k,mmlu}
RUN=${RUN:-e16-$(date -u +%Y%m%dT%H%M)}
DATA=/mnt/data/datasets
QUALITY_TIMEOUT=${QUALITY_TIMEOUT:-14400}   # seconds per attempt of one model's quality run
BIND=$(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -u | cut -d, -f1 | paste -sd,)
mkdir -p "$R" "$RS" /mnt/data/vllm-cache "$DATA"
LOG=$R/chain.log

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
save() { [ "$SAVE" = 1 ] || return 0; bash "$B/save_results.sh" quality-tasks.jsonl quality-summary.jsonl 'vllm-correct*' vllm-image.txt \
           model-revisions.txt dataset-revisions.txt chain.log 'quality-run-*.log' e16-top.log >/dev/null 2>&1; }
step_done() { log "$1"; save; }
missing_benches() {  # tag -> comma list of BENCHES without a summary line for this run and tag
  local done_b; done_b=$(jq -r --arg run "$RUN" --arg tag "$1" 'select(.run == $run and .tag == $tag) | .benchmark' \
    "$R/quality-summary.jsonl" 2>/dev/null)
  for b in ${BENCHES//,/ }; do grep -qx "$b" <<< "$done_b" || echo "$b"; done | paste -sd,
}
kill_containers() { docker ps -aq 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1; }

log "chain start run=$RUN host=$(hostname) LIMIT=$LIMIT BENCHES=$BENCHES KV_GB=$KV_GB MAX_SEQS=$MAX_SEQS"
export RUN
while [ ! -f /var/lib/azslm/setup-done ]; do sleep 10; done

# ---- 1. setup (Docker, image, model downloads) ----
rm -f "$RS/model-revisions.txt"
ok=0
for attempt in 1 2 3; do
  log "setup attempt $attempt (REPOS=$REPOS)"
  if REPOS="$REPOS" IMAGE="$IMAGE" bash "$B/vllm_setup.sh" > "$RS/setup-$attempt.log" 2>&1; then ok=1; break; fi
  log "setup attempt $attempt FAILED: $(tail -n 3 "$RS/setup-$attempt.log" | tr '\n' ' ' | cut -c1-300)"
done
sort -u -o "$RS/model-revisions.txt" "$RS/model-revisions.txt" 2>/dev/null
[ $ok -eq 1 ] && step_done "setup done: $(tr '\n' ';' < "$RS/vllm-image.txt" | cut -c1-300)" || step_done "setup FAILED (continuing anyway)"

# ---- 2. datasets ----
export HF_XET_HIGH_PERFORMANCE=1
: > "$R/dataset-revisions.txt"
dl() {  # repo local-name include-pattern
  echo "$1 $(curl -sS "https://huggingface.co/api/datasets/$1" | jq -r .sha)" >> "$R/dataset-revisions.txt"
  /opt/azslm/venv/bin/hf download "$1" --repo-type dataset --include "$3" --local-dir "$DATA/$2" >> "$R/e16-hf.log" 2>&1
}
ds_ok=0
for attempt in 1 2 3; do
  dl openai/gsm8k gsm8k 'main/test-*.parquet'; dl cais/mmlu mmlu 'all/test-*.parquet'
  docker run --rm -v /mnt/data:/mnt/data -v "$B":/bench:ro --entrypoint python3 "$IMAGE" /bench/e16_convert.py "$DATA" > "$R/e16-convert.log" 2>&1
  tail -n 4 "$R/e16-convert.log" | cut -c1-300 >> "$LOG"
  if [ -s "$DATA/gsm8k_test.jsonl" ] && [ -s "$DATA/mmlu_test.jsonl" ]; then ds_ok=1; break; fi
  log "dataset step attempt $attempt FAILED"
done
sort -u -o "$R/dataset-revisions.txt" "$R/dataset-revisions.txt"
step_done "datasets ready=$ds_ok: $(tr '\n' ';' < "$R/dataset-revisions.txt"); rows gsm8k=$(wc -l < "$DATA/gsm8k_test.jsonl" 2>/dev/null) mmlu=$(wc -l < "$DATA/mmlu_test.jsonl" 2>/dev/null)"

# ---- 3. per model: correctness check, then quality ----
quality_attempt() {  # tag repo benchmarks [extra args]
  local tag=$1 repo=$2 todo=$3; shift 3
  { echo "== $tag attempt $(date -u +%T)"; top -bn1 -o %CPU | sed -n 1,12p; } >> "$R/e16-top.log"
  timeout "$QUALITY_TIMEOUT" docker run --rm --name "e16-$tag" --privileged --shm-size 8g \
    -v /mnt/data:/mnt/data -v "$B":/bench:ro -v /mnt/data/vllm-cache:/root/.cache/vllm \
    -e VLLM_CPU_KVCACHE_SPACE="$KV_GB" -e VLLM_CPU_OMP_THREADS_BIND="$BIND" -e RUN="$RUN" \
    --entrypoint python3 "$IMAGE" /bench/quality_tasks.py "/mnt/data/models/$repo" --tag "$tag" \
    --benchmarks "$todo" --limit "$LIMIT" --out-tasks "$R/quality-tasks.jsonl" --out-summary "$R/quality-summary.jsonl" --max-num-seqs "$MAX_SEQS" --show 3 "$@" 2>&1 \
    | tr '\r' '\n' | grep -vE '^\s*$' >> "$R/quality-run-$tag.log"
  local rc=${PIPESTATUS[0]}
  kill_containers
  return $rc
}

for entry in $MODELS; do
  tag=${entry%%=*}; repo=${entry#*=}
  if [ ! -f "/mnt/data/models/$repo/config.json" ]; then
    step_done "MODEL $tag ($repo): not downloaded, skipped"; continue
  fi
  log "== model $tag ($repo) start"

  # 3a. correctness check (3 prompts), once retried
  for attempt in 1 2; do
    MODEL="$repo" TAG="$tag" IMAGE="$IMAGE" KV_GB="$KV_GB" bash "$B/vllm_correct.sh" > "$R/vllm-correct-$tag.out" 2>&1 \
      && grep -q '"prompt_id": 3' "$RS/vllm-correct-$tag.log" && { c_ok=1; break; }
    c_ok=0; kill_containers; log "correctness $tag attempt $attempt FAILED: $(grep -iE 'error|Traceback' "$RS/vllm-correct-$tag.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
  done
  step_done "correctness $tag ok=$c_ok: $(grep -o '"text": "[^"]*"' "$RS/vllm-correct-$tag.log" | cut -c1-90 | tr '\n' '|')"

  # 3b. task quality: attempt 1 as the bench runs (torch.compile); attempt 2 is eager and only runs
  # the benchmarks that have no summary line yet (finished benchmarks are saved as they complete).
  for attempt in 1 2; do
    todo=$(missing_benches "$tag"); [ -z "$todo" ] && break
    extra=(); [ $attempt -eq 2 ] && extra=(--enforce-eager)
    log "quality $tag attempt $attempt start: $todo (${extra[*]:-compile})"
    quality_attempt "$tag" "$repo" "$todo" "${extra[@]}"; rc=$?
    todo=$(missing_benches "$tag"); [ -z "$todo" ] && break
    log "quality $tag attempt $attempt FAILED rc=$rc, missing $todo: $(grep -iE 'error|Traceback' "$R/quality-run-$tag.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
    save
  done
  q_ok=0; [ -z "$(missing_benches "$tag")" ] && q_ok=1
  step_done "quality $tag ok=$q_ok: $(grep "\"run\": \"$RUN\", \"tag\": \"$tag\"" "$R/quality-summary.jsonl" 2>/dev/null | jq -r '"\(.benchmark)=\(.accuracy) (n=\(.n), \(.wall_seconds)s)"' | tr '\n' ' ')"
done
log "chain done"
save
