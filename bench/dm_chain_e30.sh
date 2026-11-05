#!/bin/bash
# E30 chain: training-free decision readout on vLLM CPU (bench/dm_readout.py), no agent in the loop.
# Smallest model first; every step logs to /mnt/data/results/chain.log, is capped by a timeout, never
# aborts the chain, removes its container, scores, and saves results to the OS disk.
# Env: SMOKE=1 (Qwen3.5-0.8B only, LIMIT items, separate files, nothing saved), LIMIT.
set -uo pipefail
IMAGE=vllm/vllm-openai-cpu@sha256:8024248339dc6878daa5349344ed29d49c8a6732f6bdf7400fda32ce33e4b30b
B=/opt/azslm/bench
D=/mnt/data/dm/data
R=/mnt/data/results
SMOKE=${SMOKE:-0}
LIMIT=${LIMIT:-8}
P=dm-e30; [ "$SMOKE" = 1 ] && P=dm-e30smoke
LOG=$R/chain.log; [ "$SMOKE" = 1 ] && LOG=$R/chain-smoke.log
BIND=$(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -u | cut -d, -f1 | paste -sd,)
mkdir -p $R /mnt/data/vllm-cache
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
SAVED=("chain.log" "summary-$P.jsonl" "$P-score.txt" "$P-p-*.jsonl" "$P-*.log" vllm-image.txt model-revisions.txt "jev-convert.log" "data-prep.log")
save() { [ "$SMOKE" = 1 ] || bash $B/save_results.sh "${SAVED[@]}" >> "$LOG" 2>&1; }

REPOS="Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Avesed/Qwen3.8-27B-INT8-W8A8"
# wait for a running setup job, then make sure everything is present
while systemctl is-active --quiet azslm-job-e30setup.service 2>/dev/null; do sleep 20; done
missing=""
for r in $REPOS; do [ -f /mnt/data/models/$r/config.json ] || missing="$missing $r"; done
if [ -n "$missing" ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  log "setup (missing:$missing)"
  IMAGE=$IMAGE REPOS="${missing:-$REPOS}" bash $B/vllm_setup.sh >> $R/$P-setup.log 2>&1 || log "setup FAILED rc=$?"
fi
if [ ! -f $D/banking77_8.jsonl ]; then python3 $B/dm_data.py >> $R/data-prep.log 2>&1 || log "dm_data FAILED"; fi
if ! ls $D/jev_*.jsonl >/dev/null 2>&1; then python3 $B/dm_jevbench_e30.py --out $D > $R/jev-convert.log 2>&1 || log "jev convert FAILED"; fi
cat $R/jev-convert.log >> "$LOG" 2>/dev/null
JEV=$(ls $D/jev_*.jsonl 2>/dev/null | paste -sd,)
log "chain start smoke=$SMOKE jev sets: $JEV"
save

# step NAME MODEL TAG DATA TIMEOUT_MIN [extra readout args]
step() {
  local name=$1 model=$2 tag=$3 data=$4 tmo=$5; shift 5
  local lim=(); [ "$SMOKE" = 1 ] && lim=(--limit "$LIMIT")
  log "STEP $name start model=$model data=$data"
  docker rm -f e30 >/dev/null 2>&1
  { echo "== $name $(date -u +%T)"; top -bn1 -o %CPU | sed -n 1,12p; } >> $R/$P-top.log
  timeout -k 30 $((tmo*60)) docker run --rm --name e30 --privileged --shm-size 8g -v /mnt/data:/mnt/data -v $B:/bench:ro \
    -v /mnt/data/vllm-cache:/root/.cache/vllm -e RUN=$name -e VLLM_CPU_KVCACHE_SPACE=16 -e VLLM_CPU_OMP_THREADS_BIND="$BIND" \
    --entrypoint python3 "$IMAGE" /bench/dm_readout.py /mnt/data/models/$model --tag "$tag" --data "$data" \
    --out $R/$P-p-$name.jsonl --summary $R/summary-$P.jsonl "${lim[@]}" "$@" > $R/$P-$name.log 2>&1
  local rc=$?
  docker rm -f e30 >/dev/null 2>&1
  log "STEP $name rc=$rc"
  grep -E "^SUMMARY" $R/$P-$name.log | cut -c1-400 | sed 's/^/  /' >> "$LOG"
  [ $rc -ne 0 ] && { grep -iE "error|exception" $R/$P-$name.log | tail -n 5 | cut -c1-300 >> "$LOG"; }
  python3 $B/dm_score.py $R/$P-p-*.jsonl --md > $R/$P-score.txt 2>&1
  tail -n 40 $R/$P-score.txt >> "$LOG"
  save
}

B8=$D/banking77_8.jsonl; B8S=$D/banking77_8_small.jsonl; B77S=$D/banking77_77_small.jsonl
if [ "$SMOKE" = 1 ]; then
  step smoke Qwen/Qwen3.5-0.8B qwen3.5-0.8b "$B8,$JEV" 20 --repeat 1
  exit 0
fi
step q08_b8jev   Qwen/Qwen3.5-0.8B qwen3.5-0.8b  "$B8,$JEV" 30 --max-num-seqs 64 --repeat 2
step q08_b77s    Qwen/Qwen3.5-0.8B qwen3.5-0.8b  "$B77S"    15 --max-num-seqs 64 --repeat 1
step q2_b8jev    Qwen/Qwen3.5-2B   qwen3.5-2b    "$B8,$JEV" 40 --max-num-seqs 64 --repeat 2
step q4_b8jev    Qwen/Qwen3.5-4B   qwen3.5-4b    "$B8,$JEV" 60 --max-num-seqs 64 --repeat 2
step q4_n16      Qwen/Qwen3.5-4B   qwen3.5-4b-n16  "$B8"    30 --max-num-seqs 16 --repeat 1 --limit 1000
step q4_n128     Qwen/Qwen3.5-4B   qwen3.5-4b-n128 "$B8"    30 --max-num-seqs 128 --repeat 1 --limit 1000
step q9_b8jev    Qwen/Qwen3.5-9B   qwen3.5-9b    "$B8,$JEV" 90 --max-num-seqs 64 --repeat 1
step q27_b8sjev  Avesed/Qwen3.8-27B-INT8-W8A8 qwen3.8-27b-w8a8 "$B8S,$JEV" 100 --max-num-seqs 64 --repeat 1
log "chain done"
save
