#!/bin/bash
# E32 + E31 chain on the cheapest 8-core AMX size (D16ls_v6, 32 GiB), no agent in the loop.
# E31: the E30 training-free readouts (Qwen3.5 0.8B/2B/4B, list format) repeated here for real $/decision.
# E32: trained letter-readout decision models in the SemIf format on vLLM CPU:
#   base Qwen3.5-4B (control), FlyMy decision-4b-v1.2 (its LoRA merged into Qwen3.5-4B by our
#   bench/dm_lora_merge.py; T = 1.45), JevK5 v0.3 (merged weights as published; T = 1.22).
# Only weights and JSON configs are downloaded from model repos; no repository code is executed.
# Every step logs to /mnt/data/results/chain.log, is capped by a timeout, never aborts the chain,
# removes its container, scores, and saves results to the OS disk. Env: SMOKE=1 (tiny, unsaved), LIMIT.
set -uo pipefail
IMAGE=vllm/vllm-openai-cpu@sha256:8024248339dc6878daa5349344ed29d49c8a6732f6bdf7400fda32ce33e4b30b
B=/opt/azslm/bench
D=/mnt/data/dm/data
R=/mnt/data/results
M=/mnt/data/models
HF=/opt/azslm/venv/bin/hf
SMOKE=${SMOKE:-0}
LIMIT=${LIMIT:-8}
KV=${KV:-8}
QWEN4_REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a   # the base revision FlyMy decision-4b-v1.2 was trained on
P=dm-e32; [ "$SMOKE" = 1 ] && P=dm-e32smoke
LOG=$R/chain.log; [ "$SMOKE" = 1 ] && LOG=$R/chain-smoke.log
BIND=$(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -u | cut -d, -f1 | paste -sd,)
mkdir -p $R /mnt/data/vllm-cache $M
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
SAVED=("chain.log" "summary-$P.jsonl" "$P-score.txt" "$P-p-*.jsonl" "$P-*.log" vllm-image.txt model-revisions.txt
       jev-convert.log data-prep.log merge.log vm-info.txt)
save() { [ "$SMOKE" = 1 ] || bash $B/save_results.sh "${SAVED[@]}" >> "$LOG" 2>&1; }

while [ ! -f /var/lib/azslm/setup-done ]; do sleep 15; done
{ echo "vmSize $(curl -s --max-time 5 -H Metadata:true 'http://169.254.169.254/metadata/instance/compute/vmSize?api-version=2021-02-01&format=text')";
  lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket"; free -g | head -2; nproc; } > $R/vm-info.txt 2>&1

# ---- setup: image, models (weights only), data ----
if ! docker image inspect "$IMAGE" >/dev/null 2>&1 || [ ! -f $M/Qwen/Qwen3.5-0.8B/config.json ]; then
  log "setup: image + Qwen3.5-0.8B/2B + JevK5"
  IMAGE=$IMAGE REPOS="Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B alibiserikbay/JevK5" bash $B/vllm_setup.sh >> $R/$P-setup.log 2>&1 || log "setup FAILED rc=$?"
fi
export HF_XET_HIGH_PERFORMANCE=1
if [ ! -f $M/Qwen/Qwen3.5-4B/config.json ]; then
  log "download Qwen/Qwen3.5-4B @ $QWEN4_REV"
  $HF download Qwen/Qwen3.5-4B --revision $QWEN4_REV --exclude '*.gguf' --local-dir $M/Qwen/Qwen3.5-4B >> $R/$P-setup.log 2>&1 || log "Qwen3.5-4B download FAILED"
  echo "Qwen/Qwen3.5-4B $QWEN4_REV (pinned)" >> $R/model-revisions.txt
fi
if [ ! -f $M/flymy-ai/decision-4b-v1.2/adapter/adapter_model.safetensors ]; then
  log "download FlyMy adapter weights + model.json only"
  $HF download flymy-ai/decision-4b-v1.2 adapter/adapter_config.json adapter/adapter_model.safetensors model.json \
    --local-dir $M/flymy-ai/decision-4b-v1.2 >> $R/$P-setup.log 2>&1 || log "FlyMy download FAILED"
  echo "flymy-ai/decision-4b-v1.2 $(curl -sS https://huggingface.co/api/models/flymy-ai/decision-4b-v1.2 | jq -r .sha) (adapter + model.json)" >> $R/model-revisions.txt
fi
[ -f $D/banking77_8.jsonl ] || python3 $B/dm_data.py >> $R/data-prep.log 2>&1 || log "dm_data FAILED"
ls $D/jev_*.jsonl >/dev/null 2>&1 || python3 $B/dm_jevbench_e30.py --out $D > $R/jev-convert.log 2>&1 || log "jev convert FAILED"
JEV=$(ls $D/jev_*.jsonl 2>/dev/null | paste -sd,)
B8=$D/banking77_8.jsonl
log "chain start smoke=$SMOKE kv=$KV bind=$BIND jev=$JEV"
save

# ---- merge FlyMy's LoRA into the pinned base (our script, in the trusted image) ----
FLY=$M/merged/flymy-decision-4b-v1.2
if [ ! -f $FLY/config.json ]; then
  log "STEP merge start"
  timeout 1800 docker run --rm --name e32m -v /mnt/data:/mnt/data -v $B:/bench:ro --entrypoint python3 "$IMAGE" \
    /bench/dm_lora_merge.py --base $M/Qwen/Qwen3.5-4B --adapter $M/flymy-ai/decision-4b-v1.2/adapter --out $FLY > $R/merge.log 2>&1
  log "STEP merge rc=$? $(tail -n 1 $R/merge.log)"
  save
fi
FLY_T=$(jq -r .temperature $M/flymy-ai/decision-4b-v1.2/model.json 2>/dev/null); FLY_T=${FLY_T:-1.45}
JK_T=$(jq -r .temperature $M/alibiserikbay/JevK5/jevk5_config.json 2>/dev/null); [ "$JK_T" = null ] || [ -z "$JK_T" ] && JK_T=1.22
log "temperatures: flymy=$FLY_T jevk5=$JK_T"

# step NAME MODEL_DIR TAG DATA TIMEOUT_MIN [extra readout args]
step() {
  local name=$1 model=$2 tag=$3 data=$4 tmo=$5; shift 5
  local lim=(); [ "$SMOKE" = 1 ] && lim=(--limit "$LIMIT")
  log "STEP $name start model=$model"
  docker rm -f e32 >/dev/null 2>&1
  { echo "== $name $(date -u +%T)"; top -bn1 -o %CPU | sed -n 1,12p; free -g | head -2; } >> $R/$P-top.log
  timeout -k 30 $((tmo*60)) docker run --rm --name e32 --privileged --shm-size 4g -v /mnt/data:/mnt/data -v $B:/bench:ro \
    -v /mnt/data/vllm-cache:/root/.cache/vllm -e RUN=$name -e VLLM_CPU_KVCACHE_SPACE=$KV -e VLLM_CPU_OMP_THREADS_BIND="$BIND" \
    --entrypoint python3 "$IMAGE" /bench/dm_readout.py $model --tag "$tag" --data "$data" \
    --out $R/$P-p-$name.jsonl --summary $R/summary-$P.jsonl "${lim[@]}" "$@" > $R/$P-$name.log 2>&1
  local rc=$?
  docker rm -f e32 >/dev/null 2>&1
  log "STEP $name rc=$rc"
  grep -E "^SUMMARY" $R/$P-$name.log | cut -c1-300 | sed 's/^/  /' >> "$LOG"
  [ $rc -ne 0 ] && { grep -iE "error|exception|killed" $R/$P-$name.log | tail -n 5 | cut -c1-300 >> "$LOG"; }
  python3 $B/dm_score.py $R/$P-p-*.jsonl --md > $R/$P-score.txt 2>&1
  save
}

if [ "$SMOKE" = 1 ]; then
  step smoke_list $M/Qwen/Qwen3.5-0.8B qwen3.5-0.8b "$B8,$JEV" 20
  step smoke_fly  $FLY flymy-decision-4b "$B8,$JEV" 30 --format semif --temperature "$FLY_T"
  exit 0
fi
# E32 first (the new question), then E31 (cheapest-size repeat of E30)
step fly_semif   $FLY                   flymy-decision-4b      "$B8,$JEV" 60 --format semif --temperature "$FLY_T" --repeat 2
step q4_semif    $M/Qwen/Qwen3.5-4B     qwen3.5-4b-semif       "$B8,$JEV" 60 --format semif --repeat 1
step jevk5_semif $M/alibiserikbay/JevK5 jevk5-v0.3             "$B8,$JEV" 60 --format semif --temperature "$JK_T" --repeat 1
step q08_list    $M/Qwen/Qwen3.5-0.8B   qwen3.5-0.8b           "$B8,$JEV" 30 --repeat 2
step q2_list     $M/Qwen/Qwen3.5-2B     qwen3.5-2b             "$B8,$JEV" 40 --repeat 2
step q4_list     $M/Qwen/Qwen3.5-4B     qwen3.5-4b             "$B8,$JEV" 60 --repeat 2
step q2_semif    $M/Qwen/Qwen3.5-2B     qwen3.5-2b-semif       "$B8,$JEV" 40 --format semif --repeat 1
log "chain done"
save
