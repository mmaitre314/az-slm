#!/bin/bash
# One self-contained, unattended chain for the vLLM experiments E17 (batch scaling, W4A16, threads,
# MTP on E16ds_v7) and E18 (same stack on E16ds_v6). No agent in the loop: every step logs its
# outcome and the chain continues after a failure. After EVERY step it calls save_results.sh, which
# copies the small result files to /var/lib/azslm/results (survives deallocation; the watchdog then
# deallocates instead of deleting the VM). Progress log: /mnt/data/results/chain.log.
#
# Start (after `deploy.py push <vm>`):
#   deploy.py run <vm> -c 'bash /opt/azslm/bench/vllm_chain.sh' -e PROFILE=e17 --background chain
#   deploy.py job <vm> chain          # state + tail of the job log (the same text goes to chain.log)
#
# Env:
#   PROFILE   e17 (default) | e18 | smoke (tiny run to test the chain; afterwards move /mnt/data/results aside by hand)
#   STEPS     space-separated step names; overrides the profile's list (see `steps_for` below)
#   FORCE=1   re-run steps that already completed (markers live in /mnt/data/results/.done)
#   DRY=1     print the commands instead of running them (RESULTS_DIR=/some/dir for a local dry run)
#   RUN_TIMEOUT  seconds allowed for one `vllm bench throughput` run (default 7200)
#
# Steps (E17 = all of them in this order; E18 = meta ref setup correct w8a8_n16 w8a8_n64 w4a16_mixed_n64 final):
#   meta      versions/CPU/IMDS size, top snapshot            ref        llama-bench Q4_K_M pp512/tg128, 8 threads
#   setup     vllm_setup.sh (docker, image, W8A8 + W4A16)     mtp_check  MTP tensors in the W8A8 / W4A16 checkpoints
#   correct   3-prompt greedy check per model (gates that model's benchmarks)
#   w8a8_n16 w8a8_n32 w8a8_n64        W8A8, prefill+decode+mixed, 8 threads (KV 16/16/24 GiB), RUN=t8
#   w4a16_n16 w4a16_n64               W4A16, prefill+decode+mixed, 8 threads, RUN=t8
#   w8a8_t16_n64                      W8A8, prefill+mixed at 64 prompts, 16 threads, RUN=t16
#   mtp_n16 mtp_n64 mtp2_n16          W8A8 + MTP (decode+mixed N=1 at 16/64; decode N=2 at 16), RUN=mtp1/mtp2
#   w4a16_mixed_n64 (E18 only), final (summary table + digests), smoke_bench smoke_mtp (smoke only)
set -uo pipefail
export HOME=${HOME:-/root}
# Run from a private copy: bash reads a script incrementally, so a later `deploy.py push` of a fixed
# vllm_chain.sh must not change the file under a running chain.
if [ -z "${CHAIN_COPY:-}" ] && [ -z "${DRY:-}" ]; then
  mkdir -p /mnt/data/jobs && cp "$0" "/mnt/data/jobs/vllm_chain.$$.sh" && CHAIN_COPY=1 exec bash "/mnt/data/jobs/vllm_chain.$$.sh" "$@"
fi

PROFILE=${PROFILE:-e17}
B=${B:-/opt/azslm/bench}
OUT=${RESULTS_DIR:-/mnt/data/results}
DONE=$OUT/.done
STATE=/var/lib/azslm
DRY=${DRY:-}
FORCE=${FORCE:-}
RUN_TIMEOUT=${RUN_TIMEOUT:-7200}
IMAGE=${IMAGE:-vllm/vllm-openai-cpu:latest-x86_64}
W8=Avesed/Qwen3.8-27B-INT8-W8A8
W4=Avesed/Qwen3.8-27B-INT4-W4A16
MTP_JSON1='{"method":"mtp","num_speculative_tokens":1}'
MTP_JSON2='{"method":"mtp","num_speculative_tokens":2}'
ALL3="prefill 512 1
decode 32 256
mixed 512 128"
SAVE_FILES=(vllm.jsonl 'vllm-*.json' 'vllm-correct*.jsonl' 'vllm-correct-*.log.onednn*' vllm-top.log vllm-image.txt
            model-revisions.txt llama-bench.jsonl mtp-check.jsonl chain-meta.txt chain.log)

mkdir -p "$OUT" "$DONE"
cd "$OUT" || exit 1

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/chain.log"; }
run_cmd() { if [ -n "$DRY" ]; then printf 'DRY:'; printf ' %q' "$@"; echo; return 0; fi; "$@"; }
save() {
  [ -n "$DRY" ] && return 0
  bash "$B/save_results.sh" "${SAVE_FILES[@]}" 2>&1 | tail -n 1 | sed 's/^/  /' | tee -a "$OUT/chain.log"
}
kill_containers() { [ -n "$DRY" ] || docker ps -q 2>/dev/null | xargs -r docker kill >/dev/null 2>&1; }
failed_rows() { local n; n=$(grep -c '"failed":true' "$OUT/vllm.jsonl" 2>/dev/null); echo "${n:-0}"; }

if [ -z "$DRY" ]; then
  exec 9>"$STATE/chain.lock"
  flock -n 9 || { echo "another vllm_chain is running on this VM; exiting"; exit 3; }
fi

BIND8=$(lscpu -p=CPU,CORE 2>/dev/null | grep -v '^#' | sort -t, -k2,2n -u | cut -d, -f1 | paste -sd,)
BIND16=$(lscpu -p=CPU 2>/dev/null | grep -v '^#' | cut -d, -f1 | paste -sd,)

steps_for() {
  case "$1" in
    e17) echo "meta ref setup mtp_check correct w8a8_n16 w8a8_n32 w8a8_n64 w4a16_n16 w4a16_n64 w8a8_t16_n64 mtp_n16 mtp_n64 mtp2_n16 final" ;;
    e18) echo "meta ref setup correct w8a8_n16 w8a8_n64 w4a16_mixed_n64 final" ;;
    smoke) echo "meta setup mtp_check correct smoke_bench smoke_mtp" ;;
    *) echo "unknown PROFILE $1" >&2; echo "" ;;
  esac
}
STEPS=${STEPS:-$(steps_for "$PROFILE")}
CORRECT_MODELS=${CORRECT_MODELS:-"w8a8 w4a16"}
[ "$PROFILE" = smoke ] && CORRECT_MODELS=${CORRECT_MODELS_SMOKE:-w8a8}
declare -A BAD=()          # model short name -> 1 if its correctness check failed twice
MTP_BROKEN=              # set when the first MTP step failed on every run: later MTP steps are skipped

model_dir() { echo "/mnt/data/models/$1"; }

# ---- benchmark helper: one vllm_bench.sh invocation per workload so each run is timed out and saved alone
# bench MODEL NPROMPTS RUN KV_GB "workload lines" [BIND] [EXTRA]   -> returns 1 if any run failed
bench() {
  local model=$1 np=$2 run=$3 kv=$4 wls=$5 bind=${6:-} extra=${7:-} rc=0 wl in out short
  case "$model" in "$W8") short=w8a8 ;; "$W4") short=w4a16 ;; *) short=other ;; esac
  if [ -n "${BAD[$short]:-}" ]; then log "SKIP $short n=$np run=$run: its correctness check failed"; return 1; fi
  if [ -z "$DRY" ] && [ ! -f "$(model_dir "$model")/config.json" ]; then log "SKIP $short n=$np run=$run: model files missing"; return 1; fi
  while read -r wl in out; do
    [ -z "$wl" ] && continue
    local envs=(MODELS="$model" PROMPT_SET="$np" KV_GB="$kv" RUN="$run" IMAGE="$IMAGE" WORKLOADS="$wl $in $out")
    [ -n "$bind" ] && envs+=(BIND="$bind")
    [ -n "$extra" ] && envs+=(EXTRA="$extra")
    local f0 t0=$SECONDS tag
    f0=$(failed_rows)
    tag=$(echo "$model" | tr '/' '_')-$wl-n$np-$run
    log "RUN $short $wl in=$in out=$out n=$np run=$run kv=${kv}G ${bind:+bind=$bind }${extra:+extra=$extra}"
    run_cmd timeout -k 60 "$RUN_TIMEOUT" env "${envs[@]}" bash "$B/vllm_bench.sh" 2>&1 | grep -v '^done ' | tee -a "$OUT/chain.log"
    local prc=${PIPESTATUS[0]}
    [ "$prc" -eq 124 ] || [ "$prc" -eq 137 ] && log "TIMEOUT after ${RUN_TIMEOUT}s: $tag"
    kill_containers
    if [ -z "$DRY" ]; then
      if [ "$(failed_rows)" -gt "$f0" ] || [ "$prc" -ne 0 ]; then
        rc=1
        # a timeout or crash before vllm_bench.sh wrote its own failure row still leaves a marker row
        [ "$(failed_rows)" -gt "$f0" ] || echo "{\"model\":\"$model\",\"workload\":\"$wl\",\"run\":\"$run\",\"num_prompts\":$np,\"failed\":true,\"rc\":$prc}" >> "$OUT/vllm.jsonl"
        log "FAILED $tag rc=$prc; last errors:"
        grep -iE "error|exception|not supported|unsupported|Traceback" "$OUT/vllm-$tag.log" 2>/dev/null | tail -n 4 | cut -c1-300 | sed 's/^/    /' | tee -a "$OUT/chain.log"
      else
        grep -hE "GPU KV cache size|Maximum concurrency|KV cache size" "$OUT/vllm-$tag.log" 2>/dev/null | tail -n 2 | cut -c1-200 | sed 's/^/    /' | tee -a "$OUT/chain.log"
        [ -n "$extra" ] && grep -hiE "accept" "$OUT/vllm-$tag.log" 2>/dev/null | tail -n 2 | cut -c1-250 | sed 's/^/    /' | tee -a "$OUT/chain.log"
      fi
    fi
    log "  took $((SECONDS - t0))s"
    save
  done <<< "$wls"
  return $rc
}

# ---- steps ----------------------------------------------------------------------------------------
step_meta() {
  {
    echo "date $(date -u +%FT%TZ)  host $(hostname)  profile $PROFILE  steps: $STEPS"
    echo "vmSize $(curl -s --max-time 5 -H Metadata:true 'http://169.254.169.254/metadata/instance/compute/vmSize?api-version=2021-02-01&format=text')" \
         "location $(curl -s --max-time 5 -H Metadata:true 'http://169.254.169.254/metadata/instance/compute/location?api-version=2021-02-01&format=text')"
    lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|MHz' | sed 's/  */ /g'
    echo "amx flags: $(grep -o 'amx[a-z_0-9]*' /proc/cpuinfo | sort -u | paste -sd' ')"
    echo "kernel $(uname -r)  $(grep PRETTY_NAME /etc/os-release)"
    echo "llama.cpp $(cat $STATE/llama.cpp.version 2>/dev/null)"
    echo "bind8=$BIND8 bind16=$BIND16 image=$IMAGE"
    echo "--- top (before any measurement)"; top -bn1 -o %CPU | sed -n 1,14p
  } > "$OUT/chain-meta.txt" 2>&1
  head -n 9 "$OUT/chain-meta.txt" | tee -a "$OUT/chain.log"
}

step_ref() {   # E17/E18 step 0: llama-bench Q4_K_M pp512/tg128, 8 threads, AMX build (single sequence)
  local m=/mnt/data/models/bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf
  if [ -z "$DRY" ] && [ ! -x /opt/llama.cpp/build/bin/llama-bench ]; then log "llama-bench missing (cloud-init build failed?)"; return 1; fi
  if [ ! -f "$m" ]; then
    # run from / so the unquoted '*Q4_K_M*' in download.sh cannot glob-match a log file in $OUT
    (cd / && run_cmd timeout -k 60 3600 env REPO=bartowski/Qwen3.8-27B-GGUF INCLUDE='*Q4_K_M*' bash "$B/download.sh") 2>&1 | tail -n 3 | cut -c1-200 | tee -a "$OUT/chain.log"
  fi
  [ -z "$DRY" ] && [ ! -f "$m" ] && { log "Q4_K_M GGUF not found at $m"; return 1; }
  run_cmd timeout -k 60 3600 env QUANTS=Q4_K_M BUILDS=build THREADS=8 REPS=3 bash "$B/llama_bench.sh" 2>&1 | tee -a "$OUT/chain.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
  [ -n "$DRY" ] || [ "$(grep -c Q4_K_M "$OUT/llama-bench.jsonl" 2>/dev/null)" -ge 2 ]
}

step_setup() {   # docker + image + the two quantized checkpoints (NOT the BF16 repo); retried, idempotent
  local try
  for try in 1 2 3; do
    log "vllm_setup.sh attempt $try"
    run_cmd timeout -k 60 5400 env IMAGE="$IMAGE" REPOS="$W8 $W4" bash "$B/vllm_setup.sh" > "$OUT/vllm-setup.log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ] && [ -z "$DRY" ]; then
      ls "$(model_dir "$W8")"/*.safetensors >/dev/null 2>&1 && ls "$(model_dir "$W4")"/*.safetensors >/dev/null 2>&1 && break
      rc=1
    fi
    [ "$rc" -eq 0 ] && break
    log "setup rc=$rc; last lines:"; tail -n 6 "$OUT/vllm-setup.log" | cut -c1-250 | sed 's/^/    /' | tee -a "$OUT/chain.log"
    [ "$try" -lt 3 ] && sleep 30
  done
  [ -n "$DRY" ] && return 0
  tail -n 2 "$OUT/vllm-image.txt" 2>/dev/null | cut -c1-250 | tee -a "$OUT/chain.log"
  cat "$OUT/model-revisions.txt" 2>/dev/null | tee -a "$OUT/chain.log"
  du -sh "$(model_dir "$W8")" "$(model_dir "$W4")" 2>/dev/null | tee -a "$OUT/chain.log"
  [ "$rc" -eq 0 ]
}

step_mtp_check() {   # are there MTP tensors in the checkpoints? (safetensors headers, stdlib only)
  [ -n "$DRY" ] && { echo "DRY: mtp_check"; return 0; }
  python3 - "$(model_dir "$W8")" "$(model_dir "$W4")" >> "$OUT/mtp-check.jsonl" <<'PY'
import glob, json, os, struct, sys
for d in sys.argv[1:]:
    names, dtypes = [], {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__" and "mtp" in k.lower():
                names.append(k); dtypes[v["dtype"]] = dtypes.get(v["dtype"], 0) + 1
    idx = os.path.join(d, "model.safetensors.index.json")
    idx_keys = []
    if os.path.exists(idx):
        idx_keys = [k for k in json.load(open(idx))["weight_map"] if "mtp" in k.lower()]
    cfg = {}
    try:
        c = json.load(open(os.path.join(d, "config.json")))
        def walk(o, p=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    if "mtp" in k.lower() and not isinstance(v, (dict, list)):
                        cfg[p + k] = v
                    walk(v, p + k + ".")
        walk(c)
        ig = json.dumps(c.get("quantization_config", {}).get("ignore", []))
        cfg["quantization_ignore_mentions_mtp"] = "mtp" in ig.lower()
        cfg["architectures"] = c.get("architectures") or c.get("text_config", {}).get("architectures")
    except Exception as e:
        cfg["error"] = str(e)
    print(json.dumps({"model": os.path.relpath(d, "/mnt/data/models"), "mtp_tensors_in_headers": len(names),
                      "mtp_keys_in_index": len(idx_keys), "dtypes": dtypes, "sample": names[:6], "config_mtp": cfg}))
PY
  tail -n 2 "$OUT/mtp-check.jsonl" | cut -c1-600 | tee -a "$OUT/chain.log"
  local n
  n=$(mtp_tensors)
  if [ "${n:-0}" -gt 0 ]; then log "MTP: W8A8 checkpoint HAS $n mtp tensors"; else log "MTP: W8A8 checkpoint has NO mtp tensors; MTP steps will be skipped"; fi
}
mtp_tensors() { jq -r --arg m "$W8" 'select(.model == $m) | .mtp_tensors_in_headers' "$OUT/mtp-check.jsonl" 2>/dev/null | tail -n 1; }

correct_one() {   # SHORT REPO VERBOSE(1 = ONEDNN_VERBOSE AMX check on the first attempt)
  local short=$1 model=$2 verbose=$3 attempt n0 n1
  for attempt in 1 2; do
    n0=$(cat "$OUT/vllm-correct.jsonl" 2>/dev/null | wc -l)
    local envs=(MODEL="$model" TAG="$short-compiled" ARGS="--no-enforce-eager" KV_GB=16 IMAGE="$IMAGE")
    [ "$attempt" = 1 ] && [ -n "$verbose" ] && envs+=(ONEDNN_VERBOSE=1)
    log "correctness $short attempt $attempt${verbose:+ (ONEDNN_VERBOSE on attempt 1)}"
    run_cmd timeout -k 60 2700 env "${envs[@]}" bash "$B/vllm_correct.sh" 2>&1 | tee -a "$OUT/chain.log"
    kill_containers
    n1=$(cat "$OUT/vllm-correct.jsonl" 2>/dev/null | wc -l)
    [ -n "$DRY" ] && return 0
    if [ $((n1 - n0)) -eq 3 ]; then log "correctness $short OK (3 answers in vllm-correct.jsonl)"; return 0; fi
    log "correctness $short attempt $attempt gave $((n1 - n0)) rows"
  done
  BAD[$short]=1
  log "correctness $short FAILED twice: its benchmarks are skipped"
  return 1
}
step_correct() {
  local m rc=0
  for m in $CORRECT_MODELS; do
    case "$m" in
      w8a8) correct_one w8a8 "$W8" 1 || rc=1 ;;     # AMX check (oneDNN verbose) on W8A8 only
      w4a16) correct_one w4a16 "$W4" "" || rc=1 ;;   # verbose throttles W4A16 (millions of tiny ukernels)
    esac
    save
  done
  return $rc
}

step_w8a8_n16()      { bench "$W8" 16 t8 16 "$ALL3"; }
step_w8a8_n32()      { bench "$W8" 32 t8 16 "$ALL3"; }
step_w8a8_n64()      { bench "$W8" 64 t8 24 "$ALL3"; }
step_w4a16_n16()     { bench "$W4" 16 t8 16 "$ALL3"; }
step_w4a16_n64()     { bench "$W4" 64 t8 24 "$ALL3"; }
step_w4a16_mixed_n64() { bench "$W4" 64 t8 24 "mixed 512 128"; }   # E18
step_w8a8_t16_n64()  { bench "$W8" 64 t16 24 "prefill 512 1
mixed 512 128" "$BIND16"; }
# EXTRA reaches `vllm bench throughput` through an unquoted ${EXTRA} in vllm_bench.sh, so it is split on
# spaces and quotes are NOT removed: the JSON must contain no spaces and no wrapping quotes.
step_mtp_n16()  { mtp_guard || return 1; bench "$W8" 16 mtp1 16 "decode 32 256
mixed 512 128" "" "--speculative-config $MTP_JSON1"; local rc=$?; mtp_after; return $rc; }
step_mtp_n64()  { mtp_guard || return 1; bench "$W8" 64 mtp1 24 "decode 32 256
mixed 512 128" "" "--speculative-config $MTP_JSON1"; }
step_mtp2_n16() { mtp_guard || return 1; bench "$W8" 16 mtp2 16 "decode 32 256" "" "--speculative-config $MTP_JSON2"; }
mtp_guard() {
  [ -n "$DRY" ] && return 0
  if [ "$(mtp_tensors || echo 0)" -le 0 ] 2>/dev/null; then log "SKIP MTP step: no MTP tensors in the W8A8 checkpoint (see mtp-check.jsonl)"; return 1; fi
  if [ -n "$MTP_BROKEN" ]; then log "SKIP MTP step: the first MTP step failed on every run (see the errors above)"; return 1; fi
}
mtp_after() {   # first MTP step: if no run produced a result row, mark MTP as broken
  [ -n "$DRY" ] && return 0
  if ! jq -e 'select(.run == "mtp1" and .failed != true)' "$OUT/vllm.jsonl" >/dev/null 2>&1; then MTP_BROKEN=1; fi
}
step_smoke_bench() { bench "$W8" 4 smoke 16 "mixed 128 16"; }
step_smoke_mtp()   { mtp_guard || return 1; bench "$W8" 4 smoke-mtp 16 "decode 32 16" "" "--speculative-config $MTP_JSON1"; }

step_final() {
  {
    echo "=== summary (vllm.jsonl, non-failed rows)"
    jq -r 'select(.failed != true) | [.model|sub("Avesed/Qwen3.8-27B-";""), .run, .workload, .num_prompts,
        (.elapsed_time*10|floor/10), (.tokens_per_second*10|floor/10), ((.num_prompts*.output_len/.elapsed_time)*10|floor/10), .peak_rss_gb] | @tsv' \
      "$OUT/vllm.jsonl" 2>/dev/null | sort -k1,1 -k2,2 -k4,4n -k3,3 | awk 'BEGIN{print "model\trun\twl\tn\twall_s\ttot_tok/s\tout_tok/s\trss_gb"}{print}'
    echo "failed rows: $(failed_rows)"
    echo "=== image / versions"; cat "$OUT/vllm-image.txt" 2>/dev/null | cut -c1-250
    docker image inspect "$IMAGE" --format '{{.Id}} {{.RepoDigests}}' 2>/dev/null
    cat "$OUT/model-revisions.txt" 2>/dev/null
  } 2>&1 | tee -a "$OUT/chain.log"
  return 0
}

# ---- driver ---------------------------------------------------------------------------------------
t_start=$SECONDS
log "CHAIN START profile=$PROFILE host=$(hostname) steps: $STEPS"
if [ -z "$DRY" ]; then
  log "waiting for cloud-init (setup-done marker)"
  for i in $(seq 1 540); do [ -f "$STATE/setup-done" ] && break; sleep 10; done
  log "cloud-init setup-done: $(cat "$STATE/setup-done" 2>/dev/null || echo 'MISSING after 90 min') (0 = ok)"
  pgrep -a -f 'vllm|llama-' | grep -v vllm_chain | head -n 3
fi

for s in $STEPS; do
  if [ -e "$DONE/$s" ] && [ -z "$FORCE" ] && [ "$PROFILE" != smoke ]; then log "step $s: already done, skipping (FORCE=1 to redo)"; continue; fi
  if ! declare -F "step_$s" >/dev/null; then log "step $s: unknown, skipping"; continue; fi
  log "STEP $s START"
  t0=$SECONDS
  "step_$s"; rc=$?
  log "STEP $s END rc=$rc ($((SECONDS - t0))s)"
  [ "$rc" -eq 0 ] && [ -z "$DRY" ] && touch "$DONE/$s"
  save
done
log "CHAIN DONE in $((SECONDS - t_start))s"
save
exit 0
