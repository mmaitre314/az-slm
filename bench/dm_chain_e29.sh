#!/bin/bash
# E29 chain: llama.cpp /v1/systemone with ggml-org decision GGUFs. No agent in the loop.
# setup -> per model: correctness (AMX vs no-AMX, 1 vs 8 slots) -> quality -> throughput.
# Results: /mnt/data/results (chain.log, *.jsonl, *.txt); saved after every step with save_results.sh.
set -uo pipefail
R=/mnt/data/results; D=/mnt/data/dm; B=/opt/azslm/bench; L=/opt/llama.cpp
BIN_AMX=$L/build/bin; BIN_NOAMX=$L/build-native-noamx/bin
PORT=8080; URL=http://127.0.0.1:$PORT
mkdir -p $R $D/models $D/data
LOG=$R/chain.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
save() { (cd $R && bash $B/save_results.sh chain.log '*.jsonl' '*.txt' '*.json' '*.env' >/dev/null 2>&1); }
killsrv() { pkill -9 -f llama-server 2>/dev/null; sleep 2; }
step() { # step NAME TIMEOUT_SECONDS command...
  local name=$1 to=$2; shift 2
  log "STEP $name start (timeout ${to}s)"
  timeout -k 20 "$to" bash -c '"$@"' _ "$@" >>$R/step_$name.log 2>&1; local rc=$?
  killsrv; log "STEP $name end rc=$rc"; save
  return 0
}

# serve BIN MODEL NP THREADS CTXPER  (starts llama-server in the background, waits for /health)
serve() {
  local bin=$1 model=$2 np=$3 t=$4 ctx=$5
  killsrv
  setsid nohup $bin/llama-server -m $D/models/$model.gguf --host 127.0.0.1 --port $PORT -np $np -t $t -tb $t \
    -c $((np*ctx)) --no-webui >$R/server_${model}.log 2>&1 < /dev/null &
  for i in $(seq 1 240); do
    curl -sf $URL/health >/dev/null 2>&1 && return 0
    pgrep -f llama-server >/dev/null || { echo "server died"; return 1; }
    sleep 3
  done
  echo "server timeout"; return 1
}
# load MODELNAME TAG ITEMS CONCURRENCY LIMIT PREDS SUMMARY
load() {
  python3 $B/dm_load_e29.py --url $URL --model "$1" --tag "$2" --items "$3" --concurrency "$4" --limit "$5" \
    --preds "$6" --summary "$7"
}

# ---- model table: name|file|ctx per slot|quality limit banking77_8|throughput items|tag
MODELS=(   # name|file|ctx per slot|banking77_8 items|throughput items|jev-bench items per set|banking77_77_small items|correctness items
 "Kev-4B|Kev-4B-Q8_0|4096|800|160|40|60|100"
 "Laya|Laya-BF16|512|3080|600|100|100|100"
 "Kev-0.8B|Kev-0.8B-Q8_0|4096|3080|600|100|100|100"
 "Kev-9B|Kev-9B-Q8_0|4096|300|60|25|40|50"
 "OpenJev|OpenJev-Q8_0|4096|60|12|8|15|24"
)

setup() {
  cd $L && git log -1 --format='llama.cpp %H' | tee $R/versions.txt
  if [ ! -x $BIN_NOAMX/llama-server ]; then
    bash $B/build_native_noamx.sh >$R/build_noamx.log 2>&1
    cmake --build $L/build-native-noamx -j16 --target llama-server >>$R/build_noamx.log 2>&1
  fi
  ls -la $BIN_NOAMX/llama-server $BIN_AMX/llama-server
  for e in "Kev-4B-GGUF Kev-4B-Q8_0" "Laya-GGUF Laya-BF16" "OpenJev-GGUF OpenJev-Q8_0" "Kev-0.8B-GGUF Kev-0.8B-Q8_0" "Kev-9B-GGUF Kev-9B-Q8_0"; do
    set -- $e
    sha=$(curl -s https://huggingface.co/api/models/ggml-org/$1 | python3 -c "import sys,json;print(json.load(sys.stdin).get('sha'))")
    echo "ggml-org/$1 revision $sha file $2.gguf" >>$R/versions.txt
    if [ ! -s $D/models/$2.gguf ] || [ -f $D/models/$2.part ]; then
      touch $D/models/$2.part
      curl -sSL -f -o $D/models/$2.gguf https://huggingface.co/ggml-org/$1/resolve/main/$2.gguf && rm -f $D/models/$2.part
    fi
  done
  ls -la $D/models
  [ -s $D/data/banking77_8.jsonl ] || python3 $B/dm_data.py
  python3 $B/dm_jevbench_e29.py --n 200 --configs banking77,clinc150,massive,go_emotions,mmlu,ledgar,boolq,helpsteer2_helpfulness
  head -100 $D/data/banking77_8.jsonl > $D/data/banking77_8_100.jsonl
  wc -l $D/data/*.jsonl
  cp $R/versions.txt $R/versions_e29.txt; nproc; free -g | head -2
  top -b -n1 | head -15
}

# correctness MODELNAME FILE CTX N : 100 items, AMX vs no-AMX, 1 vs 8 slots. Writes ok_<name>.env (BUILD, MAXPAR)
correct() {
  local name=$1 file=$2 ctx=$3 cn=$4 items=$D/data/banking77_8_100.jsonl
  local c=$R/corr_$name; rm -f $c.*.preds.jsonl
  for b in amx noamx; do
    bin=$BIN_AMX; [ $b = noamx ] && bin=$BIN_NOAMX
    for np in 1 8; do
      if serve $bin $file $np 8 $ctx; then
        load $name "$name-$b-np$np" $items $np $cn $c.$b-np$np.preds.jsonl $R/corr_summary.jsonl
      else echo "serve failed $b np$np"; fi
      killsrv
    done
  done
  : > $R/corr_$name.jsonl
  cmpf() { python3 $B/dm_compare_e29.py $c.$1.preds.jsonl $c.$2.preds.jsonl --label "$name $1 vs $2" >>$R/corr_$name.jsonl; echo $?; }
  r_amx=$(cmpf amx-np8 amx-np1); r_nox=$(cmpf noamx-np8 noamx-np1)
  cmpf amx-np1 noamx-np1 >/dev/null; cmpf amx-np8 noamx-np8 >/dev/null
  # the build must agree with itself at 1 vs 8 slots; and the single-slot results must agree across builds
  s=$(python3 $B/dm_compare_e29.py $c.amx-np1.preds.jsonl $c.noamx-np1.preds.jsonl --tol 1e-2 >/dev/null; echo $?)
  if [ "$r_amx" = 0 ]; then echo "BUILD=amx MAXPAR=8" >$R/ok_$name.env
  elif [ "$r_nox" = 0 ]; then echo "BUILD=noamx MAXPAR=8" >$R/ok_$name.env
  else echo "BUILD=amx MAXPAR=1" >$R/ok_$name.env; fi
  echo "single-slot AMX vs noAMX (tol 1e-2) rc=$s"; cat $R/corr_$name.jsonl $R/ok_$name.env
  cp $R/ok_$name.env $R/ok_$name.txt
}

# quality NAME FILE CTX LIMIT_B77_8 : uses the build/parallelism chosen by correct()
quality() {
  local name=$1 file=$2 ctx=$3 lim=$4
  . $R/ok_$name.env; bin=$BIN_AMX; [ "$BUILD" = noamx ] && bin=$BIN_NOAMX
  local np=$MAXPAR; [ $np -gt 8 ] && np=8
  serve $bin $file $np 8 $ctx || return 1
  local q=$R/q_$name; rm -f $q.*.preds.jsonl
  load $name "$name" $D/data/banking77_8.jsonl $np $lim $q.banking77_8.preds.jsonl $R/q_summary.jsonl
  python3 $B/dm_score.py $q.banking77_8.preds.jsonl >$R/score_${name}_banking77_8.txt; cat $R/score_${name}_banking77_8.txt; save
  load $name "$name" $D/data/banking77_77_small.jsonl $np $Q77 $q.banking77_77_small.preds.jsonl $R/q_summary.jsonl
  python3 $B/dm_score.py $q.banking77_77_small.preds.jsonl >$R/score_${name}_banking77_77_small.txt; cat $R/score_${name}_banking77_77_small.txt; save
  for ds in banking77 clinc150 massive go_emotions mmlu ledgar boolq helpsteer2_helpfulness; do
    [ -s $D/data/jb_$ds.jsonl ] || continue
    local jl=$QJL; [ $ds = helpsteer2_helpfulness ] && jl=$((QJL*2/3))
    load $name "$name" $D/data/jb_$ds.jsonl $np $jl $q.jb_$ds.preds.jsonl $R/q_summary.jsonl
    python3 $B/dm_score.py $q.jb_$ds.preds.jsonl >$R/score_${name}_jb_$ds.txt; cat $R/score_${name}_jb_$ds.txt; save
  done
}

# throughput NAME FILE CTX N : -np = concurrency, -t 8; plus -t 16 at c=16
throughput() {
  local name=$1 file=$2 ctx=$3 n=$4
  . $R/ok_$name.env; bin=$BIN_AMX; [ "$BUILD" = noamx ] && bin=$BIN_NOAMX
  local items=$D/data/banking77_8.jsonl
  for cfg in "1 8" "4 8" "16 8" "16 16"; do
    set -- $cfg; local c=$1 t=$2
    [ $c -gt $MAXPAR ] && { log "skip c=$c ($name has MAXPAR=$MAXPAR)"; continue; }
    serve $bin $file $c $t $ctx || { log "serve failed c=$c t=$t"; continue; }
    load $name "$name" $items $c 12 /dev/null /dev/null >/dev/null 2>&1   # warmup
    load $name "$name-c$c-t$t-$BUILD" $items $c $n $R/tp_$name-c$c-t$t.preds.jsonl $R/tp_summary.jsonl
    killsrv; save
  done
}

export R D B L BIN_AMX BIN_NOAMX PORT URL LOG
export -f setup correct quality throughput serve load killsrv log save
log "chain start"
step setup 3600 setup
for m in "${MODELS[@]}"; do
  IFS='|' read -r name file ctx qlim tpn QJL Q77 CN <<<"$m"; export QJL Q77
  if [ ! -s $D/models/$file.gguf ]; then log "SKIP $name: no model file"; continue; fi
  step correct_$name 1800 correct $name $file $ctx $CN
  [ -f $R/ok_$name.env ] || echo "BUILD=amx MAXPAR=1" >$R/ok_$name.env
  step quality_$name 4200 quality $name $file $ctx $qlim
  step throughput_$name 3000 throughput $name $file $ctx $tpn
done
log "chain done"
save
