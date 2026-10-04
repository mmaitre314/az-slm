#!/bin/bash
# One speculative-decoding configuration: start llama-server with the given spec flags, send a prompt
# set with bench/spec_client.py, stop the server, and append rows to /mnt/data/results/spec.jsonl.
# Server logs go to /mnt/data/results/spec-logs/<TAG>.log.
# Env:
#   TAG        label for this configuration (required)
#   BUILD      llama.cpp build dir name under /opt/llama.cpp (default build; use build-native-noamx for NP>1)
#   MODEL      path to the target GGUF (default Q4_K_M)
#   SPEC_ARGS  extra llama-server flags, e.g. "--spec-type draft-mtp --spec-draft-n-max 3"
#   NP         server slots (default 1)
#   SETS       prompt sets for the client, run in order on one server: std3 | ten | sixteen (default "ten")
#   INFLIGHT   concurrent requests (default NP)
#   COUNT      number of requests (default: size of the set)
#   MAXTOK     override max_tokens for all prompts (default: per-prompt values)
#   CTX        context per slot (default 1024)
#   THREADS    threads (default 8)
#   OUT        results file (default /mnt/data/results/spec.jsonl)
set -uo pipefail
: "${TAG:?set TAG}"
BUILD=${BUILD:-build}
MODEL=${MODEL:-/mnt/data/models/bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf}
SPEC_ARGS=${SPEC_ARGS:-}
NP=${NP:-1}
SETS=${SETS:-${SET:-ten}}
INFLIGHT=${INFLIGHT:-$NP}
CTX=${CTX:-1024}
THREADS=${THREADS:-8}
OUT=${OUT:-/mnt/data/results/spec.jsonl}
PORT=${PORT:-8080}
LOGDIR=/mnt/data/results/spec-logs
mkdir -p "$LOGDIR"
log=$LOGDIR/$TAG.log
pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 1

echo "== $TAG build=$BUILD np=$NP sets=$SETS inflight=$INFLIGHT spec='$SPEC_ARGS' $(date -u +%T)"
# shellcheck disable=SC2086
/opt/llama.cpp/$BUILD/bin/llama-server -m "$MODEL" -c $((NP * CTX)) -np "$NP" -t "$THREADS" -tb "$THREADS" \
  --host 127.0.0.1 --port "$PORT" --no-webui $SPEC_ARGS > "$log" 2>&1 &
spid=$!
ok=0
for i in $(seq 1 360); do
  if ! kill -0 $spid 2>/dev/null; then echo "server exited early"; tail -n 25 "$log"; exit 2; fi
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then ok=1; break; fi
  sleep 2
done
[ $ok = 1 ] || { echo "server did not come up"; tail -n 25 "$log"; kill $spid; exit 3; }
echo "server up after ~$((i * 2)) s"

meta=$(jq -nc --arg b "$BUILD" --arg m "$(basename "$MODEL")" --arg s "$SPEC_ARGS" --argjson np "$NP" \
  '{build: $b, model: $m, spec_args: $s, np: $np}')
rc=0
for set in $SETS; do
  args=(--url "http://127.0.0.1:$PORT" --tag "$TAG" --set "$set" --n "$INFLIGHT" --out "$OUT" --meta "$meta")
  [ -n "${COUNT:-}" ] && args+=(--count "$COUNT")
  [ -n "${MAXTOK:-}" ] && args+=(--max-tokens "$MAXTOK")
  python3 /opt/azslm/bench/spec_client.py "${args[@]}"
  r=$?; [ $r -ne 0 ] && rc=$r
done

# server-side speculation statistics, if the build prints them at shutdown
kill $spid 2>/dev/null; wait $spid 2>/dev/null
grep -iE "draft acceptance|accept(ed|ance) rate|statistics|n_drafted|n_accept" "$log" | tail -n 6 | cut -c1-220
echo "client exit $rc, done $(date -u +%T)"
