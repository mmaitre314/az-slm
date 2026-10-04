#!/bin/bash
# E11 driver: start OVMS (continuous batching, OpenAI API), run the correctness prompts, the
# concurrency sweep with 512-token prompts / 128 new tokens, and the shared-prefix test, then stop
# the server. Results: /mnt/data/results/ovms.jsonl, logs ovms-*.log.
# Env: MODEL, TAG (default ovms), PREFIX_CACHING (true|false, server option), NS (default "1 4 16 32"),
# ROUNDS_SMALL (rounds per client for N<=4, default 2), ROUNDS_LARGE (N>=16, default 1),
# STEPS (comma list of correct,load,shared; default all), SHARED_N (16), SHARED_PREFIX (256),
# MAX_SEQS, CACHE_GB, MBT, PLUGIN_CONFIG, EXTRA (passed to ovms_serve.sh), PROMPT_LEN (512), MAX_TOKENS (128).
set -uo pipefail
export MODEL=${MODEL:-OpenVINO/Qwen3.8-27B-int4-ov}
export TAG=${TAG:-ovms}
export PREFIX_CACHING=${PREFIX_CACHING:-true}
B=/opt/azslm/bench
R=/mnt/data/results
STEPS=${STEPS:-correct,load,shared}
mkdir -p $R
echo "== E11 $MODEL tag=$TAG prefix_caching=$PREFIX_CACHING steps=$STEPS $(date -u +%T)"
echo "-- top before starting (nothing else should use CPU):"; top -bn1 | head -12 | tail -6 | cut -c1-110

if [ ! -s $R/word_pool.json ]; then
/opt/ov/venv/bin/python - <<PY
import json, random
import openvino_genai as g
tok = g.Tokenizer("/mnt/data/models/$MODEL")
rng = random.Random(99)
cand = list(range(2000, 100000)); rng.shuffle(cand)
pool = []
for t in cand:
    w = tok.decode([t])
    if len(w) > 2 and w[0] == " " and w[1:].isascii() and w[1:].isalpha() and tok.encode(w).input_ids.data.shape[-1] == 1:
        pool.append(w)
        if len(pool) >= 3000: break
json.dump(pool, open("$R/word_pool.json", "w"))
print("pool", len(pool))
PY
fi

bash $B/ovms_serve.sh start || { echo "server failed to start"; bash $B/ovms_serve.sh stop; exit 1; }
CL="python3 $B/ovms_client.py"
# warm-up (not recorded in the results file under the main tag)
OUT=$R/ovms-warmup.jsonl $CL load --n 2 --in-tokens 128 --max-tokens 8 --tag warmup > /dev/null 2>&1
if [[ ",$STEPS," == *,correct,* ]]; then
  echo "-- correctness"; $CL correct --tag "$TAG" 2>&1 | cut -c1-500
fi
if [[ ",$STEPS," == *,load,* ]]; then
  for n in ${NS:-1 4 16 32}; do
    rounds=${ROUNDS_LARGE:-1}; [ "$n" -le 4 ] && rounds=${ROUNDS_SMALL:-2}
    echo "-- load n=$n rounds=$rounds $(date -u +%T)"; top -bn1 | head -8 | tail -2 | cut -c1-110
    $CL load --n "$n" --rounds "$rounds" --in-tokens "${PROMPT_LEN:-512}" --max-tokens "${MAX_TOKENS:-128}" --tag "$TAG" 2>&1 | tail -n 3 | cut -c1-700
  done
fi
if [[ ",$STEPS," == *,shared,* ]]; then
  echo "-- shared prefix n=${SHARED_N:-16} prefix=${SHARED_PREFIX:-256} $(date -u +%T)"
  $CL load --n "${SHARED_N:-16}" --shared-prefix "${SHARED_PREFIX:-256}" --prime --in-tokens "${PROMPT_LEN:-512}" --max-tokens "${MAX_TOKENS:-128}" --tag "$TAG-shared" 2>&1 | tail -n 3 | cut -c1-700
  echo "-- unique prompts n=${SHARED_N:-16} (control, same server) $(date -u +%T)"
  $CL load --n "${SHARED_N:-16}" --in-tokens "${PROMPT_LEN:-512}" --max-tokens "${MAX_TOKENS:-128}" --tag "$TAG-unique-control" 2>&1 | tail -n 3 | cut -c1-700
fi
docker stats --no-stream ovms 2>/dev/null | cut -c1-160
bash $B/ovms_serve.sh stop
echo "finished $(date -u +%T)"
