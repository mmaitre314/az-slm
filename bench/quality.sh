#!/bin/bash
# Quantization quality vs BF16: KL divergence and top-token agreement on wikitext-2 (llama-perplexity).
# Writes /mnt/data/results/kld-<quant>.log and appends a summary line per quant to quality.jsonl.
# Env: QUANTS, CHUNKS (512-token chunks, default 16), THREADS, BUILD (default build-noamx: llama-perplexity
# evaluates several sequences per batch, which hits llama.cpp's AMX multi-sequence bug on this model;
# weights and math are identical, so quantization quality does not depend on AMX).
set -uo pipefail
MODELS=/mnt/data/models/bartowski/Qwen3.8-27B-GGUF
OUT=/mnt/data/results
QUANTS=${QUANTS:-"Q4_0 IQ4_XS Q4_K_M Q5_K_M Q6_K Q8_0"}
CHUNKS=${CHUNKS:-16}
THREADS=${THREADS:-8}
BIN=/opt/llama.cpp/${BUILD:-build-noamx}/bin
data=/mnt/data/wikitext-2-raw/wiki.test.raw
if [ ! -f "$data" ]; then
  curl -fsSL -o /mnt/data/wikitext-2-raw-v1.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
  python3 -m zipfile -e /mnt/data/wikitext-2-raw-v1.zip /mnt/data/
fi
base=$OUT/kld-base-bf16-c$CHUNKS.bin
if [ ! -f "$base" ]; then
  echo "== bf16 base logits $(date -u +%T)"
  $BIN/llama-perplexity -m "$(ls "$MODELS"/Qwen3.8-27B-bf16/*-00001-of-*.gguf)" -f "$data" -c 512 --chunks "$CHUNKS" \
    -t "$THREADS" --kl-divergence-base "$base" > "$OUT/kld-bf16.log" 2>&1
  grep -E 'Final estimate' "$OUT/kld-bf16.log"
fi
for q in $QUANTS; do
  echo "== $q $(date -u +%T)"
  log=$OUT/kld-$q.log
  $BIN/llama-perplexity -m "$MODELS/Qwen3.8-27B-$q.gguf" --kl-divergence-base "$base" --kl-divergence \
    -t "$THREADS" > "$log" 2>&1
  python3 - "$q" "$log" >> "$OUT/quality.jsonl" <<'EOF'
import json, re, sys
q, text = sys.argv[1], open(sys.argv[2]).read()
def grab(label):
    m = re.search(rf"^{re.escape(label)}\s*:\s*([-0-9.]+)", text, re.M)
    return float(m.group(1)) if m else None
print(json.dumps({"quant": q, "ppl_q": grab("Mean PPL(Q)"), "ppl_base": grab("Mean PPL(base)"),
                  "ln_ppl_ratio": grab("Mean ln(PPL(Q)/PPL(base))"), "kld_mean": grab("Mean    KLD"),
                  "kld_p99": grab("99.0%   KLD"), "same_top_p": grab("Same top p")}))
EOF
  tail -n 1 "$OUT/quality.jsonl"
done
echo "done $(date -u +%T)"
