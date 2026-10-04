#!/usr/bin/env python3
"""Correctness check for the vLLM CPU stack: 3 fixed prompts, greedy, up to 64 new tokens.

Runs inside the vLLM container (entrypoint python3). Prints and appends one JSON line per prompt
to $OUT (default /mnt/data/results/vllm-correct.jsonl), including the raw text so a human can
compare it with llama.cpp's answers.

usage: vllm_correct.py MODEL_PATH [--tag TAG] [--thinking] [--kv-gb N]
"""
import argparse
import json
import os
import resource
import time

PROMPTS = [
    "List the three largest cities in France by population.",
    "Translate to German: The quick brown fox jumps over the lazy dog.",
    "What is 17 * 23? Answer with the number only.",
]

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", default="")
ap.add_argument("--thinking", action="store_true", help="leave the chat template's thinking on")
ap.add_argument("--max-model-len", type=int, default=2048)
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--extra", default="{}", help="JSON dict of extra LLM() kwargs")
args = ap.parse_args()

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

t0 = time.time()
llm = LLM(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len,
          limit_mm_per_prompt={"image": 0, "video": 0}, enforce_eager=True,
          **json.loads(args.extra))
load_s = time.time() - t0
sp = SamplingParams(temperature=0.0, max_tokens=64)
kw = {} if args.thinking else {"chat_template_kwargs": {"enable_thinking": False}}
rows = []
for i, p in enumerate(PROMPTS, 1):
    t = time.time()
    res = llm.chat([{"role": "user", "content": p}], sp, use_tqdm=False, **kw)[0]
    dt = time.time() - t
    o = res.outputs[0]
    row = {"tag": args.tag, "model": args.model, "vllm": vllm.__version__, "prompt_id": i, "prompt": p,
           "text": o.text, "n_prompt_tokens": len(res.prompt_token_ids), "n_out_tokens": len(o.token_ids),
           "finish_reason": o.finish_reason, "thinking": args.thinking, "seconds": round(dt, 2),
           "load_seconds": round(load_s, 1)}
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
print("peak_rss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1), flush=True)
out = os.environ.get("OUT", "/mnt/data/results/vllm-correct.jsonl")
with open(out, "a") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
