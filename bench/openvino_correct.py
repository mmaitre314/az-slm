#!/usr/bin/env python3
"""Correctness check for OpenVINO GenAI on a Qwen3.8-27B IR: 3 fixed prompts, greedy, 64 new tokens.

1. each prompt on its own (batch of 1), 2. all three prompts in one generate() call (continuous
batching, different prompt lengths) -- the batched texts must equal the single-prompt texts (the
llama.cpp AMX bug, E04, showed up exactly in this comparison). Appends JSON lines to $OUT.

usage: openvino_correct.py MODEL_DIR [--tag TAG] [--pipe auto|llm|vlm|cb] [--thinking]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openvino_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", default="")
ap.add_argument("--pipe", default="auto")
ap.add_argument("--thinking", action="store_true")
ap.add_argument("--max-new", type=int, default=64)
ap.add_argument("--props", default="{}", help="JSON dict of extra pipeline properties")
args = ap.parse_args()

import openvino_genai as g  # noqa: E402

sched = C.make_scheduler(max_num_seqs=8, cache_gb=8)
pipe, kind, load_s = C.load_pipe(args.model, args.pipe, sched, json.loads(args.props))
print(f"loaded with {kind} in {load_s:.1f}s, rss {C.read_proc_status('VmRSS')} GB", flush=True)
tok = pipe.get_tokenizer()
cfg = g.GenerationConfig()
cfg.max_new_tokens = args.max_new
cfg.do_sample = False
cfg.apply_chat_template = False
texts_in = [C.chat_prompt(tok, p, args.thinking) for p in C.PROMPTS]
print("templated prompt 1:", repr(texts_in[0]), flush=True)

# warm-up (first call compiles kernels / allocates the cache)
t = time.time()
pipe.generate([texts_in[0]], [cfg])
print(f"warmup {time.time() - t:.1f}s", flush=True)

rows = []
singles = []
for i, (p, tp) in enumerate(zip(C.PROMPTS, texts_in), 1):
    t = time.time()
    res = pipe.generate([tp], [cfg])
    dt = time.time() - t
    txt = C.gen_texts(res)[0]
    singles.append(txt)
    n_in = len(tok.encode(tp).input_ids.data[0])
    rows.append({"mode": "single", "prompt_id": i, "prompt": p, "text": txt, "n_prompt_tokens": n_in,
                 "seconds": round(dt, 2)})
t = time.time()
res = pipe.generate(texts_in, [cfg] * len(texts_in))
dt = time.time() - t
batched = C.gen_texts(res)
for i, (p, txt) in enumerate(zip(C.PROMPTS, batched), 1):
    rows.append({"mode": "batched3", "prompt_id": i, "prompt": p, "text": txt, "seconds": round(dt, 2),
                 "same_as_single": txt == singles[i - 1]})
# 12 copies of the three prompts: more sequences than the single-step batch can hold
res = pipe.generate(texts_in * 4, [cfg] * 12)
b12 = C.gen_texts(res)
ok12 = all(b12[j] == singles[j % 3] for j in range(12))
rows.append({"mode": "batched12", "all_same_as_single": ok12, "n_distinct": len(set(b12))})

meta = {"tag": args.tag, "model": args.model, "pipe": kind, "thinking": args.thinking,
        "load_seconds": round(load_s, 1), "peak_rss_gb": C.peak_rss_gb(), **C.env_info()}
out = os.environ.get("OUT", "/mnt/data/results/openvino-correct.jsonl")
with open(out, "a") as f:
    for r in rows:
        r = {**meta, **r}
        print(json.dumps(r, ensure_ascii=False)[:700], flush=True)
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("peak_rss_gb", C.peak_rss_gb(), flush=True)
