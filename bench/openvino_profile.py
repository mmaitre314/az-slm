#!/usr/bin/env python3
"""Per-node profile of the stateful Qwen3.8 language model IR on the OpenVINO CPU plugin (PERF_COUNT).

Compiles openvino_language_model.xml directly (no GenAI, no paged attention), runs one prefill of
T tokens with random embeddings and a few decode steps, and prints the time by node type and by
execution type (implementation names such as ..._amx show which kernels ran). Appends a JSON line
to $OUT (default /mnt/data/results/openvino-profile.jsonl).

usage: openvino_profile.py MODEL_DIR [--tokens 512] [--decode-steps 6] [--props JSON] [--tag T]
"""
import argparse
import collections
import json
import os
import time

import numpy as np
import openvino as ov

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tokens", type=int, default=512)
ap.add_argument("--decode-steps", type=int, default=6)
ap.add_argument("--props", default="{}")
ap.add_argument("--tag", default="")
args = ap.parse_args()

core = ov.Core()
props = {"PERF_COUNT": True, **json.loads(args.props)}
t0 = time.time()
cm = core.compile_model(os.path.join(args.model, "openvino_language_model.xml"), "CPU", props)
print(f"compile {time.time() - t0:.1f}s", flush=True)
req = cm.create_infer_request()
rng = np.random.default_rng(0)


def step(t, past):
    emb = (rng.standard_normal((1, t, 5120)) * 0.02).astype(np.float32)
    pos = np.broadcast_to(np.arange(past, past + t, dtype=np.int64)[None, None, :], (4, 1, t)).copy()
    t1 = time.perf_counter()
    req.infer({"inputs_embeds": emb, "attention_mask": np.ones((1, past + t), dtype=np.int64),
               "position_ids": pos, "beam_idx": np.array([0], dtype=np.int32)})
    return time.perf_counter() - t1


def summarize(label, secs, n_tokens):
    info = req.get_profiling_info()
    by_type, by_exec = collections.Counter(), collections.Counter()
    tot = 0.0
    for p in info:
        s = p.real_time.total_seconds() + 0.0
        tot += s
        by_type[p.node_type] += s
        by_exec[p.exec_type] += s
    print(f"== {label}: wall {secs:.2f}s for {n_tokens} tokens ({n_tokens / secs:.1f} tok/s); profiled node time {tot:.2f}s")
    rows = []
    for k, v in by_type.most_common(12):
        print(f"   type {k:28s} {v:8.3f}s {100 * v / tot:5.1f}%")
    for k, v in by_exec.most_common(12):
        print(f"   exec {k:40s} {v:8.3f}s {100 * v / tot:5.1f}%")
    return {"label": label, "wall_s": round(secs, 3), "tokens": n_tokens, "node_time_s": round(tot, 3),
            "by_type": {k: round(v, 4) for k, v in by_type.most_common(15)},
            "by_exec": {k: round(v, 4) for k, v in by_exec.most_common(15)}}


step(16, 0)  # warm-up (compiles kernels); resets state below
req.reset_state()
secs = step(args.tokens, 0)
pre = summarize(f"prefill {args.tokens}", secs, args.tokens)
dec = []
past = args.tokens
for i in range(args.decode_steps):
    dec.append(step(1, past))
    past += 1
dsum = summarize(f"decode (last of {args.decode_steps} steps)", dec[-1], 1)
row = {"tag": args.tag, "props": json.loads(args.props), "prefill": pre, "decode": dsum, "decode_step_s": [round(d, 3) for d in dec],
       "openvino": ov.__version__}
with open(os.environ.get("OUT", "/mnt/data/results/openvino-profile.jsonl"), "a") as f:
    f.write(json.dumps(row) + "\n")
