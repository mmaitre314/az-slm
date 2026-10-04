#!/usr/bin/env python3
"""Offline batch throughput of an OpenVINO GenAI continuous-batching pipeline (E10).

For each workload (name:input_len:output_len) and batch size N, submits N random-token prompts of
exactly input_len tokens in one generate() call (greedy, ignore_eos, output_len new tokens) and
times the whole call. Appends one JSON line per measurement to $OUT (default
/mnt/data/results/openvino.jsonl). Same workloads as E09 (vllm_bench.sh).

usage: openvino_bench.py MODEL_DIR --tag TAG [--workloads prefill:512:1,decode:32:256,mixed:512:128]
       [--batches 1,4,16,32] [--reps 2] [--props JSON] [--max-num-seqs N] [--cache-gb N]
       [--max-batched-tokens N] [--prefix-caching] [--shared-prefix L] [--note TEXT]
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openvino_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--workloads", default="prefill:512:1,decode:32:256,mixed:512:128")
ap.add_argument("--batches", default="1,4,16,32")
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--slow-s", type=float, default=150.0, help="skip further reps when one rep took longer")
ap.add_argument("--props", default="{}", help="JSON dict of plugin properties for the pipeline")
ap.add_argument("--max-num-seqs", type=int, default=32)
ap.add_argument("--cache-gb", type=int, default=24)
ap.add_argument("--max-batched-tokens", type=int, default=0, help="0 = GenAI default (256)")
ap.add_argument("--prefix-caching", action="store_true")
ap.add_argument("--no-split-fuse", action="store_true")
ap.add_argument("--pipe", default="cb")
ap.add_argument("--note", default="")
ap.add_argument("--warmup", default="full", help="full | quick | none")
args = ap.parse_args()

import openvino_genai as g  # noqa: E402

sched = C.make_scheduler(args.max_num_seqs, args.cache_gb, args.prefix_caching,
                         args.max_batched_tokens or None, not args.no_split_fuse)
props = json.loads(args.props)
pipe, kind, load_s = C.load_pipe(args.model, args.pipe, None if args.pipe == "vlm" else sched, props)
print(f"loaded with {kind} in {load_s:.1f}s rss {C.read_proc_status('VmRSS')} GB; {sched.to_string()!r}", flush=True)
tok = pipe.get_tokenizer()
out_path = os.environ.get("OUT", "/mnt/data/results/openvino.jsonl")
meta = {"tag": args.tag, "model": os.path.basename(args.model.rstrip("/")), "pipe": kind, "props": props,
        "max_num_seqs": args.max_num_seqs, "cache_gb": args.cache_gb,
        "max_batched_tokens": args.max_batched_tokens or 256, "prefix_caching": args.prefix_caching,
        "dynamic_split_fuse": not args.no_split_fuse, "note": args.note, **C.env_info()}

rng = random.Random(1234)


def build_pool(size=3000):
    """Words that are exactly one token with a leading space, so that n words = n tokens."""
    pool = []
    cand = list(range(2000, 100000))
    rng.shuffle(cand)
    for tid in cand:
        w = tok.decode([tid])
        if len(w) > 2 and w[0] == " " and w[1:].isascii() and w[1:].isalpha():
            if tok.encode(w).input_ids.data.shape[-1] == 1:
                pool.append(w)
                if len(pool) >= size:
                    break
    return pool


POOL = build_pool()
print(f"word pool: {len(POOL)} one-token words, e.g. {POOL[:5]}", flush=True)


def make_ids(n):
    return [rng.choice(POOL) for _ in range(n)]


def tensor(words):
    return "".join(words)


def cfg_for(out_len):
    c = g.GenerationConfig()
    c.max_new_tokens = out_len
    c.min_new_tokens = out_len
    c.ignore_eos = True
    c.do_sample = False
    c.apply_chat_template = False
    return c


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run_once(n, in_len, out_len, shared_prefix=0):
    prefix = make_ids(shared_prefix) if shared_prefix else []
    prompts = [prefix + make_ids(in_len - shared_prefix) for _ in range(n)]
    inputs = [tensor(p) for p in prompts]
    n_tok = tok.encode(inputs[0]).input_ids.data.shape[-1]
    assert n_tok == in_len, f"prompt has {n_tok} tokens, wanted {in_len}"
    cfgs = [cfg_for(out_len) for _ in range(n)]
    load1 = os.getloadavg()[0]
    t0 = time.perf_counter()
    if kind == "vlm":  # stateful (non-batching) VLM pipeline: one prompt after the other
        res = [pipe.generate(x, generation_config=c) for x, c in zip(inputs, cfgs)]
    else:
        res = pipe.generate(inputs, cfgs)
    wall = time.perf_counter() - t0
    out_tokens = 0
    in_tokens_seen = 0
    ttfts, tpots = [], []
    for r in res:
        pm = getattr(r, "perf_metrics", None)
        if pm is not None:
            try:
                out_tokens += pm.get_num_generated_tokens()
                in_tokens_seen += pm.get_num_input_tokens()
                ttfts.append(pm.get_ttft().mean / 1000.0)  # ms -> s
                tpots.append(pm.get_tpot().mean / 1000.0)
            except Exception:  # noqa: BLE001
                pass
    if not out_tokens:
        out_tokens = n * out_len  # fallback: ignore_eos + min_new_tokens guarantee it
    row = {"n": n, "input_len": in_len, "output_len": out_len, "shared_prefix": shared_prefix,
           "wall_s": round(wall, 3), "in_tokens": n * in_len, "in_tokens_seen": in_tokens_seen, "out_tokens": out_tokens,
           "total_tok_s": round((n * in_len + out_tokens) / wall, 2),
           "in_tok_s": round(n * in_len / wall, 2), "out_tok_s": round(out_tokens / wall, 2),
           "ttft_mean_s": round(mean(ttfts), 3) if ttfts else None,
           "ttft_max_s": round(max(ttfts), 3) if ttfts else None,
           "tpot_mean_s": round(mean(tpots), 4) if tpots else None,
           "loadavg1_before": round(load1, 2), "rss_gb": C.read_proc_status("VmRSS")}
    # prefill-only: everything is prefill. otherwise estimate decode rate after the last prefill finishes
    if out_len > 1 and ttfts:
        dec_time = wall - max(ttfts)
        if dec_time > 0:
            row["est_decode_tok_s"] = round(n * (out_len - 1) / dec_time, 2)
    if out_len == 1:
        row["prefill_tok_s"] = row["in_tok_s"]
    try:
        m = pipe.get_metrics()
        row["max_cache_usage"] = round(m.max_cache_usage, 2)
    except Exception:  # noqa: BLE001
        pass
    return row


def emit(row):
    row = {**meta, **row}
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in row if k not in meta}), flush=True)


# warm-up: loads the weights into page cache, creates kernels for typical shapes
t = time.time()
if args.warmup == "full":
    run_once(4, 512, 8)
    run_once(1, 32, 16)
elif args.warmup == "quick":
    run_once(1, 128, 4)
print(f"warmup {time.time() - t:.1f}s rss {C.read_proc_status('VmRSS')} GB", flush=True)

for wl in args.workloads.split(","):
    name, i, o = wl.split(":")
    i, o = int(i), int(o)
    shared = 0
    if name.startswith("shared"):  # e.g. shared256:512:128 -> 256-token shared prefix
        shared = int(name[len("shared"):])
    for n in [int(x) for x in args.batches.split(",")]:
        for rep in range(1, args.reps + 1):
            row = run_once(n, i, o, shared)
            row.update({"workload": name, "rep": rep})
            emit(row)
            if row["wall_s"] > args.slow_s:
                break
print("peak_rss_gb", C.peak_rss_gb(), flush=True)
print("done", time.strftime("%T"), flush=True)
