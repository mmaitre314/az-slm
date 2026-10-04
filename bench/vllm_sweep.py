#!/usr/bin/env python3
"""In-process vLLM CPU sweep: load the model once, then time batches of random-token prompts.

Runs inside the vLLM container (entrypoint python3). For every (workload, batch size) it does a
short warm-up generate of the same shape (same prompt length, 2 new tokens: primes oneDNN
primitives and the compile cache for that M), then one timed `llm.generate` of `batch` requests
with `ignore_eos` and the full output length, and appends a JSON line to $OUT (default
/mnt/data/results/vllm-sweep.jsonl). The max/mean time-to-first-token of the batch splits the
wall time into prefill (until the last request has its first token) and the rest (decode).

usage: vllm_sweep.py MODEL_PATH [--run t8] [--batches 1,4,16,32]
                     [--workloads prefill:512:1,decode:32:256,mixed:512:128] [--extra '{...}']
"""
import argparse
import json
import os
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--run", default="t8", help="label stored in each row (thread configuration)")
    ap.add_argument("--batches", default="1,4,16,32")
    ap.add_argument("--workloads", default="prefill:512:1,decode:32:256,mixed:512:128",
                    help="name:input_len:output_len, comma separated")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--extra", default="{}", help="JSON dict of extra LLM() kwargs")
    args = ap.parse_args()

    import vllm  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402

    out_path = os.environ.get("OUT", "/mnt/data/results/vllm-sweep.jsonl")
    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              limit_mm_per_prompt={"image": 0, "video": 0}, disable_log_stats=False,
              **json.loads(args.extra))
    load_s = time.time() - t0
    print(f"loaded in {load_s:.0f}s", flush=True)
    rng = random.Random(0)
    warm_sp = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)

    def make(n, length):
        return [{"prompt_token_ids": [rng.randrange(1000, 100000) for _ in range(length)]} for _ in range(n)]

    if not args.no_warmup:  # lazy initialisation (weight pages, thread pool, first oneDNN primitives)
        t = time.perf_counter()
        llm.generate([{"prompt_token_ids": [rng.randrange(1000, 100000) for _ in range(512)]} for _ in range(4)],
                     warm_sp, use_tqdm=False)
        print(f"global warm-up (4 x 512 in, 8 out) took {time.perf_counter() - t:.1f}s", flush=True)

    for spec in args.workloads.split(","):
        name, in_len, out_len = spec.split(":")
        in_len, out_len = int(in_len), int(out_len)
        for batch in [int(b) for b in args.batches.split(",")]:
            if not args.no_warmup:
                llm.generate(make(batch, min(in_len, 64)),
                             SamplingParams(temperature=0, max_tokens=2, ignore_eos=True), use_tqdm=False)
            sp = SamplingParams(temperature=0, max_tokens=out_len, ignore_eos=True)
            prompts = make(batch, in_len)
            t = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            dt = time.perf_counter() - t
            n_in = sum(len(o.prompt_token_ids) for o in outs)
            n_out = sum(len(o.outputs[0].token_ids) for o in outs)
            ttfts = []
            for o in outs:
                m = getattr(o, "metrics", None)
                lat = getattr(m, "first_token_latency", None) if m is not None else None
                if lat:
                    ttfts.append(lat)
            row = {"model": args.model, "workload": name, "run": args.run, "num_prompts": batch,
                   "input_len": in_len, "output_len": out_len, "vllm": vllm.__version__,
                   "elapsed_time": round(dt, 3), "total_input_tokens": n_in, "total_output_tokens": n_out,
                   "tokens_per_second": round((n_in + n_out) / dt, 2),
                   "load_seconds": round(load_s, 1)}
            if ttfts:
                row["ttft_max_s"] = round(max(ttfts), 3)
                row["ttft_mean_s"] = round(sum(ttfts) / len(ttfts), 3)
                if out_len > 1:  # decode phase = wall time after the last request's first token
                    row["decode_phase_s"] = round(dt - max(ttfts), 3)
                    row["decode_tok_per_s_after_prefill"] = round(
                        (n_out - batch) / max(dt - max(ttfts), 1e-9), 2)
                row["prefill_tok_per_s_by_ttft"] = round(n_in / max(ttfts), 2)
            print(json.dumps(row), flush=True)
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
    try:  # cgroup v2 peak memory of the container (includes page cache of the weights)
        print("container_peak_gb", round(int(open("/sys/fs/cgroup/memory.peak").read()) / 1e9, 1), flush=True)
    except OSError:
        pass


if __name__ == "__main__":  # required: vLLM spawns the engine process
    main()
