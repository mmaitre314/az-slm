#!/usr/bin/env python3
"""Load client for an OpenAI-compatible server (OVMS /v3/chat/completions), standard library only.

  ovms_client.py correct --tag T            3 fixed prompts, greedy, 64 tokens (thinking off)
  ovms_client.py load --n 16 --tag T        N concurrent clients, each sends --rounds requests back to back
        [--in-tokens 512] [--max-tokens 128] [--shared-prefix 256 --prime] [--pool pool.json]

Prompts are made of one-token words (pool from the model's tokenizer, see ovms_bench.sh) and
calibrated against the server's own prompt_tokens, so a request has --in-tokens prompt tokens
including the chat template. Streams the response to get time-to-first-token and the final usage
block. Appends one JSON line per run to $OUT (default /mnt/data/results/ovms.jsonl).
"""
import argparse
import http.client
import json
import os
import random
import statistics
import threading
import time
import urllib.parse

PROMPTS = [
    "List the three largest cities in France by population.",
    "Translate to German: The quick brown fox jumps over the lazy dog.",
    "What is 17 * 23? Answer with the number only.",
]
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["correct", "load"])
ap.add_argument("--url", default="http://127.0.0.1:8000/v3/chat/completions")
ap.add_argument("--model", default="qwen")
ap.add_argument("--tag", default="")
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("--in-tokens", type=int, default=512)
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--shared-prefix", type=int, default=0, help="tokens of prompt shared by all requests")
ap.add_argument("--prime", action="store_true", help="send the shared prefix once first (fills the prefix cache)")
ap.add_argument("--pool", default="/mnt/data/results/word_pool.json")
ap.add_argument("--no-stream", action="store_true")
ap.add_argument("--note", default="")
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
u = urllib.parse.urlparse(args.url)
OUT = os.environ.get("OUT", "/mnt/data/results/ovms.jsonl")


def post(body, timeout=3600):
    """POST and read the (possibly streamed) answer. Returns dict with text, usage, ttft, total."""
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request("POST", u.path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    out = {"status": resp.status, "ttft": None, "usage": None, "text": "", "chunks": 0}
    if resp.status != 200:
        out["error"] = resp.read().decode(errors="replace")[:300]
        out["total"] = time.perf_counter() - t0
        conn.close()
        return out
    if body.get("stream"):
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                out["usage"] = ev["usage"]
            for ch in ev.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    out["chunks"] += 1
                    out["text"] += piece
                    if out["ttft"] is None:
                        out["ttft"] = time.perf_counter() - t0
    else:
        ev = json.loads(resp.read())
        out["usage"] = ev.get("usage")
        out["text"] = ev["choices"][0]["message"].get("content") or ""
    out["total"] = time.perf_counter() - t0
    conn.close()
    return out


def body_for(text, max_tokens, stream):
    b = {"model": args.model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
         "temperature": 0, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    if stream:
        b["stream"] = True
        b["stream_options"] = {"include_usage": True}
    return b


def append(row):
    row = {"tag": args.tag, "model": args.model, "note": args.note, **row}
    with open(OUT, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if args.mode == "correct":
    for i, p in enumerate(PROMPTS, 1):
        b = body_for(p, 64, False)
        b["ignore_eos"] = False
        r = post(b)
        row = {"mode": "correct", "prompt_id": i, "prompt": p, "text": r["text"], "usage": r["usage"],
               "seconds": round(r["total"], 2), "status": r["status"], "error": r.get("error")}
        print(json.dumps(row, ensure_ascii=False)[:600], flush=True)
        append(row)
    raise SystemExit(0)

pool = json.load(open(args.pool))
rng = random.Random(args.seed)


def words(k):
    return "".join(rng.choice(pool) for _ in range(k))


# calibrate the chat-template overhead: usage.prompt_tokens - number of words
cal_words = 100
cal = post(body_for(words(cal_words), 1, False))
if cal["status"] != 200 or not cal["usage"]:
    raise SystemExit(f"calibration failed: {cal}")
overhead = cal["usage"]["prompt_tokens"] - cal_words
print(f"template overhead {overhead} tokens", flush=True)
n_words = args.in_tokens - overhead
shared_text = words(args.shared_prefix) if args.shared_prefix else ""
if args.prime and shared_text:
    pr = post(body_for(shared_text + words(n_words - args.shared_prefix), 1, False))
    print(f"primed prefix cache: {pr['total']:.1f}s status {pr['status']}", flush=True)

results = []
lock = threading.Lock()
barrier = threading.Barrier(args.n)
stream = not args.no_stream


def worker(wid):
    barrier.wait()
    for rd in range(args.rounds):
        text = shared_text + words(n_words - args.shared_prefix)
        r = post(body_for(text, args.max_tokens, stream))
        with lock:
            results.append({"worker": wid, "round": rd, "t_end": time.perf_counter(), **r})


load1 = os.getloadavg()[0]
t_start = time.perf_counter()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.n)]
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = max(r["t_end"] for r in results) - t_start
ok = [r for r in results if r["status"] == 200 and r["usage"]]
errs = [r.get("error") for r in results if r["status"] != 200]
in_tok = sum(r["usage"]["prompt_tokens"] for r in ok)
out_tok = sum(r["usage"]["completion_tokens"] for r in ok)
lat = sorted(r["total"] for r in ok)
ttft = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
pct = lambda xs, q: round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None  # noqa: E731
first = [r for r in ok if r["round"] == 0]
row = {"mode": "load", "n_concurrent": args.n, "rounds": args.rounds, "input_len": args.in_tokens,
       "max_tokens": args.max_tokens, "shared_prefix": args.shared_prefix, "primed": args.prime,
       "requests": len(results), "ok": len(ok), "errors": errs[:3], "wall_s": round(wall, 2),
       "req_s": round(len(ok) / wall, 4), "in_tokens": in_tok, "out_tokens": out_tok,
       "in_tok_s": round(in_tok / wall, 2), "out_tok_s": round(out_tok / wall, 2),
       "total_tok_s": round((in_tok + out_tok) / wall, 2),
       "lat_p50_s": pct(lat, 0.5), "lat_p90_s": pct(lat, 0.9), "lat_max_s": pct(lat, 1.0),
       "ttft_p50_s": pct(ttft, 0.5), "ttft_max_s": pct(ttft, 1.0),
       "wave1_wall_s": round(max(r["t_end"] for r in first) - t_start, 2) if first else None,
       "loadavg1_before": round(load1, 2), "stream": stream}
if ttft and len(ok):
    # decode rate after the last first token (all requests are in decode by then): the first-wave estimate
    t_last_first = max(r["ttft"] for r in first)
    dec_wall = max(r["total"] for r in first) - t_last_first
    row["est_decode_tok_s_wave1"] = round(sum(r["usage"]["completion_tokens"] - 1 for r in first) / dec_wall, 2) if dec_wall > 0 else None
print(json.dumps(row), flush=True)
append(row)
