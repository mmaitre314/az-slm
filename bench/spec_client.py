#!/usr/bin/env python3
"""Client for the speculative-decoding experiment (E12). Runs on the VM, standard library only.

Sends N chat requests to a running llama-server (greedy, thinking off, prompt cache off), N at a
time, and appends one JSON line per request plus one summary line to --out. Per-request numbers
come from the server's own `timings` block: decode tokens/s (predicted_per_second), drafted and
accepted draft tokens (draft_n, draft_n_accepted).

usage: spec_client.py --url http://127.0.0.1:8080 --tag NAME --set std3|ten|sixteen
                      [--n N] [--max-tokens M] [--out FILE] [--seq]
  --set    which prompt list (std3: the 3 fixed correctness prompts, 64 tokens; ten: 10 varied
           prompts with 64-256 output tokens; sixteen: the ten plus 6 more)
  --n      number of requests in flight at once (default 1 = sequential)
  --seq    force sequential even if --n > 1 is given for the slot count (rarely needed)
"""
import argparse
import concurrent.futures
import json
import time
import urllib.request

STD3 = [
    ("List the three largest cities in France by population.", 64),
    ("Translate to German: The quick brown fox jumps over the lazy dog.", 64),
    ("What is 17 * 23? Answer with the number only.", 64),
]

# 10 varied prompts, (text, max_tokens). Mixed task types: free text, code, extraction/copying
# (n-gram friendly), translation, reasoning, structured output.
TEN = [
    ("Explain in simple terms why the sky is blue and why sunsets are red.", 160),
    ("Write a Python function that checks whether a string is a palindrome, ignoring case and "
     "punctuation, with a short docstring and two example calls.", 192),
    ("Fix the spelling and grammar mistakes in this text and return only the corrected text: "
     "'Yesterday me and my freinds goes to the mueseum, wich was realy bored. The guide were "
     "talking to fast and we cant understood nothing. Next time we will went to the beach insted.'", 96),
    ("Extract the fields as JSON with keys name, date, amount, currency from this invoice text: "
     "'Invoice 2024-0153 issued on 14 March 2024 to Contoso Ltd for a total of 2,450.75 EUR, due "
     "within 30 days.' Return only the JSON.", 64),
    ("Write a short story opening (about 120 words) set on a research station in Antarctica.", 192),
    ("Translate to French and then to Spanish: 'The meeting has been moved to Thursday afternoon "
     "because the manager is travelling. Please update your calendars and confirm attendance.'", 128),
    ("A train leaves city A at 9:00 traveling at 80 km/h. Another leaves city B, 300 km away, at "
     "10:00 traveling toward A at 100 km/h. When and where do they meet? Think step by step.", 256),
    ("Write a SQL query that returns the top 5 customers by total order value in 2023 from tables "
     "customers(id, name) and orders(id, customer_id, total, created_at), and explain it briefly.", 192),
    ("Summarize the following in two sentences: 'Photosynthesis is the process by which green plants, "
     "algae and some bacteria convert light energy into chemical energy. Using chlorophyll, they "
     "absorb sunlight and use it to turn carbon dioxide and water into glucose and oxygen. The "
     "oxygen is released into the atmosphere, while the glucose fuels growth and metabolism.'", 96),
    ("Give me a bulleted list of 8 practical tips for reducing the cost of running batch jobs on "
     "cloud virtual machines, one sentence each.", 224),
]

EXTRA6 = [
    ("Write a Python class implementing a simple LRU cache with get and put methods.", 192),
    ("Explain the difference between TCP and UDP and when to use each.", 160),
    ("Rewrite this sentence in a formal tone: 'hey, can u send me that report asap? thx'", 64),
    ("What are the main causes of the French Revolution? Answer in a short paragraph.", 128),
    ("Write a bash one-liner to count the lines in all .py files under the current directory, "
     "and explain each part.", 128),
    ("Describe how a binary search works and give its time complexity.", 128),
]


def get_set(name):
    if name == "std3":
        return STD3
    if name == "ten":
        return TEN
    if name == "sixteen":
        return TEN + EXTRA6
    raise SystemExit("unknown prompt set " + name)


def request(url, prompt, max_tokens, think=False):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 1,
        "cache_prompt": False,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    t1 = time.time()
    return d, t0, t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", default="ten", dest="pset")
    ap.add_argument("--n", type=int, default=1, help="requests in flight at once")
    ap.add_argument("--count", type=int, default=0, help="number of requests (default: size of the set)")
    ap.add_argument("--max-tokens", type=int, default=0, help="override max_tokens for every prompt")
    ap.add_argument("--out", default="/mnt/data/results/spec.jsonl")
    ap.add_argument("--meta", default="{}", help="JSON dict merged into every output row")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    meta = json.loads(args.meta)
    prompts = get_set(args.pset)
    if args.count:
        prompts = (prompts * (args.count // len(prompts) + 1))[:args.count]
    if args.max_tokens:
        prompts = [(p, args.max_tokens) for p, _ in prompts]

    rows = []

    def one(i):
        p, mt = prompts[i]
        d, t0, t1 = request(args.url, p, mt)
        tm = d.get("timings", {})
        ch = d["choices"][0]
        row = {
            "tag": args.tag, "set": args.pset, "i": i, "n_inflight": args.n, "max_tokens": mt,
            "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
            "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"),
            "decode_tps": tm.get("predicted_per_second"),
            "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted"),
            "finish": ch.get("finish_reason"), "wall_s": round(t1 - t0, 2),
            "t_start": t0, "t_end": t1,
            "text": ch["message"].get("content", ""),
            "reasoning": ch["message"].get("reasoning_content", ""),
        }
        row.update(meta)
        if not args.quiet:
            acc = ""
            if row["draft_n"]:
                acc = f" draft {row['draft_n_accepted']}/{row['draft_n']}"
            print(f"  [{args.tag}] #{i} {row['predicted_n']} tok, {row['decode_tps']:.2f} t/s{acc}, "
                  f"prefill {row['prompt_n']} tok", flush=True)
        return row

    t_all0 = time.time()
    if args.n <= 1:
        rows = [one(i) for i in range(len(prompts))]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n) as ex:
            rows = list(ex.map(one, range(len(prompts))))
    t_all1 = time.time()

    tok = sum(r["predicted_n"] or 0 for r in rows)
    dn = sum(r["draft_n"] or 0 for r in rows)
    da = sum(r["draft_n_accepted"] or 0 for r in rows)
    dec_ms = sum(r["predicted_ms"] or 0 for r in rows)
    pre_n = sum(r["prompt_n"] or 0 for r in rows)
    pre_ms = sum(r["prompt_ms"] or 0 for r in rows)
    summary = {
        "tag": args.tag, "set": args.pset, "kind": "summary", "n_inflight": args.n, "requests": len(rows),
        "tokens": tok, "wall_s": round(t_all1 - t_all0, 2),
        "agg_wall_tps": round(tok / (t_all1 - t_all0), 3),
        # tokens / summed per-request decode time: per-sequence rate (1 in flight) or per-slot rate
        "mean_decode_tps": round(tok / (dec_ms / 1000), 3) if dec_ms else None,
        "sum_slot_tps": round(sum(r["decode_tps"] or 0 for r in rows), 3),
        "draft_n": dn, "draft_n_accepted": da, "accept_rate": round(da / dn, 4) if dn else None,
        "prefill_tps": round(pre_n / (pre_ms / 1000), 2) if pre_ms else None,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    summary.update(meta)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
