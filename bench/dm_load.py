#!/usr/bin/env python3
"""Load generator and accuracy probe for any Jev-compatible `POST /v1/systemone` server (E28-E31),
standard library only. Runs on the VM next to the server.

Each canonical item (dm_data.py format) becomes one request with one `choice` question:
  {"model": M, "state": <state>, "questions": {"q": {"type": "choice", "instructions": <question>,
   "criteria": {<option>: null, ...}}}}
Requests are sent from CONCURRENCY threads. Per item it writes a prediction row for dm_score.py
(probabilities over the item's options, parsed from the response) with the client latency; at the end
one summary row: wall time, decisions/s, latency percentiles, errors, and prompt tokens when the
server reports them (llama-server `usage`/`timings`, or a `usage` block).

  dm_load.py --url http://127.0.0.1:8080 --model kev-4b --items banking77_8.jsonl \
             --concurrency 16 [--limit N] --preds preds.jsonl --summary load.jsonl [--tag TAG]
"""
import argparse
import concurrent.futures as cf
import json
import os
import statistics
import time
import urllib.request


def request_of(it, model):
    return {"model": model, "state": it["state"],
            "questions": {"q": {"type": "choice", "instructions": it["question"],
                                "criteria": {o: None for o in it["options"]}}}}


def find_probs(resp, options):
    """Pull a distribution over `options` out of the many response shapes servers use."""
    def walk(x):
        if isinstance(x, dict):
            yield x
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, list):
            for v in x:
                yield from walk(v)
    best = None
    for d in walk(resp):
        for key in ("probabilities", "distribution", "probs", "option_probabilities", "scores"):
            v = d.get(key)
            if isinstance(v, dict) and set(v) & set(options):
                return [float(v.get(o, 0.0)) for o in options]
            if isinstance(v, list) and len(v) == len(options) and all(isinstance(z, (int, float)) for z in v):
                return [float(z) for z in v]
            if isinstance(v, list) and v and isinstance(v[0], dict):
                m = {}
                for e in v:
                    name = e.get("name", e.get("option", e.get("choice", e.get("label"))))
                    p = e.get("probability", e.get("p", e.get("prob")))
                    if name in options and isinstance(p, (int, float)):
                        m[name] = float(p)
                if m:
                    return [m.get(o, 0.0) for o in options]
        if best is None and isinstance(d.get("choice"), str) and d["choice"] in options:
            conf = float(d.get("confidence", 1.0) or 1.0)
            n = len(options)
            best = [conf if o == d["choice"] else (1 - conf) / max(1, n - 1) for o in options]
    return best


def prompt_tokens(resp):
    for path in (("usage", "prompt_tokens"), ("usage", "input_tokens"), ("timings", "prompt_n")):
        x = resp
        for k in path:
            x = x.get(k) if isinstance(x, dict) else None
        if isinstance(x, (int, float)):
            return int(x)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--path", default="/v1/systemone")
    ap.add_argument("--model", required=True, help="model id sent in the request and written to the rows")
    ap.add_argument("--items", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--tag", default="", help="row label (defaults to --model)")
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.items) if l.strip()]
    if args.limit:
        items = items[: args.limit]
    tag = args.tag or args.model
    run = os.environ.get("RUN", "")

    def one(it):
        body = json.dumps(request_of(it, args.model)).encode()
        req = urllib.request.Request(args.url + args.path, data=body, headers={"Content-Type": "application/json"})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                resp = json.loads(r.read().decode())
            dt = time.time() - t
            probs = find_probs(resp, it["options"])
            row = {"run": run, "model": tag, "dataset": it["dataset"], "id": it["id"], "gold": it["gold"],
                   "latency_s": round(dt, 4), "n_prompt_tokens": prompt_tokens(resp)}
            if probs is None:
                row.update(error="no distribution in response", n_options=len(it["options"]),
                           sample=json.dumps(resp)[:300])
            else:
                row["probs"] = [round(p, 6) for p in probs]
            return row
        except Exception as e:  # keep going; the error is a measurement
            return {"run": run, "model": tag, "dataset": it["dataset"], "id": it["id"], "gold": it["gold"],
                    "latency_s": round(time.time() - t, 4), "n_options": len(it["options"]), "error": str(e)[:300]}

    t0 = time.time()
    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        rows = list(ex.map(one, items))
    wall = time.time() - t0
    with open(args.preds, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lat = sorted(r["latency_s"] for r in rows if "error" not in r)
    toks = [r.get("n_prompt_tokens") for r in rows if r.get("n_prompt_tokens")]
    summ = {"run": run, "model": tag, "dataset": items[0]["dataset"] if items else "", "n": len(rows),
            "errors": sum(1 for r in rows if "error" in r), "concurrency": args.concurrency, "wall_s": round(wall, 2),
            "decisions_per_s": round(len(rows) / wall, 3) if wall else None,
            "p50_s": round(statistics.median(lat), 3) if lat else None,
            "p95_s": round(lat[int(0.95 * (len(lat) - 1))], 3) if lat else None,
            "prompt_tokens": sum(toks) if toks else None,
            "tok_per_s": round(sum(toks) / wall, 1) if toks and wall else None}
    with open(args.summary, "a") as f:
        f.write(json.dumps(summ) + "\n")
    print("SUMMARY " + json.dumps(summ))


if __name__ == "__main__":
    main()
