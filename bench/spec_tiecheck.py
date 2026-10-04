#!/usr/bin/env python3
"""Why does greedy output differ between speculative and plain decoding? (E12, runs on the VM.)

For every request whose text differs from the baseline run, find the first differing token, feed
the prompt plus the common token prefix to a plain (non-speculative) llama-server, and print the
log-probabilities that server assigns to the baseline's token and to the speculative run's token.
A gap near 0 (a few hundredths of a nat) means a floating-point near-tie that batched verification
resolved differently; a large gap would point at a real bug.

usage: spec_tiecheck.py RESULTS.jsonl BASE_TAG SPEC_TAG [--url URL] [--set ten]
The server at --url must run the same model without speculation. Chat template is applied by hand
(thinking disabled), which the script validates against the baseline's own first token.
"""
import argparse
import json
import urllib.request

import sys  # noqa: E402

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import spec_client  # noqa: E402


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def tokenize(url, text, special=False):
    return post(url, "/tokenize", {"content": text, "add_special": False, "parse_special": special})["tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("base_tag")
    ap.add_argument("spec_tag")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--set", default="ten")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.file) if line.strip()]
    sel = {}
    for r in rows:
        if r.get("kind") == "summary" or r["set"] != args.set:
            continue
        sel.setdefault(r["tag"], {})[r["i"]] = r
    base, spec = sel[args.base_tag], sel[args.spec_tag]
    prompts = dict(enumerate(spec_client.get_set(args.set)))
    results = []
    for i in sorted(base):
        tb, ts = base[i]["text"], spec[i]["text"]
        if tb == ts:
            continue
        ptxt = ("<|im_start|>user\n" + prompts[i][0] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        ptoks = tokenize(args.url, ptxt, special=True)
        b, s = tokenize(args.url, tb), tokenize(args.url, ts)
        d = next((k for k in range(min(len(b), len(s))) if b[k] != s[k]), min(len(b), len(s)))
        if d >= len(b) or d >= len(s):
            results.append({"i": i, "note": "one output is a prefix of the other", "len_b": len(b), "len_s": len(s)})
            continue
        res = post(args.url, "/completion", {"prompt": ptoks + b[:d], "n_predict": 1, "temperature": 0.0,
                                              "n_probs": 20, "cache_prompt": False, "post_sampling_probs": False})
        top = res["completion_probabilities"][0]["top_logprobs"]
        lp = {t["id"]: t["logprob"] for t in top}
        top1 = top[0]
        row = {"i": i, "tok_index": d, "n_tokens_base": len(b), "base_tok": b[d], "spec_tok": s[d],
               "base_lp": lp.get(b[d]), "spec_lp": lp.get(s[d]), "plain_argmax": top1["id"],
               "plain_argmax_matches_base": top1["id"] == b[d]}
        if row["base_lp"] is not None and row["spec_lp"] is not None:
            row["gap_nats"] = round(row["base_lp"] - row["spec_lp"], 4)
        results.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps({"kind": "tiecheck", "base": args.base_tag, "spec": args.spec_tag, **r}) + "\n")


if __name__ == "__main__":
    main()
