#!/usr/bin/env python3
"""Summarize spec.jsonl-style results (E12), standard library only. Runs on the VM or locally.

For each (tag, set) it prints decode tokens/s, acceptance rate, and how the greedy output compares
with the baseline run of the same set: number of requests with identical text, and for the others the
character position of the first difference.

usage: spec_analyze.py FILE [--base TAG_PREFIX] [--set std3|ten|sixteen] [--md]
"""
import argparse
import collections
import json


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--base", default="base-", help="tag prefix of the baseline configuration")
    ap.add_argument("--set", default="")
    ap.add_argument("--md", action="store_true", help="markdown table")
    ap.add_argument("--texts", action="store_true", help="also print the first 70 chars of each output")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.file) if line.strip()]
    by = collections.OrderedDict()
    for r in rows:
        if args.set and r["set"] != args.set:
            continue
        by.setdefault((r["tag"], r["set"]), []).append(r)
    base_texts = {}
    for (tag, st), rs in by.items():
        if tag.startswith(args.base):
            base_texts[st] = {r["i"]: r["text"] for r in rs if r.get("kind") != "summary"}
    # dec t/s: one sequence = decode rate; several = "all slots decoding / wall-clock incl. prefill and ramp-up"
    hdr = ["tag", "set", "n_in", "tokens", "dec t/s", "accept", "tok/step", "s/step", "same/total",
           "first diffs (req:char)"]
    out = []
    for (tag, st), rs in by.items():
        summ = [r for r in rs if r.get("kind") == "summary"]
        reqs = [r for r in rs if r.get("kind") != "summary"]
        if not summ:
            continue
        s = summ[-1]
        bt = base_texts.get(st, {})
        same, diffs = 0, []
        for r in reqs:
            if r["i"] in bt:
                d = first_diff(bt[r["i"]], r["text"])
                if d < 0:
                    same += 1
                else:
                    diffs.append(f"{r['i']}:{d}")
        acc = s.get("accept_rate")
        # one verification step always yields 1 token plus the accepted drafts (single sequence only)
        dec_s = sum((r.get("predicted_ms") or 0) for r in reqs) / 1000
        steps = s["tokens"] - (s.get("draft_n_accepted") or 0)
        tps = f"{s['tokens'] / steps:.2f}" if s["n_inflight"] == 1 and steps > 0 else "-"
        sps = f"{dec_s / steps:.3f}" if s["n_inflight"] == 1 and steps > 0 else "-"
        out.append([tag, st, s["n_inflight"], s["tokens"], s["mean_decode_tps"] if s["n_inflight"] == 1 else
                    f"{s['n_inflight'] * s['mean_decode_tps']:.2f}/{s['agg_wall_tps']}", "-" if acc is None else f"{acc:.2f}",
                    tps, sps, f"{same}/{len(reqs)}", " ".join(diffs)])
    if args.md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for o in out:
            print("| " + " | ".join(str(x) for x in o) + " |")
    else:
        for o in out:
            print(" ".join(str(x) for x in o))
    if args.texts:
        for (tag, st), rs in by.items():
            for r in rs:
                if r.get("kind") != "summary":
                    print(f"{tag} #{r['i']}: {r['text'][:70]!r}")


if __name__ == "__main__":
    main()
