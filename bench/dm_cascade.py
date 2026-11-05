#!/usr/bin/env python3
"""Simulate a calibrated two-model decision cascade from saved predictions (E31), stdlib only.

Every item goes to the SMALL model first; items whose top-1 probability is below a threshold are sent
to the BIG model, whose answer replaces the small one. With both models' per-item predictions on the
same items (dm_score.py's input format), the sweep gives accuracy and cost per 1k decisions for each
threshold, next to small-only and big-only.

  dm_cascade.py preds.jsonl [more.jsonl] --small kev-4b --big qwen3.8-27b-w8a8 --dataset banking77_8 \
                --small-cost 0.04 --big-cost 0.42 [--md]
Costs are $ per 1k decisions for each model alone (from the throughput runs); the cascade pays the
small cost for every item plus the big cost for escalated items.
"""
import argparse
import json


def load(files, model, dataset):
    out = {}
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("model") == model and r.get("dataset") == dataset and r.get("probs") and not r.get("error"):
                out[r["id"]] = r
    return out


def top(r):
    k = max(range(len(r["probs"])), key=r["probs"].__getitem__)
    return k, r["probs"][k]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--small", required=True)
    ap.add_argument("--big", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--small-cost", type=float, required=True, help="$ per 1k decisions, small model alone")
    ap.add_argument("--big-cost", type=float, required=True, help="$ per 1k decisions, big model alone")
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    s, b = load(args.files, args.small, args.dataset), load(args.files, args.big, args.dataset)
    ids = sorted(set(s) & set(b))
    if not ids:
        raise SystemExit("no items predicted by both models")
    n = len(ids)
    acc_s = sum(top(s[i])[0] == s[i]["gold"] for i in ids) / n
    acc_b = sum(top(b[i])[0] == b[i]["gold"] for i in ids) / n
    rows = [("small only", 0.0, acc_s, args.small_cost), ("big only", 1.0, acc_b, args.big_cost)]
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99):
        esc = [i for i in ids if top(s[i])[1] < th]
        correct = sum((top(b[i])[0] if top(s[i])[1] < th else top(s[i])[0]) == s[i]["gold"] for i in ids)
        frac = len(esc) / n
        rows.append((f"escalate if p < {th}", frac, correct / n, args.small_cost + frac * args.big_cost))
    hdr = ["policy", "escalated", "accuracy", "$ per 1k decisions", "gap closed"]
    gap = acc_b - acc_s
    print(f"{args.dataset}: {n} items, small {args.small} ({acc_s:.3f}), big {args.big} ({acc_b:.3f})")
    if args.md:
        print("| " + " | ".join(hdr) + " |\n|---|---:|---:|---:|---:|")
    for name, frac, acc, cost in rows:
        closed = (acc - acc_s) / gap if gap > 0 else float("nan")
        vals = [name, f"{frac:.1%}", f"{acc:.3f}", f"{cost:.4f}", f"{closed:.0%}" if gap > 0 else "-"]
        print("| " + " | ".join(vals) + " |" if args.md else "\t".join(vals))


if __name__ == "__main__":
    main()
