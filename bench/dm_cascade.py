#!/usr/bin/env python3
"""Simulate a calibrated two-model decision cascade from saved predictions (E31), stdlib only.

Every item goes to the SMALL model first; items whose top-1 probability is below a threshold are sent
to the BIG model, whose answer replaces the small one. With both models' per-item predictions on the
same items (dm_score.py's input format), the sweep gives accuracy and cost per 1k decisions for each
threshold, next to small-only and big-only.

  dm_cascade.py preds.jsonl [more.jsonl] --small kev-4b --big qwen3.8-27b-w8a8 --dataset banking77_8 \
                --small-cost 0.04 --big-cost 0.42 [--md]
Costs are $ per 1k decisions for each model alone (from the throughput runs); the cascade pays the
small cost for every item plus the big cost for escalated items. Files are raw prediction files or
dm_compact.py's compact form (both can be mixed, e.g. across experiments). A model is named by its
tag, or TAG@RUN when the same tag appears in several runs (E30 and E31 share the Qwen tags).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dm_compact  # noqa: E402


def load(files, spec, dataset):
    """id -> (gold, top-1 option, top-1 probability) for one model on one dataset."""
    model, _, run = spec.partition("@")
    out = {}
    for f in files:
        if dm_compact.is_compact(f):
            for it in dm_compact.iter_items(f):
                if it["model"] == model and it["dataset"] == dataset and (not run or it["run"] == run):
                    out[it["id"]] = (it["gold"], it["pred"], it["conf"])
            continue
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("model") == model and r.get("dataset") == dataset and r.get("probs") and not r.get("error")
                    and (not run or r.get("run") == run)):
                k, p = dm_compact.top1(r["probs"])
                out[r["id"]] = (r["gold"], k, p)
    return out


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
    if any(s[i][0] != b[i][0] for i in ids):
        raise SystemExit("gold labels differ between the two models' items: not the same item set")
    n = len(ids)
    acc_s = sum(s[i][1] == s[i][0] for i in ids) / n
    acc_b = sum(b[i][1] == b[i][0] for i in ids) / n
    rows = [("small only", 0.0, acc_s, args.small_cost), ("big only", 1.0, acc_b, args.big_cost)]
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
        esc = [i for i in ids if s[i][2] < th]
        correct = sum((b[i][1] if s[i][2] < th else s[i][1]) == s[i][0] for i in ids)
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
