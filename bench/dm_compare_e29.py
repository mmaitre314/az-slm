#!/usr/bin/env python3
"""Compare per-item probabilities of two dm_load prediction files (E29), stdlib only.
  dm_compare_e29.py A.jsonl B.jsonl [--tol 1e-3] [--label TEXT]  -> one JSON line; exit 0 if all items within tol.
Items are matched by id. An item with an error or without probs in either file counts as a mismatch."""
import argparse
import json
import sys


def load(p):
    d = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            d[r["id"]] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--label", default="")
    x = ap.parse_args()
    a, b = load(x.a), load(x.b)
    ids = sorted(set(a) & set(b))
    diffs, bad, top = [], 0, 0
    for i in ids:
        pa, pb = a[i].get("probs"), b[i].get("probs")
        if not pa or not pb or len(pa) != len(pb):
            bad += 1
            continue
        diffs.append(max(abs(u - v) for u, v in zip(pa, pb)))
        top += pa.index(max(pa)) == pb.index(max(pb))
    n = len(diffs)
    out = {"label": x.label, "n_common": len(ids), "n_compared": n, "n_missing_or_error": bad,
           "max_abs_diff": round(max(diffs), 6) if diffs else None,
           "mean_abs_diff": round(sum(diffs) / n, 7) if n else None,
           "n_over_tol": sum(1 for d in diffs if d > x.tol), "top1_agree": round(top / n, 4) if n else None,
           "tol": x.tol}
    out["match"] = bool(n) and bad == 0 and out["n_over_tol"] == 0
    print(json.dumps(out))
    sys.exit(0 if out["match"] else 1)


if __name__ == "__main__":
    main()
