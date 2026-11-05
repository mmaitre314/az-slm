#!/usr/bin/env python3
"""Score decision-model predictions (E28-E30), standard library only. Runs on the VM or locally.

Input: JSON lines, one per (model, dataset, item), as written by every decision-model runner:
  {"model": "...", "dataset": "...", "id": "...", "gold": 3, "probs": [0.1, ...], "run": "..."}
`probs` are the model's probabilities over the item's options (any non-negative scores are
renormalized); `gold` is the index of the correct option. Rows with "error" set (or no probs) are left out of
the probability metrics, counted, and treated as wrong in `acc_all`. Optional fields are ignored.

Output (stdout): one line per (run, model, dataset): n, accuracy, macro-F1, ECE (15 equal-width
bins on the top-1 probability), Brier, NLL, and the same ECE/NLL after temperature scaling with T
fitted on a deterministic half of the items (by id hash) and evaluated on the other half.
  dm_score.py preds.jsonl [more.jsonl ...] [--md] [--json out.jsonl]
"""
import argparse
import collections
import hashlib
import json
import math


def softmax_t(probs, t):
    logs = [math.log(max(p, 1e-12)) / t for p in probs]
    m = max(logs)
    e = [math.exp(x - m) for x in logs]
    s = sum(e)
    return [x / s for x in e]


def ece(rows, bins=15):
    tot = len(rows)
    if not tot:
        return float("nan")
    b = collections.defaultdict(lambda: [0, 0.0, 0])
    for conf, correct in rows:
        k = min(int(conf * bins), bins - 1)
        b[k][0] += 1
        b[k][1] += conf
        b[k][2] += correct
    return sum(abs(c / n - a / n) * n / tot for n, c, a in b.values())


def metrics(items, t=1.0):
    conf_rows, nll, brier, golds, preds = [], 0.0, 0.0, [], []
    for it in items:
        p = softmax_t(it["probs"], t) if t != 1.0 else it["probs"]
        g = it["gold"]
        k = max(range(len(p)), key=p.__getitem__)
        conf_rows.append((p[k], int(k == g)))
        nll -= math.log(max(p[g], 1e-12))
        brier += sum((pi - (1.0 if i == g else 0.0)) ** 2 for i, pi in enumerate(p))
        golds.append(g)
        preds.append(k)
    n = len(items)
    acc = sum(c for _, c in conf_rows) / n if n else float("nan")
    # macro-F1 over gold classes present (option index = class when the option list is fixed)
    f1s = []
    for c in set(golds):
        tp = sum(1 for g, q in zip(golds, preds) if g == c and q == c)
        fp = sum(1 for g, q in zip(golds, preds) if g != c and q == c)
        fn = sum(1 for g, q in zip(golds, preds) if g == c and q != c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"n": n, "acc": acc, "macro_f1": sum(f1s) / len(f1s) if f1s else float("nan"),
            "ece": ece(conf_rows), "brier": brier / n if n else float("nan"), "nll": nll / n if n else float("nan")}


def fit_t(items):
    best = (float("inf"), 1.0)
    t = 0.05
    while t <= 10.0:
        nll = -sum(math.log(max(softmax_t(it["probs"], t)[it["gold"]], 1e-12)) for it in items)
        best = min(best, (nll, t))
        t *= 1.08
    return best[1]


def half(item_id):
    return int(hashlib.sha256(str(item_id).encode()).hexdigest(), 16) % 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--md", action="store_true", help="markdown table")
    ap.add_argument("--json", help="also write the metrics as JSON lines to this file")
    args = ap.parse_args()
    groups = collections.OrderedDict()
    errors = collections.Counter()
    for f in args.files:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r.get("run", ""), r["model"], r["dataset"])
            groups.setdefault(key, {})
            if r.get("error") or not r.get("probs"):
                errors[key] += 1
                continue
            s = sum(max(0.0, x) for x in r["probs"]) or 1.0
            r["probs"] = [max(0.0, x) / s for x in r["probs"]]
            groups.setdefault(key, {})[r["id"]] = r  # last row per id wins (re-runs)
    out = []
    for key, d in groups.items():
        items = list(d.values())
        m = metrics(items)
        fit = [it for it in items if half(it["id"]) == 0]
        ev = [it for it in items if half(it["id"]) == 1]
        t = fit_t(fit) if fit else 1.0
        mt = metrics(ev, t) if ev else {}
        if not items:
            out.append({"run": key[0], "model": key[1], "dataset": key[2], "n": 0, "errors": errors[key]})
            continue
        acc_all = m["acc"] * m["n"] / (m["n"] + errors[key])
        out.append({"run": key[0], "model": key[1], "dataset": key[2], **{k: round(v, 4) for k, v in m.items() if k != "n"},
                    "n": m["n"], "errors": errors[key], "acc_all": round(acc_all, 4), "T": round(t, 3), "ece_T_heldout": round(mt.get("ece", float("nan")), 4),
                    "nll_T_heldout": round(mt.get("nll", float("nan")), 4)})
    cols = ["run", "model", "dataset", "n", "errors", "acc", "acc_all", "macro_f1", "ece", "brier", "nll", "T", "ece_T_heldout", "nll_T_heldout"]
    if args.md:
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join(["---"] * 3 + ["---:"] * (len(cols) - 3)) + "|")
    for o in out:
        vals = [str(o.get(c, "")) for c in cols]
        print("| " + " | ".join(vals) + " |" if args.md else "\t".join(vals))
    if args.json:
        with open(args.json, "a") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
