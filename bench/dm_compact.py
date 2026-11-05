#!/usr/bin/env python3
"""Compact decision-model predictions for a quick harvest (E29-E32), and read them back. Stdlib only.

Run Command returns at most 4 KiB per call, so the full prediction files (~1 MB per model run, every
option's probability for every item) take hours to fetch. The compact form keeps per item what
accuracy, ECE on the top-1 probability and cascade simulations need: the top-1 option, its
probability and the label mass (format compliance). Metrics that need the whole distribution (NLL,
Brier, temperature scaling) are computed on the VM by dm_score.py before packing (bench/dm_pack.sh).

  dm_compact.py pack OUT.jsonl preds.jsonl [more.jsonl ...]   # on the VM
  dm_compact.py stats compact.jsonl [more ...] [--md]         # accuracy and ECE per group, locally

Format, JSON lines:
  {"kind": "dataset", "dataset": D, "ids": [...], "gold": [...]}    once per dataset: union of ids, first seen first
  {"kind": "group", "file": F, "run": R, "model": M, "dataset": D, "n": N,
   "idx": [...] or null,   positions of the group's items in the dataset's ids (null: all of them, in order)
   "ok": "1101...",        1 where the top-1 option is the gold one
   "wrong_pred": [...],    top-1 option of each wrong item, in order
   "pmax": [...],          top-1 probability x 1e4, rounded (ties: the first option, as dm_score.py)
   "mass": [...] or null,  label mass x 100 (vLLM readout runs only)
   "errors": E, "error_idx": [...], "error_msg": "first error message"}
iter_items() yields one dict per scored item: file, run, model, dataset, id, gold, pred, conf, ok, mass.
"""
import argparse
import collections
import json
import os
import sys


def top1(probs):
    s = sum(max(0.0, x) for x in probs)
    p = [max(0.0, x) / s for x in probs] if s > 0 else [1.0 / len(probs)] * len(probs)
    k = max(range(len(p)), key=p.__getitem__)
    return k, p[k]


def pack(files, out):
    groups = collections.OrderedDict()   # (file, run, model, dataset) -> rows
    ds_ids = collections.OrderedDict()   # dataset -> {id: gold}
    for f in files:
        base = os.path.basename(f)
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not all(k in r for k in ("model", "dataset", "id")):
                continue
            d = ds_ids.setdefault(r["dataset"], {})
            if r["id"] not in d:
                d[r["id"]] = r.get("gold")
            groups.setdefault((base, r.get("run", ""), r["model"], r["dataset"]), []).append(r)
    pos = {ds: {i: k for k, i in enumerate(d)} for ds, d in ds_ids.items()}
    with open(out, "w") as fo:
        for ds, d in ds_ids.items():
            fo.write(json.dumps({"kind": "dataset", "dataset": ds, "ids": list(d), "gold": list(d.values())}) + "\n")
        for (base, run, model, ds), rows in groups.items():
            byid = collections.OrderedDict()
            for r in rows:
                byid[r["id"]] = r   # last row per id wins, as dm_score.py
            ok, wrong, pmax, mass, idx, err_idx, gold_over = [], [], [], [], [], [], {}
            msg = ""
            for i, r in byid.items():
                if r.get("gold") != ds_ids[ds][i]:
                    gold_over[pos[ds][i]] = r.get("gold")
                if r.get("error") or not r.get("probs"):
                    err_idx.append(pos[ds][i])
                    msg = msg or str(r.get("error") or "no probs")[:300]
                    continue
                k, p = top1(r["probs"])
                idx.append(pos[ds][i])
                ok.append("1" if k == r.get("gold") else "0")
                if k != r.get("gold"):
                    wrong.append(k)
                pmax.append(round(p * 1e4))
                mass.append(None if r.get("label_mass") is None else round(r["label_mass"] * 100))
            g = {"kind": "group", "file": base, "run": run, "model": model, "dataset": ds, "n": len(idx),
                 "idx": None if idx == list(range(len(ds_ids[ds]))) else idx, "ok": "".join(ok), "wrong_pred": wrong,
                 "pmax": pmax, "mass": mass if any(m is not None for m in mass) else None,
                 "errors": len(err_idx), "error_idx": err_idx, "error_msg": msg}
            if gold_over:
                g["gold_override"] = gold_over   # never expected: the same item id with another gold label
            fo.write(json.dumps(g, separators=(",", ":")) + "\n")


def iter_items(path):
    ds = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "dataset":
            ds[r["dataset"]] = r
            continue
        if r.get("kind") != "group":
            continue
        d = ds[r["dataset"]]
        idx = r["idx"] if r["idx"] is not None else range(len(d["ids"]))
        over = {int(k): v for k, v in (r.get("gold_override") or {}).items()}
        wrong = iter(r["wrong_pred"])
        for j, (p, okc) in enumerate(zip(idx, r["ok"])):
            gold = over.get(p, d["gold"][p])
            yield {"file": r["file"], "run": r["run"], "model": r["model"], "dataset": r["dataset"], "id": d["ids"][p],
                   "gold": gold, "pred": gold if okc == "1" else next(wrong), "conf": r["pmax"][j] / 1e4, "ok": okc == "1",
                   "mass": None if not r["mass"] or r["mass"][j] is None else r["mass"][j] / 100}


def is_compact(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                return json.loads(line).get("kind") in ("dataset", "group")
    return False


def ece(rows, bins=15):
    b = collections.defaultdict(lambda: [0, 0.0, 0])
    for conf, correct in rows:
        k = min(int(conf * bins), bins - 1)
        b[k][0] += 1
        b[k][1] += conf
        b[k][2] += correct
    tot = len(rows)
    return sum(abs(c / n - a / n) * n / tot for n, c, a in b.values()) if tot else float("nan")


def stats(paths, md):
    groups = collections.OrderedDict()
    for p in paths:
        for it in iter_items(p):
            groups.setdefault((it["run"], it["model"], it["dataset"]), []).append(it)
    cols = ["run", "model", "dataset", "n", "acc", "ece", "mean_conf", "mean_mass"]
    if md:
        print("| " + " | ".join(cols) + " |\n|---|---|---|" + "---:|" * (len(cols) - 3))
    for (run, model, ds), its in groups.items():
        n = len(its)
        masses = [it["mass"] for it in its if it["mass"] is not None]
        vals = [run, model, ds, str(n), f"{sum(it['ok'] for it in its) / n:.4f}",
                f"{ece([(it['conf'], it['ok']) for it in its]):.4f}", f"{sum(it['conf'] for it in its) / n:.4f}",
                f"{sum(masses) / len(masses):.3f}" if masses else ""]
        print("| " + " | ".join(vals) + " |" if md else "\t".join(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("out")
    p.add_argument("files", nargs="+")
    s = sub.add_parser("stats")
    s.add_argument("files", nargs="+")
    s.add_argument("--md", action="store_true")
    args = ap.parse_args()
    if args.cmd == "pack":
        pack(args.files, args.out)
        print(f"packed {len(args.files)} files into {args.out} ({os.path.getsize(args.out)} bytes)", file=sys.stderr)
    else:
        stats(args.files, args.md)


if __name__ == "__main__":
    main()
