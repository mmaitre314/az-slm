#!/usr/bin/env python3
"""One table across the decision-model experiments (E29, E30, E32): accuracy, calibration, throughput
and cost per 1k decisions per (model, stack, VM), from harvested raw files. Standard library only.

  dm_report.py --price E30=1.326 --price E32=0.744 --price E29=1.326 \
      E30=experiments/E30-vllm-decision-readout/raw E32=experiments/E32-trained-readout-vllm/raw \
      E29=experiments/E29-llamacpp-decision-models/raw [--md]

Accuracy and ECE come from dm_score.py's metrics over every *-p-*.jsonl / q_*.preds.jsonl prediction
file in a directory; throughput from summary-*.jsonl (vLLM readout: prompt tok/s, items/s per dataset,
best repeat) or tp_summary.jsonl (llama.cpp /v1/systemone load runs, best concurrency).
$ per 1k decisions = $/h / (decisions/s * 3600) * 1000 on banking77_8; $ per M tokens likewise.
"""
import argparse
import collections
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def score(files):
    out = subprocess.run([sys.executable, os.path.join(HERE, "dm_score.py"), *files, "--json", "/dev/stdout"],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def throughput(d):
    tp = {}
    for f in glob.glob(os.path.join(d, "summary-*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("n") and r.get("dataset") == "banking77_8":
                k = r["model"]
                cur = tp.get(k)
                if cur is None or (r.get("items_per_s") or 0) > cur["dps"]:
                    tp[k] = {"dps": r["items_per_s"], "tps": r["tok_per_s"], "tok_per_item": r["prompt_tokens"] / r["n"],
                             "how": f"vLLM readout, max_num_seqs {r.get('max_num_seqs')}"}
    f = os.path.join(d, "tp_summary.jsonl")
    if os.path.exists(f):
        for line in open(f):
            r = json.loads(line)
            if r.get("errors") or not r.get("decisions_per_s"):
                continue
            k = r["model"].split("-c")[0] if "-c" in r["model"] else r["model"]
            cur = tp.get(k)
            if cur is None or r["decisions_per_s"] > cur["dps"]:
                tpi = (r["prompt_tokens"] / r["n"]) if r.get("prompt_tokens") else None
                tp[k] = {"dps": r["decisions_per_s"], "tps": r.get("tok_per_s"), "tok_per_item": tpi,
                         "how": f"llama.cpp /v1/systemone, concurrency {r.get('concurrency')} ({r['model']})"}
    return tp


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", help="EXP=dir")
    ap.add_argument("--price", action="append", default=[], help="EXP=all-in $/h of the VM")
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    price = dict(p.split("=") for p in args.price)
    table = []
    for spec in args.dirs:
        exp, d = spec.split("=", 1)
        files = sorted(set(glob.glob(os.path.join(d, "*-p-*.jsonl")) + glob.glob(os.path.join(d, "q_*.preds.jsonl"))))
        if not files:
            continue
        mets = score(files)
        tp = throughput(d)
        by = collections.defaultdict(dict)
        for m in mets:
            if m.get("n"):
                by[m["model"]][m["dataset"]] = m
        for model, ds in by.items():
            t = tp.get(model) or tp.get(model.split("-c")[0]) or {}
            p = float(price.get(exp, "nan"))
            dps = t.get("dps")
            jev = [v for k, v in ds.items() if k.startswith(("jev_", "jb_"))]
            table.append({
                "exp": exp, "model": model,
                "b77_8": ds.get("banking77_8", {}).get("acc"), "b77_8_n": ds.get("banking77_8", {}).get("n"),
                "b77_8_ece": ds.get("banking77_8", {}).get("ece"), "b77_8_ece_T": ds.get("banking77_8", {}).get("ece_T_heldout"),
                "b77_77": ds.get("banking77_77", {}).get("acc"),
                "jev_mean": round(sum(v["acc"] for v in jev) / len(jev), 4) if jev else None, "jev_sets": len(jev),
                "dps": dps, "tok_s": t.get("tps"), "tok_per_item": round(t["tok_per_item"], 1) if t.get("tok_per_item") else None,
                "usd_per_1k": round(p / (dps * 3600) * 1000, 4) if dps else None,
                "usd_per_M_tok": round(p / (t["tps"] * 3600) * 1e6, 3) if t.get("tps") else None,
                "how": t.get("how", "")})
    cols = ["exp", "model", "b77_8", "b77_8_n", "b77_8_ece", "b77_8_ece_T", "b77_77", "jev_mean", "jev_sets", "dps",
            "tok_s", "tok_per_item", "usd_per_1k", "usd_per_M_tok", "how"]
    if args.md:
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join(["---"] * 2 + ["---:"] * (len(cols) - 3) + ["---"]) + "|")
    for r in table:
        vals = ["" if r[c] is None else str(r[c]) for c in cols]
        print("| " + " | ".join(vals) + " |" if args.md else "\t".join(vals))


if __name__ == "__main__":
    main()
