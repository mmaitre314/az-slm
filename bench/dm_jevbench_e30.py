#!/usr/bin/env python3
"""Convert Praveenrajus/jev-bench test records to canonical decision items (E30), stdlib only.

Fetches data/<config>/test.jsonl over HTTPS as plain data. Record fields used: id, primitive
(choice|noul|score), state (JSON text), question (JSON text: type, instructions, criteria), label.
  choice: options = criteria keys (with ": description" or the description alone when keys are
          single letters), gold = index of label
  noul:   options = ["yes", "no"]; label "1" -> yes (index 0), "0" -> no (index 1)
  score:  options = criteria levels in order, gold = int(label)
Items with more options than 52 are skipped (counted). Deterministic sample of N items per config
(sorted by sha256 of id). Writes OUT/jev_<config>.jsonl with dataset "jev_<config>".
  dm_jevbench_e30.py [--out DIR] [--n 200] [--configs a,b,...]
"""
import argparse
import hashlib
import json
import os
import urllib.request

URL = "https://huggingface.co/datasets/Praveenrajus/jev-bench/resolve/main/data/{c}/test.jsonl"
DEFAULT = "go_emotions,mmlu,sst5,sms_spam,boolq,arc_challenge,banking77,massive,clinc150,ledgar"
MAXOPT = 52


def render_state(s):
    try:
        v = json.loads(s)
    except Exception:
        return s
    if isinstance(v, dict):
        return "\n".join(f"{k}: {x}" for k, x in v.items())
    return str(v)


def convert(rec, config):
    q = json.loads(rec["question"])
    prim = rec["primitive"]
    crit = q.get("criteria")
    instr = q.get("instructions", "")
    if prim == "choice":
        keys = list(crit)
        if all(len(k) == 1 for k in keys) and all(crit[k] for k in keys):
            opts = [str(crit[k]) for k in keys]
        else:
            opts = [k + (f": {crit[k]}" if crit[k] else "") for k in keys]
        gold = keys.index(rec["label"])
    elif prim == "noul":
        opts = ["yes", "no"]
        gold = 0 if str(rec["label"]) == "1" else 1
    elif prim == "score":
        opts = [str(x) for x in crit]
        gold = int(rec["label"])
        instr += " (ordered options, lowest level first)"
    else:
        raise ValueError(prim)
    return {"id": rec["id"], "dataset": f"jev_{config}", "state": render_state(rec["state"]), "question": instr,
            "qtype": prim, "options": opts, "gold": gold, "label": str(rec["label"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/mnt/data/dm/data")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--configs", default=DEFAULT)
    ap.add_argument("--max-configs", type=int, default=6, help="stop after this many configs with items")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    done = 0
    for c in args.configs.split(","):
        if done >= args.max_configs:
            break
        try:
            with urllib.request.urlopen(URL.format(c=c), timeout=120) as r:
                recs = [json.loads(l) for l in r.read().decode("utf-8").splitlines() if l.strip()]
        except Exception as e:
            print(f"{c}: fetch failed: {e}")
            continue
        items, skipped = [], 0
        for rec in sorted(recs, key=lambda x: hashlib.sha256(x["id"].encode()).hexdigest()):
            try:
                it = convert(rec, c)
            except Exception as e:
                skipped += 1
                continue
            if len(it["options"]) > MAXOPT:
                skipped += 1
                continue
            items.append(it)
            if len(items) >= args.n:
                break
        print(f"{c}: {len(recs)} records, kept {len(items)}, skipped/failed {skipped}")
        if not items:
            continue
        with open(os.path.join(args.out, f"jev_{c}.jsonl"), "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        done += 1


if __name__ == "__main__":
    main()
