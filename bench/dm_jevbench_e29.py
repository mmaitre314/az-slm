#!/usr/bin/env python3
"""Convert Praveenrajus/jev-bench test splits (data files) into canonical decision items (E29), stdlib only.

Record: {id, source, primitive: choice|noul|score, state (JSON text), question (JSON text of the wire
question: {type, instructions, criteria?}), label, soft_label}. Output item (dm_data.py format plus "wire"):
  {id, dataset: "jb_<config>", state, question, qtype, options, gold, wire}
choice -> options = criteria keys; noul -> ["yes","no"] (label "1" = yes); score -> the levels in order
(label = index). `wire` is the original typed question; bench/dm_load_e29.py sends it as is.
Sampling is deterministic: items sorted by sha256(id), first N taken.
  dm_jevbench_e29.py [--out DIR] [--n 200] [--configs banking77,clinc150,...]
"""
import argparse
import hashlib
import json
import os
import urllib.request

URL = "https://huggingface.co/datasets/Praveenrajus/jev-bench/resolve/main/data/%s/test.jsonl"
DEFAULT = "banking77,clinc150,massive,go_emotions,mmlu,ledgar,boolq,helpsteer2_helpfulness"


def convert(rec, cfg):
    q = json.loads(rec["question"])
    st = rec["state"]
    try:
        st = json.loads(st)
    except Exception:
        pass
    state = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
    prim = rec["primitive"]
    if prim == "choice":
        options = list(q["criteria"])
        gold = options.index(rec["label"])
    elif prim == "noul":
        options = ["yes", "no"]
        gold = 0 if str(rec["label"]) == "1" else 1
    elif prim == "score":
        crit = q["criteria"]
        options = [str(i) for i in range(len(crit))] if isinstance(crit, list) else list(crit)
        gold = int(rec["label"])
        if not 0 <= gold < len(options):
            return None
    else:
        return None
    return {"id": rec["id"], "dataset": "jb_" + cfg, "state": state, "question": q["instructions"], "qtype": prim,
            "options": options, "gold": gold, "wire": q}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/mnt/data/dm/data")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--configs", default=DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for cfg in a.configs.split(","):
        try:
            with urllib.request.urlopen(URL % cfg, timeout=180) as r:
                lines = r.read().decode("utf-8").splitlines()
            items = []
            for l in lines:
                if l.strip():
                    try:
                        it = convert(json.loads(l), cfg)
                    except Exception:
                        it = None
                    if it:
                        items.append(it)
            items.sort(key=lambda it: hashlib.sha256(it["id"].encode()).hexdigest())
            items = items[: a.n]
            with open(os.path.join(a.out, "jb_%s.jsonl" % cfg), "w") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(cfg, "records", len(lines), "items", len(items), "options", sorted({len(i["options"]) for i in items})[:6])
        except Exception as e:
            print(cfg, "FAILED", e)


if __name__ == "__main__":
    main()
