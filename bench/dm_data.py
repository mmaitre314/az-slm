#!/usr/bin/env python3
"""Prepare canonical decision items (E28-E30), standard library only. Runs on the VM.

Every item is one typed question about one state, in the Jev "choice" shape:
  {"id", "dataset", "state", "question", "qtype": "choice", "options": [...], "gold": <index>}
Datasets written to OUT (default /mnt/data/dm/data):
  banking77_77.jsonl    BANKING77 test (3,080 messages), all 77 intents as options (fixed order)
  banking77_8.jsonl     same messages, 8 options: the gold intent + 7 others drawn per item
                        (deterministic seed), shuffled: the "changing menu" setting
  banking77_train.jsonl BANKING77 train (10,003), for models that fit on labels
  *_small.jsonl         a deterministic subset (SMALL per intent, default 4 -> 308 items) for slow models
JevBench is converted separately (dm_jevbench.py), once its format is known.
  dm_data.py [--out DIR] [--small 4]
"""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import urllib.request

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
QUESTION = "Which customer-support intent does this banking customer's message express?"


def fetch(name):
    with urllib.request.urlopen(BASE + name, timeout=60) as r:
        return r.read().decode("utf-8")


def human(label):
    return label.replace("_", " ")


def seed_of(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/mnt/data/dm/data")
    ap.add_argument("--small", type=int, default=4, help="items per intent in the *_small subsets")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cats = json.loads(fetch("categories.json"))
    labels = sorted(cats)
    assert len(labels) == 77, len(labels)
    names = [human(c) for c in labels]
    for split in ("test", "train"):
        rows = list(csv.DictReader(io.StringIO(fetch(f"{split}.csv"))))
        items77, items8 = [], []
        for i, r in enumerate(rows):
            gold = labels.index(r["category"])
            iid = f"b77-{split}-{i}"
            items77.append({"id": iid, "dataset": "banking77_77", "state": r["text"].strip(), "question": QUESTION,
                            "qtype": "choice", "options": names, "gold": gold, "label": r["category"]})
            rnd = random.Random(seed_of(iid))
            others = rnd.sample([k for k in range(77) if k != gold], 7)
            opts = [gold] + others
            rnd.shuffle(opts)
            items8.append({"id": iid, "dataset": "banking77_8", "state": r["text"].strip(), "question": QUESTION,
                           "qtype": "choice", "options": [names[k] for k in opts], "gold": opts.index(gold),
                           "label": r["category"]})
        if split == "train":
            with open(os.path.join(args.out, "banking77_train.jsonl"), "w") as f:
                for it in items77:
                    f.write(json.dumps(dict(it, dataset="banking77_train")) + "\n")
            continue
        for name, items in (("banking77_77", items77), ("banking77_8", items8)):
            with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
                for it in items:
                    f.write(json.dumps(it) + "\n")
            per, small = {}, []
            for it in sorted(items, key=lambda x: seed_of(x["id"])):
                if per.get(it["label"], 0) < args.small:
                    per[it["label"]] = per.get(it["label"], 0) + 1
                    small.append(it)
            with open(os.path.join(args.out, f"{name}_small.jsonl"), "w") as f:
                for it in small:
                    f.write(json.dumps(it) + "\n")
            print(f"{name}: {len(items)} items, small {len(small)}")
    print(f"banking77_train: written; labels {len(labels)}")


if __name__ == "__main__":
    main()
