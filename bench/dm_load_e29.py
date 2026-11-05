#!/usr/bin/env python3
"""dm_load.py with E29 differences (dm_load.py is shared and unchanged): an item that carries a `wire`
question (bench/dm_jevbench_e29.py) is sent as that typed question (choice with descriptions, noul,
score); noul/score responses are parsed to probabilities over item["options"]. Same arguments.
Also records the first raw response in --summary's directory as <summary>.sample.json."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dm_load  # noqa: E402

_orig_request = dm_load.request_of
_orig_find = dm_load.find_probs
_sample = {"done": False}


def request_of(it, model):
    if "wire" in it:
        return {"model": model, "state": it["state"], "questions": {"q": it["wire"]}}
    return _orig_request(it, model)


def find_probs(resp, options):
    if not _sample["done"]:
        _sample["done"] = True
        p = None
        for i, a in enumerate(sys.argv):
            if a == "--summary" and i + 1 < len(sys.argv):
                p = sys.argv[i + 1] + ".sample.json"
        if p:
            with open(p, "w") as f:
                json.dump(resp, f)
    r = _orig_find(resp, options)
    if r is not None:
        return r
    return extra(resp, options)


def extra(resp, options):
    """Fallbacks for shapes dm_load does not know; extended after reading llama-server's real output."""
    def walk(x):
        if isinstance(x, dict):
            yield x
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, list):
            for v in x:
                yield from walk(v)
    for d in walk(resp):
        if isinstance(d.get("noul"), (int, float)) and len(options) == 2:
            return [float(d["noul"]), 1.0 - float(d["noul"])]
    return None


dm_load.request_of = request_of
dm_load.find_probs = find_probs

if __name__ == "__main__":
    dm_load.main()
