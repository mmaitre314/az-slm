#!/usr/bin/env python3
"""Convert the E16 parquet test splits (openai/gsm8k, cais/mmlu) to JSONL.

Runs inside the vLLM container (python3) or on the host with pyarrow/pandas available. Reads
/mnt/data/datasets/{gsm8k,mmlu}/**/test-*.parquet and writes /mnt/data/datasets/{gsm8k,mmlu}_test.jsonl,
one object per row in the parquet's own order. Prints the row counts.

usage: e16_convert.py [DATA_DIR]
"""
import glob
import json
import sys


def read_rows(path):
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    except ImportError:
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")


def clean(v):
    if hasattr(v, "tolist"):  # numpy arrays/scalars from pandas
        return v.tolist()
    return v


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/datasets"
    for name in ("gsm8k", "mmlu"):
        files = sorted(glob.glob(f"{root}/{name}/**/test-*.parquet", recursive=True))
        if len(files) != 1:
            raise SystemExit(f"{name}: expected exactly one test parquet, found {files}")
        rows = read_rows(files[0])
        out = f"{root}/{name}_test.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps({k: clean(v) for k, v in r.items()}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows from {files[0]} -> {out}", flush=True)


if __name__ == "__main__":
    main()
