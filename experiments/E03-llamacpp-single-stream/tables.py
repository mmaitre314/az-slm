#!/usr/bin/env python3
"""E03 tables from llama-bench.jsonl (standard library only).

  python3 tables.py [--price 1.681] [--spot-price 0.325]

Prices are all-in USD/hour (VM + 0.018 disk and IP). FLOP/token = 2 x model_n_params (upper bound:
includes the embedding table and the unused MTP layer).
"""
import argparse
import collections
import json
import pathlib
import re

HERE = pathlib.Path(__file__).parent
AMX_INT8_TOPS = 59.0   # 8 cores x 3.6 GHz, from the experiment brief
AMX_BF16_TFLOPS = 29.0
ORDER = ["Q4_0", "IQ4_XS", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "bf16"]


def usd(tps, price):
    return price / (tps * 3600) * 1e6


def table(headers, rows, left=1):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i >= left else "---" for i in range(len(headers))) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, default=1.681)
    ap.add_argument("--spot-price", type=float, default=0.325)
    a = ap.parse_args()
    d = collections.defaultdict(dict)
    meta = {}
    for line in (HERE / "llama-bench.jsonl").read_text().splitlines():
        r = json.loads(line)
        kind = "pp" if r["n_prompt"] else "tg"
        d[r["quant"]][(r["build"], kind)] = (r["avg_ts"], r["stddev_ts"])
        meta[r["quant"]] = (r["model_size"], r["model_n_params"])

    def v(q, b, k):
        return d[q][(b, k)]

    print("### Throughput (llama-bench, 1 sequence, 8 threads, 3 repetitions, mean ± sd)\n")
    rows = []
    for q in ORDER:
        sz = meta[q][0] / 1e9
        rows.append([q, f"{sz:.1f}",
                     "%.1f ± %.2f" % v(q, "build", "pp"), "%.1f ± %.2f" % v(q, "build-noamx", "pp"),
                     f"{v(q, 'build', 'pp')[0] / v(q, 'build-noamx', 'pp')[0]:.2f}x",
                     "%.2f ± %.2f" % v(q, "build", "tg"), "%.2f ± %.2f" % v(q, "build-noamx", "tg"),
                     f"{v(q, 'build', 'tg')[0] / v(q, 'build-noamx', 'tg')[0]:.2f}x"])
    print(table(["quant", "file GB", "pp512 AMX (tok/s)", "pp512 no-AMX (tok/s)", "AMX/no-AMX",
                 "tg128 AMX (tok/s)", "tg128 no-AMX (tok/s)", "AMX/no-AMX"], rows))

    print("\n### Physical limits\n")
    rows = []
    for q in ORDER:
        sz, npar = meta[q]
        gf = 2 * npar / 1e9
        r = [q, f"{gf:.1f}"]
        for b in ("build", "build-noamx"):
            r.append(f"{v(q, b, 'tg')[0] * sz / 1e9:.0f}")
        for b in ("build", "build-noamx"):
            r.append(f"{v(q, b, 'pp')[0] * gf / 1e3:.2f}")
        r.append("n/a (no AMX path)" if q == "bf16" else f"{v(q, 'build', 'pp')[0] * gf / 1e3 / AMX_INT8_TOPS * 100:.1f}%")
        rows.append(r)
    print(table(["quant", "GFLOP/token", "decode GB/s AMX (tg x file size)", "decode GB/s no-AMX",
                 "prefill TFLOPS AMX", "prefill TFLOPS no-AMX", "AMX build, % of 59 TOPS INT8 peak"], rows))

    print("\n### Cost per million tokens, one sequence (all-in %.3f USD/h on-demand, %.3f USD/h Spot)\n"
          % (a.price, a.spot_price))
    rows = []
    for q in ORDER:
        for b in ("build", "build-noamx"):
            pp, tg = v(q, b, "pp")[0], v(q, b, "tg")[0]
            t_req = 512 / pp + 128 / tg   # s per 512-in/128-out request
            row = [q, "AMX" if b == "build" else "no-AMX"]
            for p in (a.price, a.spot_price):
                row += [f"{usd(pp, p):.1f}", f"{usd(tg, p):.1f}"]
            for p in (a.price, a.spot_price):
                row.append(f"{p * t_req / 3600 / 640 * 1e6:.1f}")
            row.append(f"{t_req:.0f}")
            rows.append(row)
    print(table(["quant", "build", "input USD/M on-demand", "output USD/M on-demand", "input USD/M Spot",
                 "output USD/M Spot", "blended 512+128 USD/M on-demand", "blended USD/M Spot", "s per request"], rows, left=2))

    print("\n### Weight buffers in the verbose llama-bench logs (MiB)\n")
    cur, bufs = None, collections.defaultdict(dict)
    for line in (HERE / "model-buffers.txt").read_text().splitlines():
        if line.startswith("## "):
            m = re.match(r"## llama-bench-(.+)-(build(?:-noamx)?)-t8", line)
            cur = (m.group(1), m.group(2))
        else:
            m = re.match(r"(AMX|CPU_Mapped|CPU_REPACK) model buffer size =\s*([\d.]+) MiB", line)
            if m:
                bufs[cur].setdefault(m.group(1), []).append(float(m.group(2)))
    rows = []
    for q in ["Q4_0", "IQ4_XS", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "bf16"]:
        r = [q, f"{meta[q][0] / 2**20:.0f}"]
        for b in ("build", "build-noamx"):
            x = bufs[(q, b)]
            r.append("%s" % (", ".join(f"{s:.0f}" for s in x.get("AMX", [])) or "-") if b == "build"
                     else (", ".join(f"{s:.0f}" for s in x.get("CPU_REPACK", [])) or "-"))
        rows.append(r)
    print(table(["quant", "GGUF file MiB", "AMX buffer MiB (build)", "CPU_REPACK buffer MiB (build-noamx)"], rows))


if __name__ == "__main__":
    main()
