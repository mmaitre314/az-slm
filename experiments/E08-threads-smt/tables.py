#!/usr/bin/env python3
"""E08 tables from llama-bench.jsonl (build vs build-native-noamx) and llama-threads.jsonl (standard library only).

  python3 tables.py [--price 1.681] [--spot-price 0.325]

Prices are all-in USD/hour (VM + 0.018 disk and IP). FLOP/token = 2 x 27.32 G parameters (upper bound: includes
the embedding table and the unused MTP layer). Decode GB/s = tg x GGUF file size (a lower bound on the traffic).
"""
import argparse
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).parent
E03 = HERE.parent / "E03-llamacpp-single-stream" / "llama-bench.jsonl"
N_PARAMS = 27_320_697_856
AMX_INT8_TOPS = 59.0     # 8 cores x 3.6 GHz x 2048 ops/cycle
VNNI_INT8_TOPS = 7.4     # 8 cores x 3.6 GHz x 256 ops/cycle (two 512-bit vpdpbusd ports)
QUANTS = ["Q4_0", "IQ4_XS", "Q4_K_M", "Q8_0"]


def usd(tps, price):
    return price / (tps * 3600) * 1e6


def blended(pp, tg, price, i=512, o=128):
    return price * (i / pp + o / tg) / 3600 / (i + o) * 1e6


def table(headers, rows, left=1):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i >= left else "---" for i in range(len(headers))) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, default=1.681)
    ap.add_argument("--spot-price", type=float, default=0.325)
    a = ap.parse_args()
    d, size = collections.defaultdict(dict), {}
    for r in load(HERE / "llama-bench.jsonl"):
        d[r["quant"]][(r["build"], "pp" if r["n_prompt"] else "tg")] = (r["avg_ts"], r["stddev_ts"])
        size[r["quant"]] = r["model_size"]

    def v(q, b, k):
        return d[q][(b, k)]

    print("### A/B: AMX build vs clean control (llama-bench, 1 sequence, 8 threads, 3 repetitions, mean ± sd)\n")
    rows = []
    for q in QUANTS:
        rows.append([q, f"{size[q] / 1e9:.1f}",
                     "%.1f ± %.2f" % v(q, "build", "pp"), "%.1f ± %.2f" % v(q, "build-native-noamx", "pp"),
                     f"{v(q, 'build', 'pp')[0] / v(q, 'build-native-noamx', 'pp')[0]:.2f}x",
                     "%.2f ± %.3f" % v(q, "build", "tg"), "%.2f ± %.3f" % v(q, "build-native-noamx", "tg"),
                     f"{v(q, 'build', 'tg')[0] / v(q, 'build-native-noamx', 'tg')[0]:.2f}x"])
    print(table(["quant", "file GB", "pp512 AMX (tok/s)", "pp512 control (tok/s)", "AMX/control",
                 "tg128 AMX (tok/s)", "tg128 control (tok/s)", "AMX/control"], rows))

    print("\n### Physical limits\n")
    rows = []
    for q in QUANTS:
        gb = size[q] / 1e9
        pa, pc = v(q, "build", "pp")[0], v(q, "build-native-noamx", "pp")[0]
        ta, tc = v(q, "build", "tg")[0], v(q, "build-native-noamx", "tg")[0]
        fa, fc = pa * 2 * N_PARAMS / 1e12, pc * 2 * N_PARAMS / 1e12
        rows.append([q, f"{fa:.2f}", f"{100 * fa / AMX_INT8_TOPS:.1f}%", f"{fc:.2f}", f"{100 * fc / VNNI_INT8_TOPS:.0f}%",
                     f"{ta * gb:.0f}", f"{tc * gb:.0f}"])
    print(table(["quant", "prefill TFLOPS AMX", "% of 59 TOPS AMX peak", "prefill TFLOPS control",
                 "% of 7.4 TOPS VNNI peak", "decode GB/s AMX (tg x file)", "decode GB/s control"], rows))

    print("\n### Threads, SMT and pinning (build = AMX, 3 repetitions, mean ± sd)\n")
    t = collections.defaultdict(dict)
    for r in load(HERE / "llama-threads.jsonl"):
        t[r["quant"]][(r["variant"], "pp" if r["n_prompt"] else "tg")] = (r["avg_ts"], r["stddev_ts"])
        size.setdefault(r["quant"], r["model_size"])
    rows = []
    for q in ["Q4_K_M", "Q4_0"]:
        p0, g0 = t[q][("t8", "pp")][0], t[q][("t8", "tg")][0]
        for var, label in [("t8", "8 threads"), ("t8-pin", "8 threads, one hyperthread per core, strict"), ("t16", "16 threads")]:
            p, g = t[q][(var, "pp")], t[q][(var, "tg")]
            rows.append([q, label, "%.2f ± %.2f" % p, f"{100 * (p[0] / p0 - 1):+.1f}%", f"{p[0] * 2 * N_PARAMS / 1e12:.2f}",
                         "%.3f ± %.3f" % g, f"{100 * (g[0] / g0 - 1):+.1f}%", f"{g[0] * size[q] / 1e9:.0f}"])
    print(table(["quant", "variant", "pp512 (tok/s)", "vs 8 threads", "prefill TFLOPS", "tg128 (tok/s)", "vs 8 threads",
                 "decode GB/s"], rows, left=2))

    print("\n### Instance variance: this VM (bench-lc2) over E03's VM (bench-e16v7), same `build`, 8 threads\n")
    e3 = collections.defaultdict(dict)
    for r in load(E03):
        e3[r["quant"]][(r["build"], "pp" if r["n_prompt"] else "tg")] = r["avg_ts"]
    rows = []
    for q in QUANTS:
        rows.append([q, f"{e3[q][('build', 'pp')]:.1f}", f"{v(q, 'build', 'pp')[0]:.1f}",
                     f"{100 * (v(q, 'build', 'pp')[0] / e3[q][('build', 'pp')] - 1):+.0f}%",
                     f"{e3[q][('build', 'tg')]:.2f}", f"{v(q, 'build', 'tg')[0]:.2f}",
                     f"{100 * (v(q, 'build', 'tg')[0] / e3[q][('build', 'tg')] - 1):+.0f}%"])
    print(table(["quant", "pp512 e16v7 (tok/s)", "pp512 lc2 (tok/s)", "lc2 vs e16v7", "tg128 e16v7 (tok/s)",
                 "tg128 lc2 (tok/s)", "lc2 vs e16v7"], rows))


    print("\n### AMX gain against two different baselines (E03: explicit-flag `build-noamx` on bench-e16v7; E08: clean control on bench-lc2)\n")
    e3b = collections.defaultdict(dict)
    for r in load(E03):
        e3b[r["quant"]][(r["build"], "pp" if r["n_prompt"] else "tg")] = r["avg_ts"]
    rows = []
    for q in QUANTS:
        rows.append([q,
                     f"{e3b[q][('build', 'pp')] / e3b[q][('build-noamx', 'pp')]:.2f}x", f"{v(q, 'build', 'pp')[0] / v(q, 'build-native-noamx', 'pp')[0]:.2f}x",
                     f"{e3b[q][('build', 'tg')] / e3b[q][('build-noamx', 'tg')]:.2f}x", f"{v(q, 'build', 'tg')[0] / v(q, 'build-native-noamx', 'tg')[0]:.2f}x",
                     f"{e3b[q][('build-noamx', 'tg')]:.2f}", f"{v(q, 'build-native-noamx', 'tg')[0]:.2f}"])
    print(table(["quant", "prefill gain vs build-noamx (E03)", "prefill gain vs control (E08)", "decode gain vs build-noamx (E03)",
                 "decode gain vs control (E08)", "tg128 build-noamx (E03, e16v7, tok/s)", "tg128 control (E08, lc2, tok/s)"], rows))

    print("\n### Cost per token (single sequence, bench-lc2, USD per million tokens)\n")
    rows = []
    cases = [(q, b, "8", v(q, b, "pp")[0], v(q, b, "tg")[0]) for q in QUANTS for b in ("build", "build-native-noamx")]
    cases += [(q, "build", "16", t[q][("t16", "pp")][0], t[q][("t16", "tg")][0]) for q in ["Q4_K_M", "Q4_0"]]
    for q, b, thr, pp, tg in cases:
        rows.append([q, "AMX" if b == "build" else "no-AMX control", thr,
                     f"{usd(pp, a.price):.1f}", f"{usd(tg, a.price):.1f}", f"{blended(pp, tg, a.price):.1f}",
                     f"{usd(pp, a.spot_price):.1f}", f"{usd(tg, a.spot_price):.1f}", f"{blended(pp, tg, a.spot_price):.1f}",
                     f"{512 / pp + 128 / tg:.0f}"])
    print(table(["quant", "build", "threads", "input on-demand", "output on-demand", "blended 512+128 on-demand",
                 "input Spot", "output Spot", "blended 512+128 Spot", "s per request"], rows, left=2))


if __name__ == "__main__":
    main()
