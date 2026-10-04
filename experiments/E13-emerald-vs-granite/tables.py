#!/usr/bin/env python3
"""E13 tables (standard library only). Reads this directory's llama-bench.jsonl / llama-batched.jsonl and the
Granite Rapids raw files of E03 (bench-e16v7), E05 (bench-e16v7) and E08 (bench-lc2).

  python3 tables.py

Prices are all-in USD/hour (VM + 0.018 disk and IP): v6 1.326 / Spot 0.260, v7 1.681 / Spot 0.325.
FLOP per token = 2 x 27.32 G parameters = 54.6 GFLOP (upper bound: includes the embedding table and the MTP block).
"""
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent
P = {"v6": 1.326, "v7": 1.681}
SPOT = {"v6": 0.260, "v7": 0.325}
ORDER = ["Q4_0", "IQ4_XS", "Q4_K_M", "Q8_0"]
GFLOP = 54.6
# v6: all-core turbo 3.0 GHz (Learn, Edsv6 page); v7: 3.6 GHz (brief). 8 cores. AMX INT8 = 2048 ops/cycle/core, VNNI = 256.
PEAK = {"v6": (8 * 3.0e9 * 2048 / 1e12, 8 * 3.0e9 * 256 / 1e12), "v7": (8 * 3.6e9 * 2048 / 1e12, 8 * 3.6e9 * 256 / 1e12)}


def load(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip().startswith("{")]


def bench(path):
    d, size = {}, {}
    for r in load(path):
        d[(r["quant"], r["build"], "pp" if r["n_prompt"] else "tg")] = (r["avg_ts"], r["stddev_ts"])
        size[r["quant"]] = r["model_size"]
    return d, size


def batched(path):
    return {(r["quant"], r["pl"]): r for r in load(path) if r["build"] == "build-native-noamx"}


def usd(tps, p):
    return p / (tps * 3600) * 1e6


def blended(pp, tg, p, i=512, o=128):
    return p * (i / pp + o / tg) / 3600 / (i + o) * 1e6


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(headers))) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


v6, size = bench(HERE / "llama-bench.jsonl")
e16, _ = bench(EXP / "E03-llamacpp-single-stream/llama-bench.jsonl")   # builds: build, build-noamx
lc2, _ = bench(EXP / "E08-threads-smt/llama-bench.jsonl")              # builds: build, build-native-noamx
b6 = batched(HERE / "llama-batched.jsonl")
b7 = batched(EXP / "E05-llamacpp-batched/llama-batched.jsonl")
BR = P["v7"] / P["v6"]
BRS = SPOT["v7"] / SPOT["v6"]


def main():
    print("### 1. bench-v6, one sequence (llama-bench, 8 threads, 3 repetitions, mean ± sd)\n")
    rows = []
    for q in ORDER:
        g = lambda b, k: v6[(q, b, k)]
        rows.append([q, f"{size[q] / 1e9:.1f}", "%.1f ± %.2f" % g("build", "pp"), "%.1f ± %.2f" % g("build-native-noamx", "pp"),
                     f"{g('build', 'pp')[0] / g('build-native-noamx', 'pp')[0]:.2f}x",
                     "%.2f ± %.3f" % g("build", "tg"), "%.2f ± %.3f" % g("build-native-noamx", "tg"),
                     f"{g('build', 'tg')[0] / g('build-native-noamx', 'tg')[0]:.2f}x"])
    print(table(["quant", "file GB", "pp512 AMX (tok/s)", "pp512 control (tok/s)", "AMX/control",
                 "tg128 AMX (tok/s)", "tg128 control (tok/s)", "AMX/control"], rows))

    print("\n### 2. Physical limits on bench-v6\n")
    rows = []
    for q in ORDER:
        g = lambda b, k: v6[(q, b, k)][0]
        gb = size[q] / 1e9
        tf = lambda b: g(b, "pp") * GFLOP / 1e3
        rows.append([q, f"{g('build', 'tg') * gb:.0f}", f"{g('build-native-noamx', 'tg') * gb:.0f}",
                     f"{tf('build'):.2f}", f"{100 * tf('build') / PEAK['v6'][0]:.1f}%",
                     f"{tf('build-native-noamx'):.2f}", f"{100 * tf('build-native-noamx') / PEAK['v6'][1]:.0f}%"])
    print(table(["quant", "decode GB/s AMX (tg x file)", "decode GB/s control", "prefill TFLOPS AMX",
                 f"% of {PEAK['v6'][0]:.0f} TOPS AMX INT8 peak", "prefill TFLOPS control", f"% of {PEAK['v6'][1]:.1f} TOPS VNNI peak"], rows))

    print("\n### 3. Same `build` (AMX), one sequence, three instances: speed and v7/v6 speedup\n")
    rows = []
    for kind, unit in (("pp", "pp512"), ("tg", "tg128")):
        for q in ORDER:
            a, b, c = v6[(q, "build", kind)][0], e16[(q, "build", kind)][0], lc2[(q, "build", kind)][0]
            rows.append([unit, q, f"{a:.2f}", f"{b:.2f}", f"{c:.2f}", f"{b / a:.2f}x", f"{c / a:.2f}x", f"{c / b:.2f}x"])
    print(table(["test", "quant", "v6 (tok/s)", "v7 e16v7 (tok/s)", "v7 lc2 (tok/s)", "e16v7/v6", "lc2/v6", "lc2/e16v7"], rows))

    print("\n### 4. Clean control (`build-native-noamx`), one sequence: v6 vs lc2 (E08); E05 decode on e16v7 for reference\n")
    rows = []
    for kind, unit in (("pp", "pp512"), ("tg", "tg128")):
        for q in ORDER:
            a, c = v6[(q, "build-native-noamx", kind)][0], lc2[(q, "build-native-noamx", kind)][0]
            e5 = ""
            if kind == "tg" and (q, 1) in b7:
                e5 = f"{b7[(q, 1)]['speed_tg']:.2f} ({b7[(q, 1)]['speed_tg'] / b6[(q, 1)]['speed_tg']:.2f}x of v6 {b6[(q, 1)]['speed_tg']:.2f})"
            rows.append([unit, q, f"{a:.2f}", f"{c:.2f}", f"{c / a:.2f}x", e5 or "-"])
    print(table(["test", "quant", "v6 (tok/s)", "v7 lc2 (tok/s)", "lc2/v6", "E05 npl=1 decode on e16v7 (tok/s), ratio to v6 npl=1 decode"], rows))

    print("\n### 5. Batched control build, 128-token prompts, 128 generated per sequence: v6 and e16v7 (E05)\n")
    rows = []
    for q in ["Q4_0", "Q4_K_M", "Q8_0"]:
        for n in (1, 4, 8, 16, 32):
            a, b = b6[(q, n)], b7[(q, n)]
            rows.append([q, n, f"{a['speed_pp']:.1f}", f"{b['speed_pp']:.1f}", f"{b['speed_pp'] / a['speed_pp']:.2f}x",
                         f"{a['speed_tg']:.2f}", f"{b['speed_tg']:.2f}", f"{b['speed_tg'] / a['speed_tg']:.2f}x",
                         f"{a['speed_tg'] / n:.2f}", f"{a['t']:.0f}", f"{b['t']:.0f}"])
    print(table(["quant", "sequences", "prefill v6 (tok/s)", "prefill e16v7 (tok/s)", "e16v7/v6", "decode v6 all seq (tok/s)",
                 "decode e16v7 all seq (tok/s)", "e16v7/v6", "decode v6 per seq (tok/s)", "wall v6 (s)", "wall e16v7 (s)"], rows))

    print("\n### 6. Cost, one sequence (USD per million tokens; blended = 512 in + 128 out per request)\n")
    rows = []
    for q in ORDER:
        for label, src, bld, key in (("v6 AMX", v6, "build", "v6"), ("v6 control", v6, "build-native-noamx", "v6"),
                                     ("e16v7 AMX", e16, "build", "v7"), ("lc2 AMX", lc2, "build", "v7"),
                                     ("lc2 control", lc2, "build-native-noamx", "v7")):
            pp, tg = src[(q, bld, "pp")][0], src[(q, bld, "tg")][0]
            rows.append([q, label, f"{usd(pp, P[key]):.1f}", f"{usd(tg, P[key]):.1f}", f"{blended(pp, tg, P[key]):.1f}",
                         f"{usd(pp, SPOT[key]):.1f}", f"{usd(tg, SPOT[key]):.1f}", f"{blended(pp, tg, SPOT[key]):.1f}",
                         f"{(512 / pp + 128 / tg):.0f}"])
    print(table(["quant", "VM / build", "input on-demand", "output on-demand", "blended on-demand", "input Spot", "output Spot",
                 "blended Spot", "s per request"], rows))

    print("\n### 7. Cost, batched control build (USD per million tokens)\n")
    print("Blended 512+128 uses the 128-token prefill rate for the 512-token prompt (as in E05); blended 128+128 uses the measured wall time.\n")
    rows = []
    for q in ["Q4_0", "Q4_K_M", "Q8_0"]:
        for n in (1, 4, 8, 16, 32):
            r = b6[(q, n)]
            m = P["v6"] * r["t"] / 3600 / (n * 256) * 1e6
            ms = SPOT["v6"] * r["t"] / 3600 / (n * 256) * 1e6
            rows.append([q, n, f"{usd(r['speed_pp'], P['v6']):.1f}", f"{usd(r['speed_tg'], P['v6']):.1f}",
                         f"{blended(r['speed_pp'], r['speed_tg'], P['v6']):.1f}", f"{m:.1f}",
                         f"{usd(r['speed_pp'], SPOT['v6']):.1f}", f"{usd(r['speed_tg'], SPOT['v6']):.1f}",
                         f"{blended(r['speed_pp'], r['speed_tg'], SPOT['v6']):.1f}", f"{ms:.1f}"])
    print(table(["quant", "sequences", "input on-demand", "output on-demand", "blended 512+128 on-demand", "blended 128+128 on-demand",
                 "input Spot", "output Spot", "blended 512+128 Spot", "blended 128+128 Spot"], rows))

    print("\n### 8. v6 against v7, batched control at 32 sequences: cost per million tokens (blended 512+128)\n")
    rows = []
    for q in ["Q4_0", "Q4_K_M", "Q8_0"]:
        a, b = b6[(q, 32)], b7[(q, 32)]
        c6, c7 = blended(a["speed_pp"], a["speed_tg"], P["v6"]), blended(b["speed_pp"], b["speed_tg"], P["v7"])
        s6, s7 = blended(a["speed_pp"], a["speed_tg"], SPOT["v6"]), blended(b["speed_pp"], b["speed_tg"], SPOT["v7"])
        w6, w7 = a["t"], b["t"]
        rows.append([q, f"{c6:.1f}", f"{c7:.1f}", f"{c7 / c6:.2f}x", f"{s6:.1f}", f"{s7:.1f}", f"{s7 / s6:.2f}x",
                     f"{w7 / w6:.2f}"])
    print(table(["quant", "v6 on-demand", "e16v7 on-demand", "e16v7 cost / v6 cost", "v6 Spot", "e16v7 Spot", "e16v7 cost / v6 cost",
                 "e16v7 wall / v6 wall (same work)"], rows))

    print("\n### 9. v7 cost relative to v6 (blended 512+128, one sequence; >1 means v6 is cheaper)\n")
    rows = []
    for q in ORDER:
        r = [q]
        for bld6, bld7, srcs in (("build", "build", (("e16v7", e16), ("lc2", lc2))),
                                 ("build-native-noamx", "build-native-noamx", (("lc2", lc2),))):
            c6 = blended(v6[(q, bld6, "pp")][0], v6[(q, bld6, "tg")][0], P["v6"])
            for name, s in srcs:
                c7 = blended(s[(q, bld7, "pp")][0], s[(q, bld7, "tg")][0], P["v7"])
                r.append(f"{c7 / c6:.2f}x")
        rows.append(r)
    print(table(["quant", "AMX build: e16v7 / v6", "AMX build: lc2 / v6", "control: lc2 / v6"], rows))
    print(f"\nPrice ratio v7/v6 all-in: on-demand {BR:.3f}, Spot {BRS:.3f}")


if __name__ == "__main__":
    main()
