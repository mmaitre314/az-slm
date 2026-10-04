#!/usr/bin/env python3
"""E05 tables from llama-batched.jsonl (standard library only).

  python3 tables.py [--price 1.681] [--spot-price 0.325]

llama-batched-bench: npl independent sequences, each with a 128-token prompt and 128 generated tokens.
speed_pp = npl*128 / t_pp (prefill, tok/s); speed_tg = npl*128 / t_tg (aggregate decode, tok/s).
FLOP/token = 2 x 27.32 B parameters (upper bound).
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).parent
GFLOP_PER_TOKEN = 2 * 27.320697856
REF_IN, REF_OUT = 512, 128


def usd(tps, price):
    return price / (tps * 3600) * 1e6


def table(headers, rows, left=2):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i >= left else "---" for i in range(len(headers))) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, default=1.681)
    ap.add_argument("--spot-price", type=float, default=0.325)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (HERE / "llama-batched.jsonl").read_text().splitlines()]
    base = {(r["quant"], r["build"]): r["speed_tg"] for r in rows if r["pl"] == 1}

    for build, title in (("build-native-noamx", "clean control, native build with AMX compiled out (output presumed valid, see threats)"),
                         ("build", "AMX build: speed reference only (output corrupt for > 1 sequence, E04)")):
        print(f"\n### Throughput, {title}\n")
        out = []
        for r in rows:
            if r["build"] != build:
                continue
            n = r["pl"]
            step = n / r["speed_tg"]
            out.append([r["quant"], n, f"{r['speed_pp']:.1f}", f"{r['speed_tg']:.2f}", f"{r['speed_tg'] / n:.2f}",
                        f"{r['speed_tg'] / base[(r['quant'], build)]:.2f}x", f"{step:.2f}",
                        f"{r['speed_tg'] * GFLOP_PER_TOKEN / 1e3:.2f}", f"{r['speed']:.1f}", f"{r['t']:.0f}"])
        print(table(["quant", "sequences", "prefill (tok/s)", "decode, all sequences (tok/s)", "decode per sequence (tok/s)",
                     "decode vs 1 sequence", "decode step (s)", "decode TFLOPS", "prefill+decode (tok/s)", "wall (s)"], out))

    print(f"\n### Cost per million tokens (all-in {a.price} USD/h on-demand, {a.spot_price} USD/h Spot)\n")
    out = []
    for r in rows:
        n, pp, tg = r["pl"], r["speed_pp"], r["speed_tg"]
        ref_t = REF_IN / pp + REF_OUT / tg   # seconds per request at the batch rate
        row = [r["quant"], "no-AMX" if r["build"] == "build-native-noamx" else "AMX (invalid output)", n]
        for p in (a.price, a.spot_price):
            row += [f"{usd(pp, p):.1f}", f"{usd(tg, p):.1f}"]
        for p in (a.price, a.spot_price):
            row.append(f"{p * r['t'] / 3600 / (n * (r['pp'] + r['tg'])) * 1e6:.1f}")
        for p in (a.price, a.spot_price):
            row.append(f"{p * ref_t / 3600 / (REF_IN + REF_OUT) * 1e6:.1f}")
        out.append(row)
    print(table(["quant", "build", "sequences", "input USD/M on-demand", "output USD/M on-demand", "input USD/M Spot",
                 "output USD/M Spot", "blended 128+128 measured USD/M on-demand", "blended 128+128 USD/M Spot",
                 "blended 512+128 USD/M on-demand", "blended 512+128 USD/M Spot"], out, left=3))

    print("\n### AMX build over the clean control (same quant, same number of sequences; speed only)\n")
    d = {(r["quant"], r["build"], r["pl"]): r for r in rows}
    out = []
    for (q, b, n), r in d.items():
        c = d.get((q, "build-native-noamx", n))
        if b == "build" and c:
            out.append([q, n, f"{r['speed_pp'] / c['speed_pp']:.2f}x", f"{r['speed_tg'] / c['speed_tg']:.2f}x"])
    print(table(["quant", "sequences", "prefill AMX / no-AMX", "decode AMX / no-AMX"], out))


if __name__ == "__main__":
    main()
