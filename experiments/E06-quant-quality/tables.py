#!/usr/bin/env python3
"""E06 tables from kld-stats.jsonl (standard library only). Cost columns read the raw data of E03, E05 and E08.

  python3 tables.py [--price 1.681] [--spot-price 0.325]

Prices are all-in USD/hour (VM + 0.018 disk and IP). Bits per weight = GGUF file bytes x 8 / 27.32 G parameters
(file includes the unused MTP block, about 1.5%, and the type mix differs per file).
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent
N_PARAMS = 27_320_697_856
ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q4_0"]
H1 = {"Q8_0": "< 0.002", "Q6_K": "~0.005", "Q5_K_M": "~0.01", "Q4_K_M": "0.02-0.03", "IQ4_XS": "~0.03", "Q4_0": "~0.05"}
H2 = {"Q8_0": ">= 98", "Q4_K_M": "93-95", "IQ4_XS": "93-95", "Q4_0": "93-95"}


def usd(tps, price):
    return price / (tps * 3600) * 1e6


def blended(pp, tg, price, i=512, o=128):
    return price * (i / pp + o / tg) / 3600 / (i + o) * 1e6


def table(headers, rows, left=1):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i >= left else "---" for i in range(len(headers))) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def pm(v, e, f):
    return (f % v) + (" ± " + (f % e) if e is not None else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, default=1.681)
    ap.add_argument("--spot-price", type=float, default=0.325)
    a = ap.parse_args()
    k = {r["quant"]: r for r in load(HERE / "kld-stats.jsonl")}

    print("### KL divergence vs BF16 (wikitext-2 test, 8 chunks x 512 tokens = 4096 scored tokens; mean ± standard error)\n")
    rows = [[q, f"{k[q]['kld_mean']:.5f} ± {k[q]['kld_mean_err']:.5f}", f"{k[q]['kld_median']:.5f}", f"{k[q]['kld_p99']:.4f}",
             f"{k[q]['kld_p999']:.3f}", f"{k[q]['kld_max']:.3f}", pm(k[q]['same_top_p_pct'], k[q].get('same_top_p_pct_err'), "%.2f"),
             pm(k[q]['dp_rms_pct'], k[q].get('dp_rms_pct_err'), "%.2f")] for q in ORDER]
    print(table(["quant", "mean KLD (nats)", "median KLD (nats)", "99% KLD (nats)", "99.9% KLD (nats)", "max KLD (nats)",
                 "same top-1 token (%)", "RMS Δp (%)"], rows))

    print("\n### Perplexity on the same 4096 tokens (BF16: 6.6104 ± 0.3589 from the saved logits, 6.6159 ± 0.3600 'Final estimate' of the BF16 run)\n")
    rows = [[q, pm(k[q]['ppl_q'], k[q]['ppl_q_err'], "%.4f"), pm(k[q]['ppl_ratio'], k[q]['ppl_ratio_err'], "%.4f"),
             pm(k[q]['ln_ppl_ratio'], k[q]['ln_ppl_ratio_err'], "%.4f"), f"{k[q]['cor_lnppl_pct']:.2f}", f"{k[q]['s_per_pass']:.0f}"]
            for q in ORDER]
    print(table(["quant", "PPL", "PPL / PPL(BF16)", "ln(PPL ratio)", "correlation of ln PPL with BF16 (%)", "s per evaluation pass"], rows))

    size, e3 = {}, {}
    for r in load(EXP / "E03-llamacpp-single-stream" / "llama-bench.jsonl"):
        size[r["quant"]] = r["model_size"]
        e3.setdefault(r["quant"], {})[(r["build"], "pp" if r["n_prompt"] else "tg")] = r["avg_ts"]
    e8 = {}
    for r in load(EXP / "E08-threads-smt" / "llama-bench.jsonl"):
        if r["build"] == "build":
            e8.setdefault(r["quant"], {})["pp" if r["n_prompt"] else "tg"] = r["avg_ts"]
    e5 = {}
    for r in load(EXP / "E05-llamacpp-batched" / "llama-batched.jsonl"):
        if r["build"] == "build-native-noamx" and r["pl"] == 32:
            e5[r["quant"]] = (r["speed_pp"], r["speed_tg"])

    print("\n### Quality against cost (USD per million tokens, reference request 512 input + 128 output, 100% utilization)\n")
    rows = []
    for q in ORDER:
        b = e3[q]
        s1 = blended(b[("build", "pp")], b[("build", "tg")], a.price)
        s1s = blended(b[("build", "pp")], b[("build", "tg")], a.spot_price)
        l8 = blended(e8[q]["pp"], e8[q]["tg"], a.price) if q in e8 else None
        b32 = blended(e5[q][0], e5[q][1], a.price) if q in e5 else None
        b32s = blended(e5[q][0], e5[q][1], a.spot_price) if q in e5 else None
        rows.append([q, f"{size[q] / 1e9:.1f}", f"{size[q] * 8 / N_PARAMS:.2f}", f"{k[q]['kld_mean']:.4f}",
                     f"{k[q]['same_top_p_pct']:.1f}", f"{s1:.1f}", f"{s1s:.1f}", f"{l8:.1f}" if l8 else "-",
                     f"{b32:.1f}" if b32 else "-", f"{b32s:.1f}" if b32s else "-"])
    print(table(["quant", "file GB", "bits per weight", "mean KLD (nats)", "same top-1 (%)",
                 "1 seq, AMX, bench-e16v7 (E03), on-demand", "1 seq, AMX, bench-e16v7 (E03), Spot",
                 "1 seq, AMX, bench-lc2 (E08), on-demand", "32 seq, no-AMX, bench-e16v7 (E05), on-demand",
                 "32 seq, no-AMX, bench-e16v7 (E05), Spot"], rows))

    print("\n### Hypothesis check\n")
    rows = []
    for q in ORDER:
        rows.append([q, H1[q], f"{k[q]['kld_mean']:.4f}", H2.get(q, "-"), f"{k[q]['same_top_p_pct']:.1f} ± {k[q]['same_top_p_pct_err']:.1f}"])
    print(table(["quant", "H1 mean KLD predicted", "measured", "H2 same top-1 predicted (%)", "measured (%)"], rows))


if __name__ == "__main__":
    main()
