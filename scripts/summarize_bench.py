#!/usr/bin/env python3
"""Turn benchmark JSON lines fetched from a VM into markdown tables (stdout).

  summarize_bench.py results/<run-dir> --price 1.663 --spot-price 0.307

Reads llama-bench.jsonl, llama-batched.jsonl, quality.jsonl and vllm.jsonl when present.
Cost columns: USD per million tokens at the given hourly VM price.
"""

import argparse
import collections
import json
import pathlib


def load(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip().startswith("{")]


def usd_per_mtok(tps, price):
    return price / (tps * 3600) * 1e6 if tps else float("nan")


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=pathlib.Path)
    ap.add_argument("--price", type=float, required=True, help="VM price actually paid, USD/hour")
    ap.add_argument("--spot-price", type=float, help="Spot price for the same size, USD/hour")
    ap.add_argument("--control", help="name of the no-AMX build to compare with `build` "
                    "(default: build-native-noamx if present, else build-noamx)")
    args = ap.parse_args()
    prices = [("price", args.price)] + ([("spot", args.spot_price)] if args.spot_price else [])

    bench = load(args.dir / "llama-bench.jsonl")
    if bench:
        control = args.control or ("build-native-noamx" if any(b["build"] == "build-native-noamx" for b in bench)
                                   else "build-noamx")
        r = collections.defaultdict(dict)
        sizes = {}
        for b in bench:
            kind = "pp" if b["n_prompt"] else "tg"
            r[b["quant"]][(b["build"], kind)] = b["avg_ts"]
            sizes[b["quant"]] = b.get("model_size", 0) / 1e9
        rows = []
        for q, v in r.items():
            pp, ppn = v.get(("build", "pp")), v.get((control, "pp"))
            tg, tgn = v.get(("build", "tg")), v.get((control, "tg"))
            rows.append([q, f"{sizes[q]:.1f}", f"{pp:.1f}" if pp else "-", f"{ppn:.1f}" if ppn else "-",
                         f"{pp / ppn:.2f}x" if pp and ppn else "-", f"{tg:.2f}" if tg else "-",
                         f"{tgn:.2f}" if tgn else "-", f"{tg / tgn:.2f}x" if tg and tgn else "-"])
        print(f"### llama.cpp, one sequence (llama-bench pp512 / tg128, 8 threads; no-AMX = {control})\n")
        print(table(["quant", "GB", "prefill t/s AMX", "prefill t/s no-AMX", "AMX gain",
                     "decode t/s AMX", "decode t/s no-AMX", "AMX gain"], rows))
        print()

    batched = load(args.dir / "llama-batched.jsonl")
    if batched:
        rows = []
        for b in batched:
            row = [b["quant"], b["build"], b["pl"], f"{b['speed_pp']:.1f}", f"{b['speed_tg']:.2f}", f"{b['speed']:.1f}"]
            row += [f"{usd_per_mtok(b['speed_pp'], p):.2f} / {usd_per_mtok(b['speed_tg'], p):.2f}" for _, p in prices]
            rows.append(row)
        print(f"### llama.cpp, parallel sequences (llama-batched-bench, {batched[0].get('pp', 512)}-token prompts, "
              f"{batched[0].get('tg', 128)} generated each)\n")
        print(table(["quant", "build", "sequences", "prefill t/s", "decode t/s", "total t/s"]
                    + [f"$/M tok prefill / decode ({n})" for n, _ in prices], rows))
        print()

    quality = load(args.dir / "quality.jsonl")
    if quality:
        rows = [[q["quant"], q.get("kld_mean"), q.get("kld_p99"), q.get("same_top_p"), q.get("ppl_q"), q.get("ppl_base")]
                for q in quality]
        print("### Quality vs BF16 (wikitext-2, KL divergence; same top-1 token %)\n")
        print(table(["quant", "mean KLD", "99% KLD", "same top-1 %", "PPL", "PPL BF16"], rows))
        print()

    vllm = load(args.dir / "vllm.jsonl")
    if vllm:
        rows = []
        for v in vllm:
            total = v.get("tokens_per_second", 0)
            out_tps = v["num_prompts"] * v["output_len"] / v["elapsed_time"] if v.get("elapsed_time") else 0
            row = [v["model"].split("/")[-1], v["workload"], f"{v['input_len']}/{v['output_len']}", v["num_prompts"],
                   f"{total:.1f}", f"{out_tps:.2f}", f"{v.get('elapsed_time', 0):.0f}"]
            row += [f"{usd_per_mtok(total, p):.2f}" for _, p in prices]
            rows.append(row)
        print("### vLLM CPU, offline batch (vllm bench throughput)\n")
        print(table(["model", "workload", "in/out", "prompts", "total t/s", "output t/s", "seconds"]
                    + [f"$/M tok ({n})" for n, _ in prices], rows))
        print()


if __name__ == "__main__":
    main()
