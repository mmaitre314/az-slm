#!/usr/bin/env python3
"""E07 tables from e07-profile.txt (perf report --no-children --sort dso,symbol, top 25 per build; standard library only).

  python3 tables.py [--price 1.681] [--spot-price 0.325]

The "Q4_0 pp512" rates are the rates printed by the profiled llama-bench runs (perf cpu-clock, 499 Hz, call graphs on).
"""
import argparse
import collections
import pathlib
import re

HERE = pathlib.Path(__file__).parent
N_PARAMS = 27_320_697_856
AMX_INT8_TOPS = 59.0
VNNI_INT8_TOPS = 7.4
RATE = {"build": 25.258409, "build-native-noamx": 19.150248}   # tok/s under perf, from e07-profile.txt
SAMPLES = {"build": 154141, "build-native-noamx": 201769}       # perf samples in the data files (from perf script)

RULES = [  # (category, regex on symbol) first match wins
    ("GEMM: AMX tile kernel (lambda in ggml_backend_amx_mul_mat)", r"ggml_backend_amx_mul_mat"),
    ("GEMM: AMX tinygemm for q4_1 / q8_0 / q6_K tensors", r"tinygemm_kernel_amx"),
    ("GEMM: repacked Q4_0 kernel (LUT, AVX-512)", r"gemm_q4_b32_8x8"),
    ("GEMM: ggml_vec_dot q4_1 / q6_K (generic)", r"ggml_vec_dot_q(4_1|6_K)"),
    ("GEMM: tinyBLAS q8_0 / tiled VNNI micro-kernels", r"tinyBLAS_Q0|tiled_run_micro|tiled_unpack_src0"),
    ("weight repack / unpack (load time or per tile)", r"pack_qs|unpack_B|repack_q4_0|clear_page_erms"),
    ("Gated DeltaNet scan", r"gated_delta_net"),
    ("ssm_conv (linear-attention conv)", r"ssm_conv"),
    ("concat", r"compute_forward_concat"),
    ("full attention (flash_attn_ext)", r"flash_attn"),
    ("F32 matmul (tinyBLAS f32, vec_dot_f32)", r"tinyBLAS<|ggml_vec_dot_f32"),
    ("norms, activations, adds", r"rms_norm|swiglu|silu|add_non_quantized"),
    ("activation quantization", r"quantize_row|quantize_mat"),
    ("OpenMP wait (libgomp)", r"libgomp"),
]


def usd(tps, price):
    return price / (tps * 3600) * 1e6


def table(headers, rows, left=1):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if i >= left else "---" for i in range(len(headers))) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def parse():
    sections, cur = {}, None
    for line in (HERE / "e07-profile.txt").read_text().splitlines():
        m = re.match(r"== top functions \(Q4_0, (\S+)\)", line)
        if m:
            cur = m.group(1)
            sections[cur] = []
            continue
        m = re.match(r"\s+([0-9.]+)%\s+(\S+)\s+\[.\]\s+(.*)", line)
        if m and cur:
            sections[cur].append((float(m.group(1)), m.group(2), m.group(3).strip()))
    return sections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, default=1.681)
    ap.add_argument("--spot-price", type=float, default=0.325)
    a = ap.parse_args()
    sec = parse()
    cats = {}
    for b, rows in sec.items():
        c = collections.OrderedDict()
        for pct, dso, sym in rows:
            name = next((n for n, rx in RULES if re.search(rx, sym) or re.search(rx, dso)), "other (ld.so TLS lookup)")
            c[name] = c.get(name, 0) + pct
        cats[b] = (c, sum(p for p, _, _ in rows))
    print("### Where the pp512 time goes (Q4_0, 8 threads; % of all perf samples, top 25 symbols grouped)\n")
    names = [n for n, _ in RULES] + ["other (ld.so TLS lookup)"]
    rows = [[n, f"{cats['build'][0].get(n, 0):.1f}", f"{cats['build-native-noamx'][0].get(n, 0):.1f}"] for n in names]
    rows.append(["listed (top 25 symbols)", f"{cats['build'][1]:.1f}", f"{cats['build-native-noamx'][1]:.1f}"])
    rows.append(["not listed (beyond the top 25)", f"{100 - cats['build'][1]:.1f}", f"{100 - cats['build-native-noamx'][1]:.1f}"])
    print(table(["category", "AMX build (%)", "clean control, no AMX (%)"], rows))
    print("\n### Roll-up\n")
    rows = []
    for b, label in [("build", "AMX build"), ("build-native-noamx", "clean control")]:
        c = cats[b][0]
        gemm = sum(v for n, v in c.items() if n.startswith("GEMM"))
        rows.append([label, f"{RATE[b]:.2f}", f"{gemm:.1f}", f"{c.get('Gated DeltaNet scan', 0):.1f}",
                     f"{c.get('Gated DeltaNet scan', 0) + c.get('ssm_conv (linear-attention conv)', 0) + c.get('concat', 0):.1f}",
                     f"{c.get('OpenMP wait (libgomp)', 0):.1f}", f"{SAMPLES[b] / 499:.0f}",
                     f"{RATE[b] * 2 * N_PARAMS / 1e12:.2f}", f"{RATE[b] * 2 * N_PARAMS / 1e12 / (gemm / 100):.2f}",
                     f"{100 * RATE[b] * 2 * N_PARAMS / 1e12 / (gemm / 100) / (AMX_INT8_TOPS if b == 'build' else VNNI_INT8_TOPS):.1f}%"])
    print(table(["build", "pp512 under perf (tok/s)", "GEMM kernels (% of samples)", "Gated DeltaNet scan (%)",
                 "scan + ssm_conv + concat (%)", "OpenMP wait (%)", "CPU-seconds sampled (samples / 499 Hz)",
                 "prefill TFLOPS (whole run)", "TFLOPS inside GEMM kernels (whole-run TFLOPS / GEMM share)",
                 "% of INT8 peak (59 TOPS AMX, 7.4 TOPS VNNI)"], rows))
    print("\n### Cost per token implied by the profiled runs, and an Amdahl what-if (Q4_0, prefill only, USD per million input tokens)\n")
    rows = []
    for b, label in [("build", "AMX build"), ("build-native-noamx", "clean control")]:
        rows.append([label, "measured", f"{RATE[b]:.1f}", f"{usd(RATE[b], a.price):.1f}", f"{usd(RATE[b], a.spot_price):.1f}"])
    c = cats["build"][0]
    gemm = sum(v for n, v in c.items() if n.startswith("GEMM")) / 100
    for k in (2, 4, 9, 1e9):
        t = 1 - gemm + gemm / k
        rate = RATE["build"] / t
        label = "GEMM time to zero (bound)" if k > 1e6 else f"GEMM {k:g}x faster"
        rows.append(["AMX build", f"what-if: {label}", f"{rate:.0f}", f"{usd(rate, a.price):.1f}", f"{usd(rate, a.spot_price):.1f}"])
    print(table(["build", "case", "pp512 (tok/s)", "input USD/M on-demand", "input USD/M Spot"], rows, left=2))


if __name__ == "__main__":
    main()
