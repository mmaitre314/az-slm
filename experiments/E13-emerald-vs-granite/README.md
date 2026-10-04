# E13: Emerald Rapids (E16ds_v6) vs Granite Rapids (E16ds_v7)

| | |
| --- | --- |
| Status | done (measured 2026-10-04 02:48–04:19 UTC; reported and reviewed 2026-10-04) |
| VM | `bench-v6` (Standard_E16ds_v6, westus2, Regular): Xeon Platinum 8573C; compare with `bench-e16v7` and `bench-lc2` |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-noamx` = explicit AVX-512 flag list without AMX; `build-native-noamx` = -march=native minus AMX (clean control) |
| Model | `bartowski/Qwen3.8-27B-GGUF` @ `0c92138c51` (Q4_0, IQ4_XS, Q4_K_M, Q8_0) |
| Dates | `llama-bench` 02:48-03:18 UTC, `llama-batched-bench` 03:18-04:19 UTC (job `e13`, started on a freshly booted VM) |
| Raw data | `llama-bench.jsonl` (16 rows, key fields incl. the three repetitions), `llama-batched.jsonl` (15 rows), `check_amx.txt` (CPU and AMX probe), `model-buffers.txt` (weight-buffer lines from the verbose logs), `tables.py` (regenerates every table below, reading the E03, E05 and E08 raw files for the Granite Rapids side) |

## Question

Is the newer, 27% more expensive Granite Rapids VM faster enough per token to be worth it for llama.cpp?

## Hypotheses

_Written before the results were known._

- H1: Decode scales with memory bandwidth (Granite Rapids: 12 channels of DDR5-6400 vs 8 of DDR5-5600 for Emerald Rapids), so v7 decodes 1.2-1.6x faster.
- H2: Prefill is ~1.2x faster on v7 (3.6 vs 3.0 GHz all-core); llama.cpp doesn't use AMX-FP16, so v7's extra ISA doesn't matter.
- H3: v6 is cheaper per token only if v7's speedup is below the 1.27x price ratio.

## Setup

- VM `bench-v6`: Intel Xeon Platinum 8573C (Emerald Rapids), 16 vCPU = 8 cores with SMT, 2 MiB L2 per core, 260 MiB L3 as seen by the guest, 128 GiB, kernel 6.17 (`check_amx.txt`). CPU flags `amx_tile amx_int8 amx_bf16` are present and the kernel grants AMX permission; **AMX-FP16 is absent** (cpuid leaf 7 says `AMX-FP16 = false`). Microsoft Learn documents an all-core turbo of 3.0 GHz for Edsv6. The guest reports a 2.3 GHz base clock, and the clock under AMX load was not measured.
- Granite Rapids comparison data, all from the repo and none re-measured: `bench-e16v7` (eastus2, Xeon 6 6973P-C): E03 (`build`, single sequence) and E05 (`build-native-noamx`, batched); `bench-lc2` (northcentralus, same CPU model): E08 (`build` and `build-native-noamx`, single sequence). E08 found these two instances differ by 9-23% on the same build and quant (lc2 faster), so every comparison below is shown against both.
- Builds: llama.cpp `11fe02151f79` compiled on the VM with GCC 13.3.0. `-march=native` resolves to `sapphirerapids` on this VM and to `graniterapids` on `bench-lc2` (probed with `gcc -march=native -Q --help=target`; `bench-e16v7` was not probed, same CPU model and image). So the "same" control build is not byte-identical code on the two generations: on v7 it also gets `-mprefetchi` and Granite Rapids tuning, and the AMX-FP16 flag is disabled by `-mno-amx-fp16`. The job log shows the same instruction counts as on lc2 (E08): `libggml-cpu.so` contains 203 AMX and 2269 `vpdpbusd` instructions in `build`, and 0 AMX and 1189 `vpdpbusd` in both `build-noamx` and `build-native-noamx`.
- Weight buffers (`model-buffers.txt`) have the same sizes as in E03 and E08: AMX buffer 14661 / 13980 / 16057 / 28990 MiB for Q4_0 / IQ4_XS / Q4_K_M / Q8_0, `CPU_REPACK` buffer 11993 / 101 / 10238 MiB for the control and none for Q8_0. All VMs therefore use the same weight layouts.
- `llama-bench`: 8 threads, `n_batch` 2048, `n_ubatch` 512, flash attention auto, f16 KV cache, 3 repetitions run back to back, `pp512` = 512 prompt tokens on an empty context, `tg128` = 128 generated tokens. `llama-batched-bench`: N independent sequences, each a 128-token prompt followed by 128 generated tokens, `-b 2048 -ub 512 -t 8 -tb 8`, one run per cell, `build-native-noamx` only.
- No other benchmark job overlapped: the VM was booted at ~02:38 UTC, the `e13` job (the only one in `/mnt/data/jobs` for that period) ran from 02:44 to 04:19, and the next job (E12, `specbuild`) started at ~07:05. Light Run Command calls from other agents during the run were not logged.
- **Output correctness was not checked on this VM.** E04 verified the AMX build's single-sequence output for Q4_K_M on `bench-e16v7` only; E06 verified `build-native-noamx` (multi-sequence perplexity) on `bench-lc2`. Nothing was compared on `bench-v6`.

## Method

Same scripts and parameters as E03/E08 (`llama_bench.sh`, `build` vs `build-native-noamx`, 4 quants) and E05 (`llama_batched.sh`, `build-native-noamx`, npl 1-32).

Commands actually run (job script `e13.sh`): `QUANTS="Q4_0 IQ4_XS Q4_K_M Q8_0" BUILDS="build build-native-noamx" bash llama_bench.sh`, then `QUANTS="Q4_0 Q4_K_M Q8_0" BUILDS="build-native-noamx" NPL=1,4,8,16,32 PP=128 TG=128 bash llama_batched.sh`. Same parameters as E05, so the batched numbers are directly comparable with `bench-e16v7`. The AMX build was not run with several sequences (E04: corrupt output, E05: crashes).

## Measurements

Units: tok/s unless stated; "ratio" columns are the left-named VM over the right-named one. Raw data: `llama-bench.jsonl`, `llama-batched.jsonl`. No row is marked invalid by a check, but see the Setup note on output correctness. One cell is noisy: the control's Q4_K_M `pp512` (19.6, 17.3, 18.0 tok/s, sd 6.4%), so its 1.61x AMX/control ratio carries about +-7%. E08 saw the same cell (22.3-23.7 on lc2) as the noisiest.

### bench-v6, one sequence

`pp512` and `tg128` on `build` (AMX) and `build-native-noamx` (control), mean +- sample sd of 3 repetitions:

| quant | file GB | pp512 AMX (tok/s) | pp512 control (tok/s) | AMX/control | tg128 AMX (tok/s) | tg128 control (tok/s) | AMX/control |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_0 | 16.3 | 20.2 ± 0.05 | 14.9 ± 0.00 | 1.36x | 5.03 ± 0.010 | 3.29 ± 0.005 | 1.53x |
| IQ4_XS | 15.5 | 30.8 ± 0.01 | 34.9 ± 0.02 | 0.88x | 3.47 ± 0.006 | 3.13 ± 0.003 | 1.11x |
| Q4_K_M | 17.4 | 29.5 ± 0.03 | 18.3 ± 1.18 | 1.61x | 3.60 ± 0.004 | 3.21 ± 0.025 | 1.12x |
| Q8_0 | 29.1 | 20.8 ± 0.01 | 12.2 ± 0.01 | 1.70x | 3.56 ± 0.005 | 2.21 ± 0.003 | 1.61x |

Physical limits (assumptions: decode GB/s = tg128 x GGUF file size, a lower bound on traffic; prefill FLOP/token = 2 x 27.32 G parameters = 54.6 GFLOP, an upper bound; v6 peaks at the documented 3.0 GHz all-core turbo with 8 cores: AMX INT8 = 8 x 3.0 GHz x 2048 ops/cycle = 49 TOPS, AVX-512 VNNI = 8 x 3.0 GHz x 256 = 6.1 TOPS, both theoretical, frequency under load not measured):

| quant | decode GB/s AMX (tg x file) | decode GB/s control | prefill TFLOPS AMX | % of 49 TOPS AMX INT8 peak | prefill TFLOPS control | % of 6.1 TOPS VNNI peak |
|---|---:|---:|---:|---:|---:|---:|
| Q4_0 | 82 | 54 | 1.10 | 2.2% | 0.81 | 13% |
| IQ4_XS | 54 | 48 | 1.68 | 3.4% | 1.90 | 31% |
| Q4_K_M | 63 | 56 | 1.61 | 3.3% | 1.00 | 16% |
| Q8_0 | 104 | 64 | 1.13 | 2.3% | 0.67 | 11% |

### One sequence, `build` (AMX), three instances

v6 (this experiment), `bench-e16v7` (E03) and `bench-lc2` (E08). The `e16v7/v6` and `lc2/v6` columns are the speedup of v7 over v6 (the price break-even is 1.27x on demand, 1.25x on Spot); `lc2/e16v7` is the spread between the two v7 instances:

| test | quant | v6 (tok/s) | v7 e16v7 (tok/s) | v7 lc2 (tok/s) | e16v7/v6 | lc2/v6 | lc2/e16v7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pp512 | Q4_0 | 20.21 | 21.06 | 25.46 | 1.04x | 1.26x | 1.21x |
| pp512 | IQ4_XS | 30.84 | 29.71 | 35.59 | 0.96x | 1.15x | 1.20x |
| pp512 | Q4_K_M | 29.46 | 28.72 | 35.12 | 0.97x | 1.19x | 1.22x |
| pp512 | Q8_0 | 20.78 | 22.35 | 27.47 | 1.08x | 1.32x | 1.23x |
| tg128 | Q4_0 | 5.03 | 5.29 | 5.79 | 1.05x | 1.15x | 1.09x |
| tg128 | IQ4_XS | 3.47 | 3.97 | 4.57 | 1.14x | 1.32x | 1.15x |
| tg128 | Q4_K_M | 3.60 | 3.88 | 4.45 | 1.08x | 1.24x | 1.15x |
| tg128 | Q8_0 | 3.56 | 2.78 | 3.42 | 0.78x | 0.96x | 1.23x |

Per-phase speedups of the Granite Rapids instances over v6 (AMX build): prefill 0.96-1.08x (geometric mean 1.01x) for `bench-e16v7` and 1.15-1.32x (1.23x) for `bench-lc2`; decode 0.78-1.14x (1.00x) for `bench-e16v7` and 0.96-1.32x (1.16x) for `bench-lc2`. Across all 24 per-phase ratios in this and the next table, 4 exceed the 1.27 price ratio (lc2 AMX `pp512` Q8_0 1.32x, lc2 AMX `tg128` IQ4_XS 1.32x, lc2 control `pp512` Q4_0 1.30x and Q8_0 1.33x) and 7 are below 1.0.

### One sequence, clean control (`build-native-noamx`)

v6 against `bench-lc2` (E08). The last column adds the one-sequence decode of E05's batched run on `bench-e16v7` (128-token prompt, 128 generated, a slightly different shape from `tg128`) with its ratio to the same measurement on v6 (`llama-batched.jsonl`, `pl` = 1):

| test | quant | v6 (tok/s) | v7 lc2 (tok/s) | lc2/v6 | E05 npl=1 decode on e16v7 (tok/s), ratio to v6 npl=1 decode |
|---|---:|---:|---:|---:|---:|
| pp512 | Q4_0 | 14.89 | 19.37 | 1.30x | - |
| pp512 | IQ4_XS | 34.86 | 40.34 | 1.16x | - |
| pp512 | Q4_K_M | 18.26 | 23.01 | 1.26x | - |
| pp512 | Q8_0 | 12.23 | 16.31 | 1.33x | - |
| tg128 | Q4_0 | 3.29 | 2.57 | 0.78x | 2.44 (0.74x of v6 3.28) |
| tg128 | IQ4_XS | 3.13 | 2.10 | 0.67x | - |
| tg128 | Q4_K_M | 3.21 | 3.88 | 1.21x | 3.54 (1.09x of v6 3.24) |
| tg128 | Q8_0 | 2.21 | 2.16 | 0.98x | 1.97 (0.90x of v6 2.20) |

The AMX/control ratios on v6 are prefill 1.36x, 0.88x, 1.61x, 1.70x and decode 1.53x, 1.11x, 1.12x, 1.61x (Q4_0, IQ4_XS, Q4_K_M, Q8_0); on `bench-lc2` (E08) they are prefill 1.31x, 0.88x, 1.53x, 1.68x and decode 2.26x, 2.18x, 1.15x, 1.58x. Prefill ratios agree within 0.08; the decode ratios differ for Q4_0 and IQ4_XS because the **control's** decode is 28% and 49% faster on v6 than on lc2 (table above), while the AMX build's decode is 13% and 24% slower.

### Batched, clean control, v6 against `bench-e16v7` (E05)

N sequences of 128 prompt + 128 generated tokens, aggregate rates over all sequences (E05 numbers for `bench-e16v7`, identical command line). `bench-lc2` was not run in this configuration:

| quant | sequences | prefill v6 (tok/s) | prefill e16v7 (tok/s) | e16v7/v6 | decode v6 all seq (tok/s) | decode e16v7 all seq (tok/s) | e16v7/v6 | decode v6 per seq (tok/s) | wall v6 (s) | wall e16v7 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_0 | 1 | 15.3 | 17.8 | 1.16x | 3.28 | 2.44 | 0.74x | 3.28 | 47 | 60 |
| Q4_0 | 4 | 15.8 | 18.0 | 1.14x | 8.93 | 9.14 | 1.02x | 2.23 | 90 | 84 |
| Q4_0 | 8 | 15.8 | 18.2 | 1.15x | 9.50 | 9.52 | 1.00x | 1.19 | 173 | 164 |
| Q4_0 | 16 | 15.8 | 18.1 | 1.15x | 11.76 | 12.77 | 1.09x | 0.74 | 304 | 273 |
| Q4_0 | 32 | 15.8 | 18.1 | 1.15x | 12.13 | 13.04 | 1.07x | 0.38 | 597 | 540 |
| Q4_K_M | 1 | 20.2 | 20.7 | 1.03x | 3.24 | 3.54 | 1.09x | 3.24 | 46 | 42 |
| Q4_K_M | 4 | 20.9 | 21.7 | 1.04x | 9.06 | 9.03 | 1.00x | 2.27 | 81 | 80 |
| Q4_K_M | 8 | 20.9 | 21.7 | 1.04x | 9.46 | 9.20 | 0.97x | 1.18 | 157 | 158 |
| Q4_K_M | 16 | 20.7 | 21.5 | 1.04x | 13.60 | 13.43 | 0.99x | 0.85 | 250 | 248 |
| Q4_K_M | 32 | 20.7 | 21.6 | 1.04x | 14.35 | 14.22 | 0.99x | 0.45 | 483 | 478 |
| Q8_0 | 1 | 12.5 | 14.4 | 1.15x | 2.20 | 1.97 | 0.90x | 2.20 | 68 | 74 |
| Q8_0 | 4 | 12.7 | 14.9 | 1.17x | 8.29 | 8.10 | 0.98x | 2.07 | 102 | 98 |
| Q8_0 | 8 | 12.7 | 14.9 | 1.18x | 9.20 | 9.91 | 1.08x | 1.15 | 192 | 172 |
| Q8_0 | 16 | 12.7 | 14.9 | 1.17x | 9.84 | 10.71 | 1.09x | 0.62 | 369 | 329 |
| Q8_0 | 32 | 12.7 | 14.9 | 1.18x | 10.01 | 11.04 | 1.10x | 0.31 | 732 | 646 |

At 32 sequences prefill is 73.5-75.9% of the busy time of a 512+128 request on v6 (computed from the rates; the 128-token prefill rate stands in for 512 tokens, see Cost). Aggregate decode at 32 sequences is 0.55-0.78 TFLOPS (9-13% of the 6.1 TOPS VNNI peak) and 69-79% of the same cell's prefill rate, the same compute-bound behavior E05 found on v7.

## Cost per token

Formulas from [COST_MODEL.md](../COST_MODEL.md): input = `P / (prefill tok/s x 3600) x 1e6`, output = `P / (decode tok/s x 3600) x 1e6` with the aggregate decode rate, blended for the reference request (512 input + 128 output) = `P x (512 / prefill + 128 / decode) / 3600 / 640 x 1e6`. All numbers are USD per million tokens.

Assumptions:

1. All-in hourly price = VM + $0.018 for disk and IP, taken from the brief: v6 (`Standard_E16ds_v6`, westus2) $1.326/h on demand and $0.260/h Spot (VM $1.308 + $0.018, $0.2417 + $0.018); v7 (`Standard_E16ds_v7`) $1.681/h on demand and $0.325/h Spot for both Granite Rapids instances. The price ratio v7/v6 is 1.268 on demand and 1.250 on Spot, the break-even speedup for "v7 is cheaper per token". Spot is not available on this subscription (the on-demand column is what we pay); Spot figures are hypothetical. The northcentralus price of `bench-lc2` was not checked (E08 notes the same).
2. 100% utilization and steady state; VM boot, download and model load are excluded (each `llama-bench` run loads and repacks the model first, about a minute or more for the larger files).
3. Single-sequence rows use `pp512` and `tg128` on an empty context; the reference request decodes at a longer context (up to 640 tokens), which was not measured. The AMX-build single-sequence rows are speed references for every quant on v6 (output unchecked, see Setup).
4. Batched rows use the clean control build only. "Blended 512+128" uses the **128-token prefill rate measured in the batched run for the 512-token prompt** (same assumption as E05; E05's sensitivity check showed +8-13% if `pp512` of another build is used instead) and the aggregate decode rate at the same N. "Blended 128+128" uses the measured wall time of the run (exactly what was benchmarked).
5. List prices, no reservations; Spot evictions are not modeled. One VM instance and one run per cell on each side.

### Headline: best valid llama.cpp configuration (Q4_K_M, 32 sequences, clean control)

| VM | input on-demand | output on-demand | blended 512+128 on-demand | input Spot | output Spot | blended 512+128 Spot |
|---|---:|---:|---:|---:|---:|---:|
| `bench-v6` (E16ds_v6), this experiment | 17.8 | 25.7 | 19.3 | 3.5 | 5.0 | 3.8 |
| `bench-e16v7` (E16ds_v7), E05 | 21.7 | 32.8 | 23.9 | 4.2 | 6.3 | 4.6 |
| `bench-lc2` (E16ds_v7) | not measured | not measured | about 19-21 (extrapolated, below) | | | about 3.7-4.0 |

The `bench-lc2` row is an extrapolation, not a measurement: scaling E05's `bench-e16v7` Q4_K_M prefill and decode rates up by 15-25% (E08's lc2/e16v7 single-sequence ratios are 1.20-1.23x for prefill and 1.09-1.23x for decode) gives $19.1-$20.8/M blended on demand, that is 1% below to 8% above v6's $19.3/M. Treat it as "about equal to v6".

### Batched cost on v6 (control build, all sequence counts)

| quant | sequences | input on-demand | output on-demand | blended 512+128 on-demand | blended 128+128 on-demand | input Spot | output Spot | blended 512+128 Spot | blended 128+128 Spot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_0 | 1 | 24.1 | 112.2 | 41.7 | 68.1 | 4.7 | 22.0 | 8.2 | 13.4 |
| Q4_0 | 4 | 23.4 | 41.2 | 26.9 | 32.3 | 4.6 | 8.1 | 5.3 | 6.3 |
| Q4_0 | 8 | 23.3 | 38.8 | 26.4 | 31.0 | 4.6 | 7.6 | 5.2 | 6.1 |
| Q4_0 | 16 | 23.3 | 31.3 | 24.9 | 27.3 | 4.6 | 6.1 | 4.9 | 5.4 |
| Q4_0 | 32 | 23.3 | 30.4 | 24.7 | 26.8 | 4.6 | 6.0 | 4.9 | 5.3 |
| Q4_K_M | 1 | 18.3 | 113.9 | 37.4 | 66.1 | 3.6 | 22.3 | 7.3 | 13.0 |
| Q4_K_M | 4 | 17.6 | 40.6 | 22.2 | 29.1 | 3.5 | 8.0 | 4.4 | 5.7 |
| Q4_K_M | 8 | 17.6 | 38.9 | 21.9 | 28.3 | 3.5 | 7.6 | 4.3 | 5.5 |
| Q4_K_M | 16 | 17.8 | 27.1 | 19.7 | 22.4 | 3.5 | 5.3 | 3.9 | 4.4 |
| Q4_K_M | 32 | 17.8 | 25.7 | 19.3 | 21.7 | 3.5 | 5.0 | 3.8 | 4.3 |
| Q8_0 | 1 | 29.4 | 167.4 | 57.0 | 98.4 | 5.8 | 32.8 | 11.2 | 19.3 |
| Q8_0 | 4 | 28.9 | 44.4 | 32.0 | 36.7 | 5.7 | 8.7 | 6.3 | 7.2 |
| Q8_0 | 8 | 29.0 | 40.1 | 31.2 | 34.5 | 5.7 | 7.9 | 6.1 | 6.8 |
| Q8_0 | 16 | 29.0 | 37.4 | 30.7 | 33.2 | 5.7 | 7.3 | 6.0 | 6.5 |
| Q8_0 | 32 | 29.1 | 36.8 | 30.6 | 32.9 | 5.7 | 7.2 | 6.0 | 6.5 |

### v6 against `bench-e16v7`, batched control at 32 sequences (blended 512+128)

| quant | v6 on-demand | e16v7 on-demand | e16v7 cost / v6 cost | v6 Spot | e16v7 Spot | e16v7 cost / v6 cost | e16v7 wall / v6 wall (same work) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_0 | 24.7 | 27.8 | 1.12x | 4.9 | 5.4 | 1.11x | 0.90 |
| Q4_K_M | 19.3 | 23.9 | 1.23x | 3.8 | 4.6 | 1.22x | 0.99 |
| Q8_0 | 30.6 | 33.5 | 1.09x | 6.0 | 6.5 | 1.08x | 0.88 |

### One sequence: v6, `bench-e16v7` and `bench-lc2`

Rows labeled "AMX" use `build`, "control" uses `build-native-noamx`; `bench-e16v7` has no control row here (E03 measured `build-noamx`, a different build, and E05's control run is batched). "s per request" is the wall time of one 512+128 request at these rates.

| quant | VM / build | input on-demand | output on-demand | blended on-demand | input Spot | output Spot | blended Spot | s per request |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_0 | v6 AMX | 18.2 | 73.2 | 29.2 | 3.6 | 14.4 | 5.7 | 51 |
| Q4_0 | v6 control | 24.7 | 112.0 | 42.2 | 4.9 | 22.0 | 8.3 | 73 |
| Q4_0 | e16v7 AMX | 22.2 | 88.2 | 35.4 | 4.3 | 17.1 | 6.8 | 48 |
| Q4_0 | lc2 AMX | 18.3 | 80.6 | 30.8 | 3.5 | 15.6 | 6.0 | 42 |
| Q4_0 | lc2 control | 24.1 | 181.8 | 55.7 | 4.7 | 35.2 | 10.8 | 76 |
| IQ4_XS | v6 AMX | 11.9 | 106.2 | 30.8 | 2.3 | 20.8 | 6.0 | 54 |
| IQ4_XS | v6 control | 10.6 | 117.8 | 32.0 | 2.1 | 23.1 | 6.3 | 56 |
| IQ4_XS | e16v7 AMX | 15.7 | 117.7 | 36.1 | 3.0 | 22.8 | 7.0 | 49 |
| IQ4_XS | lc2 AMX | 13.1 | 102.2 | 30.9 | 2.5 | 19.8 | 6.0 | 42 |
| IQ4_XS | lc2 control | 11.6 | 222.3 | 53.7 | 2.2 | 43.0 | 10.4 | 74 |
| Q4_K_M | v6 AMX | 12.5 | 102.4 | 30.5 | 2.5 | 20.1 | 6.0 | 53 |
| Q4_K_M | v6 control | 20.2 | 114.9 | 39.1 | 4.0 | 22.5 | 7.7 | 68 |
| Q4_K_M | e16v7 AMX | 16.3 | 120.3 | 37.1 | 3.1 | 23.3 | 7.2 | 51 |
| Q4_K_M | lc2 AMX | 13.3 | 105.0 | 31.6 | 2.6 | 20.3 | 6.1 | 43 |
| Q4_K_M | lc2 control | 20.3 | 120.4 | 40.3 | 3.9 | 23.3 | 7.8 | 55 |
| Q8_0 | v6 AMX | 17.7 | 103.4 | 34.9 | 3.5 | 20.3 | 6.8 | 61 |
| Q8_0 | v6 control | 30.1 | 166.9 | 57.5 | 5.9 | 32.7 | 11.3 | 100 |
| Q8_0 | e16v7 AMX | 20.9 | 167.8 | 50.3 | 4.0 | 32.4 | 9.7 | 69 |
| Q8_0 | lc2 AMX | 17.0 | 136.7 | 40.9 | 3.3 | 26.4 | 7.9 | 56 |
| Q8_0 | lc2 control | 28.6 | 215.7 | 66.0 | 5.5 | 41.7 | 12.8 | 91 |

### v7 cost per token relative to v6

Blended 512+128 cost of the Granite Rapids instance divided by v6's, one sequence (from the table above); values above 1 mean v6 is cheaper:

| quant | AMX build: e16v7 / v6 | AMX build: lc2 / v6 | control: lc2 / v6 |
|---|---:|---:|---:|
| Q4_0 | 1.21x | 1.05x | 1.32x |
| IQ4_XS | 1.17x | 1.00x | 1.68x |
| Q4_K_M | 1.22x | 1.04x | 1.03x |
| Q8_0 | 1.44x | 1.17x | 1.15x |

## Analysis

### Orchestrator review and conclusions

- **Verdicts accepted.** H1 is refuted (decode v7/v6 ≈ 1.0×; both VMs hit a ~100 GB/s ceiling), H2's magnitude is
  inconclusive (it lands between the two v7 instances), and H3 is supported (nothing reaches the 1.27× price break-even).
- **For llama.cpp, buy Emerald Rapids (E16ds_v6)**: best valid batch configuration (Q4_K_M, 32 sequences) costs $19.3/M
  tokens blended on-demand ($3.8/M at Spot) vs $23.9/M on bench-e16v7. Granite Rapids' extra channels and clocks
  don't reach an 8-core slice running llama.cpp, whose kernels use only a few percent of AMX peak (E07).
- **The ~100 GB/s ceiling is per VM slice, not per socket**: decode bandwidth is set by the core count of the VM
  (memory-level parallelism of 8 cores), so cost per output token in llama.cpp is roughly constant across these
  sizes. Larger VMs (more cores) should scale it, but that can't be tested on this subscription (20-vCPU regional quota).
- **This conclusion is specific to llama.cpp.** A stack that actually uses AMX efficiently (oneDNN in vLLM/OpenVINO,
  E09/E10) may separate the generations, because compute is where v7 differs (clock, AMX-FP16). If E09/E10 show much
  higher prefill, repeat their best configuration on bench-v6 before choosing hardware (follow-up in E15).

### Reporter's analysis (reviewed)

**Which VM is cheaper per token for llama.cpp: v6 (Emerald Rapids).** In every blended comparison measured, v7 costs the same or more per token than v6: 1.17-1.44x (against `bench-e16v7`) and 1.00-1.17x (against `bench-lc2`) for the AMX build at one sequence, 1.03-1.68x (against `bench-lc2`) for the control at one sequence, and 1.09-1.23x (against `bench-e16v7`) for the control at 32 sequences. The best valid configuration costs $19.3/M tokens on v6 ($3.8/M Spot) against $23.9/M on `bench-e16v7` ($4.6/M Spot), 19% less, and about the same as the extrapolated `bench-lc2` figure. The comparison range against both Granite Rapids instances is therefore: for the AMX build at one sequence v6 is 0-15% cheaper per token than `bench-lc2` and 15-31% cheaper than `bench-e16v7`; for the control it is 3-40% cheaper than `bench-lc2` at one sequence (the top end is the anomalous IQ4_XS cell, A1; 3-24% without it) and 8-19% cheaper than `bench-e16v7` at 32 sequences. The spread between the two v7 instances (9-23% in E08, 15-25% in the brief) is comparable to the 27% price premium, so the size of v6's advantage depends on which v7 instance you get and no single-instance comparison should be trusted. The direction of the answer does not change: the v7 premium is not repaid for llama.cpp.

**H1 (decode scales with memory bandwidth; v7 decodes 1.2-1.6x faster): refuted.** Decode speedup of v7 over v6 on the AMX build is 0.78-1.14x for `bench-e16v7` (geometric mean 1.00x) and 0.96-1.32x for `bench-lc2` (1.16x). Only two lc2 cells reach the predicted range (IQ4_XS 1.32x, Q4_K_M 1.24x); no `bench-e16v7` cell does, and the batched control at 32 sequences gives 1.07x, 0.99x, 1.10x (Q4_0, Q4_K_M, Q8_0). The mechanism in H1 also fails the physical check: Q8_0 on the AMX build streams about 104 GB/s on v6 (3.56 tok/s x 29.1 GB; 108 GB/s counting the 30.4 GB repacked AMX buffer), the same as lc2's 99 GB/s (E08: 3.42 tok/s) and above `bench-e16v7`'s 81 GB/s, although the hypothesis's own channel arithmetic gives socket peaks of 358 GB/s (8 x DDR5-5600) and 614 GB/s (12 x DDR5-6400), a 1.7x gap. Neither VM gets near its socket peak (29% and 16% of those figures, which are the hypothesis's numbers and were not measured here). With 8 cores both are limited by what 8 cores can pull (about 12-13 GB/s per core, probably the number of outstanding cache-line requests per core; not tested), not by the channels behind them. For quants that are not at that ceiling (IQ4_XS 54 GB/s, Q4_K_M 63 GB/s on v6) decode is probably limited by dequantization compute (an inference, as in E03/E08; no decode profile exists), and there the results depend on the kernel and on the CPU in ways I can't explain (anomaly A1).

**H2 (prefill ~1.2x faster on v7; extra ISA irrelevant): the magnitude is inconclusive, the ISA claim is supported.**

- Magnitude: AMX `pp512` speedup of v7 over v6 is 0.96-1.08x for `bench-e16v7` (geometric mean 1.01x) and 1.15-1.32x for `bench-lc2` (1.23x); control `pp512` against lc2 is 1.16-1.33x; batched control prefill against `bench-e16v7` is 1.03-1.18x (Q4_K_M 1.04x, Q4_0 1.15x, Q8_0 1.18x). The predicted 1.2x (3.6/3.0 GHz) lies between the two Granite Rapids instances, which themselves differ by 1.20-1.23x on `pp512` (E08). One instance matches the clock-ratio prediction and the other shows no gain, so a single v6-v7 pair could have produced either answer.
- ISA: v6 has AMX INT8 and BF16 but not AMX-FP16, and the AMX build behaves as on v7. AMX/control prefill ratios are 1.36x, 0.88x, 1.61x, 1.70x on v6 against 1.31x, 0.88x, 1.53x, 1.68x on lc2 (within 0.08). IQ4_XS prefill is slower with AMX on v6 too (0.88x, same as E03 0.93x and E08 0.88x), so that anomaly (E03 A1) is not specific to Granite Rapids.
- Efficiency is the same on both generations and tiny: v6's AMX prefill is 1.10-1.68 TFLOPS, 2.2-3.4% of the 49 TOPS AMX peak (E03/E08: 2.0-3.3% of 59 TOPS). The AMX units are not the bottleneck on either, which is why the generation barely matters; peak-proportional gains do not show up when the software reaches 3% of the peak. The best no-AMX prefill (IQ4_XS control, 1.90 TFLOPS) is 31% of the VNNI peak, which is the highest utilization of any configuration on v6.

**H3 (v6 cheaper unless v7's speedup is below 1.27x): supported.** Break-even is 1.268 on demand and 1.250 on Spot. The v7 speedup in request time (512 prefill + 128 decode tokens, one sequence) is 0.88-1.08x against `bench-e16v7` and 1.08-1.26x against `bench-lc2` for the AMX build, 0.76-1.23x for lc2's control, and 1.01-1.14x for `bench-e16v7` over v6 in the batched control at 32 sequences (inverse of the wall-time ratio for identical work). No blended cell reaches 1.27x. The nearest is lc2 IQ4_XS AMX at 1.26x (cost ratio 1.00x), i.e. a tie. Only 4 of 24 per-phase ratios exceed 1.27x (all against `bench-lc2`: AMX Q8_0 prefill, AMX IQ4_XS decode, control Q4_0 and Q8_0 prefill), and the phases that matter most for the reference workload (prefill is 73-76% of busy time when batched) lie at 1.0-1.2x. The Spot break-even is 1.25x, nearly the same, so Spot does not change the ranking. The condition could flip for a stack that uses v7-only features. H2's premise is that llama.cpp does not use AMX-FP16, and the matching AMX/control ratios on both generations are consistent with that; E09 and E10/E11 should check their own paths, where AMX-FP16 or more memory bandwidth might matter.

**Compare with physical limits (summary).**

| quantity | v6 | v7 (E03/E08) | reading |
|---|---|---|---|
| best decode GB/s (8 threads, AMX build) | 104 (Q8_0), 82 (Q4_0) | 99 / 95 (lc2), 86 / 81 (e16v7) | same ~100 GB/s ceiling on both, far below socket peaks, not channel-bound |
| prefill, AMX build (TFLOPS, % of AMX INT8 peak) | 1.10-1.68, 2.2-3.4% | 1.4-1.9, 2.4-3.3% (lc2); 1.15-1.62, 2.0-2.8% (e16v7) | software-limited; the clock-scaled peak is irrelevant |
| batched decode at 32 sequences (TFLOPS) | 0.55-0.78 | 0.60-0.78 (e16v7) | compute-bound, like a small prefill |

**Anomalies / items for investigation**

- A1: the control's decode is not slower on v6 uniformly. Against lc2, v6 is 28% faster for Q4_0 (3.29 vs 2.57), 49% faster for IQ4_XS (3.13 vs 2.10), 17% slower for Q4_K_M (3.21 vs 3.88) and level for Q8_0; E05 on `bench-e16v7` agrees for Q4_0 (2.44 vs v6's 3.28 at one sequence). For IQ4_XS the two v7 no-AMX builds (E03's explicit-flags `build-noamx`, 2.05 tok/s; E08's control, 2.10) are both about 1.5x slower than v6's control, so this is not (only) the `-march=graniterapids` versus `sapphirerapids` compile target. A CPU-microarchitecture or VM-level effect on the IQ4_XS and Q4_0 repack lookups is possible; unexplained. It also reopens E03-A2/E05-B5 (the "slow Q4_0 control on v7"). A test: rebuild the control on lc2 with `-march=sapphirerapids`, and profile IQ4_XS decode on both VMs with E07's method.
- A2: Q8_0 AMX decode on `bench-e16v7` is 2.78 tok/s against 3.56 on v6 and 3.42 on lc2, and E05's one-sequence Q8_0 control decode on that instance is also 10% below v6's (1.97 vs 2.20). This one cell is why the v7/v6 decode range starts at 0.78x. Instance variance (E03 reported 4% sd inside that cell) or a bandwidth-sensitive instance; not examined.
- A3: output correctness on v6 is unchecked for both builds (see Setup). The v6 control binary is built for `sapphirerapids`, a different code path from the v7 builds checked in E06. The speed numbers are only usable if a short output comparison passes.
- A4: the control's Q4_K_M `pp512` is noisy on v6 (sd 6.4%) and on lc2 (3.2%), the only cell with a noticeable spread in both experiments; the cause is not known.
- A5: AMX multi-sequence on v6 was not run (E04 bug, E05 crashes), so the comparison for the fastest llama.cpp configuration (AMX build, ~$16/M on v7 if its output were valid) is not available for v6.

## Threats to validity

- One instance of v6, one of each v7 side, one run per configuration with 3 back-to-back repetitions (tiny sd, but no instance or time-of-day averaging); the batched cells are single runs. E08's 9-23% spread between two v7 instances is comparable to the price premium, so the v6-v7 verdict is only as good as the range shown; a second v6 instance could move v6 by a similar amount.
- Regions differ (westus2, eastus2, northcentralus); prices are the brief's list prices and `bench-lc2`'s northcentralus price is unverified. Different regions may also run different host generations or loads.
- The batched comparison exists for `bench-e16v7` only, not `bench-lc2`; the lc2 batched figure is an extrapolation from single-sequence ratios.
- The control builds are "-march=native" on each VM: sapphirerapids (v6) vs graniterapids (v7) target and tuning. The code differs, so part of any difference may be compiler output and not the CPU (A1).
- Output not verified on v6 (A3). Prompts in `llama-batched-bench` are synthetic, and prompt length 128 differs from the reference 512 (see cost assumption 4).
- The 3.0 GHz (v6) and 3.6 GHz (v7) clocks come from Learn and the brief; actual all-core frequency under AMX load was not measured, so the peak percentages are approximate. The DDR5 channel counts and speeds in H1 are from the hypothesis, not from Azure documentation or a bandwidth test.
- Cost figures assume 100% utilization and a lock-step batch; real jobs with ragged lengths cost more on either VM, and the ranking should hold since both are affected equally.

## Next steps

- Check output on v6: a short single-sequence and 4-sequence comparison of `build` and `build-native-noamx` (the E04 sanity script), before E12 or others use the v6 numbers. Needs a few minutes of the VM that E12 is using now.
- Close the instance-variance gap: run E05's batched control (Q4_K_M at least, 1-32 sequences, ~20 minutes per quant) on `bench-lc2`, and repeat `llama-bench` on a second v6 instance, so the comparison has two instances per side.
- Run the E08 16-thread variant on v6 (E08: +25% prefill on lc2); if it gains the same, the break-even comparison stands, otherwise recompute.
- Explain A1: STREAM-like bandwidth probe at 8 and 16 threads on v6 and a v7 VM, and an E07-style profile of IQ4_XS/Q4_0 decode on both; cross-build the control with `-march=sapphirerapids` on a v7 VM.
- Re-test the v6/v7 ranking for the other stacks (E09 vLLM, E10/E11 OpenVINO): they may use AMX-BF16/FP16 and memory bandwidth differently, and v6 lacks AMX-FP16.
