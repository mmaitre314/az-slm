# E08: Clean AMX control build; threads, SMT and pinning

| | |
| --- | --- |
| Status | done (measured 2026-10-04 ~02:55–03:40 UTC; reported and reviewed 2026-10-04) |
| VM | `bench-lc2` (Standard_E16ds_v7, northcentralus, Regular) |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-noamx` = explicit AVX-512 flag list without AMX; `build-native-noamx` = -march=native minus AMX (clean control) |
| Model | `bartowski/Qwen3.8-27B-GGUF` @ `0c92138c51` (Q4_0, IQ4_XS, Q4_K_M, Q8_0) |
| Dates | measured 2026-10-04 02:49-03:30 UTC (A/B 02:49-03:16, threads 03:16-03:30) |
| Raw data | `llama-bench.jsonl` (16 rows) and `llama-threads.jsonl` (12 rows), both reduced on the VM to the key fields incl. the three per-repetition samples; `model-buffers.txt` (weight-buffer lines from the verbose logs); `tables.py` (regenerates every table below) |

## Question

(a) E03's no-AMX build also differs in compiler flags. With a control build identical except for AMX, what does AMX alone contribute? (b) Do 16 threads (SMT) or pinning help?

## Hypotheses

_Written before the results were known._

- H1: With the clean control, the AMX effect on decode shrinks to ~0 (E03's decode differences come from other flags) and prefill gains remain at 1.3–2×.
- H2: 16 threads add ≤10% to prefill and may slow decode (two hyperthreads share one AMX unit and the same bandwidth); pinning adds a few percent.

## Setup

- VM `bench-lc2`: Xeon 6 (Granite Rapids), 16 vCPU = 8 cores with SMT, 128 GiB, pay-as-you-go. This is a different instance from E03/E05's `bench-e16v7` and runs 9-23% faster on the same quant and build (instance table below), so absolute numbers are not comparable across the two VMs; ratios inside this experiment are.
- Builds (all llama.cpp `11fe02151f79`): AMX instruction count in `libggml-cpu.so` (`objdump -d | grep -cE 'tdpb|tileloadd'`) is 203 for `build` and 0 for `build-noamx` and `build-native-noamx`; `vpdpbusd` (AVX-512 VNNI) counts are 2269, 1189 and 1189. The control was compiled with `-march=native -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-amx-fp16 -mno-amx-complex` and all `GGML_AVX*` options off (`bench/build_native_noamx.sh`).
- `llama-bench` defaults otherwise (`n_batch` 2048, `n_ubatch` 512, flash attention auto, f16 KV cache, weight repacking on), 8 threads unless stated, 3 repetitions run back to back; `pp512` = 512 prompt tokens on an empty context, `tg128` = 128 generated tokens.
- Pinning: `-C 0x5555 --cpu-strict 1`, a mask with the first hyperthread of each physical core as listed by `lscpu -p=CPU,CORE` inside the guest.
- The job chain `lc2` (E08, then E07, then E06) ran alone on the VM from 02:49 to 03:54 UTC; the next jobs on the VM (OpenVINO) began at 07:00. Light Run Command calls from other agents during that window were not logged.
- Output correctness was not checked here. The AMX build's single-sequence output was verified for Q4_K_M only (E04); Q4_0, IQ4_XS and Q8_0 outputs on the AMX build were not compared, and no 16-thread output was compared.

## Method

`bench/build_native_noamx.sh` (−march=native −mno-amx-*; verified 0 AMX instructions), `bench/llama_bench.sh` for 4 quants on `build` vs `build-native-noamx`, `bench/llama_threads.sh` (8 threads, 8 pinned to one hyperthread per core, 16 threads) for Q4_K_M and Q4_0 on `build`.

## Measurements

All rows are `llama-bench` on one sequence, mean ± sample standard deviation of 3 repetitions. Raw data: `llama-bench.jsonl`, `llama-threads.jsonl`. No row is marked invalid. The largest repetition-to-repetition spreads are Q4_K_M control `pp512` (22.3, 23.0, 23.7 tok/s, 3.2%) and Q4_0 control `pp512` (1.2%); everything else is under 0.5%.

### A/B: AMX build against the clean control

| quant | file GB | pp512 AMX (tok/s) | pp512 control (tok/s) | AMX/control | tg128 AMX (tok/s) | tg128 control (tok/s) | AMX/control |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_0 | 16.3 | 25.5 ± 0.01 | 19.4 ± 0.23 | 1.31x | 5.79 ± 0.003 | 2.57 ± 0.001 | 2.26x |
| IQ4_XS | 15.5 | 35.6 ± 0.02 | 40.3 ± 0.02 | 0.88x | 4.57 ± 0.018 | 2.10 ± 0.001 | 2.18x |
| Q4_K_M | 17.4 | 35.1 ± 0.01 | 23.0 ± 0.73 | 1.53x | 4.45 ± 0.007 | 3.88 ± 0.046 | 1.15x |
| Q8_0 | 29.1 | 27.5 ± 0.03 | 16.3 ± 0.01 | 1.68x | 3.42 ± 0.002 | 2.16 ± 0.001 | 1.58x |

AMX/control is `build` over `build-native-noamx`. Physical limits (assumptions: prefill FLOP per token = 2 × 27.32 G parameters = 54.6 GFLOP, an upper bound that includes the embedding table and the unused MTP block; AMX INT8 peak 59 TOPS = 8 cores × 3.6 GHz × 2048 ops/cycle; AVX-512 VNNI peak 7.4 TOPS = 8 × 3.6 GHz × 256 ops/cycle with two 512-bit `vpdpbusd` ports; decode GB/s = tg × GGUF file size, a lower bound on traffic because the file includes the embedding table and MTP block):

| quant | prefill TFLOPS AMX | % of 59 TOPS AMX peak | prefill TFLOPS control | % of 7.4 TOPS VNNI peak | decode GB/s AMX (tg x file) | decode GB/s control |
|---|---:|---:|---:|---:|---:|---:|
| Q4_0 | 1.39 | 2.4% | 1.06 | 14% | 95 | 42 |
| IQ4_XS | 1.94 | 3.3% | 2.20 | 30% | 71 | 32 |
| Q4_K_M | 1.92 | 3.3% | 1.26 | 17% | 78 | 68 |
| Q8_0 | 1.50 | 2.5% | 0.89 | 12% | 99 | 63 |

Weight buffers that each build used (verbose logs, `model-buffers.txt`). The AMX build holds the large 2-D weight matrices in an `AMX` buffer; the control uses a `CPU_REPACK` buffer only for the types that have a repacked CPU kernel (Q4_0, Q4_K, a sliver of IQ4_XS) and runs Q8_0 straight from the mmap'd file. These sizes are identical to the ones E03 logged for `build-noamx`, so the two no-AMX builds use the same weight layouts:

| quant | AMX buffer, `build` (MiB) | CPU_REPACK buffer, `build-native-noamx` (MiB) |
|---|---:|---:|
| Q4_0 | 14661 | 11993 |
| IQ4_XS | 13980 | 101 |
| Q4_K_M | 16057 | 10238 |
| Q8_0 | 28990 | none |

### Threads, SMT and pinning

| quant | variant | pp512 (tok/s) | vs 8 threads | prefill TFLOPS | tg128 (tok/s) | vs 8 threads | decode GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 8 threads | 34.88 ± 0.39 | +0.0% | 1.91 | 4.476 ± 0.003 | +0.0% | 78 |
| Q4_K_M | 8 threads, one hyperthread per core, strict | 35.37 ± 0.01 | +1.4% | 1.93 | 4.485 ± 0.001 | +0.2% | 78 |
| Q4_K_M | 16 threads | 43.62 ± 0.05 | +25.1% | 2.38 | 4.595 ± 0.006 | +2.6% | 80 |
| Q4_0 | 8 threads | 25.40 ± 0.01 | +0.0% | 1.39 | 5.803 ± 0.004 | +0.0% | 95 |
| Q4_0 | 8 threads, one hyperthread per core, strict | 25.46 ± 0.02 | +0.2% | 1.39 | 5.824 ± 0.001 | +0.4% | 95 |
| Q4_0 | 16 threads | 28.97 ± 0.04 | +14.1% | 1.58 | 5.998 ± 0.004 | +3.4% | 98 |

### Same `build`, two instances of the same VM size

| quant | pp512 e16v7 (tok/s) | pp512 lc2 (tok/s) | lc2 vs e16v7 | tg128 e16v7 (tok/s) | tg128 lc2 (tok/s) | lc2 vs e16v7 |
|---|---:|---:|---:|---:|---:|---:|
| Q4_0 | 21.1 | 25.5 | +21% | 5.29 | 5.79 | +9% |
| IQ4_XS | 29.7 | 35.6 | +20% | 3.97 | 4.57 | +15% |
| Q4_K_M | 28.7 | 35.1 | +22% | 3.88 | 4.45 | +15% |
| Q8_0 | 22.3 | 27.5 | +23% | 2.78 | 3.42 | +23% |

### AMX gain against two different baselines

| quant | prefill gain vs build-noamx (E03) | prefill gain vs control (E08) | decode gain vs build-noamx (E03) | decode gain vs control (E08) | tg128 build-noamx (E03, e16v7, tok/s) | tg128 control (E08, lc2, tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| Q4_0 | 1.34x | 1.31x | 1.49x | 2.26x | 3.55 | 2.57 |
| IQ4_XS | 0.93x | 0.88x | 1.94x | 2.18x | 2.05 | 2.10 |
| Q4_K_M | 1.48x | 1.53x | 1.08x | 1.15x | 3.60 | 3.88 |
| Q8_0 | 1.76x | 1.68x | 1.33x | 1.58x | 2.09 | 2.16 |

The prefill ratios barely move between the two baselines (within 0.08), the decode ratios do (Q4_0 1.49x becomes 2.26x).

## Cost per token

USD per million tokens, single sequence, `bench-lc2` rates. Formulas from [COST_MODEL.md](../COST_MODEL.md): input = `P / (pp512 × 3600) × 1e6`, output = `P / (tg128 × 3600) × 1e6`, blended for the reference request (512 input + 128 output) = `P × (512/pp + 128/tg) / 3600 / 640 × 1e6`. `s per request` is the wall time of one such request at these rates.

| quant | build | threads | input on-demand | output on-demand | blended 512+128 on-demand | input Spot | output Spot | blended 512+128 Spot | s per request |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_0 | AMX | 8 | 18.3 | 80.6 | 30.8 | 3.5 | 15.6 | 6.0 | 42 |
| Q4_0 | no-AMX control | 8 | 24.1 | 181.8 | 55.7 | 4.7 | 35.2 | 10.8 | 76 |
| IQ4_XS | AMX | 8 | 13.1 | 102.2 | 30.9 | 2.5 | 19.8 | 6.0 | 42 |
| IQ4_XS | no-AMX control | 8 | 11.6 | 222.3 | 53.7 | 2.2 | 43.0 | 10.4 | 74 |
| Q4_K_M | AMX | 8 | 13.3 | 105.0 | 31.6 | 2.6 | 20.3 | 6.1 | 43 |
| Q4_K_M | no-AMX control | 8 | 20.3 | 120.4 | 40.3 | 3.9 | 23.3 | 7.8 | 55 |
| Q8_0 | AMX | 8 | 17.0 | 136.7 | 40.9 | 3.3 | 26.4 | 7.9 | 56 |
| Q8_0 | no-AMX control | 8 | 28.6 | 215.7 | 66.0 | 5.5 | 41.7 | 12.8 | 91 |
| Q4_K_M | AMX | 16 | 10.7 | 101.6 | 28.9 | 2.1 | 19.6 | 5.6 | 40 |
| Q4_0 | AMX | 16 | 16.1 | 77.8 | 28.5 | 3.1 | 15.1 | 5.5 | 39 |

Assumptions:

1. All-in hourly price = VM + $0.018 disk and IP: $1.663 + 0.018 = $1.681/h pay-as-you-go (what we paid), $0.3073 + 0.018 = $0.325/h Spot (not available on this subscription; shown for comparison). PLAN.md lists the same $1.681/h all-in for `bench-lc2` (northcentralus) as for `bench-e16v7` (eastus2); COST_MODEL.md lists eastus2 and westus3 but not northcentralus, so the northcentralus price itself was not checked.
2. 100% utilization, steady state; VM boot, model download and model load (a minute or more per `llama-bench` invocation, repack included) are excluded.
3. One sequence at a time. E05 shows the effect of batching, but E05's AMX batched numbers are invalid output (E04), so batched cost comes from the no-AMX control only.
4. Prefill rate = `pp512` on an empty context, decode rate = `tg128` (context 0 to 128); the reference request decodes at a longer context, and E03's A3 shows that prefill speed depends on prompt length on the AMX build.
5. Output correctness of the AMX rows is verified for Q4_K_M only (E04). The 16-thread rows were not checked for output.
6. List prices; Spot evictions not modeled; instance variance (+9-23% for this VM over `bench-e16v7`) is inside every number, so the blended cost figures here are 13-19% lower than E03's for the same configuration.

## Analysis

### Orchestrator review and conclusions

- **H1**: the prefill half is supported (AMX ×1.31–1.68 for Q4_0/Q4_K_M/Q8_0; IQ4_XS ×0.88). The decode half is refuted. The decode
  gain (×1.15–2.26) is real and not a compiler-flag artifact; it comes from the AMX backend's weight layout and
  single-row kernels, not from tiles (see E03's review). **H2 is refuted favorably**: 16 threads give +14–25% prefill and
  +3% decode, and pinning makes no difference.
- **Operational guidance (added to AGENTS.md benchmarking notes)**: run llama.cpp prefill-heavy work with 16 threads on
  these 8-core/16-vCPU sizes; don't bother pinning.
- **Consequence for E05's baseline**: E05 ran 8 threads. With 16 threads its prefill-dominated blended cost would
  likely drop by ~10–20% (from $23.9/M toward ~$20/M on-demand). That's not measured; it's noted in E15 as an adjustment
  range, not a number.
- **Instance variance** (bench-lc2 runs 9–25% faster than bench-e16v7 on identical configurations) means cross-VM
  comparisons need a common reference run. E15 normalizes by quoting each stack against the llama.cpp numbers from the
  same VM where possible.

### Reporter's analysis (reviewed)

**H1 (clean control: AMX decode effect ~0, prefill gain stays 1.3-2x): decode half refuted, prefill half supported for three of four quants.**

- Decode: AMX/control is 2.26x (Q4_0), 2.18x (IQ4_XS), 1.15x (Q4_K_M) and 1.58x (Q8_0). Nothing shrinks to ~0. For Q4_0, IQ4_XS and Q8_0 the gain is larger than against E03's explicit-flag `build-noamx` (1.49x, 1.94x, 1.33x), for Q4_K_M about the same (1.08x, 1.15x), so E03's decode gains are not a compiler-flag artifact. Against the faster of the two no-AMX builds (E03's `build-noamx`, see A2) the Q4_0 gain is 1.5x, not 2.3x. The gain comes from what the AMX build does differently at M = 1: it routes the weights through its own repacked `AMX` buffer and kernels. Single-token decode cannot fill a 16-row tile, so this is the AMX backend's weight layout and its non-tile kernels, not tile hardware (a decode profile would show it; E07 profiles prefill only, so this is inference). It shows up as bandwidth utilization: the AMX build reads 71-99 GB/s, the control 32-68 GB/s (table above), so the control leaves 30-70% of the observed ~100 GB/s unused and is probably limited by dequantization compute, not DRAM.
- Prefill: 1.31x (Q4_0), 1.53x (Q4_K_M), 1.68x (Q8_0) are inside the predicted 1.3-2x band. IQ4_XS is 0.88x: AMX is slower than the control, and the control's 40.3 tok/s is the highest prefill of any configuration measured on this VM. The ratios match E03's against `build-noamx` within 0.08, so compile flags do not matter for prefill and the earlier AMX/no-AMX prefill ratios stand. AMX peak utilization is 2.4-3.3% (1.4-1.9 TFLOPS of 59 TOPS), the control reaches 12-30% of the VNNI peak (that peak is an assumption).
- H1 as a whole is half right: the "AMX gain" for prefill is real but small (1.3-1.7x for three quants, a loss for IQ4_XS), and for decode it is large and not a compiler-flag artifact.

**H2 (16 threads add ≤10% to prefill, may slow decode; pinning adds a few percent): refuted on all three points, in the favorable direction.**

- 16 threads (both hyperthreads of each of the 8 cores) raise prefill by +25.1% (Q4_K_M, 34.9 to 43.6 tok/s) and +14.1% (Q4_0, 25.4 to 29.0 tok/s), more than the predicted ≤10%, and raise decode by +2.6% and +3.4% instead of slowing it. The two hyperthreads of a core share one AMX unit, so the gain means the unit is not saturated by one thread. That is consistent with E07's profile, which finds the AMX kernel stalled on tile loads (69% of its samples sit right after `tileloadd`, 0.1% on `tdpbssd`), leaving idle cycles a second thread can fill. That mechanism is a hypothesis (E07's profile is time-based sampling, and no experiment isolates it). Even at 16 threads prefill reaches only 1.6-2.4 TFLOPS (2.7-4.0% of AMX peak).
- Decode barely moves with 16 threads (+3%), and at 95-99 GB/s for Q4_0 and Q8_0 it sits on a plateau near 100 GB/s on 8 cores. Extra hyperthreads do not lift the plateau; whether more cores would was not tested (this size has 8 cores), so it may be a per-core limit (line-fill buffers are shared by the two hyperthreads of a core) or a per-VM limit. E13's hypotheses quote 12 DDR5-6400 channels for the socket; neither that peak nor this VM's STREAM bandwidth was measured.
- Pinning (one hyperthread per core, strict) changes prefill by +1.4% (Q4_K_M) and +0.2% (Q4_0) and decode by +0.2% and +0.4%: below the predicted "few percent". The Q4_K_M +1.4% is partly one low repetition in the unpinned run (34.4, 35.1, 35.1 tok/s; against the median of the unpinned runs the gain is +0.9%), so pinning is a null result. Guest-level pinning only fixes guest threads to vCPUs; whether the hypervisor keeps the two vCPUs of a pair on one core was not checked.
- Practical consequence: use 16 threads for prefill-heavy llama.cpp jobs on this VM. It costs nothing (the VM bills 16 vCPUs anyway) and lowers the blended single-sequence cost by 7-9% (Q4_0 $30.8 to $28.5/M, Q4_K_M $31.6 to $28.9/M on-demand). Not checked: 16 threads with several sequences, or with the control build.

**Anomalies**

- A1: IQ4_XS prefill is faster on the control (40.3 tok/s, 2.2 TFLOPS) than on the AMX build (35.6 tok/s), and this is the fastest prefill measured; E03 saw the same direction (0.93x). The AMX kernel for IQ4_XS is worse than the generic AVX-512 path. Only Q4_0 was profiled (E07).
- A2: Q4_0 decode on the control is 2.57 tok/s on this VM, and E05 measured 2.44 tok/s for the same build on `bench-e16v7`, while E03's explicit-flag `build-noamx` gets 3.55 tok/s on `bench-e16v7` (about 3.9 tok/s if scaled by this VM's +9%). The two no-AMX builds use identical weight layouts (same `CPU_REPACK` sizes) and differ only in compiler flags, yet differ by about 1.4-1.5x in Q4_0 decode (not in IQ4_XS, Q4_K_M, Q8_0). So "no-AMX" is not a single baseline: the control (-march=native) is the slower one. E07's control profile shows a `gemm_q4_b32_8x8_q8_0_lut_avx<block_iq4_nlx8>` kernel; whether `build-noamx` selects a different kernel was not checked.
- A3: control Q4_K_M `pp512` is noisy (22.3-23.7 tok/s) compared with every other row; the 1.53x ratio carries about ±5%.
- A4: Decode bandwidth differs by quant on the same build (AMX: Q8_0 99 GB/s, Q4_0 95, Q4_K_M 78, IQ4_XS 71), which suggests decode is near the bandwidth plateau for Q8_0 and Q4_0 but limited by dequantization compute for IQ4_XS and Q4_K_M (not profiled).

## Threats to validity

- One VM instance, one run per configuration. The 3 repetitions run back to back and their spread is tiny, but they do not capture instance or time-of-day drift. The same-size instances differ by 9-23% (table above); the ratios in this report are within one VM and are not affected, the absolute numbers are.
- The control differs from `build` only in AMX, but "only AMX" includes everything the AMX backend changes (weight buffer type, repack, kernels for all matmuls, including decode), not just the tile instructions. A2 shows the choice of no-AMX baseline moves the Q4_0 decode gain from 1.5x to 2.3x.
- Pinning inside a guest cannot control the vCPU-to-core mapping done by the hypervisor, and the "first hyperthread of each core" mask assumes the guest's `lscpu` topology is real.
- The AMX and VNNI peaks and the 54.6 GFLOP/token figure are theoretical (3.6 GHz all-core assumed; frequency under AMX load not measured). GB/s are lower bounds (file size includes the embedding table and MTP block, about 5% and 1.5%).
- Only Q4_K_M and Q4_0 were run at 16 threads, only on the AMX build, and only for single-sequence `pp512`/`tg128`. Correctness of the AMX outputs was not checked (see Setup).
- Cost figures are for a single sequence; batched cost is much lower (E05) and uses the control build.

## Next steps

- Use 16 threads for later prefill-bound runs and re-measure batched decode with 16 threads on the control build (E05's setup), which bears on cost at 8-32 sequences.
- Profile decode (`tg128`) on `build` and `build-native-noamx` with the E07 method to find out what the AMX build's M = 1 path does better, and settle A2 by profiling `build-noamx` against `build-native-noamx` for Q4_0 decode.
- Profile IQ4_XS prefill on both builds (A1), and Q8_0, to see whether the AMX kernels for other types stall on tile loads as Q4_0 does (E07).
- Check AMX output for Q4_0, IQ4_XS and Q8_0 single-sequence against the control before using these cost rows in the final recommendation.
