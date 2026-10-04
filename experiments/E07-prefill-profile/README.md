# E07: Why is prefill slow? CPU profile

| | |
| --- | --- |
| Status | done (profiled 2026-10-04; reported and reviewed 2026-10-04) |
| VM | `bench-lc2` (Standard_E16ds_v7, northcentralus, Regular) |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-noamx` = explicit AVX-512 flag list without AMX; `build-native-noamx` = -march=native minus AMX (clean control); `perf` (cpu-clock sampling) |
| Model | `bartowski/Qwen3.8-27B-GGUF` @ `0c92138c51` (Q4_0) |
| Dates | profiles recorded 2026-10-04 ~03:30-03:32 UTC; instruction-level look at the saved profiles ~07:45 UTC |
| Raw data | `e07-profile.txt` (the job log's profile section: both builds' top-25 tables and the AMX/VNNI instruction counts), `e07-annotate.txt` (output of the two annotate scripts), `annotate_classes.py` and `annotate_prev.py` (the scripts), `tables.py` (regenerates every table below from `e07-profile.txt`). The `perf-*.data` files (24 MB each) stay on the VM. |

## Question

Prefill reaches only 20–35 tok/s, about 1–2 TFLOPS, against tens of TFLOPS of AMX peak. Where does the time go?

## Hypotheses

_Written before the results were known._

- H1: Prefill time is dominated by non-GEMM work, especially the sequential Gated DeltaNet recurrence of the 48 linear-attention layers, not by matrix multiplications.
- H2 (alternative): the GEMMs dominate but run at low efficiency, because the AMX path isn't used for most shapes (llama.cpp's AMX gating conditions, hidden size 5120).

## Setup

- VM `bench-lc2` (Granite Rapids, 8 cores / 16 vCPU, 128 GiB). `perf` (linux-tools) sampling with `-e cpu-clock -F 499 -g`: a software timer (the script's comment says it uses `cpu-clock` because it works in VMs without hardware PMU access; I did not re-check that the PMU is unavailable here). So the profile is time-based: each sample is the instruction a core was executing when the timer fired.
- The profiled command is `llama-bench -p 512 -n 0 -r 1 -t 8` (8 threads, one repetition) on the Q4_0 file, once on `build` (AMX) and once on `build-native-noamx`. The Q4_0 file is a mixed-type file (E03's verbose log lists `q4_0`, `q4_1`, `q8_0` and `q6_K` tensors); the profile's kernels agree (most time is in a `block_q4_0` kernel, a few percent each in `q4_1`, `q8_0` and `q6_K` ones).
- Sample counts: 154,141 samples on `build` and 201,769 on the control at 499 Hz, i.e. 309 and 404 CPU-seconds. One 512-token pass takes 20.3 s (AMX) and 26.7 s (control) on 8 threads, or 162 and 214 CPU-seconds, so the data files hold about two passes plus model load. I infer that `llama-bench` ran a full warm-up pass before the timed repetition, which was not checked in its source; both passes have the same shape, so the profile stays representative, but it includes load-time work (repack, page clearing: about 3% on `build`).
- The rate printed by the profiled runs is 25.26 tok/s (AMX) and 19.15 tok/s (control); E08 measured 25.46 and 19.37 tok/s for the same configurations without `perf`, so the sampling overhead is about 1%.
- Instruction-level look (after the fact): `annotate_classes.py` and `annotate_prev.py` read the saved `perf` data with `perf script -F ip,sym,symoff`, map each sampled address to the disassembly (`objdump -d` of the symbol's address range) and aggregate by instruction. They ran with `nice -n 19` on the VM while an OpenVINO job was running there; the profiles themselves were recorded when the VM was otherwise idle (job chain `lc2`, 02:49-03:54 UTC).
- No output check applies here (the run discards the generated text); AMX single-sequence correctness is E04's and covers Q4_K_M only.

## Method

`bench/profile_prefill.sh`: `perf record -e cpu-clock -g` of `llama-bench -p 512 -n 0` for Q4_0 on `build` and `build-native-noamx`; top functions by self time.

## Measurements

### Where the time goes

Self time per symbol from `perf report --no-children --sort dso,symbol`, top 25 symbols per build grouped by what they do (rules in `tables.py`). Percent of all samples, which include threads spinning in OpenMP barriers.

| category | AMX build (%) | clean control, no AMX (%) |
|---|---:|---:|
| GEMM: AMX tile kernel (lambda in ggml_backend_amx_mul_mat) | 78.0 | 0.0 |
| GEMM: AMX tinygemm for q4_1 / q8_0 / q6_K tensors | 6.3 | 0.0 |
| GEMM: repacked Q4_0 kernel (LUT, AVX-512) | 0.0 | 78.7 |
| GEMM: ggml_vec_dot q4_1 / q6_K (generic) | 0.0 | 8.1 |
| GEMM: tinyBLAS q8_0 / tiled VNNI micro-kernels | 0.0 | 4.2 |
| weight repack / unpack (load time or per tile) | 3.3 | 0.9 |
| Gated DeltaNet scan | 2.2 | 1.7 |
| ssm_conv (linear-attention conv) | 1.0 | 0.7 |
| concat | 1.4 | 1.0 |
| full attention (flash_attn_ext) | 0.7 | 0.5 |
| F32 matmul (tinyBLAS f32, vec_dot_f32) | 1.4 | 1.1 |
| norms, activations, adds | 1.1 | 0.8 |
| activation quantization | 0.4 | 0.2 |
| OpenMP wait (libgomp) | 2.8 | 1.5 |
| other (ld.so TLS lookup) | 0.3 | 0.1 |
| listed (top 25 symbols) | 98.7 | 99.3 |
| not listed (beyond the top 25) | 1.3 | 0.7 |

| build | pp512 under perf (tok/s) | GEMM kernels (% of samples) | Gated DeltaNet scan (%) | scan + ssm_conv + concat (%) | OpenMP wait (%) | CPU-seconds sampled (samples / 499 Hz) | prefill TFLOPS (whole run) | TFLOPS inside GEMM kernels (whole-run TFLOPS / GEMM share) | % of INT8 peak (59 TOPS AMX, 7.4 TOPS VNNI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMX build | 25.26 | 84.3 | 2.2 | 4.5 | 2.8 | 309 | 1.38 | 1.64 | 2.8% |
| clean control | 19.15 | 90.9 | 1.7 | 3.4 | 1.5 | 404 | 1.05 | 1.15 | 15.5% |

Notes on the table: the 78% AMX symbol is one function (`ggml_backend_amx_mul_mat(...)::{lambda(int, int)#3}`, a lambda that inlines the tile loop, the 4-bit-to-int8 unpacking and the dequantization epilogue). "Weight repack / unpack" on `build` is `pack_qs` and `unpack_B` (AMX weight preparation) plus the kernel's page clearing; `clear_page_erms` (1.3%) happens when the repacked weight copy is allocated at load. The "TFLOPS inside GEMM kernels" column assumes the GEMM share of time is spent at the average rate, and uses 54.6 GFLOP/token (2 × 27.32 G parameters, an upper bound that includes the embedding table and the unused MTP block); the peaks are 59 TOPS (AMX INT8, 8 cores × 3.6 GHz × 2048 ops/cycle) and 7.4 TOPS (AVX-512 VNNI, two `vpdpbusd` ports).

### Inside the GEMM kernels (instruction level)

All samples in the AMX symbol (78.0% of all samples, 120,287 samples) grouped by the instruction the sampled address points at. The symbol is 30,641 bytes and contains 24 `tdpbssd` (tile multiply), 23 `tileloadd`, 24 `tilestored` and 22 `tilezero` instructions. `e07-annotate.txt` has the raw output.

| instruction at the sampled address | share of the AMX kernel's samples (%) |
|---|---:|
| `tilezero` (tile load/store/zero class) | 75.3 |
| AVX-512 (zmm) arithmetic and moves, not VNNI | 14.2 |
| scalar and other | 10.4 |
| `tdpbssd` (the tile multiply) | 0.1 |

The same samples grouped by the instruction *before* the sampled one (the one that had just executed): `tileloadd` 69.4%, `vfmadd213ps` 8.3%, `tilestored` 8.0%, `vmulps` 3.3%, `vcvtdq2ps` 2.2%. Two sites, both a `tilezero` that follows a `tileloadd`, hold 34.6% and 33.1% of the kernel's samples; two more, a `tilezero` after a `tilestored`, hold 3.5% and 3.4%. The disassembly before the hottest site is a run of AVX-512 `vpsrlw`/`vpandd`/`vpaddb` that unpack 4-bit weights, `vmovdqa64` stores of the unpacked int8 data to a scratch buffer, and then `tileloadd`.

For the clean control, the same method on the `gemm_q4_b32_8x8_q8_0_lut_avx<block_iq4_nlx8>` kernel (78.7% of all samples): 78.3% of its samples are on AVX-512 instructions other than VNNI, 10.5% on `vpdpbusd` (the multiply-accumulate), 8.4% on AVX2. Top instructions: `vmovdqa64` 27.3%, `vpabsb` 15.9%, `vpdpbusd` 10.5%, `vpsubb` 7.6%, `vfmadd213ps` 6.0%.

## Cost per token

Prefill only: this experiment measures no decode. For the output-token and blended cost of the same configurations (Q4_0, AMX and control, 8 and 16 threads) see E08. Rates are the profiled `pp512` runs (single sequence, 8 threads); USD per million input tokens = `P / (tok/s × 3600) × 1e6` at the all-in price P. The "what-if" rows are an Amdahl bound, not a measurement: they assume the GEMM kernels' share of time (84.3% on `build`) shrinks by the stated factor, and everything else stays as profiled (including load-time work and OpenMP waiting, which a steady-state server would not pay, so the bound is a little pessimistic on that count and optimistic if the other work grows).

| build | case | pp512 (tok/s) | input USD/M on-demand | input USD/M Spot |
|---|---|---:|---:|---:|
| AMX build | measured | 25.3 | 18.5 | 3.6 |
| clean control | measured | 19.2 | 24.4 | 4.7 |
| AMX build | what-if: GEMM 2x faster | 44 | 10.7 | 2.1 |
| AMX build | what-if: GEMM 4x faster | 69 | 6.8 | 1.3 |
| AMX build | what-if: GEMM 9x faster | 101 | 4.6 | 0.9 |
| AMX build | what-if: GEMM time to zero (bound) | 161 | 2.9 | 0.6 |

Assumptions: all-in $1.681/h on-demand (VM $1.663 + $0.018 disk and IP; PLAN.md lists the same $1.681/h for `bench-lc2` in northcentralus as for `bench-e16v7` in eastus2), $0.325/h Spot (not available on this subscription); 100% utilization; model load, VM boot and the warm-up pass excluded; a 512-token prompt on an empty context; E08's 16-thread Q4_0 run gives 28.97 tok/s, or $16.1/M on-demand ($3.1/M Spot), and is not profiled here.

## Analysis

### Orchestrator review and conclusions

- **H1 is refuted, decisively**: the 48 Gated DeltaNet layers cost ~2% of prefill. The hybrid architecture is not why
  CPU prefill is slow.
- **The bottleneck is software**: 84–91% of prefill is quantized GEMM, and llama.cpp's AMX tile kernel spends its time
  around tile loads (69% of samples), not tile multiplies (0.1%), reaching ~3% of AMX peak. That's consistent with E08
  (a second hyperthread per core adds 14–25%, so the AMX unit is idle much of the time) and with E03's A3 (prefill
  efficiency depends on how the batch is formed). A well-blocked AMX GEMM (oneDNN, used by vLLM, OpenVINO and PyTorch) should
  run several times faster on the same hardware. The Amdahl ceiling here is ~160 tok/s if GEMMs cost nothing.
- **Decision impact**: don't invest further in llama.cpp tuning for this workload. E09 (vLLM) and E10 (OpenVINO)
  test whether oneDNN-based stacks collect that prize. If one does, its prefill rate is the main input to the final cost.
- The open anomaly A5 (4×128 vs 1×512 prompts on the same build) stays open; a hardware-counter profile (not
  available in this VM: `perf` falls back to cpu-clock) or a llama.cpp-side timing breakdown would be needed.

### Reporter's analysis (reviewed)

**H1 (non-GEMM work, especially the Gated DeltaNet recurrence, dominates prefill): refuted.** GEMM kernels take 84.3% of the samples on the AMX build and 90.9% on the control. The Gated DeltaNet scan takes 2.2% (AMX) and 1.7% (control), and with `ssm_conv` and `concat` (the other linear-attention ops that show up) 4.5% and 3.4%. Full attention is 0.7% and 0.5%, norms and activations about 1%. Everything that is not a GEMM, a weight repack or OpenMP waiting adds up to 8.5% (AMX build) and 6.1% (control). The 48 linear-attention layers are not what holds prefill back at 512 tokens.

**H2 (GEMMs dominate but run at low efficiency, because the AMX path is not used for most shapes): first half supported, second half refuted, and the real cause is different.**

- Dominance: supported (84-91% above).
- Low efficiency: supported. Counting only GEMM time, the AMX build runs at about 1.64 TFLOPS equivalent, 2.8% of the 59 TOPS AMX peak, and only 1.43x faster than the control's AVX-512 kernel (1.15 TFLOPS in-kernel, 15.5% of the 7.4 TOPS VNNI peak), although the AMX peak is 8x the VNNI peak. The control's kernel is itself far from its peak: the multiply-accumulate `vpdpbusd` holds 10.5% of its samples, and byte shuffles, sign handling and loads the rest.
- "AMX is not used for most shapes": refuted. The AMX tile kernel is 78.0% of all samples and `tinygemm_kernel_amx` for the `q4_1`, `q8_0` and `q6_K` tensors another 6.3%; no non-AMX quantized-GEMM symbol appears among the AMX build's top 25 (the control, by contrast, spends 8.1% in generic `ggml_vec_dot_q4_1_q8_1` and `q6_K`, the types without a repacked kernel). So all the quantized matmuls of this file take the AMX path. The profile cannot say how many distinct matrix shapes there are, only that none visibly falls back.
- What the AMX kernel is doing instead of multiplying: only 0.1% of its samples sit on `tdpbssd`, and 69% sit right after a `tileloadd` (8% after a `tilestored`), at two hot sites. Reading of the timer-based samples (hedged): the core's oldest unretired instruction is a tile load or its immediate successor, so the kernel spends most of its time waiting for tile loads to complete, not multiplying. The tile multiply units are therefore likely idle most of the time, which also fits E08's finding that a second hyperthread per core adds 14-25% prefill. What the loads wait for cannot be told from a software-timer profile; candidates, none tested: (a) the activation tile misses L2 at 512 rows (the int8 activations of one 512-token ubatch are 2.6 MB for a projection with K = 5120, more for any with larger K, against a 2 MB L2 per core on Granite Rapids, which I did not verify on this VM); (b) the load reads the scratch buffer that the immediately preceding AVX-512 stores just wrote, which a tile load cannot serve by store forwarding; (c) the 16-row strided tile load itself is slow. E03's A3 argues against (a) as the whole story: on the AMX build four 128-token prompts in one 512-row ubatch prefill at 35 tok/s, against 21 tok/s for one 512-token prompt (both on `bench-e16v7`), although the row count and the matrix shapes are the same. The profile cannot explain that gap either, since attention and the recurrence are small at `pp512`; a profile of the 4 x 128 case is the direct comparison.
- Ceiling: if the GEMM share could be accelerated 4x, 9x or without bound, prefill would reach about 69, 101 or 161 tok/s on this VM (Amdahl, table above), against 25 measured. That is the size of the prize if the kernel stall is fixable in software; it is not a prediction.

**Answer to the question.** Nearly all the time (84%) is in the quantized GEMMs, which run through AMX but use only ~3% of its peak because the tile kernel stalls on tile loads; all non-GEMM compute (recurrence, convolution, attention, norms) together takes about 8%.

**Anomalies**

- A1: the AMX kernel's samples concentrate at two `tilezero` instructions right after `tileloadd`. A sample could be attributed one instruction late (skid), so this should be treated as "stalled around the tile load", and the load's target (activations or the unpacked weight scratch) is not identified.
- A2: the AMX build is only 1.43x faster in-kernel than the AVX-512 LUT kernel of the control, and for IQ4_XS and Q6_K it was slower than the non-AMX builds in E03/E08. Only Q4_0 is profiled, so those cases are unexplained.
- A3: `clear_page_erms` (1.3% on `build`) and `pack_qs`/`unpack_B` (2.0%) show that the profile includes model-load work; and a profile that holds two passes can't separate warm-up from timed execution. The kernel shares are not affected by more than a few percent.
- A4: the OpenMP wait share is small (2.8% on `build`, 1.5% on the control): thread imbalance is not what limits prefill.
- A5: E03's A3 (the same AMX build prefills 4 x 128 tokens in one 512-row ubatch at 35 tok/s but 1 x 512 tokens at 21 tok/s) is not explained: the `pp512` profile shows GEMM-dominated time with the stall at tile loads, and the GEMM shapes are the same in both cases. Something other than shape (sequence length, sequence count, or how `llama-bench` and `llama-batched-bench` drive the model) changes the GEMM's efficiency.

## Threats to validity

- Software-timer sampling at 499 Hz, no hardware counters: no cache-miss, stall or frequency data, so the mechanism behind the tile-load stall is inference. Sample skid and virtualized timers can shift attribution by an instruction.
- One quant (the mixed-type Q4_0 file), one prompt length (512 tokens, empty context), 8 threads, one VM instance. E03/E08 show other quants behave differently (IQ4_XS and Q6_K lose with AMX), and A3 in E03 shows prompt length matters on the AMX build; neither was profiled.
- The profile holds about two passes and the load. The category totals use the top 25 symbols per build (98.7% and 99.3% of samples); the remainder (1.3%, 0.7%) is not categorized.
- The FLOP and peak figures are theoretical (3.6 GHz assumed under load, 54.6 GFLOP/token upper bound), so the efficiency percentages carry a few-percent error from the FLOP count and an unknown error from the frequency.
- The instruction-level scripts ran hours after the recording, against the same binaries and `perf` data; the build files were not rebuilt in between (checked only by the matching 78.0% share against `perf report`'s 78.04%).

## Next steps

- Profile E05's prefill shape (4 sequences x 128 tokens in one 512-row ubatch, AMX build, Q4_0) next to this `pp512` profile: same matrix shapes, 1.7x the speed (E03 A3). Whatever differs in the instruction-level profile is the stall's cause. Add a prompt-length / ubatch sweep on the AMX build (`pp` 64-1024, `-ub` 64-512, 8 and 16 threads); if prefill per token rises as the ubatch shrinks, the activation tile loads miss cache (hypothesis a) and a smaller ubatch is a free speed-up.
- Repeat the profile for IQ4_XS and Q8_0 on both builds to explain E03/E08's IQ4_XS and Q6_K results, and profile `tg128` on both builds, since E08 finds a large decode gain that cannot come from tile multiplies at M = 1.
- Profile with 16 threads (does the second hyperthread fill the stall?) and, if a bare-metal or PMU-enabled size is available, collect `perf stat` cache and tile counters; check whether Azure exposes the PMU on any E16-class size.
- If hypothesis (b) or (a) holds, the fix is in llama.cpp's AMX backend (`ggml_backend_amx_mul_mat`; blocking or prefetch for the activation tile, a direct packed-weight layout that skips the scratch buffer). Whether to take this upstream is for the orchestrator.
