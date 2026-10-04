# E03: llama.cpp single-sequence prefill and decode per quantization, AMX vs no AMX

| | |
| --- | --- |
| Status | done (measured 2026-10-04 02:23–03:25 UTC; reported and reviewed 2026-10-04) |
| VM | `bench-e16v7` (Standard_E16ds_v7, eastus2, Regular) |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-noamx` = explicit AVX-512 flag list without AMX; `build-native-noamx` = -march=native minus AMX (clean control) |
| Model | `bartowski/Qwen3.8-27B-GGUF` @ `0c92138c51` (Q4_0, IQ4_XS, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16) |
| Raw data | `llama-bench.jsonl` (28 rows, reduced to the key fields on the VM), `model-buffers.txt` (weight-buffer lines extracted from the verbose logs), `tables.py` (regenerates every table below) |

## Question

How fast is one sequence at prefill (prompt processing) and decode (generation) for each quantization, and how much does AMX contribute?

## Hypotheses

_Written before the results were known._

- H1: Prefill is compute-bound and benefits from AMX-INT8 for the quant types with AMX kernels (Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS): 2–4× over AVX-512-VNNI.
- H2: Decode is memory-bandwidth-bound (decode tok/s × model bytes ≈ constant, ~80–100 GB/s) and gains little from AMX.
- H3: BF16 has no AMX path in llama.cpp, so it's the slowest at both phases.

## Setup

- VM `bench-e16v7`: Intel Xeon 6973P-C (Granite Rapids), 16 vCPU = 8 cores with SMT, 128 GiB, pay-as-you-go. All numbers in this report are from this one VM instance (E08 saw 15–25% differences between two E16ds_v7 instances).
- llama.cpp `11fe02151f79`, builds `build` (AMX) and `build-noamx` as above. `llama-bench` defaults otherwise: `n_batch` 2048, `n_ubatch` 512, flash-attention auto, f16 KV cache, weight repacking on.
- Model files: GGUF sizes in the first table (10^9 bytes). The Q4_0 file is a mixed-type file (the verbose log shows `q4_0`, `q4_1`, `q8_0` and `q6_K` tensors); the type mix of the other files was not inspected. `model_size` includes the unused NextN/MTP block `blk.64` (about 1.5% of the file), which llama.cpp skips.
- No other benchmark job overlapped: the job list shows E04's sanity runs ended 02:21 and E05 waited for this job to finish; the next jobs (vLLM) started at 07:00. Light Run Command calls from other agents during the hour were not logged.

## Method

`bench/llama_bench.sh`: `llama-bench -p 512 -n 128 -t 8 -r 3 -o jsonl` per quant on `build` and `build-noamx`. Output correctness for one sequence verified in E04.

## Measurements

All rows are `llama-bench` on one sequence, 8 threads, 3 repetitions (mean ± sample standard deviation); `pp512` = 512 prompt tokens, `tg128` = 128 generated tokens. Raw data: `llama-bench.jsonl`. No row is marked invalid, but note that output correctness was verified only for Q4_K_M (E04); the other quants' outputs were not checked.

| quant | file GB | pp512 AMX (tok/s) | pp512 no-AMX (tok/s) | AMX/no-AMX | tg128 AMX (tok/s) | tg128 no-AMX (tok/s) | AMX/no-AMX |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_0 | 16.3 | 21.1 ± 0.02 | 15.8 ± 0.10 | 1.34x | 5.29 ± 0.05 | 3.55 ± 0.01 | 1.49x |
| IQ4_XS | 15.5 | 29.7 ± 0.03 | 31.8 ± 0.11 | 0.93x | 3.97 ± 0.03 | 2.05 ± 0.00 | 1.94x |
| Q4_K_M | 17.4 | 28.7 ± 0.17 | 19.4 ± 0.28 | 1.48x | 3.88 ± 0.01 | 3.60 ± 0.03 | 1.08x |
| Q5_K_M | 20.9 | 26.1 ± 0.21 | 23.8 ± 0.35 | 1.09x | 2.90 ± 0.03 | 2.07 ± 0.01 | 1.40x |
| Q6_K | 23.8 | 18.3 ± 0.07 | 22.1 ± 0.28 | 0.83x | 2.81 ± 0.04 | 2.42 ± 0.08 | 1.16x |
| Q8_0 | 29.1 | 22.3 ± 0.02 | 12.7 ± 0.88 | 1.76x | 2.78 ± 0.11 | 2.09 ± 0.01 | 1.33x |
| bf16 | 54.6 | 14.6 ± 0.05 | 13.4 ± 0.00 | 1.09x | 1.36 ± 0.01 | 1.38 ± 0.00 | 0.99x |

"AMX/no-AMX" is `build` over `build-noamx`, and **`build-noamx` is not a clean control**: it differs from `build` in more than AMX (explicit AVX-512 flags instead of `-march=native`, E08 has the clean control). The weight-buffer lines in the verbose logs show that the two builds also pick different weight layouts.

Physical limits (assumptions: decode bytes per token = GGUF file size; prefill FLOP per token = 2 × 27.32 G parameters = 54.6 GFLOP, an upper bound because it includes the embedding table and the unused MTP block; AMX INT8 peak 59 TOPS = 8 cores × 3.6 GHz × 2048 ops/cycle):

| quant | GFLOP/token | decode GB/s AMX (tg x file size) | decode GB/s no-AMX | prefill TFLOPS AMX | prefill TFLOPS no-AMX | AMX build, % of 59 TOPS INT8 peak |
|---|---:|---:|---:|---:|---:|---:|
| Q4_0 | 54.6 | 86 | 58 | 1.15 | 0.86 | 2.0% |
| IQ4_XS | 54.6 | 61 | 32 | 1.62 | 1.74 | 2.8% |
| Q4_K_M | 54.6 | 68 | 63 | 1.57 | 1.06 | 2.7% |
| Q5_K_M | 54.6 | 61 | 43 | 1.42 | 1.30 | 2.4% |
| Q6_K | 54.6 | 67 | 58 | 1.00 | 1.21 | 1.7% |
| Q8_0 | 54.6 | 81 | 61 | 1.22 | 0.69 | 2.1% |
| bf16 | 54.6 | 75 | 75 | 0.80 | 0.73 | n/a (no AMX path) |

Weight buffers that each build used (from the verbose logs; `model-buffers.txt`). In `build`, the AMX buffer holds the large 2-D weight matrices repacked for AMX; the remaining ~450 tensors (the log names `token_embd.weight` and 449 others, presumably norms, SSM parameters and embeddings) stay in the regular CPU buffer. In `build-noamx` the `CPU_REPACK` buffer exists only for Q4_0 (q4_0_8x8), Q4_K_M and, marginally, IQ4_XS; Q5_K_M, Q6_K, Q8_0 and BF16 run from the unrepacked mmap'd weights:

| quant | GGUF file MiB | AMX buffer MiB (build) | CPU_REPACK buffer MiB (build-noamx) |
|---|---:|---:|---:|
| Q4_0 | 15581 | 14661 | 11992 |
| IQ4_XS | 14749 | 13980 | 101 |
| Q4_K_M | 16624 | 16057 | 10238 |
| Q5_K_M | 19944 | 19666 | - |
| Q6_K | 22745 | 22143 | - |
| Q8_0 | 27757 | 28990 | - |
| bf16 | 52115 | - | - |

## Cost per token

Single-sequence figures. Formulas from [COST_MODEL.md](../COST_MODEL.md): `$/M input = P / (pp512 tok/s × 3600) × 1e6`, `$/M output = P / (tg128 tok/s × 3600) × 1e6`, blended for the reference request (512 input + 128 output) = `P × (512/pp + 128/tg) / 3600 / 640 × 1e6`.

| quant | build | input USD/M on-demand | output USD/M on-demand | input USD/M Spot | output USD/M Spot | blended 512+128 USD/M on-demand | blended USD/M Spot | s per request |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_0 | AMX | 22.2 | 88.2 | 4.3 | 17.1 | 35.4 | 6.8 | 48 |
| Q4_0 | no-AMX | 29.6 | 131.7 | 5.7 | 25.5 | 50.0 | 9.7 | 69 |
| IQ4_XS | AMX | 15.7 | 117.7 | 3.0 | 22.8 | 36.1 | 7.0 | 49 |
| IQ4_XS | no-AMX | 14.7 | 228.2 | 2.8 | 44.1 | 57.4 | 11.1 | 79 |
| Q4_K_M | AMX | 16.3 | 120.3 | 3.1 | 23.3 | 37.1 | 7.2 | 51 |
| Q4_K_M | no-AMX | 24.0 | 129.8 | 4.6 | 25.1 | 45.2 | 8.7 | 62 |
| Q5_K_M | AMX | 17.9 | 160.8 | 3.5 | 31.1 | 46.5 | 9.0 | 64 |
| Q5_K_M | no-AMX | 19.6 | 225.6 | 3.8 | 43.6 | 60.8 | 11.8 | 83 |
| Q6_K | AMX | 25.5 | 166.2 | 4.9 | 32.1 | 53.7 | 10.4 | 74 |
| Q6_K | no-AMX | 21.2 | 193.2 | 4.1 | 37.4 | 55.6 | 10.7 | 76 |
| Q8_0 | AMX | 20.9 | 167.8 | 4.0 | 32.4 | 50.3 | 9.7 | 69 |
| Q8_0 | no-AMX | 36.7 | 223.4 | 7.1 | 43.2 | 74.1 | 14.3 | 101 |
| bf16 | AMX | 32.0 | 342.2 | 6.2 | 66.2 | 94.0 | 18.2 | 129 |
| bf16 | no-AMX | 34.9 | 339.3 | 6.7 | 65.6 | 95.7 | 18.5 | 131 |

Assumptions:

1. All-in hourly price = VM + $0.018 disk and IP: $1.663 + 0.018 = $1.681/h pay-as-you-go (what we paid); $0.3073 + 0.018 = $0.325/h Spot (not available on this subscription, shown for comparison).
2. 100% utilization, steady state; VM boot, model download and model load excluded (each `llama-bench` run loads and repacks the model first, about a minute for the large files, not counted).
3. One sequence at a time (no batching): E05 shows what batching changes.
4. Prefill rate = `pp512` (512 tokens on an empty context); decode rate = `tg128` (context grows from 0 to 128). The reference request is 512 + 128 tokens, so its decode runs at a longer context than measured. Only the 16 full-attention layers see the context, and the effect was not measured.
5. Output correctness of the AMX build verified for Q4_K_M only (E04). The no-AMX build is E04's verified-correct reference for that quant.
6. List prices, no reservations; Spot evictions not modeled.

## Analysis

### Orchestrator review and conclusions

- **The reporter's verdicts stand.** H1 is refuted against `build-noamx` (prefill ×0.83–1.76) but holds against the clean control
  for Q4_0, Q4_K_M and Q8_0 (E08: ×1.32–1.68, E05: ×1.9–2.6 on 128-token prompts). H2 is refuted (decode is not one
  bandwidth constant; AMX helps decode). H3 is supported.
- **In llama.cpp, "AMX" is as much a kernel library as an instruction set.** Most of the decode gain (single-row
  matvec) can't come from tiles; it comes from the AMX backend's weight layout and its AVX-512-VNNI single-row
  kernels, which are better than the CPU_REPACK kernels the no-AMX builds use. The practical meaning is the same (the
  native AMX build is the fastest single-sequence configuration), but other runtimes' AMX claims shouldn't be
  extrapolated from this.
- **Prefill efficiency is the problem.** All quants prefill at 18–36 tok/s ≈ 1–2 TFLOPS, 2–3% of the AMX INT8 peak.
  At one sequence, output tokens cost 4–15× as much as input tokens; with batching (E05) prefill becomes ~75% of
  the busy time. Anything that raises prefill efficiency several-fold (E09 vLLM, E10 OpenVINO) dominates the cost picture.
- **A3 (prefill depends on sequence length) matters for the cost model**: one 512-token prompt prefills at 21 tok/s on
  the AMX build, while four 128-token prompts in the same micro-batch reach 35 tok/s. E07's profile of the AMX
  build and a prompt-length sweep are the follow-up.
- **Cheapest single-stream configuration**: Q4_0 or Q4_K_M with AMX, $35–37/M tokens blended on-demand ($6.8–7.2 Spot).
  Q4_K_M has half the KL divergence of Q4_0 (E06: 0.013 vs 0.025) for 5% more cost, so it's the better default.

### Reporter's analysis (reviewed)

**H1 (prefill gains 2–4× from AMX): refuted as measured.** The AMX/no-AMX prefill ratios at `pp512` are Q4_0 1.34×, IQ4_XS 0.93×, Q4_K_M 1.48×, Q5_K_M 1.09×, Q6_K 0.83×, Q8_0 1.76×. No quant reaches 2×, and IQ4_XS and Q6_K are slower with AMX (anomaly A1). The best AMX prefill is 29.7 tok/s (IQ4_XS) at only 1.6 TFLOPS, and every quant on the AMX build sits at 1.7–2.8% of the 59 TOPS INT8 peak, so prefill is nowhere near AMX-compute-bound. Two things limit what this ratio can tell us:

- The no-AMX baseline varies by 2.5× across quants (12.7 to 31.8 tok/s) for reasons unrelated to AMX: Q4_0 and Q4_K_M get the repacked CPU kernels, Q8_0 does not (it is the slowest baseline at 12.7 ± 0.9 tok/s, which inflates its ratio to 1.76×), IQ4_XS and Q6_K run generic kernels that beat the AMX path. The AMX build's own prefill sits in a narrow 18–30 tok/s band across the six quants.
- E05 measures AMX against the clean control (`build-native-noamx`) at 128-token prompts and finds 1.9× (Q4_0) and 2.3× (Q8_0) for one sequence, and 1.9–2.6× for 4+ sequences. So H1's range is reached for Q4_0 and Q8_0 with a clean control and shorter prompts, but those shapes differ from `pp512` (see anomaly A3), so it can't be decided here which part of the gap is control and which is prompt shape. Verdict: refuted for `pp512` against `build-noamx`, open against a clean control (E08, E05).

A hypothesis for why IQ4_XS and Q6_K lose with AMX (not tested): the AMX kernels for 6-bit and codebook-coded weights must unpack them to int8 tiles on the fly, and that unpacking costs more than the tile math saves, while the generic AVX-512-VNNI path is already efficient for them. The log confirms AMX kernels exist for both types (symbols `tinygemm_kernel_amx<block_q8_K, block_q6_K>` and `<..., block_iq4_xs>`).

**H2 (decode is memory-bound at ~80–100 GB/s and gains little from AMX): refuted as stated, partly supported at the top end.** `tg × file size` ranges from 32 to 86 GB/s, not a constant. The values near the ceiling are BF16 (75 GB/s on both builds, a plain streaming kernel and so the cleanest bandwidth probe) and the AMX build's Q4_0 (86 GB/s) and Q8_0 (81 GB/s, or about 85 GB/s counting the 4.4% larger repacked AMX buffer). So the VM sustains at least ~86 GB/s with 8 threads (a floor, not a STREAM measurement), and most other combinations leave roughly 20–60% of it unused: IQ4_XS without AMX gets only 32 GB/s, Q5_K_M 43 GB/s. Those cases are limited by dequantization compute in the matvec kernel, not by DRAM. AMX does not "gain little" for decode either: the AMX build decodes 1.49× (Q4_0), 1.94× (IQ4_XS), 1.40× (Q5_K_M), 1.33× (Q8_0) faster, but only 1.08× (Q4_K_M), 1.16× (Q6_K) and 0.99× (BF16). Single-token decode cannot fill AMX tiles, so this gain most likely comes from the AMX path's repacked weight layout and its AVX-512-VNNI single-row kernel (a `tinygemm_kernel_vnni<..., 1, ...>` single-row kernel is present in the library, see the E05 crash trace), not from tile instructions, and part of it may come from the compiler flags (E08). Hypothesis for the ~86 GB/s ceiling (not tested): per-core memory-level parallelism with 8 threads on 8 cores, not DRAM peak; E08's 16-thread runs speak to this.

**H3 (BF16 slowest at both phases): supported.** BF16 decodes slowest by far (1.36 tok/s against 2.78 for the next, Q8_0, as expected from 54.6 GB of weights) and decodes identically on both builds (1.36 vs 1.38 tok/s), and the verbose log shows no AMX weight buffer for BF16, confirming there is no AMX path. At prefill BF16 is the slowest on `build` (14.6 tok/s; next is Q6_K at 18.3) and on `build-noamx` it is tied with Q8_0 (13.4 vs 12.7 ± 0.9 tok/s). The 9% prefill difference between the two BF16 builds (14.6 vs 13.4, both with sd under 0.1) can't come from AMX since neither uses it, so it is a build difference (flags); the cause was not checked.

**Compute check.** Prefill reaches 0.7–1.7 TFLOPS. The AVX-512 VNNI peak is about 7.4 TOPS (assuming two 512-bit VNNI ports per core, 8 cores at 3.6 GHz), so the best no-AMX case (IQ4_XS, 1.74 TFLOPS) is ~23% of that peak, while the best AMX case is ~3% of the AMX peak. The AMX prefill is not limited by the AMX units; the profile in E07 should say where the time goes.

**Anomalies**

- A1: IQ4_XS and Q6_K prefill is faster without AMX (31.8 vs 29.7 and 22.1 vs 18.3 tok/s, differences of 7% and 21%, with sd under 2%); see the hypothesis above.
- A2: decode differs between builds by up to 1.94× (IQ4_XS), although single-sequence decode should be bandwidth-bound. E03's `build-noamx` also differs in compiler flags. In E05 the clean control `build-native-noamx` decodes Q4_0 at 2.44 tok/s with one sequence, 31% below `build-noamx`'s 3.55 tok/s here (Q4_K_M 3.54 vs 3.60, Q8_0 1.97 vs 2.09), so the two no-AMX builds themselves differ for Q4_0, and the AMX effect on Q4_0 decode is 1.49× against `build-noamx` but 2.1× against `build-native-noamx` (using E05's 5.09 tok/s for the AMX build). E08 (other VM) is the clean comparison; E05's numbers are on this VM.
- A3: AMX `pp512` (single sequence) is much slower than the same build on shorter or shared prompts in E05: Q4_0 21.1 tok/s here vs 33.3 tok/s for one 128-token prompt and 35.2 tok/s for four 128-token prompts in one 512-token micro-batch (Q8_0 22.3 vs 33.8 and 38.6). The 4 × 128 case has the same micro-batch size (`n_ubatch` 512) as `pp512`, so the gap comes from sequence length or sequence count (attention span, Gated DeltaNet scan), not from matrix shape. This matters for E07 and for any prompt-length assumption in the cost model.
- A4: Q8_0 `build-noamx` `pp512` is noisy (12.7 ± 0.9 tok/s, samples 12.5, 12.0, 13.7), and Q8_0 AMX decode has 4% sd; treat those two ratios with caution.

**Cost reading.** At one sequence the cheapest blended cost for 512 + 128 tokens is Q4_0 with AMX: $35.4/M tokens on-demand ($6.8/M Spot), 48 s per request. Output tokens cost 4× (Q4_0 AMX, $88 vs $22 per million on-demand) to 15× (IQ4_XS without AMX) as much as input tokens, and BF16 is the most expensive at $94/M. Q4_K_M with AMX is within 5% of Q4_0 ($37.1/M); E06 gives the quality comparison. See E05 for the effect of batching: the same hardware reaches $23.9/M with 32 sequences.

## Threats to validity

- `build-noamx` is not a clean control (A2); read AMX/no-AMX ratios as "this build vs that build".
- One VM instance, one run per configuration (3 repetitions inside `llama-bench`, run back to back, so they don't average out instance or time-of-day effects). E08 measured 15–25% differences between two E16ds_v7 instances.
- Output correctness was verified only for Q4_K_M single-sequence (E04). The other quants' AMX single-sequence outputs weren't compared; E04 flagged Q4_0 and Q8_0 as unchecked.
- Empty-context prompts of exactly 512 tokens and 128 generated tokens; the reference workload's decode runs at a longer context. A3 shows prompt length matters on the AMX build.
- GGUF `model_size` includes the unused MTP block (~1.5%), and AMX repacked buffers differ from file sizes (Q8_0 +4.4%), so the GB/s figures carry a few percent uncertainty.
- The AMX and FLOP peaks are theoretical, derived from 8 cores × 3.6 GHz as given in the brief; actual all-core frequency under AMX load was not measured.

## Next steps

- E08 (clean control, threads/SMT) decides A2 and H1; compare its `build-native-noamx` single-sequence numbers with the Q4_0 gap seen here.
- E07 (prefill profile) should include a prompt-length sweep on the AMX build (128/256/512/1024, one sequence versus several) to explain A3, and the IQ4_XS / Q6_K AMX kernel cost (A1).
- Run a STREAM-like bandwidth probe with 8 and 16 threads on this VM to turn the ~86 GB/s floor into a measured ceiling.
- Verify AMX single-sequence output for the quants that E03 does not cover (at least Q4_0 and Q8_0) before using their cost figures.
