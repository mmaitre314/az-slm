# E06: Quantization quality: KL divergence vs BF16

| | |
| --- | --- |
| Status | done (measured 2026-10-04 ~03:15–03:53 UTC; reported and reviewed 2026-10-04) |
| VM | `bench-lc2` (Standard_E16ds_v7, northcentralus, Regular) |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-noamx` = explicit AVX-512 flag list without AMX; `build-native-noamx` = -march=native minus AMX (clean control). This experiment ran on `build-native-noamx`. |
| Model | `bartowski/Qwen3.8-27B-GGUF` @ `0c92138c51` (Q4_0, IQ4_XS, Q4_K_M, Q5_K_M, Q6_K, Q8_0 vs BF16) |
| Dates | 2026-10-04 03:32:46-03:53:56 UTC (BF16 reference 03:32-03:37, then 6 quants of 2-4 min each) |
| Raw data | `kld-stats.jsonl` (every statistic of each quant's `llama-perplexity` log, parsed on the VM), `quality.jsonl` (the 6-field summary written by `bench/quality.sh`), `kld-bf16.txt` (BF16 run: tokens/chunks and 'Final estimate'), `tables.py` (regenerates every table below, including the cost pairing from E03/E05/E08 raw data). The 1.0 GB BF16 logits file and the verbose logs stay on the VM. |

## Question

How much output quality does each quantization lose, so throughput gains can be weighed against quality?

## Hypotheses

_Written before the results were known._

- H1: Typical llama.cpp ordering holds: mean KLD ≈ Q8_0 <0.002, Q6_K ~0.005, Q5_K_M ~0.01, Q4_K_M ~0.02–0.03, IQ4_XS ~0.03, Q4_0 ~0.05.
- H2: Top-1 token agreement ≥98% for Q8_0 and ~93–95% for 4-bit quants.

## Setup

- VM `bench-lc2` (8 cores / 16 vCPU Granite Rapids, 128 GiB), 8 threads, `build-native-noamx` (the `system_info` line of every log lists no AMX feature). `llama-perplexity` evaluates 4 sequences per batch (`n_seq=4`, `batch_size=2048`), which is why the AMX build was excluded (E04: corrupt output with several sequences).
- Data: wikitext-2 raw test split (`ggml-org/ci` copy), `-c 512 --chunks 8`. `llama-perplexity` scores the second half of each window, so each run scores 255 tokens per chunk, **2,040 tokens in total** (not 4,096). This matches the size of the saved BF16 logits file (1,013,178,324 bytes = 2,040 × (2 B × 248,320 vocabulary entries + 8 B) + 16,404 B of header and token ids; the vocabulary size is my assumption). The 8 windows are the first 4,096 tokens of the test file, so the sample is narrow. `bench/quality.sh` defaults to 16 chunks; the run used `CHUNKS=8` to save time.
- BF16 reference: the BF16 GGUF (shards) run once with `--kl-divergence-base`; then each quant with `--kl-divergence` against it.
- Reported errors are the standard errors printed by `llama-perplexity`, which treat tokens as independent; tokens within a window are correlated, so the true uncertainty is larger.
- The quant files are bartowski's; their type mix is not uniform (E03's verbose log shows `q4_0`, `q4_1`, `q8_0` and `q6_K` tensors in the Q4_0 file), so "Q4_0" here is 4.78 bits per weight, not 4.5. Whether they are imatrix quants, and on which calibration text, was not checked.
- No other benchmark job was recorded on the VM during this experiment (job chain `lc2`: E08, E07, then E06, 02:49-03:54 UTC; light Run Command calls from other agents were not logged).

## Method

`bench/quality.sh` with `CHUNKS=8` (wikitext-2 test, 8 × 512-token chunks): BF16 logits saved with `--kl-divergence-base`, then `llama-perplexity --kl-divergence` per quant, on `build-native-noamx` (perplexity batches several sequences; AMX output is corrupt then, E04).

## Measurements

### KL divergence and token agreement

| quant | mean KLD (nats) | median KLD (nats) | 99% KLD (nats) | 99.9% KLD (nats) | max KLD (nats) | same top-1 token (%) | RMS Δp (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 | 0.00064 ± 0.00003 | 0.00031 | 0.0057 | 0.015 | 0.044 | 98.14 ± 0.30 | 0.68 ± 0.04 |
| Q6_K | 0.00267 ± 0.00010 | 0.00143 | 0.0215 | 0.050 | 0.059 | 95.83 ± 0.44 | 1.36 ± 0.05 |
| Q5_K_M | 0.00494 ± 0.00027 | 0.00232 | 0.0470 | 0.103 | 0.376 | 95.64 ± 0.45 | 1.99 ± 0.23 |
| Q4_K_M | 0.01348 ± 0.00064 | 0.00531 | 0.1371 | 0.384 | 0.418 | 93.63 ± 0.54 | 3.00 ± 0.17 |
| IQ4_XS | 0.01874 ± 0.00096 | 0.00756 | 0.2104 | 0.517 | 0.780 | 92.84 ± 0.57 | 4.17 ± 0.34 |
| Q4_0 | 0.02522 ± 0.00130 | 0.01137 | 0.2120 | 0.758 | 1.473 | 92.06 ± 0.60 | 4.13 ± 0.23 |

KLD is in nats per token, averaged over the 2,040 scored positions; "same top-1" is the share of positions where the quant's most likely token is BF16's; RMS Δp is the root-mean-square change in the probability of the correct next token. No row is marked invalid. `llama-perplexity` also prints BF16's 'Final estimate' PPL = 6.6159 ± 0.3600; its value computed from the saved logits is 6.6104 (a 0.08% difference between two estimates of the same quantity; not investigated, possibly the compressed storage of the saved logits).

### Perplexity (a weaker signal on this sample)

| quant | PPL | PPL / PPL(BF16) | ln(PPL ratio) | correlation of ln PPL with BF16 (%) | s per pass (4 windows = 2048 tokens) |
|---|---:|---:|---:|---:|---:|
| Q8_0 | 6.6121 ± 0.3596 | 1.0003 ± 0.0011 | 0.0003 ± 0.0011 | 99.98 | 122 |
| Q6_K | 6.6256 ± 0.3607 | 1.0023 ± 0.0019 | 0.0023 ± 0.0019 | 99.94 | 68 |
| Q5_K_M | 6.6479 ± 0.3615 | 1.0057 ± 0.0028 | 0.0057 ± 0.0028 | 99.87 | 63 |
| Q4_K_M | 6.6127 ± 0.3575 | 1.0004 ± 0.0044 | 0.0004 ± 0.0044 | 99.67 | 77 |
| IQ4_XS | 6.6943 ± 0.3649 | 1.0127 ± 0.0048 | 0.0126 ± 0.0048 | 99.62 | 45 |
| Q4_0 | 6.7503 ± 0.3690 | 1.0212 ± 0.0053 | 0.0209 ± 0.0052 | 99.55 | 100 |

Perplexity differences are smaller than the 0.36 standard error of PPL itself; the ratio's standard error (0.1-0.5%) is the usable one. The PPL ordering is not monotonic with quality (Q4_K_M, ratio 1.0004 ± 0.0044, comes out "better" than Q5_K_M and Q6_K), unlike the KLD ordering.

### Hypothesis check

| quant | H1 mean KLD predicted | measured | H2 same top-1 predicted (%) | measured (%) |
|---|---:|---:|---:|---:|
| Q8_0 | < 0.002 | 0.0006 | >= 98 | 98.1 ± 0.3 |
| Q6_K | ~0.005 | 0.0027 | - | 95.8 ± 0.4 |
| Q5_K_M | ~0.01 | 0.0049 | - | 95.6 ± 0.5 |
| Q4_K_M | 0.02-0.03 | 0.0135 | 93-95 | 93.6 ± 0.5 |
| IQ4_XS | ~0.03 | 0.0187 | 93-95 | 92.8 ± 0.6 |
| Q4_0 | ~0.05 | 0.0252 | 93-95 | 92.1 ± 0.6 |

## Cost per token

E06 measures quality, not throughput, so there is no cost of its own to report beyond the experiment's price: the six quants plus the BF16 reference took 21 minutes of VM time, about $0.59 on-demand at $1.681/h all-in ($0.11 at the $0.325/h Spot price). The table pairs each quant's quality with the cost per token from the experiments that measured throughput. Formulas as in [COST_MODEL.md](../COST_MODEL.md): blended USD/M tokens for the reference request (512 input + 128 output) = `P × (512/pp + 128/tg) / 3600 / 640 × 1e6`, with `pp` the prefill and `tg` the decode rate.

| quant | file GB | bits per weight | mean KLD (nats) | same top-1 (%) | 1 seq, AMX, bench-e16v7 (E03), on-demand | 1 seq, AMX, bench-e16v7 (E03), Spot | 1 seq, AMX, bench-lc2 (E08), on-demand | 32 seq, no-AMX, bench-e16v7 (E05), on-demand | 32 seq, no-AMX, bench-e16v7 (E05), Spot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 | 29.1 | 8.52 | 0.0006 | 98.1 | 50.3 | 9.7 | 40.9 | 33.5 | 6.5 |
| Q6_K | 23.8 | 6.98 | 0.0027 | 95.8 | 53.7 | 10.4 | - | - | - |
| Q5_K_M | 20.9 | 6.12 | 0.0049 | 95.6 | 46.5 | 9.0 | - | - | - |
| Q4_K_M | 17.4 | 5.10 | 0.0135 | 93.6 | 37.1 | 7.2 | 31.6 | 23.9 | 4.6 |
| IQ4_XS | 15.5 | 4.53 | 0.0187 | 92.8 | 36.1 | 7.0 | 30.9 | - | - |
| Q4_0 | 16.3 | 4.78 | 0.0252 | 92.1 | 35.4 | 6.8 | 30.8 | 27.8 | 5.4 |

Where each cost column comes from (columns are not directly comparable, because the VMs and builds differ):

- "1 seq, AMX, bench-e16v7 (E03)": `llama-bench` pp512/tg128, one sequence, AMX build, 8 threads, on `bench-e16v7`; all six quants; correct output verified only for Q4_K_M (E04).
- "1 seq, AMX, bench-lc2 (E08)": same but on `bench-lc2`, which runs 9-23% faster than `bench-e16v7`; four quants.
- "32 seq, no-AMX, bench-e16v7 (E05)": `llama-batched-bench` with 32 independent sequences on the clean control build (valid output), prefill rate measured on 128-token prompts and applied to 512 (E05's assumption), aggregate decode rate; only Q4_0, Q4_K_M and Q8_0 were measured.

Assumptions: all-in $1.681/h on-demand (VM $1.663 + $0.018 disk and IP) and $0.325/h Spot (not available on this subscription); 100% utilization; model load, boot and setup excluded; list prices; throughput as measured in the referenced experiments (see their threats to validity). Cost is per token for a given file; it ignores that bigger quants need more RAM (not binding on a 128 GiB VM).

## Analysis

### Orchestrator review and conclusions

- **Verdicts accepted.** H1's ordering is supported and its magnitudes are better than predicted (bartowski's imatrix files keep
  sensitive tensors at higher precision). H2 is mostly supported.
- **Choice of quantization for llama.cpp**: **Q4_K_M** is the default. It has half the KL divergence of Q4_0 and
  costs the same or less (E03/E05/E08). IQ4_XS is the size-optimal 4-bit choice but untested for batching. **Q8_0** is the quality option
  (KLD 0.0006, 98% top-1 agreement) at +30–40% cost. Q6_K is dominated by Q8_0 on single-sequence AMX throughput.
- **Limits of this measurement**: 2,040 scored tokens of wikitext-2. KLD on prose measures closeness to BF16, not
  task accuracy. For the batch use case (extraction/classification), the decision needs a task-level check
  (agreement of structured outputs with BF16 on a representative prompt set). That check also has to cover the OpenVINO
  INT4/INT8 and vLLM W8A8/W4A16 formats, which use different quantizers and can't be compared through llama.cpp's KLD tool.
  This is follow-up **E16** in PLAN.md.

### Reporter's analysis (reviewed)

**H1 (typical ordering and magnitudes): ordering supported, magnitudes refuted for five of six (every quant is better than predicted).** The mean KLD rises monotonically from Q8_0 (0.00064) to Q6_K (0.0027), Q5_K_M (0.0049), Q4_K_M (0.0135), IQ4_XS (0.0187) and Q4_0 (0.0252); each adjacent step is at least 4 combined standard errors (smallest: IQ4_XS to Q4_0, 0.0065 against 0.0016). Against the predicted values, Q8_0 meets "<0.002", and the others come out at 0.45-0.67 of the prediction: Q6_K 0.0027 (~0.005), Q5_K_M 0.0049 (~0.01), Q4_K_M 0.0135 (0.02-0.03), IQ4_XS 0.0187 (~0.03), Q4_0 0.0252 (~0.05). Candidate reasons, none tested: these files keep sensitive tensors in higher precision (4.78 bits per weight for "Q4_0", 5.10 for Q4_K_M) and may be imatrix-calibrated; wikitext-2 may overlap with the calibration text (which would flatter the quants); 2,040 tokens from the first part of the test file are not representative of the whole corpus.

**H2 (top-1 agreement ≥98% for Q8_0, ~93-95% for 4-bit quants): mostly supported.** Q8_0 98.14 ± 0.30% (meets the bound at the point estimate), Q4_K_M 93.63 ± 0.54% (inside the range), IQ4_XS 92.84 ± 0.57% (0.2 points below the range, within one standard error) and Q4_0 92.06 ± 0.60% (0.9 points below, 1.6 standard errors). The agreement metric saturates: Q6_K (95.83%) and Q5_K_M (95.64%) are statistically indistinguishable from each other, and the whole range from Q6_K down to Q4_0 spans 3.8 points, while the mean KLD spans 9x. Top-1 agreement is a coarse metric (near-ties flip the top token); KLD and the tail percentiles discriminate better. The tails differ: IQ4_XS and Q4_0 share a 99th-percentile KLD of 0.21, but Q4_0's 99.9th percentile is 0.76 against 0.52 and its maximum 1.47 against 0.78.

**What the numbers say about the choice of quantization (before throughput is considered).**

- Q4_0 has the worst quality of the six. By size and quality alone it is dominated by IQ4_XS (15.5 GB, 5% smaller than Q4_0's 16.3 GB, with 0.74x the KLD); against Q4_K_M (17.4 GB) it saves 6% of the file at 1.87x the KLD.
- Q4_K_M versus Q4_0 on cost: +5% at one sequence on the AMX build (E03: $37.1 vs $35.4/M; E08: $31.6 vs $30.8/M) and 14% cheaper at 32 sequences on the control build (E05: $23.9 vs $27.8/M), so at equal or lower cost Q4_K_M has about half the KLD.
- Q8_0 costs 29-40% more than Q4_K_M per token in all three cost columns and has 21x lower KLD (0.0006 vs 0.0135) and 4.5 points higher top-1 agreement. Whether that quality difference matters for an extraction or classification batch job is not measured; KLD on prose is not task accuracy.
- Q5_K_M (+25% cost over Q4_K_M at one sequence, KLD 2.7x lower) and Q6_K (KLD 5.0x lower than Q4_K_M) sit between. Q6_K is dominated by Q8_0 on the single-sequence AMX numbers ($53.7 vs $50.3/M with 4.2x the KLD), because Q6_K has the slowest AMX prefill of the quantized files (E03, 18.3 tok/s); no batched Q5_K_M or Q6_K numbers exist (E05 skipped them), so that comparison is open for batch workloads.
- IQ4_XS is the smallest file with a mid-field KLD (0.0187), and its AMX prefill is slow relative to its no-AMX prefill (E03, E08); it has no batched measurement either.

**Anomalies**

- A1: PPL(Q4_K_M) is as close to BF16 as PPL(Q8_0) (ratios 1.0004 and 1.0003) although its mean KLD is 21x higher; with 2,040 tokens PPL is dominated by sampling noise, and only the paired KL statistics rank the quants correctly. Do not use the PPL columns to choose a quant.
- A2: the BF16 'Final estimate' (6.6159) and the BF16 PPL computed from the saved logits (6.6104) differ by 0.08%.
- A3: evaluation time per pass (one batch of 4 windows, 2,048 tokens) does not follow file size (Q8_0 122 s, Q4_0 100 s, Q4_K_M 77 s, Q6_K 68 s, Q5_K_M 63 s, IQ4_XS 45 s). It tracks the control build's prefill speed (Q8_0 2,048 tokens / 122 s = 16.8 tok/s against E08's pp512 16.3; Q4_0 20.5 against 19.4; IQ4_XS 45.9 against 40.3), a consistency check on the run.

## Threats to validity

- Small, narrow sample: 2,040 tokens from the first four thousand tokens of wikitext-2, English encyclopedic prose, 512-token windows, one run. Standard errors assume independent tokens. The chosen workload (extraction/classification batches with longer prompts) may behave differently, and quantization damage differs between tasks and across languages.
- Calibration overlap: if bartowski's imatrix text overlaps wikitext-2, the KLD of the imatrix-calibrated files is optimistic (not checked).
- The reference is BF16 computed on a CPU (`AVX512_BF16` path) and the logits are stored compressed (uint16), so the reference carries its own numerical error, small against the 0.0006 KLD of Q8_0 but not zero (A2).
- Numerics are those of `build-native-noamx`; the AMX build uses different kernels for the same weights and its quality was not measured (single-sequence correctness is only verified for Q4_K_M in E04).
- Cost pairing mixes VM instances (+9-23% for `bench-lc2`), builds (AMX single sequence vs control with 32 sequences) and prompt lengths (E05's prefill rate is measured at 128-token prompts), so the cost columns rank quants within a column, not across columns.
- Only llama.cpp GGUF quants were measured; vLLM and OpenVINO INT8/INT4 (E09-E11) are on a different quality scale until their KLD against the same BF16 reference is measured.

## Next steps

- Add a larger and more relevant evaluation for the finalists (Q4_K_M, Q5_K_M, Q8_0): 16+ chunks (the script default) on several text types, including instruction/extraction-style data, plus a small task-level accuracy check, to find out whether KLD 0.0135 vs 0.0006 changes results for the intended batch job.
- Measure KLD for the other stacks' quantizations (vLLM W8A8/W4A16, OpenVINO INT4/INT8) against this BF16 reference so E15 can compare cost on a common quality axis.
- Measure batched throughput (control build, 32 sequences, E05's method) for Q5_K_M, Q6_K and IQ4_XS so every row in the quality-against-cost table has a valid batched cost.
- Check which bartowski files are imatrix quants and on what data, and add a quant calibrated on non-wikitext text if the overlap matters.
