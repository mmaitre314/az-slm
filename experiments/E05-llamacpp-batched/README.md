# E05: llama.cpp batch throughput with 1–32 sequences

| | |
| --- | --- |
| Status | done (2026-10-04, finished 05:01 UTC), awaiting report |
| VM | `bench-e16v7` (Standard_E16ds_v7, eastus2, Regular) |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-noamx` = explicit AVX-512 flag list without AMX; `build-native-noamx` = -march=native minus AMX (clean control) |
| Model | `bartowski/Qwen3.8-27B-GGUF` @ `0c92138c51` (Q4_0, Q4_K_M, Q8_0) |
| Raw data | files in this directory |

## Question

For offline batch processing, how much does decoding several independent sequences together raise aggregate throughput, and where does it saturate?

## Hypotheses

_Written before the results were known._

- H1: Aggregate decode tok/s rises almost linearly with sequences while memory-bound (weights are read once per step for all sequences), then flattens when compute-bound, around 8–16 sequences, at 15–30 tok/s.
- H2: Per-sequence recurrent state of the 48 Gated DeltaNet layers adds memory traffic that grows with the number of sequences.
- H3: Prefill throughput doesn't change with batching (already compute-bound).

## Method

`bench/llama_batched.sh`: `llama-batched-bench -npp 128 -ntg 128 -npl 1,4,8,16,32 -t 8 -b 2048 -ub 512` on `build-native-noamx` (valid output, E04), plus `build` for Q4_0 and Q8_0 as a speed-only reference (its multi-sequence output is corrupt).

## Measurements

_To be filled by the reporter._

## Cost per token

_To be filled (see [COST_MODEL.md](../COST_MODEL.md))._

## Analysis

## Threats to validity

## Next steps
