# E03: llama.cpp single-sequence prefill and decode per quantization, AMX vs no AMX

| | |
| --- | --- |
| Status | done (2026-10-04 02:23–03:25 UTC), awaiting report |
| VM | `bench-e16v7` (Standard_E16ds_v7, eastus2, Regular) |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-noamx` = explicit AVX-512 flag list without AMX; `build-native-noamx` = -march=native minus AMX (clean control) |
| Model | `bartowski/Qwen3.8-27B-GGUF` @ `0c92138c51` (Q4_0, IQ4_XS, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16) |
| Raw data | files in this directory |

## Question

How fast is one sequence at prefill (prompt processing) and decode (generation) for each quantization, and how much does AMX contribute?

## Hypotheses

_Written before the results were known._

- H1: Prefill is compute-bound and benefits from AMX-INT8 for the quant types with AMX kernels (Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS): 2–4× over AVX-512-VNNI.
- H2: Decode is memory-bandwidth-bound (decode tok/s × model bytes ≈ constant, ~80–100 GB/s) and gains little from AMX.
- H3: BF16 has no AMX path in llama.cpp, so it's the slowest at both phases.

## Method

`bench/llama_bench.sh`: `llama-bench -p 512 -n 128 -t 8 -r 3 -o jsonl` per quant on `build` and `build-noamx`. Output correctness for one sequence verified in E04.

## Measurements

_To be filled by the reporter._

## Cost per token

_To be filled (see [COST_MODEL.md](../COST_MODEL.md))._

## Analysis

## Threats to validity

## Next steps
