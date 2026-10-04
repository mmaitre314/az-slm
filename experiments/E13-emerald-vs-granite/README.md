# E13: Emerald Rapids (E16ds_v6) vs Granite Rapids (E16ds_v7)

| | |
| --- | --- |
| Status | done (2026-10-04, finished 04:19 UTC), awaiting report |
| VM | `bench-v6` (Standard_E16ds_v6, westus2, Regular): Xeon Platinum 8573C; compare with `bench-e16v7` and `bench-lc2` |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-noamx` = explicit AVX-512 flag list without AMX; `build-native-noamx` = -march=native minus AMX (clean control) |
| Model | `bartowski/Qwen3.8-27B-GGUF` @ `0c92138c51` (Q4_0, IQ4_XS, Q4_K_M, Q8_0) |
| Raw data | files in this directory |

## Question

Is the newer, 27% more expensive Granite Rapids VM faster enough per token to be worth it for llama.cpp?

## Hypotheses

_Written before the results were known._

- H1: Decode scales with memory bandwidth (Granite Rapids: 12 channels of DDR5-6400 vs 8 of DDR5-5600 for Emerald Rapids), so v7 decodes 1.2–1.6× faster.
- H2: Prefill is ~1.2× faster on v7 (3.6 vs 3.0 GHz all-core); llama.cpp doesn't use AMX-FP16, so v7's extra ISA doesn't matter.
- H3: v6 is cheaper per token only if v7's speedup is below the 1.27× price ratio.

## Method

Same scripts and parameters as E03/E08 (`llama_bench.sh`, `build` vs `build-native-noamx`) and E05 (`llama_batched.sh`, `build-native-noamx`, npl 1–32).

## Measurements

_To be filled by the reporter._

## Cost per token

_To be filled (see [COST_MODEL.md](../COST_MODEL.md))._

## Analysis

## Threats to validity

## Next steps
