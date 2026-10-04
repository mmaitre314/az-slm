# E07: Why is prefill slow? CPU profile

| | |
| --- | --- |
| Status | done (2026-10-04), awaiting report |
| VM | `bench-lc2` (Standard_E16ds_v7, northcentralus, Regular) |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-noamx` = explicit AVX-512 flag list without AMX; `build-native-noamx` = -march=native minus AMX (clean control); `perf` (cpu-clock sampling) |
| Model | `bartowski/Qwen3.8-27B-GGUF` @ `0c92138c51` (Q4_0) |
| Raw data | files in this directory |

## Question

Prefill reaches only 20–35 tok/s, about 1–2 TFLOPS, against tens of TFLOPS of AMX peak. Where does the time go?

## Hypotheses

_Written before the results were known._

- H1: Prefill time is dominated by non-GEMM work, especially the sequential Gated DeltaNet recurrence of the 48 linear-attention layers, not by matrix multiplications.
- H2 (alternative): the GEMMs dominate but run at low efficiency, because the AMX path isn't used for most shapes (llama.cpp's AMX gating conditions, hidden size 5120).

## Method

`bench/profile_prefill.sh`: `perf record -e cpu-clock -g` of `llama-bench -p 512 -n 0` for Q4_0 on `build` and `build-native-noamx`; top functions by self time.

## Measurements

_To be filled by the reporter._

## Cost per token

_To be filled (see [COST_MODEL.md](../COST_MODEL.md))._

## Analysis

## Threats to validity

## Next steps
