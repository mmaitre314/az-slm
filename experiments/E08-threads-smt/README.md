# E08: Clean AMX control build; threads, SMT and pinning

| | |
| --- | --- |
| Status | done (2026-10-04), awaiting report |
| VM | `bench-lc2` (Standard_E16ds_v7, northcentralus, Regular) |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-noamx` = explicit AVX-512 flag list without AMX; `build-native-noamx` = -march=native minus AMX (clean control) |
| Model | `bartowski/Qwen3.8-27B-GGUF` @ `0c92138c51` (Q4_0, IQ4_XS, Q4_K_M, Q8_0) |
| Raw data | files in this directory |

## Question

(a) E03's no-AMX build also differs in compiler flags. With a control build identical except for AMX, what does AMX alone contribute? (b) Do 16 threads (SMT) or pinning help?

## Hypotheses

_Written before the results were known._

- H1: With the clean control, the AMX effect on decode shrinks to ~0 (E03's decode differences come from other flags) and prefill gains remain at 1.3–2×.
- H2: 16 threads add ≤10% to prefill and may slow decode (two hyperthreads share one AMX unit and the same bandwidth); pinning adds a few percent.

## Method

`bench/build_native_noamx.sh` (−march=native −mno-amx-*; verified 0 AMX instructions), `bench/llama_bench.sh` for 4 quants on `build` vs `build-native-noamx`, `bench/llama_threads.sh` (8 threads, 8 pinned to one hyperthread per core, 16 threads) for Q4_K_M and Q4_0 on `build`.

## Measurements

_To be filled by the reporter._

## Cost per token

_To be filled (see [COST_MODEL.md](../COST_MODEL.md))._

## Analysis

## Threats to validity

## Next steps
