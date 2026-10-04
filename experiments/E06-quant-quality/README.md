# E06: Quantization quality: KL divergence vs BF16

| | |
| --- | --- |
| Status | done (2026-10-04, finished ~03:53 UTC), awaiting report |
| VM | `bench-lc2` (Standard_E16ds_v7, northcentralus, Regular) |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-noamx` = explicit AVX-512 flag list without AMX; `build-native-noamx` = -march=native minus AMX (clean control) |
| Model | `bartowski/Qwen3.8-27B-GGUF` @ `0c92138c51` (Q4_0, IQ4_XS, Q4_K_M, Q5_K_M, Q6_K, Q8_0 vs BF16) |
| Raw data | files in this directory |

## Question

How much output quality does each quantization lose, so throughput gains can be weighed against quality?

## Hypotheses

_Written before the results were known._

- H1: Typical llama.cpp ordering holds: mean KLD ≈ Q8_0 <0.002, Q6_K ~0.005, Q5_K_M ~0.01, Q4_K_M ~0.02–0.03, IQ4_XS ~0.03, Q4_0 ~0.05.
- H2: Top-1 token agreement ≥98% for Q8_0 and ~93–95% for 4-bit quants.

## Method

`bench/quality.sh` with `CHUNKS=8` (wikitext-2 test, 8 × 512-token chunks): BF16 logits saved with `--kl-divergence-base`, then `llama-perplexity --kl-divergence` per quant, on `build-native-noamx` (perplexity batches several sequences; AMX output is corrupt then, E04).

## Measurements

_To be filled by the reporter._

## Cost per token

_To be filled (see [COST_MODEL.md](../COST_MODEL.md))._

## Analysis

## Threats to validity

## Next steps
