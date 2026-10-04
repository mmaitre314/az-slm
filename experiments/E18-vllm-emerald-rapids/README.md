# E18: vLLM on Emerald Rapids (E16ds_v6) vs Granite Rapids (E16ds_v7)

| | |
| --- | --- |
| Status | planned |
| VM | `b2-v6` (Standard_E16ds_v6, westus2, Regular): Xeon Platinum 8573C, 8 cores / 16 threads, 128 GiB |
| Stack | same vLLM image as E17 (record the digest; it must match E17's) |
| Model | `Avesed/Qwen3.8-27B-INT8-W8A8`, `Avesed/Qwen3.8-27B-INT4-W4A16`; Q4_K_M GGUF for the reference run |
| Dates | |
| Raw data | `raw/` |

## Question

E16ds_v6 costs 21% less per hour than E16ds_v7 ($1.326 vs $1.681 all-in), and for llama.cpp it
was cheaper per token (E13). Does that hold for vLLM, whose prefill leans on AMX compute
rather than memory bandwidth?

## Hypotheses

_Written before the results were known._

- **H1**: both CPUs have AMX-BF16/INT8 (v6 lacks only AMX-FP16, unused here) and similar clocks,
  so W8A8 prefill per core is within ±15% of v7. Decode is bound by the same ~100 GB/s per
  8-core slice (E13). So the v6 blended cost is 10–25% lower, in line with its price.
- **H2**: the difference between the two instances is within instance-to-instance variance
  (9–25%, E08) unless it exceeds ~20%. The llama.cpp reference run is used to tell instance
  effects from CPU-generation effects.

## Method

Self-contained chain `bench/e18_chain.sh` (a subset of E17's, same scripts and settings), saving
after every step:

0. Reference: `llama-bench` Q4_K_M pp512/tg128, 8 threads (E13 measured this on another v6 instance).
1. `bench/vllm_setup.sh` and the correctness check.
2. W8A8: prefill, decode and mixed at 16 and 64 prompts, 8 threads.
3. W4A16: mixed at 64 prompts.

Compare with E17's runs of the same configurations.

## Measurements

## Cost per token

## Analysis

## Threats to validity

## Next steps
