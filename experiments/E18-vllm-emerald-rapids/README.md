# E18: vLLM on Emerald Rapids (E16ds_v6) vs Granite Rapids (E16ds_v7)

| | |
| --- | --- |
| Status | running: background job `chain` on `b2-v6` (`PROFILE=e18`), started 2026-10-04 12:32 UTC; expected ~1.3 h |
| VM | `b2-v6` (Standard_E16ds_v6, westus2, Regular): Xeon Platinum 8573C, 8 cores / 16 threads, 128 GiB |
| Stack | same vLLM image as E17 (record the digest; it must match E17's) |
| Model | `Avesed/Qwen3.8-27B-INT8-W8A8`, `Avesed/Qwen3.8-27B-INT4-W4A16`; Q4_K_M GGUF for the reference run |
| Dates | chain started 2026-10-04 12:32 UTC; finished: – |
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

## Setup

- Same chain, image, model revisions, KV sizes (16 GiB at 16 prompts, 24 GiB at 64) and run labels as
  [E17](../E17-vllm-scaling-mtp/README.md#setup); background job `chain` on `b2-v6`, progress in
  `/mnt/data/results/chain.log`. All runs are `t8` (8 OpenMP threads, one per physical core). The VM
  was deployed in westus2 and is a Xeon Platinum 8573C (Emerald Rapids), 2.3 GHz base.

## Method

Self-contained chain `bench/vllm_chain.sh` with `PROFILE=e18` (a subset of E17's steps, same scripts and
settings), saving after every step:

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
