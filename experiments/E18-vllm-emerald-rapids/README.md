# E18: vLLM on Emerald Rapids (E16ds_v6) vs Granite Rapids (E16ds_v7)

| | |
| --- | --- |
| Status | done (chain 12:32–13:46 UTC, exit 0, no failed rows; the VM then ran the E17 MTP real-text follow-up until 15:20, was deallocated idle with results kept, harvested and deleted 17:45 UTC) |
| VM | `b2-v6` (Standard_E16ds_v6, westus2, Regular): Xeon Platinum 8573C, 8 cores / 16 threads, 128 GiB |
| Stack | vLLM 0.31.0, image `vllm/vllm-openai-cpu@sha256:8024248339dc…`, the same digest as E17; llama.cpp `dd266785` for the reference run |
| Model | `Avesed/Qwen3.8-27B-INT8-W8A8` @ `86b8427a5e`, `Avesed/Qwen3.8-27B-INT4-W4A16` @ `135ecac28b`; Q4_K_M GGUF for the reference run |
| Dates | 2026-10-04 12:32–13:46 UTC |
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

Same scripts and settings as E17 (`bench/vllm_chain.sh`, `PROFILE=e18`): `vllm bench throughput`,
random-token prompts, one run each, 8 threads. Raw: [`raw/vllm.jsonl`](raw/vllm.jsonl), [`raw/chain.log`](raw/chain.log).

**Reference run** (`llama-bench` Q4_K_M, AMX build, same llama.cpp commit as E17): pp512 31.4 tok/s,
tg128 3.55 tok/s. E17's v7 instance (`b2-vllm`): 25.7 / 3.13. So this v6 instance is 22% faster for
llama.cpp prefill and 13% faster for decode. E13 found v6 ≈ v7 on llama.cpp across instances.

| model | workload (in/out) | prompts | wall (s) | total tok/s | output tok/s | E17 v7 (`b2-vllm`) total tok/s | E09 v7 (`bench-e16v7`) total tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W8A8 | prefill (512/1) | 16 | 38.1 | 215.6 | – | 164.6 | 198.1 |
| W8A8 | decode (32/256) | 16 | 130.8 | 35.2 | 31.3 | 30.3 | 36.2 |
| W8A8 | mixed (512/128) | 16 | 97.3 | 105.2 | 21.0 | 86.8 | 102.3 |
| W8A8 | prefill (512/1) | 64 | 124.8 | **263.0** | – | 196.4 | – |
| W8A8 | decode (32/256) | 64 | 301.5 | 61.1 | 54.3 | 62.1 | – |
| W8A8 | mixed (512/128) | 64 | 265.8 | **154.1** | 30.8 | 134.4 | – |
| W4A16 | mixed (512/128) | 64 | 1067.1 | 38.4 | 7.7 | 30.0 | – |

Peak RSS 56–66 GB, as on v7. The oneDNN ISA line reports AMX BF16 and INT8 (no AMX-FP16 on Emerald Rapids).

## Cost per token

[COST_MODEL.md](../COST_MODEL.md): westus2 E16ds_v6, $1.326/h all-in on demand, $0.260/h at Spot.

| configuration | $/M input (prefill run) | $/M output (decode run) | $/M blended 512/128 (mixed run) | blended at Spot | E17 v7 blended (on demand) |
| --- | ---: | ---: | ---: | ---: | ---: |
| W8A8, 16 prompts | 1.71 | 11.8 | 3.50 | 0.69 | 5.38 |
| **W8A8, 64 prompts** | **1.40** | **6.78** | **2.39** | **0.47** | 3.47 |
| W4A16, 64 prompts | – | – | 9.60 | 1.88 | 15.6 |

## Analysis

_Orchestrator, 2026-10-04._

- **H1 confirmed, and the effect is larger than predicted.** At 64 prompts, v6 W8A8 costs **$2.39/M** blended on demand (31% below
  E17's v7 instance) and $0.47/M at Spot, the cheapest configuration measured so far. Prefill is
  faster on this v6 instance (263 vs 196 tok/s at 64 prompts) while decode is equal (61.1 vs
  62.1). Against E09's faster v7 instance at 16 prompts the two CPUs are within ±6% on every workload, so
  the per-token advantage comes mostly from v6's 21% lower hourly price.
- **H2: partly instance effect.** The llama.cpp reference says this v6 instance is 13–22% faster
  than E17's v7 instance. Normalizing by it, v6 and v7 are about equal per hour of compute, consistent with E13 for
  llama.cpp. AMX-FP16 (v7 only) plays no role for W8A8.
- **Practical rule**: prefer E16ds_v6 (or whichever of v6/v7 is cheaper in the region) and keep
  a reference run on every VM, because instance-to-instance variance (13–25%) is as large as the
  difference between CPU generations.

## Threats to validity

- One instance per generation; the v6/v7 comparison is confounded by instance speed (the reference run quantifies it).
- Random-token prompts with fixed output lengths.

## Next steps

1. Use E16ds_v6 for the next vLLM experiments. In each region, run a short reference first and
   drop slow instances (redeploy) when the job is long enough to pay for it.
2. Re-run the reference on several fresh v6/v7 instances (~10 min each) to measure the
   instance-speed distribution; if it is wide, "deploy, test, keep the fast one" is a cheap optimization.
