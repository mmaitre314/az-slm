# E09: vLLM CPU backend for Qwen3.8-27B (BF16, INT8 W8A8, INT4 W4A16)

| | |
| --- | --- |
| Status | partial (2026-10-04 07:00–08:45 UTC); BF16 and W8A8 at 16 prompts measured, W4A16 throughput, batch scaling and 16 threads lost. Numbers recovered from the runner's transcript; raw files lost (VM deleted by the idle watchdog) |
| VM | `bench-e16v7` (Standard_E16ds_v7, eastus2, Regular): Xeon 6 6973P-C, 8 cores / 16 threads, 128 GiB. (First assigned `bench-vllm` in westus3; that VM was deleted idle by the watchdog after its runner hit a usage limit.) |
| Stack | vLLM CPU Docker image `vllm/vllm-openai-cpu:latest-x86_64` = vLLM 0.31.0 (digest not recovered) |
| Model | `Qwen/Qwen3.8-27B` (BF16), `Avesed/Qwen3.8-27B-INT8-W8A8`, `Avesed/Qwen3.8-27B-INT4-W4A16` @ `135ecac28b` (20 GB) (community compressed-tensors quants) |
| Raw data | lost; values below are the runner's printed summaries (`bench/vllm_bench.sh` output lines) |

## Question

Is vLLM's CPU backend (oneDNN, AMX-BF16/INT8, continuous batching) cheaper per token than
llama.cpp for offline batch processing of Qwen3.8-27B?

## Hypotheses

- **H1 (prefill)**: oneDNN uses AMX-BF16 for the linear layers, so BF16 prefill reaches 100–200
  tokens/s. Peak AMX-BF16 on 8 cores at 3.6 GHz is about 29 TFLOPS; at 54 GFLOP/token that's
  ~540 tok/s at 100% efficiency, and 20–40% is realistic. That would be 5–10× llama.cpp's 21–30 t/s (E03).
- **H2 (decode)**: BF16 decode is memory-bound. Each step reads ~54 GB of weights, so at
  ~80 GB/s effective bandwidth (see E03) that's ~1.5 steps/s. With 16 sequences, aggregate
  decode is ~20–25 tok/s. INT8 W8A8 halves the bytes and uses AMX-INT8, so roughly 2× BF16 decode.
- **H3 (risk)**: the 48 Gated DeltaNet layers may run through a slow PyTorch-native path on
  CPU, capping both phases well below H1/H2. The model may also not load at all on the CPU backend.
- **H4**: W4A16 may be unsupported on x86 CPU. vLLM lists AWQ/GPTQ and INT8 W8A8 for x86;
  compressed-tensors W4A16 may need GPU kernels.

## Method

1. `bench/vllm_setup.sh` (Docker, image pull, IMDS blocked for containers, HF downloads to `/mnt/data/models`).
2. **Correctness**: generate 64 tokens greedily for 3 fixed prompts with `vllm.LLM(...)` in the
   container. They must be coherent and should broadly match llama.cpp Q8_0's answers on the same prompts.
3. **AMX check**: one short run with `ONEDNN_VERBOSE=1`; the log must show an `amx` ISA.
4. `bench/vllm_bench.sh`, per model: prefill-only (512 in / 1 out), decode-heavy (32 in / 256 out),
   mixed (512 in / 128 out), 16 prompts each. Then batch scaling for the best model: 1, 4, 16, 32
   prompts. One OpenMP thread per physical core (`VLLM_CPU_OMP_THREADS_BIND`). Also try 16 threads once.
5. Record vLLM version, image digest, model revisions, and peak memory.

## Measurements

> **Provenance.** The runner's subagent stopped at a usage limit at 08:51 UTC and the VM deleted
> itself an hour later (idle watchdog), taking `/mnt/data/results` with it. Every number below was
> printed by `bench/vllm_bench.sh` in the runner's transcript and copied from there by the
> orchestrator. They are single runs; the jsonl rows, logs and image digest are lost.

Setup notes recovered from the transcript:

- vLLM 0.31.0 loads the hybrid `qwen3_5` architecture on CPU (text-only, `--limit-mm-per-prompt`
  image/video 0). `ONEDNN_VERBOSE` showed AMX `brgemm` kernels for BF16 (`avx10_1_512_amx`).
- Correctness (`bench/vllm_correct.sh`, 3 fixed prompts, greedy): BF16, W8A8 and W4A16 all gave
  coherent, correct answers (Paris/Marseille/Lyon; the German pangram translation; `17 × 23 = 391`).
  One early BF16 correctness run failed with a oneDNN engine-init error (exit 2); a re-run passed.
- 8 OpenMP threads, one per physical core (`VLLM_CPU_OMP_THREADS_BIND=0,2,…,14`), KV cache 16 GiB,
  `--max-model-len 2048`, random prompts (`vllm bench throughput`, offline, all 16 requests submitted at once).

| model | workload (in/out) | prompts | wall (s) | total tok/s | output tok/s | peak RSS (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | prefill (512/1) | 16 | 61.6 | 133.1 | – | 77.2 |
| BF16 | decode (32/256) | 16 | 215.6 | 21.3 | 19.0 | 76.6 |
| BF16 | mixed (512/128) | 16 | 163.1 | 62.7 | 12.6 | 77.2 |
| W8A8 | prefill (512/1) | 16 | 41.4 | 198.1 | – | 56.7 |
| W8A8 | decode (32/256) | 16 | 126.9 | 36.2 | 32.3 | 56.6 |
| W8A8 | mixed (512/128) | 16 | 100.0 | 102.3 | 20.5 | 57.2 |
| W4A16 | – | – | – | lost | – | – |

Output tok/s = 16 × output length / wall. "Total" counts input and output tokens.

## Cost per token

[COST_MODEL.md](../COST_MODEL.md), eastus2 E16ds_v7: $1.681/h all-in on demand, $0.325/h all-in at
Spot. Assumptions 1–4 apply. Each workload's wall time is charged entirely to it, so "decode"
includes its short 32-token prefill and "prefill" includes one output token per request.

| model | $/M input (prefill run) | $/M output (decode run) | $/M blended, 512/128 (mixed run) | same at Spot |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 3.51 | 24.6 | 7.45 | 0.68 / 4.76 / 1.44 |
| W8A8 | **2.36** | **14.5** | **4.56** | 0.46 / 2.80 / **0.88** |

For comparison, the best llama.cpp configuration (E05, Q4_K_M, 32 sequences, no-AMX build) costs
$21.7/M input, $32.8/M output and $23.9/M blended on demand; on E16ds_v6 (E13) $19.3/M blended.
At 16 prompts, **vLLM W8A8 is 5.2× cheaper than llama.cpp for the reference workload**, and 9× cheaper for input tokens.

## Analysis

_Orchestrator, 2026-10-04._

- **H1 (prefill) confirmed.** BF16 prefill is 133 tok/s and W8A8 198 tok/s, against 21–30 tok/s for
  llama.cpp (E03/E05). That is 6–9× faster, in the predicted 5–10× range. At ~54 GFLOP per token,
  198 tok/s is ~10.7 TOPS, ~18% of the 8-core AMX-INT8 peak (~58 TOPS); BF16 is at ~7.2 TFLOPS, ~25% of
  the AMX-BF16 peak (~29 TFLOPS). oneDNN brgemm is doing what llama.cpp's AMX kernel doesn't (E07: ~3% of peak).
- **H2 (decode) confirmed.** BF16 decode at 16 sequences is 19 output tok/s (predicted 20–25); W8A8
  is 32 tok/s (1.7×; predicted ~2×). Per decode step (16 tokens): BF16 ≈ 0.82 s, W8A8 ≈ 0.48 s,
  i.e. ~66 GB/s and ~56 GB/s of weight reads. That's below the ~100 GB/s ceiling seen in E13, so
  at 16 sequences decode is not purely bandwidth-bound: per-sequence work (GDN recurrence,
  attention, sampling) adds a share that more sequences won't amortize. Batch scaling (lost) is
  needed to see whether 32–64 sequences still raise throughput.
- **H3 rejected.** The Gated DeltaNet layers didn't make vLLM slow; both phases met the
  oneDNN-based predictions.
- **H4 rejected.** W4A16 compressed-tensors loads and gives correct output on x86 CPU. Its speed is unknown
  (runs lost). Its oneDNN verbose log showed "millions of tiny ukernel calls", a hint that the
  INT4 path dequantizes per small tile and may be slower than W8A8.
- For the 512/128 reference workload, decode takes about 60% of W8A8's wall time
  (16×512/198 ≈ 41 s prefill, ~59 s decode). Raising decode throughput is the next lever:
  larger batches, MTP speculative decoding (E12), or W4A16 if its decode is faster.
- **Quality caveat**: W8A8 and W4A16 are community quantizations; only 3 prompts were checked.
  E16 must measure their agreement with BF16 before W8A8 is the recommendation.

## Threats to validity

- Single runs and one VM instance (instance variance 9–25%, E08). Numbers were copied from a transcript, not from the raw files.
- Random-token prompts (`--dataset-name random`): fine for throughput, but they say nothing about
  output length, since generation is forced to the requested length.
- `top` before each run was logged but is lost; Defender/monitoring agents could have taken CPU.
- 16 prompts is a small batch; the cost may still be falling with more prompts.

## Next steps

1. Re-run (E09b): W8A8 and W4A16 at 16/32/64 prompts, 8 vs 16 threads, KV cache sized for 64 × 640 tokens.
2. vLLM MTP (`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`) on the decode and mixed workloads.
3. Same runs on E16ds_v6 (westus2), which is 21% cheaper per hour.
4. E16: agreement of W8A8/W4A16 with BF16 on a task-level prompt set.
