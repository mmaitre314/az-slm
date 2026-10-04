# E09: vLLM CPU backend for Qwen3.8-27B (BF16, INT8 W8A8, INT4 W4A16)

| | |
| --- | --- |
| Status | queued |
| VM | `bench-e16v7` (Standard_E16ds_v7, eastus2, Regular): Xeon 6 6973P-C, 8 cores / 16 threads, 128 GiB. (First assigned `bench-vllm` in westus3; that VM was deleted idle by the watchdog after its runner hit a usage limit.) |
| Stack | vLLM CPU Docker image `vllm/vllm-openai-cpu:latest-x86_64` (record version and digest) |
| Model | `Qwen/Qwen3.8-27B` (BF16), `Avesed/Qwen3.8-27B-INT8-W8A8`, `Avesed/Qwen3.8-27B-INT4-W4A16` (community quants; record revisions) |
| Raw data | `vllm.jsonl`, short logs |

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

_To be filled by the runner/reporter._

## Cost per token

_To be filled (COST_MODEL.md; eastus2 E16ds_v7 at $1.681/h all-in, Spot $0.325/h all-in)._

## Analysis

## Threats to validity

## Next steps
