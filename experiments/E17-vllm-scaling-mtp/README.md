# E17: vLLM CPU batch scaling, W4A16, threads and MTP (E09 continued)

| | |
| --- | --- |
| Status | planned |
| VM | `b2-vllm` (Standard_E16ds_v7, eastus2, Regular): Xeon 6 6973P-C, 8 cores / 16 threads, 128 GiB |
| Stack | vLLM CPU Docker image `vllm/vllm-openai-cpu:latest-x86_64` (record vLLM version and image digest); llama.cpp from cloud-init for the reference run |
| Model | `Avesed/Qwen3.8-27B-INT8-W8A8`, `Avesed/Qwen3.8-27B-INT4-W4A16` (record revisions); `bartowski/Qwen3.8-27B-GGUF` Q4_K_M for the reference run |
| Dates | |
| Raw data | `raw/` (harvested with `deploy.py harvest`) |

## Question

E09 found vLLM W8A8 the cheapest stack so far ($4.6/M blended at 16 prompts, 5× cheaper than
llama.cpp), but its batch scaling, the W4A16 variant, thread count and speculative decoding
were lost with the VM. How low does the cost go with larger batches, and which knobs matter?

## Hypotheses

_Written before the results were known._

- **H1 (batch scaling)**: at 16 sequences, W8A8 decode reads weights at only ~56 GB/s (E09), so
  per-step cost is not all weight traffic, but weight reads are still shared across sequences.
  Expect mixed 512/128 throughput to rise 30–60% from 16 to 64 prompts, mostly from decode;
  prefill tok/s stays flat (compute-bound already).
- **H2 (W4A16)**: weights are half of W8A8's, so decode at 16 sequences is faster (up to 1.5×),
  but prefill dequantizes INT4 to BF16 and runs AMX-BF16 or slower (≤ BF16's 133 tok/s; E09 saw
  "millions of tiny ukernel calls"). Blended 512/128 cost ends up equal to or worse than W8A8's.
- **H3 (threads)**: AMX has one tile unit per core, so 16 threads (SMT) gives ≤ 5% on prefill
  and possibly a loss, unlike llama.cpp's +14–25% (E08).
- **H4 (MTP)**: Qwen3.8's MTP head accepted 89% of single draft tokens in llama.cpp (E12). With
  `num_speculative_tokens` 1, decode-heavy throughput at 16 prompts rises 20–40%, less at 64.
  Risks: the CPU backend or the hybrid model may not support MTP; the W8A8 checkpoint may lack
  the MTP weights.

## Method

One self-contained background chain, `bench/e17_chain.sh`, that calls `bench/save_results.sh`
after every step:

0. **Reference**: `llama-bench` Q4_K_M pp512/tg128, 8 threads, AMX build (comparable with E03/E13
   to place this VM instance relative to earlier ones).
1. `bench/vllm_setup.sh`, then the 3-prompt correctness check for each model.
2. W8A8: prefill (512/1), decode (32/256) and mixed (512/128) at 16, 32 and 64 prompts, 8
   threads (16 repeats E09 on this instance). KV cache sized for 64 × 768 tokens plus the GDN state.
3. W4A16: the same three workloads at 16 and 64 prompts.
4. W8A8 with 16 threads (all logical CPUs): prefill and mixed at 64 prompts.
5. MTP: check whether the W8A8 checkpoint contains the MTP weights (`mtp` tensors in the
   safetensors index). If so, W8A8 with `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`,
   N = 1 for decode and mixed at 16 and 64 prompts and N = 2 for decode at 16. If not, or if the
   CPU backend rejects it, record the error and move on.
6. Save `top` snapshots, versions and digests.

## Measurements

## Cost per token

## Analysis

## Threats to validity

## Next steps
