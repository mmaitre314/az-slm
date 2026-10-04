# E17: vLLM CPU batch scaling, W4A16, threads and MTP (E09 continued)

| | |
| --- | --- |
| Status | running: background job `chain` on `b2-vllm` (`PROFILE=e17`), started 2026-10-04 12:49 UTC; expected ~3 h |
| VM | `b2-vllm` (Standard_E16ds_v7, eastus2, Regular): Xeon 6 6973P-C, 8 cores / 16 threads, 128 GiB |
| Stack | vLLM CPU Docker image `vllm/vllm-openai-cpu:latest-x86_64` (record vLLM version and image digest); llama.cpp from cloud-init for the reference run |
| Model | `Avesed/Qwen3.8-27B-INT8-W8A8`, `Avesed/Qwen3.8-27B-INT4-W4A16` (record revisions); `bartowski/Qwen3.8-27B-GGUF` Q4_K_M for the reference run |
| Dates | chain started 2026-10-04 12:49 UTC (smoke test 12:32-12:47 UTC); finished: – |
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

## Setup

- Chain: `bench/vllm_chain.sh`, run as background job `chain` (a transient systemd unit) with no agent in
  the loop. A failing step is logged and the chain goes on; a model whose 3-prompt correctness check
  fails twice is not benchmarked. Progress log: `/mnt/data/results/chain.log` (copied to the OS disk
  after every step). One `vllm_bench.sh` call per workload (own 2 h timeout, own save). Each run is a
  fresh container, so the model is loaded again every time; torch.compile artefacts are cached in
  `/mnt/data/vllm-cache`.
- Image `vllm/vllm-openai-cpu:latest-x86_64` = vLLM 0.31.0, torch 2.13.0+cpu, digest
  `sha256:8024248339dc6878daa5349344ed29d49c8a6732f6bdf7400fda32ce33e4b30b` (same as E09's tag; the
  digest was not recorded there). Model revisions: W8A8 `86b8427a5e621c18f203bbce795dd83122496412`,
  W4A16 `135ecac28b03e7f3e0d3458df40eea8dc10dc973`.
- Only the two quantized repos are downloaded (no BF16). KV cache (`VLLM_CPU_KVCACHE_SPACE`) 16 GiB for
  16 and 32 prompts, 24 GiB for 64 prompts; `--max-model-len 2048`; random prompts, all requests
  submitted at once (`vllm bench throughput`, as in E09). Runs are labelled in `vllm.jsonl` by `run`:
  `t8` (8 OpenMP threads, one per physical core), `t16` (all 16 logical CPUs), `mtp1`, `mtp2`
  (`num_speculative_tokens`).
- MTP: `--speculative-config {"method":"mtp","num_speculative_tokens":N}` goes through `EXTRA` and
  `vllm_bench.sh` expands it unquoted, so the JSON has no spaces and no wrapping quotes. The chain
  scans the safetensors headers for `mtp` tensors first (`mtp-check.jsonl`) and skips the MTP steps if
  there are none, or if the first MTP step fails on every run.
- Smoke test before the real chain: `PROFILE=smoke` (setup, MTP check, one correctness run, a 4-prompt
  mixed 128/16 W8A8 run and a 4-prompt MTP run), results moved to `/mnt/data/results-smoke` on the VM.

## Method

One self-contained background chain, `bench/vllm_chain.sh` (`PROFILE=e17`, the default), that calls
`bench/save_results.sh` after every step (and after every single `vllm bench throughput` run):

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
