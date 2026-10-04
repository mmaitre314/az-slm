# E15: Cross-stack cost and quality summary

| | |
| --- | --- |
| Status | done for rounds 1–2 (2026-10-04 18:00 UTC); extended by the RESEARCH.md queue |
| Owner | orchestrator |
| Sources | E03, E05, E06, E09, E10, E12, E13, E16, E17, E18 |

## Question

What is the cheapest way to run Qwen3.8-27B batch jobs (latency irrelevant) on Azure Intel AMX
VMs, at acceptable quality, and what does it cost per million tokens?

## Reference workload and assumptions

512 input + 128 output tokens per request, the VM busy 100% of the time, list prices
([COST_MODEL.md](../COST_MODEL.md)): E16ds_v7 $1.681/h all-in on demand ($0.325/h Spot),
E16ds_v6 $1.326/h ($0.260/h Spot). The subscription can't use Spot (MSDN offer); Spot columns
are what the same throughput would cost on an EA or pay-as-you-go subscription.

## Best configuration per stack

| stack | configuration | VM | blended $/M, 512/128, on demand | Spot | quality evidence | source |
| --- | --- | --- | ---: | ---: | --- | --- |
| **vLLM 0.31.0 CPU** | **W8A8 (community), 64 prompts, 8 threads** | **E16ds_v6** | **2.39** | **0.47** | = BF16 on GSM8K; −2.75 points on MMLU (p = 0.035) | E18, E16 |
| vLLM 0.31.0 CPU | W8A8, 64 prompts, 8 threads | E16ds_v7 (slow instance) | 3.47 | 0.67 | same | E17 |
| vLLM 0.31.0 CPU | W8A8, 16 prompts | E16ds_v7 | 4.56 | 0.88 | same | E09 |
| vLLM 0.31.0 CPU | BF16, 16 prompts | E16ds_v7 | 7.45 | 1.44 | reference | E09 |
| vLLM 0.31.0 CPU | W4A16 (community), 64 prompts | E16ds_v6 | 9.60 | 1.88 | = BF16 on GSM8K; −3.0 on MMLU | E18, E16 |
| llama.cpp `11fe021` | Q4_K_M, 32 sequences, no-AMX build | E16ds_v6 | 19.3 | 3.8 | KLD 0.0135, 93.6% top-1 (E06) | E13 |
| llama.cpp `11fe021` | Q4_K_M, 32 sequences, no-AMX build | E16ds_v7 | 23.9 | 4.6 | same | E05 |
| OpenVINO GenAI 2026.4.1 | INT4 IR, 4 requests measured; 8+ estimated | E16ds_v7 | 34.1 (est. 19–34 at 8+) | 6.6 (est. 3.6–6.6) | 3 prompts correct | E10 |
| llama.cpp, one sequence | Q4_0, AMX build | E16ds_v7 | 35.4 | 6.8 | KLD 0.0252 | E03 |

Batched llama.cpp rows were measured with 128-token prompts and reuse that prefill rate for 512
(E05). vLLM rows are single runs with random-token prompts. Every row is one VM instance, and
identical sizes differ by 9–25% (E08, E18).

## Findings

1. **The serving stack matters most.** vLLM's oneDNN path reaches ~18–25% of AMX peak in prefill
   (E09). llama.cpp's AMX kernel reaches ~3% (E07) and corrupts multi-sequence output for this
   architecture (E04). OpenVINO GenAI prefills one request at a time through the hybrid model's 48
   recurrent `Loop` ops (E10). vLLM is ≥ 8× cheaper than either at its best batch size.
2. **Batch size is the next lever.** From 16 to 64 prompts, vLLM W8A8's blended cost drops 32–36%
   (E17, E18), mostly from decode (output tok/s ×2). The curve flattens: +34% from 16 to 32, +16% from 32 to 64.
3. **INT8, not INT4, and one thread per core.** On vLLM CPU, W4A16 is 4–6× slower than W8A8, and
   16 threads (SMT) halve throughput (E17).
4. **Quality**: W8A8 equals BF16 on GSM8K (92.5%) and loses 2.75 points on MMLU (90.0 → 87.25%,
   paired p = 0.035). W4A16 is the same as W8A8 (E16).
5. **v6 vs v7**: about equal per hour of compute (E13, E18), so the cheaper E16ds_v6 wins per token.
   Instance-to-instance variance (13–25%) is as large as the generation gap.
6. **Speculative decoding (MTP head)**: 96% acceptance on real text with 1 draft token, but only
   +16% throughput at 64 sequences because decode is compute-bound at that batch size (E17). It gives 1.5×
   for a single sequence in llama.cpp (E12).
7. **Spot** would cut every number by ~81% (E01) but isn't available on this subscription; GPUs
   and HBM CPUs have zero quota ([RESEARCH.md](../RESEARCH.md)).

## Recommendation

For offline batch jobs with Qwen3.8-27B on Azure CPUs:

- **vLLM CPU (0.31.0 image) with INT8 W8A8 weights on an E16ds_v6**: submit large batches (64+
  requests in flight), bind one OpenMP thread per physical core (`VLLM_CPU_OMP_THREADS_BIND`),
  and size the KV cache for the batch (24 GiB for 64 × 768 tokens). Expected **~$2.4 per million
  tokens** on demand for 512-in/128-out requests (~$0.47 at Spot prices). For decode-heavy jobs, add
  `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` (+16% at 64 sequences).
- **Check quality on the actual task** (a few hundred labelled items, BF16 vs W8A8). If ~3 points
  of knowledge-style accuracy matter, use BF16 at roughly 1.6× the cost (~$3.8/M at 64 prompts on
  v6, extrapolated from the BF16/W8A8 ratio in E09 and E16).
- Run a 5-minute reference benchmark on each new VM: instances vary by up to 25%.

The comparison with managed per-token APIs, CPU-friendlier models (smaller dense, MoE), diffusion
LMs and other stacks (SGLang) is in [RESEARCH.md](../RESEARCH.md) and its queue.

## Open questions

- Scaling beyond 64 prompts (128) and real-text 512/128 workloads with prefix caching.
- SGLang's CPU backend (E14), CPU-friendlier models, and managed-API prices: see RESEARCH.md.
