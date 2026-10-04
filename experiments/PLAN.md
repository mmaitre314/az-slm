# Experiment plan: Qwen3.8-27B batch inference on Azure Intel AMX CPUs

## Goal

Find the cheapest way to process batches of prompts (latency doesn't matter) with
**Qwen3.8-27B** on Azure CPU VMs with Intel AMX, at acceptable quality. Output: USD per million
input and output tokens per stack, quantization and VM, with the quality cost of each choice.

## Key questions

1. How much does AMX actually buy for this model, for prefill and for decode? (E03, E07, E08, E13)
2. How much does batching raise throughput, and where does it saturate? (E05, E09, E10, E11)
3. Which quantization gives the best cost at acceptable quality? (E03, E05, E06)
4. Which serving stack is fastest on CPU: llama.cpp, vLLM, OpenVINO GenAI/OVMS, or SGLang? (E09–E11, E14)
5. Does speculative decoding (draft model, MTP head) help batch throughput? (E12)
6. Granite Rapids (v7) vs Emerald Rapids (v6): which is cheaper per token? (E13)

## Constraints

- **No Spot**: the subscription is a Visual Studio (MSDN) offer, so VMs are pay-as-you-go
  (E01); costs are also reported at Spot prices for comparison.
- **Quota**: 20 vCPUs per region, so one 16-vCPU VM per region; parallelism comes from
  using several regions.
- **Budget**: the subscription has a monthly credit with a spending limit. Four VMs cost about
  **$6.30/h** all-in. Keep the total for this round under about $40 and tear VMs down when their queue is empty.
- **Model**: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a dense 27B hybrid
  (48 Gated DeltaNet linear-attention layers + 16 full-attention layers, hidden 5120, vocab 248k,
  vision encoder unused). GGUF quants from `bartowski/Qwen3.8-27B-GGUF`.

## VMs

| VM | Region | Size | CPU | Stack / queue | $/h all-in |
| --- | --- | --- | --- | --- | ---: |
| `bench-e16v7` | eastus2 | E16ds_v7 | Xeon 6 6973P-C (Granite Rapids) | llama.cpp: E03 → E05 → E06 → E07 → E08 → E12a | 1.681 |
| `bench-vllm` | westus3 | E16ds_v7 | Granite Rapids | vLLM: E09 (then E14 SGLang if warranted) | 1.681 |
| `bench-ov` | centralus | E16ds_v7 | Granite Rapids | OpenVINO: E10 → E11 → E12b | 1.615 |
| `bench-v6` | westus2 | E16ds_v6 | Xeon Platinum 8573C (Emerald Rapids) | llama.cpp: E13 | 1.326 |

## Experiments

| ID | Title | VM | Owner | Status |
| --- | --- | --- | --- | --- |
| [E01](E01-region-price-survey/) | AMX VM families, Spot prices and regions | – | orchestrator | done |
| [E02](E02-amx-enablement/) | Is AMX usable inside an Azure VM? | bench-e16v7 | orchestrator | done |
| [E03](E03-llamacpp-single-stream/) | llama.cpp single-sequence prefill/decode per quant, AMX vs no-AMX | bench-e16v7 | runner | running |
| [E04](E04-llamacpp-amx-multiseq-bug/) | llama.cpp AMX output corruption with several sequences | bench-e16v7 | orchestrator | done (finding) |
| [E05](E05-llamacpp-batched/) | llama.cpp batch throughput, 1–32 sequences | bench-e16v7 | runner | queued |
| [E06](E06-quant-quality/) | Quantization quality: KL divergence vs BF16 | bench-e16v7 | runner | queued |
| [E07](E07-prefill-profile/) | Why is prefill slow? CPU profile | bench-e16v7 | orchestrator + runner | queued |
| [E08](E08-threads-smt/) | Threads, SMT and pinning; native no-AMX control build | bench-e16v7 | runner | queued |
| [E09](E09-vllm-cpu/) | vLLM CPU backend: BF16, INT8 W8A8, INT4 W4A16 | bench-vllm | runner | queued |
| [E10](E10-openvino-genai/) | OpenVINO GenAI: INT4/INT8 weights, continuous batching | bench-ov | runner | queued |
| [E11](E11-ovms/) | OpenVINO Model Server with continuous batching (OpenAI API) | bench-ov | runner | queued |
| [E12](E12-speculative-decoding/) | Speculative decoding: draft model, MTP head | bench-e16v7, bench-ov | runner | planned |
| [E13](E13-emerald-vs-granite/) | Emerald Rapids (v6) vs Granite Rapids (v7), same llama.cpp runs | bench-v6 | runner | queued |
| E14 | SGLang CPU backend (Intel AMX kernels) | bench-vllm | runner | candidate |
| E15 | Cross-stack cost and quality summary, recommendation | – | orchestrator | planned |

Status values: planned, queued, running, done, blocked, candidate (only if earlier results warrant it).

## Decision log

- 2026-10-04: Spot is unavailable on this subscription (E01). Using Regular priority, E16ds_v7 in
  eastus2 first, then one VM per region for parallel stacks.
- 2026-10-04: llama.cpp's AMX path corrupts multi-sequence decoding for this model (E04). Batched
  llama.cpp numbers come from the no-AMX build; quality (E06) also uses it.
- 2026-10-04: The first AMX vs no-AMX comparison (E03) also differs in compiler flags (native vs
  explicit). E08 adds a native build with only AMX disabled as a clean control.
