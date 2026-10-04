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
- **Budget**: the subscription has a monthly credit with a spending limit. Round 1 (2026-10-04
  02:00–11:15 UTC, five VMs) cost about **$50** (activity-log VM lifetimes × list price), of which
  ~$20 was VMs idling after their runners stopped. Round 2 budget: about **$15** (three VMs, ~3 h each).
- **Model**: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a dense 27B hybrid
  (48 Gated DeltaNet linear-attention layers + 16 full-attention layers, hidden 5120, vocab 248k,
  vision encoder unused). GGUF quants from `bartowski/Qwen3.8-27B-GGUF`.

## VMs

Round 2 (planned 2026-10-04 ~12:45 UTC). VM names are new per deployment.

| VM | Region | Size | Queue | $/h all-in | State |
| --- | --- | --- | --- | ---: | --- |
| `b2-vllm` | eastus2 | E16ds_v7 (Granite Rapids) | E17 | 1.681 | running (job `chain` since 12:49, ~16:00 end) |
| `b2-v6` | westus2 | E16ds_v6 (Emerald Rapids) | E18 | 1.326 | running (job `chain` since 12:32, ~13:50 end) |
| `b2-qual` | centralus | E16ds_v7 (Granite Rapids) | E16 | 1.615 | running (job `e16` since 13:02, ~16:00 end) |

Round 1 VMs, all deleted by the idle watchdog on 2026-10-04: `bench-e16v7` (eastus2, E02–E05,
E09; 02:07–10:31), `bench-lc2` (northcentralus, E06–E08, E10; 02:42–11:14), `bench-v6` (westus2,
E13, E12; 02:37–10:49), `bench-vllm` (westus3) and `bench-ov` (centralus), both unused (02:35–06:03).

## Experiments

| ID | Title | VM | Owner | Status |
| --- | --- | --- | --- | --- |
| [E01](E01-region-price-survey/) | AMX VM families, Spot prices and regions | – | orchestrator | done |
| [E02](E02-amx-enablement/) | Is AMX usable inside an Azure VM? | bench-e16v7 | orchestrator | done |
| [E03](E03-llamacpp-single-stream/) | llama.cpp single-sequence prefill/decode per quant, AMX vs no-AMX | bench-e16v7 | runner | done |
| [E04](E04-llamacpp-amx-multiseq-bug/) | llama.cpp AMX output corruption with several sequences | bench-e16v7 | orchestrator | done (finding) |
| [E05](E05-llamacpp-batched/) | llama.cpp batch throughput, 1–32 sequences | bench-e16v7 | runner | done |
| [E06](E06-quant-quality/) | Quantization quality: KL divergence vs BF16 | bench-lc2 | runner | done |
| [E07](E07-prefill-profile/) | Why is prefill slow? CPU profile | bench-lc2 | orchestrator + runner | done |
| [E08](E08-threads-smt/) | Threads, SMT and pinning; native no-AMX control build | bench-lc2 | runner | done |
| [E09](E09-vllm-cpu/) | vLLM CPU backend: BF16, INT8 W8A8, INT4 W4A16 | bench-e16v7 | runner | done (partial; rest in E17) |
| [E10](E10-openvino-genai/) | OpenVINO GenAI: INT4/INT8 weights, continuous batching | bench-lc2 | runner | done (partial; INT4 only) |
| [E11](E11-ovms/) | OpenVINO Model Server with continuous batching (OpenAI API) | – | runner | deprioritized (E10) |
| [E12](E12-speculative-decoding/) | Speculative decoding in llama.cpp: draft model, MTP head, n-gram | bench-v6 | runner | done (partial; vLLM MTP in E17) |
| [E13](E13-emerald-vs-granite/) | Emerald Rapids (v6) vs Granite Rapids (v7), same llama.cpp runs | bench-v6 | runner (scripted) | done |
| E14 | SGLang CPU backend (Intel AMX kernels) | – | runner | candidate |
| E15 | Cross-stack cost and quality summary, recommendation (after E16–E18) | – | orchestrator | planned |
| [E16](E16-task-quality/) | Task-level quality of vLLM BF16, W8A8, W4A16 (GSM8K, MMLU) | b2-qual | runner | running |
| [E17](E17-vllm-scaling-mtp/) | vLLM batch scaling, W4A16, threads, MTP | b2-vllm | runner | running |
| [E18](E18-vllm-emerald-rapids/) | vLLM on E16ds_v6 vs E16ds_v7 | b2-v6 | runner | running |

Status values: planned, queued, running, done, blocked, candidate (only if earlier results warrant it).

## Decision log

- 2026-10-04: Spot is unavailable on this subscription (E01). Using Regular priority, E16ds_v7 in
  eastus2 first, then one VM per region for parallel stacks.
- 2026-10-04: llama.cpp's AMX path corrupts multi-sequence decoding for this model (E04). Batched
  llama.cpp numbers come from the no-AMX build; quality (E06) also uses it.
- 2026-10-04: The first AMX vs no-AMX comparison (E03) also differs in compiler flags (native vs
  explicit). E08 adds a native build with only AMX disabled as a clean control.
- 2026-10-04 03:10–06:50 UTC: subagents hit a usage limit; the vLLM/OpenVINO runners died before
  starting work and their two idle VMs were deleted by the watchdog after 3 h (self-delete
  cascade verified: VM, disk, NIC, IP, schedule and role assignments all removed). Idle threshold
  lowered to 1 h. E09/E10/E11 move to the VMs that finished the llama.cpp experiments.
- 2026-10-04: E13 shows Emerald Rapids ≈ Granite Rapids for llama.cpp (within instance noise);
  E08 shows 15–25% variance between two E16ds_v7 instances. Compare stacks on the same VM where
  possible, and report the VM instance with every number.
- 2026-10-04 08:51–12:00 UTC: subagents hit the usage limit again mid-run. With nobody issuing
  Run Commands, the three VMs went idle and deleted themselves 1–2 h later, taking E09/E10/E12's
  raw results on `/mnt/data` with them. The orchestrator recovered the printed summaries from the
  runners' transcripts (E09, E10, E12 are marked partial, with provenance notes). Changes: chains
  run every step without an agent in the loop and save results to the OS disk after each step
  (`bench/save_results.sh`); while results are unfetched the watchdog deallocates instead of
  deleting; `deploy.py harvest` fetches them.
- 2026-10-04: **vLLM W8A8 leads** at $4.6/M blended (512/128, 16 prompts) vs $19–24/M for
  llama.cpp and $19–34/M for OpenVINO GenAI. Round 2 focuses on vLLM: batch scaling, W4A16 and MTP (E17),
  the cheaper v6 VM (E18), and whether the community quants keep quality (E16). OVMS (E11) is
  dropped because it shares OpenVINO GenAI's serial prefill (E10). llama.cpp MTP gives 1.5× for one sequence but
  only +19% at 4 slots (E12), not enough to close a 5× gap.

