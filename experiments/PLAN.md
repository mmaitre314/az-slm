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
7. What else could cut cost: cheaper sizes of the same CPU, CPU-friendlier models, decode tuning,
   other hardware, managed per-token APIs, diffusion LMs? [RESEARCH.md](RESEARCH.md) holds the answers
   and the prioritized idea queue (R01–R44) that feeds E19 onward.
8. Decisions only the user can make (RESEARCH.md **R33**, blocked until answered): (a) what the real
   batch jobs look like (lengths, classification vs extraction, duplicates, volume); (b) whether data
   may go to Azure AI Foundry managed models (if yes, R08 runs at P0 and may make CPU self-hosting
   moot); (c) the quality bar (proposed default: ≤ 2 points below Qwen3.8-27B on the task metric in a
   paired non-inferiority test, ≥ 98% JSON-schema validity, ≤ 3 points on IFEval strict); (d) the
   monthly credit and round-3 spend.

## Constraints

- **No Spot**: the subscription is a Visual Studio (MSDN) offer, so VMs are pay-as-you-go
  (E01); costs are also reported at Spot prices for comparison.
- **Quota**: 20 vCPUs per region, so one 16-vCPU VM per region; parallelism comes from
  using several regions.
- **Budget**: the subscription has a monthly credit with a spending limit. Round 1 (2026-10-04
  02:00–11:15 UTC, five VMs) cost about **$50** (activity-log VM lifetimes × list price), of which
  ~$20 was VMs idling after their runners stopped. Round 2 budget: about **$15** (three VMs, ~3 h each); actual ~**$20** (~14 VM-hours: chains
  took longer than planned, and each VM idled ~1 h after its chain before the watchdog deallocated it).
- **Round-3 budget** (2026-10-05): spend to date this month ~**$70** (rounds 1–2). The credit amount
  is unknown to agents (Visual Studio subscriptions get $50, $100 or $150 a month depending on the
  level; the user confirms it in R33(d)). Round-3 P0 chains (RESEARCH.md: A, B1, B2, C) are est.
  18–21 VM-hours, **~$22–27** at v6 prices; cap per chain = its estimate × 1.5 (A ~$4 at D16s_v6 prices, B1 ~$12,
  B2 ~$10, C ~$14). Stop rule: if a chain exceeds 1.5× its VM-hour estimate, stop it, harvest and
  re-plan. Every chain ends by deallocating its own VM through the managed identity, so no ~1 h idle
  tail. At the best measured cost ($2.39/M, $1.53 per 1k requests of 512/128), if the credit is
  $150, the remaining ~$80 buys ~33M tokens (~52k requests) and a full month ~63M tokens (~98k
  requests); the D16s_v6 estimate ($1.51/M) would stretch that by ~1.6×.
- **Model**: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a dense 27B hybrid
  (48 Gated DeltaNet linear-attention layers + 16 full-attention layers, hidden 5120, vocab 248k,
  vision encoder unused). GGUF quants from `bartowski/Qwen3.8-27B-GGUF`.

## VMs

Round 2 (planned 2026-10-04 ~12:45 UTC). VM names are new per deployment.

| VM | Region | Size | Queue | $/h all-in | State |
| --- | --- | --- | --- | ---: | --- |
| `b2-vllm` | eastus2 | E16ds_v7 (Granite Rapids) | E17 | 1.681 | deleted 17:08 (chain done 16:56, harvested) |
| `b2-v6` | westus2 | E16ds_v6 (Emerald Rapids) | E18, E17 MTP follow-up | 1.326 | deleted 17:45 (chains done 13:46 and 15:20, deallocated idle with results kept, harvested) |
| `b2-qual` | centralus | E16ds_v7 (Granite Rapids) | E16 | 1.615 | deleted 17:20 (chain done 15:24, deallocated idle with results kept, harvested) |

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
| E14 | SGLang CPU backend vs vLLM on the same VM (shares vLLM's AMX INT8/GDN kernels; hybrid models need workaround flags; R02) | – | runner | candidate |
| [E15](E15-summary/) | Cross-stack cost and quality summary, recommendation | – | orchestrator | done (rounds 1–2) |
| [E16](E16-task-quality/) | Task-level quality of vLLM BF16, W8A8, W4A16 (GSM8K, MMLU) | b2-qual | runner | done |
| [E17](E17-vllm-scaling-mtp/) | vLLM batch scaling, W4A16, threads, MTP | b2-vllm | runner | done (MTP real-text follow-up on b2-v6) |
| [E18](E18-vllm-emerald-rapids/) | vLLM on E16ds_v6 vs E16ds_v7 | b2-v6 | runner | done |
| E19 | CPU-friendlier models on vLLM CPU: Qwen3.6-35B-A3B (MoE, ~3B active) and Qwen3.5-9B (dense), BF16/INT8/FP8, 64–256 prompts (R09, R10, R29) | – | runner | candidate |
| E20 | Quality gate for E19 models, thinking off, paired non-inferiority: GSM8K, MMLU, IFEval, JSON extraction, classification (R11) | – | runner | candidate |
| E21 | Same CPU, cheaper size: W8A8 at 64 prompts on D16s_v6 / D16ds_v6 (64 GiB) (R26, R21) | – | runner | candidate |
| E22 | vLLM W8A8 decode tuning: 128–256 sequences, BF16 GDN state, decode-step profile (R12, R13, R16) | – | runner | candidate |
| E23 | Shared-prefix workloads with prefix caching (R03) | – | runner | candidate |
| E24 | Real-text 512/128: output budget, structured output, MTP at 64–128 sequences (R14, R15) | – | runner | candidate |
| E25 | Gemma 4 26B-A4B (autoregressive) vs DiffusionGemma-26B-A4B on vLLM CPU (R04, R17) | – | runner | candidate |
| E26 | AMD Turin F16as_v7 (16 full cores, no AMX) with vLLM CPU + zentorch, Central India (R05) | – | runner | candidate |
| E27 | Managed per-token APIs (Azure AI Foundry, keyless): quality per dollar on R11's task sets and a hard knowledge set (R08; runs only if the user allows managed APIs, R33(b)) | – | orchestrator + runner | candidate |

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
- 2026-10-04 18:00 UTC: round 2 done, all VMs deleted. **vLLM W8A8 on E16ds_v6 at 64 prompts:
  $2.39/M blended** (E18), $3.47/M on a slower v7 instance (E17). W4A16 (4–6× slower) and 16
  threads (−50%) are out. W8A8 = BF16 on GSM8K, −2.75 points on MMLU (E16). vLLM MTP: 96% acceptance on
  real text, +16% at 64 sequences (E17 follow-up). The E15 recommendation is final for rounds 1–2.
  Next: the [RESEARCH.md](RESEARCH.md) queue. The safety net worked: b2-qual and b2-v6 were
  deallocated idle with results kept, and harvested after a restart.
- 2026-10-04: seven literature-research topics synthesized into [RESEARCH.md](RESEARCH.md) (queue
  R01–R32). Verdicts: **Ollama is not competitive** (it forces one sequence at a time for Qwen3.8's
  architecture; est. $20–29/M on E16ds_v6, corrected 2026-10-05). The cheapest same-model step is **a
  cheaper size of the same CPU** (D16s_v6, 64 GiB, est. $1.51/M if the job fits; E21). The biggest
  model-side lever is **fewer active parameters** (Qwen3.6-35B-A3B, Qwen3.5-9B: est. 2–4× cheaper,
  quality gate required; E19, E20); Phi-4 is dominated. Decode tuning (larger batches, BF16 GDN state,
  MTP) is est. −13% stacked on 128 GiB sizes (~$2.07/M on E16ds_v6; corrected 2026-10-05 from ~−16%,
  see below; E22, E24). **Diffusion LMs (DiffusionGemma) are the wrong regime** for batch CPU work: est. at best equal
  to their autoregressive twin, with lower quality (E25 only to confirm). SGLang is est. 0.85–1.3× of
  vLLM (E14). All current GPU families have quota 0 in all 63 regions. **Managed APIs cost
  $0.06–0.30/M** at list price, 8–40× below our best measured CPU cost; whether any matches Qwen3.8-27B
  is unknown until E27 (corrected 2026-10-05: gpt-oss-120b and Phi-4 are weaker, not comparable).
- 2026-10-05: review pass on RESEARCH.md. Corrections: the 64 GiB D16s_v6 fit is unproven (the $2.39
  baseline used 24 GiB of KV and peaked at ~65 GB in its largest process alone; R26 now measures
  16 GiB of KV and whole-system memory first). The decode levers don't stack on 64 GiB: est.
  **~$1.36/M on D16s_v6** (64 sequences, BF16 state + MTP-1, needs R13) vs ~$1.70/M on E16s_v6 and
  ~$2.07/M on E16ds_v6 with every lever (128 sequences); the earlier "~$1.27/M on D16s_v6" combined
  levers that don't fit together. MTP's gain shrinks with batch (+16% at 64, ~+10% at 128). A BF16
  GDN state must also be checked for longer outputs. R11 needs ~2,800 paired items for a 2-point
  non-inferiority call. Added R33 (user decisions, P0), R34–R36 (classification and extraction
  alternatives: one constrained label token est. −53% per request; distillation), R37 (stack on the
  chosen size), R38 (new vLLM release A/B), R39 (own W8A8), R40 (instance selection), R41 (D16ls_v6
  for small models), R42–R43 (CPU-side overhead), R44 (VM image). Priorities: R27 and R14 to P0, R08 to
  P0 if the user allows managed APIs; ranking by expected value per VM-hour with a confidence column.
  Primary metric from round 3: $ per 1,000 requests next to $/M tokens. Chains pin the vLLM image
  digest, measure run-to-run CV, and deallocate their VM at the end.
