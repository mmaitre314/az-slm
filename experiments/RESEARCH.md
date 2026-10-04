# Research plan and work queue

Updated 2026-10-04 22:30 UTC by the orchestrator from four of seven literature-research topics
(Ollama, other CPU stacks, CPU-friendly models, stack-agnostic techniques). Raw findings with
sources: [research/findings-2026-10-04.json](research/findings-2026-10-04.json).
**Not yet adversarially verified**: PR numbers, release dates and benchmark scores come from the
researchers' web searches. The remaining topics (diffusion LMs, CPU-inference literature,
hardware/GPU/managed-API prices), the synthesis and the verification are running; they will
replace this version. If they never land, this file is the queue.

Baseline for every estimate below: vLLM 0.31 CPU, W8A8, 64 prompts, **$2.39/M** blended on E16ds_v6
($3.47/M on an E16ds_v7 instance), from E17/E18.

How to use the queue: take the top `queued` row, set it to `in-progress` with an experiment ID, and
record the result and link when done. Add new ideas at the bottom with the next R-number.

## Answers so far

- **Ollama: not competitive.** Since v0.30.0 (2026-06) Ollama on Linux runs upstream llama.cpp
  `llama-server` (same ggml kernels as E03–E08, including the slow AMX path). Its scheduler forces
  `num_parallel=1` for the qwen35 architecture family that Qwen3.8 belongs to, so requests are
  served one at a time: about $28–37/M on demand, 12–15× the vLLM baseline. Patching that out just
  gives llama.cpp batched, with its AMX multi-sequence bug (E04).
- **Other stacks: SGLang's CPU backend is the only plausible challenger, and the gain is modest (0.85–1.3×).**
  vLLM's CPU backend already vendors SGLang's AMX kernels (including the INT8 GEMM and the GDN kernel), so W8A8 runs
  the same kernels in both. SGLang needs workaround flags for hybrid models on CPU. Intel xFasterTransformer, IPEX
  (end of life 2026-03), IPEX-LLM and TGI CPU are dead or archived. KTransformers, PowerInfer, T-MAC
  and bitnet.cpp don't fit a dense 27B hybrid on AMX. ik_llama.cpp has faster AVX-512 kernels but no AMX.
  OpenVINO added a paged GDN op in 2026 that might fix E10's serial prefill, but it is unlikely to beat vLLM.
- **Models better suited to CPU: fewer active parameters is the lever.** At 64 sequences the decode step is
  ~0.29 s of weight reads + ~13 ms per sequence, and prefill is about half of each request. Cost tracks active-parameter
  FLOPs. Candidates:
  Qwen3.6-35B-A3B (MoE, 3B active, same GDN-hybrid family; estimated $0.8–1.1/M INT8, $1.2–1.7/M BF16),
  Qwen3.5-9B (dense, the same vLLM code path as the 27B; ~$1.0–1.2/M INT8), and Gemma 4 26B-A4B
  (MoE hedge; ~$0.9–1.4/M). **Phi-4 is dominated**: it is from Dec 2024, scores 70.4 on MMLU-Pro, has a 16k context
  and, as a 14B dense model, would cost ~$1.7/M. Every candidate needs an E16-style quality gate.
- **Diffusion LMs (DiffusionGemma): unlikely to help batch CPU work.** DiffusionGemma-26B-A4B (released 2026-06,
  Apache-2.0, 3.8B active, 256-token canvas) has a vLLM CPU recipe for Xeon 6. Its advantage is latency at low
  concurrency (memory-bound decode). It spends ~25× more FLOPs per generated token, and on a FLOP-limited CPU
  that already batches 64+ requests it is estimated at $2.7–3.9/M, worse than an autoregressive model of the same size.
  The dedicated diffusion topic is still running.
- **GPUs: none available** (capacity checks below).

## Capacity checks (ARM usages, all 63 regions, 2026-10-04)

- **GPUs: none usable.** Every current GPU family has a quota limit of 0 in every region:
  NCasT4_v3, NVadsA10_v5, NCadsA10_v4, NCadsA100_v4, NDasv4/NDamsv4 A100, NCadsH100_v5,
  NDsH100_v5, ND H200, ND MI300X, GB200/GB300, RTX PRO 6000, NVadsV710_v5, NCsv3 and others.
  Only these have nonzero limits: `standardNCFamily`/`NCPromo` (K80) and `standardNVFamily`/`NVPromo`
  (M60), all retired with no SKUs offered; `standardNVSv4Family` (4 vCPU: NV4as_v4 = 1/8 of an
  AMD MI25, 2 GiB, useless for a 27B model); and `standardNPSFamily` (Xilinx FPGA). A quota increase is
  a subscription-level request that this sandbox can't make. The account owner can try it in the
  portal, but Visual Studio subscriptions are often restricted.
- **HBM CPUs: none.** HBv3/HBv4/HBv5, HX, HC and the M-series families all have limit 0.
- **AMD Turin (v7) is available**: Dasv7, Dadsv7, Dalsv7, Easv7, Eadsv7, Fasv7, Fadsv7, Famsv7 and
  others each have 20 vCPU per region, like the Intel families. No AMX, but more physical cores
  per vCPU on some families and 12-channel DDR5. This is a hardware option to test.
- Intel families with quota: Dsv6/Ddsv6/Dlsv6/Dldsv6, Dsv7/Ddsv7/Dlsv7/Dldsv7, Edsv6, Edsv7
  (20 vCPU each per region).

## Queue

Priorities: P0 = next, best expected $/M reduction per VM-hour. Costs at $1.326/h (E16ds_v6) unless noted.

| ID | Pri | Idea | Hypothesis / expected effect on $/M | Cost to test | Status |
| --- | --- | --- | --- | --- | --- |
| R01 | – | Round 2: E16 quality, E17 scaling/W4A16/threads/MTP, E18 v6 | see the READMEs | done | done |
| R09 | P0 | **Qwen3.6-35B-A3B (MoE, 3B active) on vLLM CPU** (E19): BF16, then INT8/FP8 if available; 64–128 prompts | $2.39 → ~$0.8–1.4/M (2–3×) if MoE kernels are efficient | ~2.5 VM-h | queued |
| R10 | P0 | **Qwen3.5-9B dense on vLLM CPU** (E19b), BF16 and a self-made W8A8 | $2.39 → ~$1.0–1.5/M; lowest technical risk (same code path) | ~1.5 VM-h | queued |
| R11 | P0 | Quality gate for R09/R10 (E20): E16 harness (GSM8K, MMLU) + IFEval subset, thinking off | decides whether the 2–3× is usable | ~1 VM-h per model | queued |
| R12 | P0 | Larger batches: 128–256 sequences with the KV cache sized to fit (W8A8, v6) | −7 to −13% ($2.39 → ~$2.1–2.2) | ~1.5 VM-h | queued |
| R13 | P0 | BF16 GDN state (`--mamba-ssm-cache-dtype bfloat16`): FP32 state is 154 MB per sequence | −4 to −8%, 2× sequences per GiB; needs an E16-style accuracy check | ~1.5 VM-h | queued |
| R03 | P0 | Shared-prefix workloads with prefix caching | 0% for 512-token prompts (vLLM's 896-token hybrid block); −35 to −50% when the shared instruction is ≥ 896 tokens | ~1.5 VM-h | queued |
| R14 | P1 | Output budget: structured/JSON output, max_tokens, thinking off (Qwen3.8 thinks by default) | halving output length: −19% per M tokens, −27% per request | ~0.5 VM-h | queued |
| R15 | P1 | MTP k=1 on real 512/128 text at 64–128 sequences, stacked with R12/R13 | −5 to −8% alone, ~−16% stacked | ~1 VM-h | queued |
| R16 | P1 | Profile one decode step at 64 sequences (where do 13 ms per sequence go?) | no direct gain; decides between kernels, state dtype and batch size | ~0.5 VM-h | queued |
| R02 | P1 | SGLang CPU backend vs vLLM, same VM (E14) | 0.85–1.3× of vLLM; mostly matters for BF16/FP8 | ~2.5 VM-h | queued |
| R17 | P1 | DiffusionGemma-26B-A4B vs its autoregressive twin Gemma 4 26B-A4B on vLLM CPU | AR twin ~$0.9–1.4/M; diffusion likely worse at batch; answers the user's question with data | ~2.5 VM-h | queued |
| R18 | P2 | Official Qwen3.8-27B-FP8 (block FP8, W8A16 kernel) as a quality-safe 8-bit option | ~$2.9–3.3/M, probably BF16-level accuracy; only if W8A8's MMLU loss matters | ~1.5 VM-h | queued |
| R19 | P2 | DFlash block-diffusion speculative decoding for Qwen3.8-27B in vLLM CPU | −0 to −20% if supported on CPU | ~1.5 VM-h | queued |
| R20 | P2 | n-gram / suffix speculation for copy-heavy extraction | 0 to −16% on extraction jobs | ~1 VM-h | queued |
| R21 | P2 | Runtime knob bundle (THP, tcmalloc, no prefix caching when unused, AOT compile-cache fix) | 0 to −8% | ~0.7 VM-h | queued |
| R22 | P2 | Prompt compression (LLMLingua-2) for long unique inputs | −22% per request, quality risk | ~1 VM-h | queued |
| R05 | P2 | AMD Turin (Easv7/Fasv7) with vLLM CPU | pending the hardware topic | ~2 VM-h | queued |
| R08 | P1 | Managed per-token APIs (Azure AI Foundry, Azure OpenAI Batch) as the price to beat | pending the hardware topic | desk research | queued |
| R07 | P3 | Ollama sanity run (num_parallel forced to 1 for qwen35) | confirms ~$28–37/M; only to close the question with data | ~0.5 VM-h | optional |
| R23 | P3 | OpenVINO nightly with the paged GDN fusion fix + full-INT8 IR | $19–34 → maybe $8–12/M; unlikely to win | ~2 VM-h | parked |
| R24 | P3 | ik_llama.cpp refresh (no AMX, faster AVX-512, MTP) | ~$10–15/M; no win expected | ~1.5 VM-h | parked |
| R25 | P3 | Watch list: SGLang CPU MTP/DFlash and INT4 AMX routing; Qwen3.8-Flash-Next (6B active) CPU support | – | – | watch |

