# Research plan and work queue

Seed written by the orchestrator on 2026-10-04 13:45 UTC, before the literature-research workflow
finished. The workflow's synthesis replaces and extends this file. If it never lands (session out
of tokens), resume from the queue below.

How to use the queue: take the top `queued` row, set it to `in-progress` with the experiment ID,
and record the result and link when done. Add new ideas at the bottom with the next R-number.

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

## Queue (seed)

| ID | Idea | Hypothesis | Status |
| --- | --- | --- | --- |
| R01 | vLLM batch scaling, W4A16, 16 threads, MTP (E17); vLLM on v6 (E18); W8A8/W4A16 accuracy (E16); MTP on GSM8K text | see the experiment READMEs | in-progress |
| R02 | SGLang CPU backend with AMX kernels (E14) | beats vLLM prefill if it supports the hybrid GDN architecture | queued |
| R03 | Prefix caching for batch jobs with a shared instruction prompt | a 400-token shared prefix out of 512 cuts prefill cost ~4×; blended $/M −40–60% | queued |
| R04 | CPU-friendlier models (Phi-4 14B, MoE with ~3B active such as Qwen3-30B-A3B or gpt-oss-20b) | 2–5× cheaper per token at some quality cost; needs a task-quality check like E16 | queued |
| R05 | AMD Turin (Easv7/Fasv7) with vLLM CPU | decode-heavy jobs cheaper per token than E16ds_v7; prefill slower without AMX | queued |
| R06 | vLLM CPU runtime knobs: tcmalloc, Intel OpenMP, fixing the AOT compile cache | 5–15% throughput, ~3 min less startup per run | queued |
| R07 | Ollama | no better than llama.cpp (same ggml kernels); low priority, sanity check only | queued |
| R08 | Managed per-token APIs (Azure AI Foundry, Azure OpenAI Batch) as the price to beat | self-hosted CPU at $4.6/M on demand is not competitive with small managed models | queued |
