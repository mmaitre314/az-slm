# Research plan and work queue

Last updated **2026-10-05**, after round 2, seven literature-research topics (Ollama, other CPU
serving stacks, CPU-friendly models, stack-agnostic techniques, diffusion LMs, hardware/GPU/managed
APIs, CPU-inference literature) and a review pass (memory fit on 64 GiB, stacking arithmetic,
managed-API comparability, quality-gate power, task-specific alternatives, R33–R44 added).

This file answers the user's open questions and holds the durable **work queue** of ideas to test.
It is written so that work can resume from this file alone: the baseline, the arithmetic, the
queue, the chains to run next and the open questions are all here.

- Raw findings (seven topics; every claim with sources and a confidence level):
  [research/findings-2026-10-04.json](research/findings-2026-10-04.json).
- Status board and decision log: [PLAN.md](PLAN.md). Cost formulas: [COST_MODEL.md](COST_MODEL.md).
  Measured results and the current recommendation: [E15-summary](E15-summary/README.md).
- **How much to trust this**: most external claims come from the researchers' web searches. Many
  primary pages (huggingface.co, arxiv.org, vllm.ai, lmsys.org, intel.com, amd.com and others) were
  blocked from the sandbox, so their figures come from search-result snippets. PR numbers, merge
  states, release dates and benchmark scores need a spot check before they drive a decision.
  The Azure VM, disk and Foundry prices in this file were re-checked against the Retail Prices API on
  2026-10-04. Every $/M figure is labelled **measured** (our experiments) or **est.** (an estimate,
  with its arithmetic or source); estimates are never measurements.

## Baseline and how to use the queue

**Baseline (measured, round 2)**: vLLM 0.31.0 CPU
(`vllm/vllm-openai-cpu@sha256:8024248339dc6878daa5349344ed29d49c8a6732f6bdf7400fda32ce33e4b30b`,
torch 2.13.0+cpu; E18's `raw/vllm-image.txt`), community INT8 W8A8
(`Avesed/Qwen3.8-27B-INT8-W8A8`), 64 prompts, 8 threads, KV cache 24 GiB, E16ds_v6 at $1.326/h
all-in: prefill 263 tok/s, decode 54.3 output tok/s, mixed 512/128 154.1 tok/s total, **$2.39/M
blended** on demand ($1.53 per 1,000 requests of 512/128), $0.47/M at Spot prices
([E18](E18-vllm-emerald-rapids/README.md)). The same configuration on
an E16ds_v7 instance: $3.47/M ([E17](E17-vllm-scaling-mtp/README.md)). Quality: W8A8 equals BF16 on
GSM8K (92.5%) and loses 2.75 points on MMLU (90.0 → 87.25%, paired p = 0.035)
([E16](E16-task-quality/README.md)). MTP with 1 draft token: 96% acceptance and +16% output tok/s
on GSM8K at 64 sequences (E17 follow-up).

Anatomy of one 512/128 request at 64 in flight on E16ds_v6, used for the arithmetic in this file:

- Wall time per request: 640 / 154.1 = **4.15 s**. Prefill 512 / 263 = 1.95 s (~45%); decode
  128 / 54.3 = 2.36 s (~55%). The two sum to 4.31 s, more than the measured 4.15 s, because vLLM
  overlaps the phases a little.
- A decode step takes **≈ 0.29 s + 13.9 ms × (sequences in flight)** (fit to the E17/E18 decode
  runs; 1.18 s at 64). The 0.29 s is reading ~28 GB of INT8 weights at ~95 GB/s, the bandwidth of an
  8-core slice. The per-sequence 13.9 ms dominates at 64 sequences: decode at 64 is the same on v6 and
  v7 (54.3 vs 55.2 output tok/s) although their prefill differs by 34%.
- $/M = all-in $/h ÷ (total tok/s × 3600) × 1e6. At E18's 154 tok/s that is **$/h ÷ 0.554**.
- **Two metrics from round 3 on.** $/M tokens misranks levers that change the token count (R14, R22,
  R34, R35) and comparisons across tokenizers (Qwen 248k vocab, Gemma 262k, Phi and the OpenAI
  tokenizers in (e)). The primary metric is **$ per 1,000 requests of a stated workload**
  (baseline: 4.15 s of VM time per 512/128 request × $1.326/h = **$1.53 per 1k requests**); $/M tokens
  is reported next to it. R03, R14, R22, R34–R36 and every cross-model or managed comparison are
  ranked by $/1k requests.
- **Job-level cost** (from E18's `chain.log`; not yet in [COST_MODEL.md](COST_MODEL.md), which
  assumes 100% utilization): per VM, boot and cloud-init ~5 min, llama-bench reference 3.5 min, image
  pull and model download 2.5 min, ~2.4 min of engine start and compile per vLLM launch, 12 min for
  the 3-prompt correctness check, and an idle tail of ~1 h before the watchdog acts (~$1.33 on
  E16ds_v6; ~$20 of round 1's $50). Without the idle tail a job pays ~15 min (~$0.33) of fixed
  cost, so fixed cost stays ≤ 10% for jobs of ≥ ~2.5 h (~2,200 requests of 512/128); with the
  idle tail it takes ≥ ~12 h. Chains therefore end with a self-deallocation step (see the chains).

**Queue rules**

1. Take the top row with status `queued` whose prerequisites are met. Rows are sorted by priority,
   then by **expected value per VM-hour = (expected gain × probability of success) ÷ VM-hours**,
   with the `Conf.` column high / medium / low = 0.8 / 0.5 / 0.2. Rows that switch models count at
   about half their gain, because they need the R11 quality gate and the user's acceptance. Rows
   that cost 0 VM-hours come first within their priority; a gate (R11, R16) inherits the value of
   the rows it unlocks and runs with them.
2. Set it to `in-progress`, give it an experiment ID (add the PLAN.md row and an
   `EXX-<slug>/README.md` from [TEMPLATE.md](TEMPLATE.md); E19–E27 are already mapped below), and run it
   as one self-contained background chain that saves results after every step ([README.md](README.md)).
3. When it is done, set `done` or `rejected`, write the measured result in the row
   ("**measured**: ..."), link the README, and re-check the rows whose estimates depend on it (named
   in their prerequisites). A row is `done` only if its effect exceeds 2× the coefficient of
   variation (CV) measured in that chain's prologue; otherwise it is `inconclusive` (re-run or drop).
4. Add new ideas at the bottom of the queue with the next R-number (next free: **R45**). Never
   renumber; retired IDs stay retired.

Status values: `queued`, `in-progress`, `done`, `inconclusive`, `rejected`, `blocked` (waiting for a
decision, a trigger or an upstream change; the row says what unblocks it). Priorities: P0 = next
round; P1 = after P0, or in parallel in another region; P2 = when cheap to append or when a trigger
fires; P3 = record only.

**Current state (2026-10-05)**: rounds 1–2 are done (~$70 spent this month), all VMs are deleted,
nothing is running. Next: ask the user R33's questions, then the P0 chains below. Round-3 budget and
stop rule: [PLAN.md](PLAN.md) → Constraints.

### Suggested chains for the next round

Quota allows one 16-vCPU VM per region, so chains run in parallel in different regions. None of the
new chain profiles exists yet: `bench/vllm_chain.sh` has only the `e17`, `e18` and `smoke` profiles,
so each chain below lists the scripting it needs before its VM is deployed.

**Shared prologue and epilogue (every chain; ~2 h of scripting once, then reused)**

- **Pin the engine**: pass `-e IMAGE=vllm/vllm-openai-cpu@sha256:8024248339dc6878daa5349344ed29d49c8a6732f6bdf7400fda32ce33e4b30b`.
  Every script (`vllm_setup.sh`, `vllm_bench.sh`, `vllm_chain.sh`, `e16_chain.sh`) defaults to the
  floating `latest-x86_64` tag, so an unpinned chain silently runs whatever vLLM is current and mixes
  engine versions into every "vs $2.39" comparison. Newer releases are tested on purpose (R38).
- `meta`, `ref` (5-minute llama-bench), then **`probe`** (~5 min on a VM the chain already pays for, output < 4 KiB): in the
  pinned image, `vllm serve --help` for `--mamba-ssm-cache-dtype`, `--enable-mamba-fine-grained-prefix-cache`,
  `--mamba-cache-mode` and `--async-scheduling`; grep the installed `vllm` package for sgl-kernel
  sources and `VLLM_CPU_SGL_KERNEL`; `nm -D` on `vllm/_C*.so` for the INT8 scaled-mm, fused-MoE and
  W4A8 symbols; print the `cpu_moe` activation list (GELU_TANH, R04); look for
  `CPUFp8BlockScaledMMKernel` (R18) and the DiffusionGemma CPU fixes (R17). This settles the
  upstream claims marked *unverified* below.
- **Reference ×3**: the W8A8 Qwen3.8-27B 512/128 × 64 mixed run three times (KV 16 GiB on 64 GiB
  sizes, 24 GiB otherwise), giving the run-to-run CV (never measured: every E17/E18 configuration was a
  single run). A/B pairs run interleaved (A B A B) on the same VM. Instances of the same size differ by
  9–25% (E08, E18), so cross-chain comparisons go through this reference. Once R40 has data: redeploy
  if the reference is > 8% below the best seen on that size and the chain is longer than ~2 h.
- **Memory**: log min `MemAvailable` (`/proc/meminfo`, every 5 s) and the container's cgroup
  `memory.peak` (`bench/vllm_bench.sh` today samples only the largest process's RSS; ~30 min to add).
- Sweeps over batch sizes use `bench/vllm_sweep.py` (one engine start per model and configuration)
  instead of `vllm_bench.sh`, which restarts the container and engine for every workload (~2.4 min each).
- **Epilogue**: `save_results.sh` after every step; a final `release` step that requests deallocation
  through the VM's managed identity (the watchdog's code path; ~20 min to script), saving the ~1 h idle
  tail; then `harvest --release` and `teardown` from the sandbox.
- **Stop rule**: if a chain exceeds 1.5× its VM-hour estimate, stop it, harvest and re-plan.

**Commands** (`<vm>` is a new name per deployment; `-p location=<region>` picks the region):

```
python3 scripts/deploy.py deploy infra/main.bicep --name <vm> -p vmName=<vm> -p location=<region> \
    -p vmSize=Standard_D16s_v6 -p osDiskSizeGB=128 -p priority=Regular
python3 scripts/deploy.py push <vm>
python3 scripts/deploy.py run <vm> -c 'bash /opt/azslm/bench/vllm_chain.sh' -e PROFILE=e21 \
    -e IMAGE=vllm/vllm-openai-cpu@sha256:8024248339dc6878daa5349344ed29d49c8a6732f6bdf7400fda32ce33e4b30b --background chain
python3 scripts/deploy.py job <vm> chain
python3 scripts/deploy.py start <vm>     # only if the chain deallocated itself
python3 scripts/deploy.py harvest <vm> experiments/E21-<slug>/raw --release
python3 scripts/deploy.py teardown --run <vm>
```

**Done criterion (every chain)**: every listed configuration has a non-failed row in `vllm.jsonl`
(or a recorded OOM/failure with its log line), the EXX README states the measured $/M, $/1k
requests and the prologue CV, the queue rows and PLAN.md are updated, results are committed, and
the VM is deleted.

| chain | VM | rows | profile to write (scripting effort) | experiments | est. VM-hours |
| --- | --- | --- | --- | --- | ---: |
| A: same model, cheaper size | D16s_v6 (or D16ds_v6), `-p osDiskSizeGB=128` | R26 step 2 (KV 16 GiB only), then R21's knobs at 64 sequences | `PROFILE=e21`: e18's W8A8 steps with KV_GB=16, the memory sampler, the knob runs (~1 h) | E21 | 2.5–3 |
| B1: CPU-friendlier models, throughput | E16ds_v6 (2 × 440 GiB local NVMe holds all ~200 GB of models; E16s_v6 has no local disk: there use `-p osDiskSizeGB=256` and delete each model after its steps, and expect ~10 min per engine start re-reading ~70 GB from a P15/P20 disk) | R10 (RedHatAI W8A8 first), R09 (BF16, AMD W8A8, FP8; real-text prompts), R29 as a 10-minute add-on | `PROFILE=e19`: downloads, one `vllm_sweep.py` engine per model and variant, real-text prompt files (~2 h) | E19 | 5–6 (incl. ~0.5 h of downloads and engine starts) |
| B2: quality gate, stage 1 | E16ds_v6 in another region | R11 stage 1 (27B W8A8 reference + candidates) | `PROFILE=e20` on `e16_chain.sh`; ifeval/json/classify builders in `bench/quality_tasks.py` (~2–3 h) | E20 | 4.5–5 |
| C: decode tuning | E16ds_v6 (128 GiB) | R26 step 1 (KV 16 vs 24 GiB at 64 prompts), R14, R34 if classification, R13 (incl. real-text GSM8K), R12, R16, then R15 | `PROFILE=e22`: `EXTRA='--max-num-seqs N'` runs, real-text harness reuse, profiler step (~1.5 h) | E21, E22, E24 | 6–7 |
| D: no VM | – | R33 (ask the user), R27 (ARM quota probes), R30 (owner decision), R25 watch-list re-check, R38 release check | – | – | 0 |

P0 total ~18–21 VM-hours (~$22–27 at v6 prices, with self-deallocation). After that, P1 in
expected-value order: R34/R03/R35 (if R33(a) makes them relevant), R40, R37, R38, R04 + R17 (E25),
R15 if not in chain C, R05 (E26), R02 (E14), R39.

## Answers to the user's questions

### (a) Is Ollama competitive? No.

**Verdict: not competitive. Close R07 on desk evidence** (an optional 0.5 VM-hour smoke test stays at P3).

- Since v0.30.0 (released ~2026-06-01/02; release trackers say June 2) Ollama on Linux has no engine of its own. It starts upstream llama.cpp
  `llama-server` ([commit 9db4bdb](https://github.com/ollama/ollama/commit/9db4bdbad6a4981ad761aa2b603e69e8fb83212c),
  #16031), pinned to b11232 (2026-09-28, [LLAMA_CPP_VERSION](https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/LLAMA_CPP_VERSION)).
  Its Linux CPU build includes ggml's `sapphirerapids` variant (AMX_TILE, AMX_INT8), so on our VMs it
  uses the AMX path that ran at ~3% of AMX peak ([E07](E07-prefill-profile/)) and corrupts
  multi-sequence decode for this architecture ([E04](E04-llamacpp-amx-multiseq-bug/)).
- The decisive fact: Ollama's scheduler forces `numParallel=1` ("model architecture does not
  currently support parallel requests") for the qwen35, qwen35moe and qwen3next families, whatever
  `OLLAMA_NUM_PARALLEL` says ([sched.go](https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/server/sched.go#L502-L515)).
  The list is the same in v0.30.0, v0.35.1 (2026-10-01) and main (2026-10-02). Qwen3.8 maps to
  `qwen35` ([create/metadata.go](https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/create/metadata.go)),
  so requests are served one at a time; users report it in
  [#14510](https://github.com/ollama/ollama/issues/14510) and [#14879](https://github.com/ollama/ollama/issues/14879).
- Est. cost from our single-sequence measurements ([E03](E03-llamacpp-single-stream/), E16ds_v7:
  Q4_K_M with AMX, prefill 28.7 and decode 3.88 tok/s): 512/28.7 + 128/3.88 = 50.8 s per request,
  $37/M at the v7 price ($1.681/h). On the baseline's E16ds_v6 price (× 0.79, same throughput
  assumed): **~$29/M**; with the MTP head that Ollama turns on when the GGUF contains it
  ([E12](E12-speculative-decoding/): 1.55× decode) **~$23/M**; with `num_thread=16` ([E08](E08-threads-smt/))
  **~$20–21/M**. That is **9–12× the $2.39/M baseline** and no better than llama.cpp batched
  ($19.3/M on v6, $23.9 on v7, E15). Whether the `qwen3.8:27b` library tag carries the MTP tensors is
  unverified.
- Patching the limit out only gives llama-server `-np N` with Ollama's defaults. With the AMX
  variant loaded that produces wrong output (E04) unless `libggml-cpu-sapphirerapids` is deleted, which
  is llama.cpp no-AMX batched again. Published Ollama vs vLLM comparisons are GPU-only and agree
  (Red Hat, A100: 793 vs 41 tok/s peak; snippet).

### (b) Models better suited to CPUs (Phi-4 and others)

**Verdict: fewer active parameters is the lever. Test Qwen3.6-35B-A3B and Qwen3.5-9B (P0) behind a
quality gate (R11). Phi-4 and the other "CPU-friendly" Microsoft models (Phi-4-mini,
Phi-4-mini-flash-reasoning, BitNet b1.58 2B4T) are dominated or far below the bar.**

Why: at 64 sequences a decode step is ~0.29 s of weight reads plus ~13.9 ms per sequence, and
prefill is ~45% of each request. Both scale with FLOPs per token (≈ 2 × active parameters), not
with total parameters. For the 27B, larger batches alone floor at ~$3.1/M on the E17 instance (est.).
An MoE still reads nearly all its experts every decode step at our batch sizes (with 8 of 256
experts per token, 87% of experts are touched at 64 sequences and 98% at 128, assuming uniform
routing; [arXiv 2511.02237](https://arxiv.org/abs/2511.02237)), so its saving comes from compute
(prefill and the per-sequence term), not from bandwidth. Total parameters matter for the 128 GiB limit.

Estimates by the models researcher, calibrated on E17 (E16ds_v7, $1.681/h; the same model predicts
$3.54/M for the 27B W8A8 at 64 prompts against $3.47 measured), 512/128 at 128 prompts. Scores are
published **thinking-mode** numbers; our jobs run non-thinking. On E16ds_v6 multiply the $/M by
~0.69 (the measured v6/v7 ratio, 2.39/3.47); on D16s_v6 (64 GiB) by ~0.44 and on D16ls_v6 (32 GiB,
$0.744/h all-in, small models only) by ~0.39, if the model fits and throughput holds (R26, R41).

| model | type, active params | license | MMLU-Pro / GPQA Diamond / IFEval | est. $/M (range, mid) | vs 27B W8A8 | notes |
| --- | --- | --- | --- | --- | ---: | --- |
| Qwen3.8-27B (current) | dense GDN hybrid, 27B | – | 84–86 (third-party) / 89.2 (model card; 90.5 Artificial Analysis) / IFBench 79.5 | **3.47 measured** (W8A8, 64 prompts) | 1× | |
| **Qwen3.6-35B-A3B** | MoE GDN hybrid, ~3B (256 experts, 8 + 1 shared) | Apache 2.0 | 85.2 / 86.0 / – | INT8 0.76–1.11 (0.92); FP8-W8A16 0.94–1.45 (1.16); BF16 1.20–1.70 (1.42) | 2.0–4.6× cheaper | same family and tokenizer; official FP8; AMD publishes an RTN W8A8 (35.1 GiB, EPYC-validated, Intel path unverified; snippet); BF16 (~70 GB) needs a 128 GiB VM, the W8A8 fits 64 GiB; R09 |
| **Qwen3.5-9B** | dense GDN hybrid, 9B | Apache 2.0 | 82.5 / 81.7 / 91.5 | W8A8 0.96–1.24 (1.05); BF16 1.39–1.80 (1.53) | 1.9–3.6× | the exact code path that already works; lowest risk; `RedHatAI/Qwen3.5-9B-quantized.w8a8` exists (snippet); R10, R41 |
| Qwen3.5-4B | dense hybrid, 4B | Apache 2.0 | 79.1 / – / – | W8A8 0.60–0.79 (0.66) | 4.4–5.8× | quality drop likely too large |
| Gemma 4 26B-A4B | MoE, 3.8B (128 experts, top-8) | Apache 2.0 | 82.6 / 82.3 / – | INT8 0.72–1.11 (0.89); BF16 1.11–1.69 (1.36) | 2.1–4.8× | cross-family hedge and DiffusionGemma's autoregressive twin; R04 |
| gpt-oss-20b | MoE, 3.6B | Apache 2.0 | MMLU 85.3; GPQA 71.5 (high effort), 56.8 (low) | BF16 1.02–1.65 (1.30), before reasoning tokens | 2.1–3.4× | always emits a reasoning channel (1.5–3× more output) |
| Phi-4 | dense, 14B | MIT | 70.4 / 56.1 / 63.0 | W8A8 1.67–1.92 (1.74) | 1.8–2.1× | Dec 2024, 16k context: **dominated** by Qwen3.5-9B (cheaper and 12–28 points better) |
| DiffusionGemma 26B-A4B | MoE block diffusion, 3.8B | Apache 2.0 | 77.6 / 73.2 / 94.5 (no thinking) | 1.1–7.7 (see (f)) | ≤ its AR twin | |

Considered and dropped on desk evidence (R32): Qwen3.8-Flash-Next (176B total including a 51B n-gram
embedding, 6B active; its GPQA 91.7 is unverified, not found in the cited sources. At 4.0–4.85 bits
per weight it is 88–107 GB, so it would fit a 128 GiB VM (137 GB) only if the n-gram table is
quantized too; with the table in BF16 it is ~170 GB. It has no mainline CPU support, so it is on the
R25 watch list), Qwen3.5-122B-A10B (doesn't fit at INT8; INT4 is slow on
vLLM CPU), Nemotron 3 Nano 30B-A3B (MMLU-Pro 78.3, GPQA 73.0), Granite 4.x, LFM2, Mistral Small 4
(119B total) and ERNIE-4.5-21B-A3B (scores unverified). No Qwen3.8 MoE in the 30–40B range exists, only
a [community request](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/120). A "Phi-5" appears only on
third-party hosting blogs; no Microsoft source was found. Microsoft's other CPU-marketed models
(scores from search snippets of their model cards and reports): Phi-4-mini (3.8B dense, MMLU-Pro
52.8, GPQA 25.2) and Phi-4-mini-flash-reasoning (3.8B SambaY hybrid built for long generations,
GPQA Diamond 45.1; it always emits reasoning tokens, a poor fit for short extraction) are dominated
by Qwen3.5-4B (MMLU-Pro 79.1) at a similar active size. BitNet b1.58 2B4T (MMLU 53.2, GSM8K 58.4;
[report](https://arxiv.org/pdf/2504.12285)) is far below the quality bar; its bitnet.cpp kernels
are covered in R32.

Quality risk: every published score above is thinking mode. For extraction and classification the
risk is instruction and format adherence in non-thinking mode. Qwen3.8's largest gain over Qwen3.6
is instruction following (IFBench 79.5 vs 69.1 for Qwen3.6-27B), and within Qwen3.5 the 3B-active MoE
and the 9B dense lose ~3–3.5 IFEval points to the 27B (95.0). R11 gates any switch with the E16
harness plus IFEval, a JSON-extraction set and a classification set, with enough items to show
non-inferiority (see R11). INT4 is not ruled out by
quality (Q4_K_M/AWQ of Qwen3.8-27B keep ~97%, [Quesma](https://quesma.com/blog/qwen38-27b-quantizations-benchmarked/))
but by vLLM CPU's slow W4A16 kernels (E17).

### (c) Other optimization techniques

**Verdict: the biggest same-model step is a cheaper size of the same CPU (D16s_v6, est. $1.51/M,
−37%), not software, if the 64-prompt job fits in 64 GiB: unmeasured, since the $2.39 baseline ran
with a 24 GiB KV cache and its largest process peaked at ~65 GB (R26). The decode levers don't all fit
on that size: on D16s_v6 at 64 sequences a BF16 GDN state plus MTP-1 is est. ~$1.36/M (−10% per
request); on 128 GiB sizes larger batches, a BF16 state and MTP stack to est. ×1.3 decode, −13%
(~$2.07/M on E16ds_v6, ~$1.70/M on E16s_v6). Shorter outputs cut cost per request, and **for
classification jobs, task-specific alternatives (one constrained label token, R34; a distilled
encoder, R35) dominate every engine-level lever**. Our stack matches Intel's MLPerf v6.0 result per
core; Intel's v6.1 software is ~1.5× faster on Llama-3.1-8B, so ~1.2–1.3× per-core headroom from
newer software is plausible (R29, R38), and the remaining same-model headroom (est. 1.3–1.6×) is in
decode's per-sequence work.**

| technique | est. effect (512/128, vs $2.39/M or $1.53 per 1k requests) | evidence | queue |
| --- | --- | --- | --- |
| Same CPU, cheaper size: D16s_v6 ($0.806/h, 64 GiB) or D16ds_v6 ($0.997/h, 64 GiB) | **−37% ($1.51) / −24% ($1.83)** if throughput holds and the job fits in 64 GiB | Dsv6, Ddsv6 and Edsv6 all run the Xeon Platinum 8573C ([Learn](https://learn.microsoft.com/azure/virtual-machines/sizes/general-purpose/dsv6-series)). Memory is unproven: the 64-prompt baseline (24 GiB KV) peaked at 65.1–65.7 GB in its largest process alone (~61 GiB); the 56–57 GB peaks were 16-prompt runs at 16 GiB KV. 16 GiB of KV is ~293 blocks of 896 tokens and 64 sequences need ~256 (87%) | R26 |
| 128–256 sequences (128 GiB sizes only) | −7 to −13% | decode step model above; measured +34% from 16 to 32 and +16% from 32 to 64 (E17); needs > ~20 GiB KV | R12 |
| BF16 GDN state (`--mamba-ssm-cache-dtype`) | −4 to −8% if output length holds; ~1.4–1.7× sequences per GiB for 512/128 | the FP32 state is ~154 MB per sequence (vLLM log: 896-token attention block, mamba page padded 14.43%); a plain low-precision cast lengthened outputs by up to ~40% in NVIDIA's Nemotron 3 Super report | R13 |
| MTP, 1 draft token | −8% at 64 sequences, ~−5% at 128, ~−3% at 256 (gain shrinks with batch) | **measured** +16% decode on GSM8K at 64 sequences (E17 follow-up); the extra position costs ~12.7 ms per sequence | R15 |
| Stack on D16s_v6 (64 GiB): 64 sequences, BF16 state + MTP-1 | decode × ~1.22, −10% per request: **~$1.36/M** (est.) | 1.05 × 1.16; MTP at 64 sequences needs ≥ 19.5 GiB KV with FP32 state (too tight on 64 GiB), est. ~14 GiB with BF16 state, so this branch needs R13 to pass | R26, R13, R15, R37 |
| Stack on 128 GiB sizes: 128 sequences, BF16 state, MTP-1 | decode × ~1.32, −13%: **~$2.07/M** on E16ds_v6, **~$1.70/M** on E16s_v6 ($1.088/h) (est.) | 1.14 × ~1.05 × ~1.10; per request 1.95 + 2.36/1.32 = 3.74 s vs 4.31 s | R12, R13, R15, R37 |
| Shorter outputs (JSON, terse prompts, thinking off) | halving output: −27% per request, −19% per M tokens | output tokens cost ~5× input tokens ($6.78 vs $1.40/M on v6, E18) | R14 |
| Classification as one constrained label token (guided choice or max label logprob) | −53% per request ($1.53 → ~$0.72 per 1k) | prefill-only: 512/263 = 1.95 s vs 4.15 s per request | R34 |
| Distil the 27B's labels into a small encoder classifier (or embeddings + a linear head) | est. 30–70× cheaper per request after a one-off labelling cost (**low confidence**) | a ~0.4B encoder needs ~0.4/27 of the 27B's prefill FLOPs; labelling 5k items costs ~$7.7 at the baseline | R35 |
| LoRA SFT of a 4B model on the 27B's extraction outputs | est. −80% for extraction (**low confidence**) | Qwen3.5-4B W8A8 est. ~$0.46/M on E16ds_v6 | R36 |
| Prefix caching | 0% for unique 512-token prompts; −35 to −50% per request when the shared prefix is ≥ the attention block (896 tokens with FP32 state; est. 448–512 with BF16 state) or with fine-grained mode | hybrid models cache per block ([vLLM #40696](https://github.com/vllm-project/vllm/issues/40696), [#45238](https://github.com/vllm-project/vllm/issues/45238), [PR #46384](https://github.com/vllm-project/vllm/pull/46384)) | R03 |
| Remove part of decode's per-sequence overhead | up to −13 to −28% if half of the 13.9 ms can go | GEMM (~3.4–3.8 ms), FP32 state traffic (~3.2 ms) and attention (< 0.5 ms) explain only half of it; async scheduling and two engines per VM if it is CPU-side overhead | R16, R42, R43 |
| SGLang CPU backend | 0.85–1.3× of vLLM ($1.84–2.81) | *unverified* (snippets; github.com blocked): vLLM vendors SGLang's CPU kernels ([PR #50387](https://github.com/vllm-project/vllm/pull/50387)) and sends W8A8 to its AMX INT8 kernel ([PR #50801](https://github.com/vllm-project/vllm/pull/50801)). E18's log shows `CPUInt8ScaledMMLinearKernel` and CPU GDN kernels: consistent with, not proof of, SGLang's kernels | R02 |
| AMD Turin F16as_v7 in Central India ($0.706/h, 16 full cores, no AMX) | est. $1.3–1.9/M (mid ~$1.6; low confidence) | vLLM's in-tree ZenCpuPlatform with zentorch; prefill without AMX needs AVX-512 VNNI at 50–65% of peak for 0.55–0.75× of v6; GDN without AMX untested | R05 |
| Runtime knobs (THP, prefix caching off when unused, larger chunk budget, allocator check, AOT compile cache) | 0 to −10% | Phoronix THP review; vLLM CPU docs; E17's compile-cache error | R21 |
| Newer vLLM CPU release | 0 to −20% (low confidence) | Intel: +56% Offline from software alone between MLPerf v6.0 and v6.1 | R38 |
| W4A8 (INT4 weights, INT8 AMX compute) for the dense model | −6 to −8% if a dense x86 kernel exists | vLLM has W4A8 for CPU MoE experts only; [#38064](https://github.com/vllm-project/vllm/issues/38064) reports a silent W4A16 fallback | R28 |
| n-gram / suffix speculation | 0 to −16%, copy-heavy extraction only | [suffix decoding](https://arxiv.org/abs/2411.04975) | R20 |
| Prompt compression (LLMLingua-2) | −22% per request, lossy | [LLMLingua-2](https://arxiv.org/abs/2403.12968v1) | R22 |
| Exact deduplication of identical prompts | the duplicate share, free | application-level; ask for the share in R33(a) | R33 |
| FP8 | no speedup: no AMX-FP8 before Diamond Rapids; FP8 W8A16 prefills at BF16 speed. Quality option at +20–40% cost (a better INT8 recipe, R39, is cheaper) | [Phoronix](https://phoronix.com/news/Intel-AMX-FP8-In-LLVM), [vLLM PR #41186](https://github.com/vllm-project/vllm/pull/41186) | R18 |
| Not worth VM time | FP8/INT8 KV cache (~5% of decode traffic at 512/128 for this hybrid: 64 × ~576 tokens × 64 KB = 2.4 GB vs 28 GB of weights and ~20 GB of state per step), W4A16 (**measured** 4–6× slower), 16 threads (**measured** −50%), activation sparsity, LUT kernels, archived Intel stacks | see R32 | R32 |

**How far from the state of the art?** (literature researcher; external figures from snippets)
Per core, our numbers match Intel's MLPerf v6.0 Xeon 6 result. W8A8 prefill on E16ds_v6 is
263 tok/s × 54.6 GFLOP = 14.4 TOPS, 1.8 TOPS per core, ~29% of the nominal INT8 peak at 3.0 GHz.
KTransformers' hand-tuned AMX kernels reach ~1.2 TOPS per core in INT8, and a study of real
transformer GEMM shapes averaged 35% of AMX peak ([arXiv 2507.03522](https://arxiv.org/pdf/2507.03522)).
Our mixed 512/128 run is 1.05 TOPS-equivalent per core, against ~0.84 for Intel's MLPerf v6.0 Xeon 6
Llama-3.1-8B Offline result ([Lenovo](https://lenovopress.lenovo.com/lp2414-sr650-v4-scalable-enterprise-ai-performance-in-mlperf-60);
775.98 tok/s on 172 cores, scaled by (778 + 73)/73 and 16.06 GFLOP per token). That is not the best
published result: Intel reports +56% Offline from software alone in MLPerf v6.1 on the same silicon
([Intel](https://www.intel.com/content/www/us/en/newsroom/news/data-center/intel-software-optimizations-boost-ai-inference-in-mlperf-v6-1.html)),
i.e. ~1.3 TOPS-equivalent per core, ~25% above ours, so ~1.2–1.3× per-core headroom from newer
software is plausible (R29 calibrates, R38 tests releases). AMX also runs below the nominal clock (Intel's
Xeon 600 tables: 2.0 GHz all-core AMX turbo vs 3.0 GHz non-AVX; low confidence for our parts), so
the true efficiency is higher than these percentages. Kernel headroom in prefill is ~1.1–1.3×; the
realistic same-model, same-precision headroom is ~1.3–1.6× (est. $1.5–1.85/M on E16ds_v6), mostly in decode.

### (d) GPU sanity check

**Verdict: no usable GPU on this subscription. A GPU would be the biggest hardware lever (est. 3–10×
cheaper than today's measured CPU cost of $2.39/M, ~2–6× cheaper than the best projected CPU option
of ~$1.36–1.51/M on D16s_v6; the pessimistic A100 case, $1.42/M, is at parity), but only the account
owner can try to get quota, and Visual Studio offers are usually refused.**

What the orchestrator found through ARM usages in all 63 regions (2026-10-04):

- Every current GPU family has a quota limit of **0 in every region**: NCasT4_v3, NVadsA10_v5,
  NCadsA10_v4, NCadsA100_v4, NDasv4/NDamsv4 (A100), NCadsH100_v5, NDsH100_v5, H200, MI300X,
  GB200/GB300, RTX PRO 6000, V710, NCsv3 (V100), NDs, NDv2, NVv3 and the rest.
- The only nonzero limits on GPU-like families are retired or unsuitable: `standardNCFamily`/`NCPromo`
  (K80) and `standardNVFamily`/`NVPromo` (M60), retired with no SKUs offered; `standardNVSv4Family`
  with 4 vCPU (NV4as_v4 = 1/8 of an AMD Radeon Instinct MI25, 2 GiB, useless for a 27B model); and
  `standardNPSFamily` (Xilinx FPGA).
- Quota increases are subscription-level requests, which fail from this sandbox.

What the account owner could do:

1. Request GPU quota in the Azure portal (the Quotas page, or a support request for subscription
   limits). Microsoft Q&A answers say Visual Studio and trial offers don't get GPU quota increases,
   and even new pay-as-you-go accounts are often refused for insufficient payment history
   ([Q&A](https://learn.microsoft.com/answers/a/12302701), [Q&A](https://learn.microsoft.com/answers/a/12957599)).
2. Move the work to a pay-as-you-go (0003P) subscription, which unlocks Spot (−81%; Learn lists Spot
   for EA, pay-as-you-go 003P, Sponsored and CSP offers only, [Learn](https://learn.microsoft.com/azure/virtual-machines/spot-vms#limitations)),
   reservations and GPU quota requests (R30). An MCA subscription supports reservations and savings
   plans; Spot on MCA is unconfirmed (not in that list). Reservations pay off only at high
   utilization: D16s_v6 1-year $0.50/h ($4,380 committed for 12 months, −38%, break-even at 62%
   utilization) and 3-year $0.315/h ($8,265, −61%, break-even at 39%).
   Visual Studio subscriptions are also dev/test only and may suspend instances that run more than
   120 hours continuously ([FAQ](https://learn.microsoft.com/visualstudio/subscriptions/faq/subscriber/azure/)),
   so long batch jobs have to be split across VMs.
3. Separate quota pools are worth a free ARM probe (R27): Azure Container Apps serverless GPUs (A100,
   T4; their own quota, enabled by default only for EA and pay-as-you-go,
   [Learn](https://learn.microsoft.com/azure/container-apps/gpu-serverless-overview)), Azure Batch
   accounts (own core quotas, [Learn](https://learn.microsoft.com/azure/batch/batch-quota-limit)) and
   Azure ML compute quotas (GPU families 0 by default).

What a GPU would cost (est., **low confidence**: there is no public Qwen3.8-27B GPU benchmark; the
throughput range is extrapolated from Qwen3-32B results: GPUStack H100 2,353 tok/s with default vLLM;
DatabaseMart at 300 concurrent requests: RTX PRO 6000 1,655, H100 1,482, A100 721 tok/s). Prices:
Linux pay-as-you-go, eastus2.

| GPU VM | $/h | assumed tok/s (512/128) | est. $/M | Spot $/h (if the offer allowed it) |
| --- | ---: | ---: | ---: | ---: |
| NC72lds RTX PRO 6000 BSE, 1/2 GPU (48 GB; needs FP8, the official Qwen3.8-27B-FP8: BF16 weights are ~54 GB) | 2.44 | 1,500–3,000 | 0.23–0.45 | 0.45 |
| NC144lds RTX PRO 6000 BSE, 1 GPU (96 GB) | 5.50 | 3,000–6,000 | 0.25–0.51 | 1.02 |
| NC24ads_A100_v4 (80 GB) | 3.673 | 1,500–3,000 (721 pessimistic) | 0.34–0.68 (1.42) | 0.68 |
| NC40ads_H100_v5 (H100 NVL, 94 GB) | 6.98 | 2,350–6,000 | 0.32–0.82 | 3.22 |
| Container Apps serverless A100 (westus3, 24 vCPU / 220 GiB replica) | ~6.35 | 1,500–3,000 | 0.6–1.2 | – |

HBM CPUs don't help either: HBv3/HBv4/HBv5, HX, HC and the M-series have limit 0, one VM would need 176
or 368 vCPUs of quota ($7.20/h HBv4, $19.80/h HBv5), and batch-64 decode here is not bandwidth-bound.

### (e) Managed per-token APIs: the price to beat

**Verdict: managed models cost $0.06–0.30/M blended at list price, 8–40× below our best measured
CPU cost ($2.39/M) and ~5–25× below the best projected CPU cost (~$1.36–1.51/M). Whether any of them
matches Qwen3.8-27B on our tasks is unknown until R08. The cheap models we can already rule out are
weaker: gpt-oss-120b scores 24 on the Artificial Analysis Intelligence Index vs 43 for Qwen3.8-27B
(snippet), below even Qwen3.5-9B on MMLU-Pro and GPQA, and Phi-4 is dominated (see (b)). The
candidates are GPT-6 Luna, GPT-5 mini (Batch) and DeepSeek V4 Flash; DeepSeek V4 Flash and the GPT
reasoning models need thinking or reasoning on to be competitive (V4 Flash non-thinking: GPQA 71.2,
MMLU-Pro 83.0 vs 88.1/86.2 in its Max mode; snippet), which raises billed output tokens. If one
passes, self-hosting Qwen3.8-27B on CPU makes sense only for data control (R33(b)); at
Spot-equivalent prices (~$0.3–0.5/M) it would come close.** R08 measures quality per dollar, in $
per 1,000 requests (tokenizers differ), before any conclusion.

Azure AI Foundry, eastus2, global deployments unless noted, blended for 512/128 as 0.8 × input +
0.2 × output, before reasoning tokens. Re-checked against the Retail Prices API
(`serviceName eq 'Foundry Models'`) on 2026-10-04 unless marked.

| model | input / output $/M | blended $/M | our $2.39 ÷ this | R08 role (quality evidence from snippets) |
| --- | --- | ---: | ---: | --- |
| GPT-5 nano, Batch | 0.025 / 0.20 | 0.06 | 40× | low-cost control; reasoning tokens billed |
| gpt-4.1-nano, Batch | 0.05 / 0.20 | 0.08 | 30× | not tested (older, small) |
| Phi-4-mini | 0.075 / 0.30 | 0.12 | 20× | below the bar (MMLU-Pro 52.8) |
| GPT-6 Luna (short context, standard; no Batch meter yet) | 0.10 / 0.50 | 0.18 | 13× | **candidate** |
| DeepSeek V4 Flash (Fireworks on Foundry, data zone) | 0.15 / 0.31 | 0.18 | 13× | **candidate**, thinking off and on |
| DeepSeek V4 Flash (per the hardware researcher, not re-checked) | 0.19 / 0.51 | 0.25 | 10× | same model |
| Phi-4 | 0.125 / 0.50 | 0.20 | 12× | below the bar (dominated, see (b)) |
| gpt-oss-120b | 0.15 / 0.60 | 0.24 | 10× | below the bar (AA index 24 vs 43) |
| GPT-5 mini, Batch | 0.125 / 1.00 | 0.30 | 8× | **candidate**; reasoning tokens billed |
| Mercury 2 / Mercury 2.5 (Inception, diffusion LM; Foundry and OpenRouter listings, not in the Retail Prices API) | 0.25 / 0.75; 0.20 / 0.75 | 0.35 / 0.31 | 7–8× | optional (Mercury 2: AA index 21) |
| Qwen3-32B fine-tuned (the only Qwen meter on Foundry) | 0.30 / 1.20, plus $0.30/h hosting | 0.48 + hosting | 5× | – |
| Qwen3.8-27B on third-party APIs (DeepInfra $0.40 / $3.00, Alibaba Cloud $0.50 / $3.00; snippets) | 0.40–0.50 / 3.00 | 0.92–1.00 | 2.4–2.6× | same model, not on Foundry |

Caveats: these are different models, so the comparison is only meaningful after a task-level
quality check on sets that separate them (R08, E27): E16's GSM8K and MMLU subsets are near ceiling
(~90%) and can't. Reasoning models (gpt-oss, GPT-5 nano/mini, DeepSeek V4 Flash with thinking) bill
hidden reasoning tokens as output, so the blended figures are lower bounds for them. On a Visual
Studio subscription, Foundry deployment quota may be restricted and partner (Marketplace) models may
not be covered by the credits; accepting Marketplace terms is a subscription-scope action. For the
same model, CPU self-hosting beats the third-party Qwen3.8-27B APIs only at Spot-equivalent prices
(~$0.32/M on D16s_v6, $0.47/M on E16ds_v6).

### (f) Diffusion language models ("Diffusion Gemma")

**Verdict: DiffusionGemma exists, is open (Apache 2.0) and can in principle run on our vLLM CPU stack,
but diffusion is the wrong regime for offline batch work on a compute-limited CPU. Est. at best equal
to its autoregressive twin per token, with lower quality, and well below Qwen3.8-27B in quality. Test
it only as an add-on to the Gemma 4 run (R17 with R04), to answer the question with data.**

What exists (2026-10-04):

- **DiffusionGemma-26B-A4B-it** (Google DeepMind, released 2026-06-10, Apache 2.0; one size; the only
  open "Diffusion Gemma"). It is the Gemma 4 26B-A4B MoE (25.2B total, 3.8B active, 128 experts top-8,
  262k vocab) converted to block discrete diffusion with < 10% of the original training tokens. A causal
  encoder prefills the prompt; a bidirectional decoder denoises a 256-token canvas in up to 48 steps
  (~12 on average, ~15–20 tokens finalized per forward pass) and then commits it to the KV cache. FP8
  (RedHatAI) and NVFP4 (NVIDIA) checkpoints exist. Quality against its autoregressive twin: MMLU-Pro
  77.6 vs 82.6, GPQA Diamond 73.2 vs 82.3, MMMLU 81.5 vs 86.3, also lower on AIME 2026, LiveCodeBench
  and Codeforces; Google recommends standard Gemma 4 when quality matters most. Qwen3.8-27B's GPQA is
  89–90. ([model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it),
  [tech report](https://arxiv.org/abs/2608.00146), [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/))
- Other open diffusion LMs, all smaller or weaker for this job: LLaDA 8B and LLaDA-MoE-7B-A1B;
  LLaDA2.0/2.1/2.2 (16B-A1.4B mini, 100B-A6B and 103B-A13B flash); Dream 7B; SDAR (Qwen3-based up to
  30B-A3B); Fast-dLLM v2 (Qwen2.5 1.5B/7B); RND1-30B-A3B (base only); WeDLM-8B; Efficient-DLM 8B; and
  dQwen3.5 (0.8B–9B base models, the only conversions of the Qwen3.5/3.6 hybrid; the 9B scores MMLU 74.6).
  **No diffusion version of Qwen3.8-27B or Qwen3.6-27B exists.**
- Closed: Gemini Diffusion (an experimental demo since May 2025, still waitlisted per secondary
  sources) and Inception Mercury 2 / 2.5 (API only; on Azure Foundry at $0.31–0.35/M blended, see (e);
  Mercury 2 scores 21 on the Artificial Analysis Intelligence Index, GPQA 77).

Can it run on our stack?

- **vLLM: yes in principle.** DiffusionGemma was the first diffusion LM supported natively (model
  runner v2), and the [vLLM recipe](https://github.com/vllm-project/recipes/blob/cbf76c0/models/Google/diffusiongemma-26B-A4B-it.yaml)
  lists Xeon 6 with the CPU image (BF16, AMX). CPU fixes: [PR #59107](https://github.com/vllm-project/vllm/pull/59107)
  (narrower canvases with sync scheduling) and [PR #59829](https://github.com/vllm-project/vllm/pull/59829)
  (async output buffers aliasing on CPU corrupted outputs; tested on a 22-vCPU Xeon 8481C). Whether the
  0.31.0 image we use contains them is unverified. The recipe sets `--max-num-seqs 4` because sampler
  state is max_seqs × canvas × 262k vocab in fp32, ~268 MB per sequence: harmless to raise to 32–64 on
  128 GiB. The recipe also mentions ≥ 4 NUMA nodes; our VM has one.
- **llama.cpp: no** on master (the DiffusionGemma PR #24423/#24427 is unmerged; `llama-diffusion-cli`
  serves Dream, LLaDA and RND1 one sequence at a time, no server). **Ollama: no** ("unknown model
  architecture"). SGLang's dLLM framework supports it on GPUs; CPU is unverified.

Expected effect on $/M:

- The published speedups (~1,500 output tok/s on one H100, 1.9–7× over autoregressive) are batch-1
  GPU numbers, where autoregressive decode is bandwidth-bound and compute sits idle. Google and vLLM
  say the advantage fades as the batch grows (autoregressive serving catches up beyond ~32 concurrent
  users on an H100), and independent studies find autoregressive models have the highest throughput at
  every batch size (LLaMA3-8B 13.7× LLaDA-8B, [arXiv 2510.18480](https://arxiv.org/abs/2510.18480)).
- Our CPU is in the other regime, and our measurements show it. An autoregressive decode step on vLLM
  CPU costs ~3.5–4.8 prefill-equivalent positions per output token at 64 sequences; DiffusionGemma needs
  ~13 (12 steps × 256 positions per 256 tokens, plus the commit pass). Extra positions per sequence are
  expensive on this stack: in E17's real-text MTP runs, 1, 2 and 3 draft tokens gave +16%, +14% and +6%
  at 64 sequences, i.e. ~13–17 ms per extra position (12.7 ms from the MTP-1 step time; 14.4, 15.7
  and 17.0 ms on the GSM8K whole-run basis for MTP-1/2/3), 3.3–4.5× the cost of a prefill token (3.8 ms).
- Est. cost for 512/128: DiffusionGemma **$1.1–7.7/M** depending on VM, precision and step count (five
  independent estimates, all low confidence); its autoregressive twin Gemma 4 26B-A4B **$0.7–1.7/M**. Any
  saving against Qwen3.8 comes from the 3.8B-active MoE backbone (R04), not from diffusion. Prefill
  (~45% of our wall time) is the same work for both, and diffusion adds nothing for classification or
  short JSON extraction.
- The one diffusion technique that keeps Qwen3.8's quality is **DFlash2**
  ([z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)), a block-diffusion draft
  model for lossless speculative decoding (3.4× at concurrency 1 on an H200, 1.0–1.45× at 32). After
  E17's MTP-2/3 results it is unlikely to help at 64 sequences on CPU, and vLLM support is only in
  unmerged PR branches (R19, P3).

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
  others each have 20 vCPU per region, like the Intel families. No AMX, but the Fasv7/Famsv7/Falsv7
  sizes have no SMT (16 vCPU = 16 physical cores) and the CPU has 12-channel DDR5.
- Intel families with quota: Dsv6/Ddsv6/Dlsv6/Dldsv6, Dsv7/Ddsv7/Dlsv7/Dldsv7, Edsv6, Edsv7
  (20 vCPU each per region). Esv6 (E16s_v6) was not checked. E16ds_v6 has 2 × 440 GiB of local
  NVMe ([Learn](https://learn.microsoft.com/azure/virtual-machines/sizes/memory-optimized/edsv6-series));
  the D16s_v6, D16ls_v6 and E16s_v6 sizes have none, so `/mnt/data` sits on the OS disk there.

Prices for the CPU sizes in the queue (Linux pay-as-you-go, Retail Prices API, re-checked 2026-10-04).
All-in = VM + OS disk + $0.005/h IP; sizes without a local disk need a 128 GiB P10 OS disk ($17.92/month
= $0.0245/h), the others keep the P6 ($0.0127/h).

| size | CPU | physical cores | RAM | local NVMe | VM $/h (region) | Spot $/h | all-in $/h | $/M at E18's 154 tok/s |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| E16ds_v7 | Xeon 6 6973P-C (Granite Rapids, AMX) | 8 | 128 GiB | yes | 1.663 (eastus2) | 0.307 | 1.681 | 3.03 (**measured 3.47** on a slow instance) |
| E16ds_v6 | Xeon Platinum 8573C (Emerald Rapids, AMX) | 8 | 128 GiB | yes | 1.308 (westus2) | 0.242 | 1.326 | **2.39 measured** |
| E16s_v6 | Xeon Platinum 8573C | 8 | 128 GiB | no | 1.058 (eastus2) | 0.196 | ~1.088 | 1.96 (est.) |
| D16ds_v6 | Xeon Platinum 8573C | 8 | 64 GiB | yes | 0.997 (eastus2, westus2) | 0.184 | ~1.015 | 1.83 (est.) |
| D16s_v6 | Xeon Platinum 8573C | 8 | 64 GiB | no | 0.806 (eastus2, westus2) | 0.149 | ~0.836 | 1.51 (est., if the 64-prompt job fits: R26) |
| D16ls_v6 | Xeon Platinum 8573C | 8 | 32 GiB | no | 0.714 (eastus2, westus2, centralindia; re-checked 2026-10-05) | 0.132 | ~0.744 | the 27B doesn't fit (37 GiB loaded); small models only (R41), × ~0.39 vs the v7 basis |
| F16as_v7 | AMD EPYC 9005 (Turin, no AMX, no SMT) | 16 | 64 GiB | no | 0.706 (centralindia), 1.093 (eastus2) | 0.130, 0.202 | ~0.736, ~1.123 | depends on throughput (R05) |

## Queue

Sorted by priority, then expected value per VM-hour (queue rule 1). "Conf." is the probability that
the expected effect materializes (high / medium / low = 0.8 / 0.5 / 0.2; "–" for gates, decisions
and records). "Cost to test" is VM-hours and dollars at the reference $1.681/h (cheaper sizes cost
less). "Exp." is the experiment the row maps to (E14 and E19–E27 are candidates in PLAN.md).
Effects are against the $2.39/M ($1.53 per 1k requests) baseline unless a different basis is named.
*Unverified* marks upstream PR, issue and recipe claims that came from search snippets (github.com,
docs.vllm.ai and recipes.vllm.ai are blocked from the sandbox); the chain `probe` step checks most
of them in ~5 minutes of VM time.

| ID | Pri | Idea | Hypothesis (falsifiable) | Expected effect on $/M | Conf. | Cost to test | Prerequisites / blockers | Status | Exp. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | – | Round 2: task quality (E16); vLLM batch scaling, W4A16, threads, MTP (E17); vLLM on E16ds_v6 (E18); MTP on real text (E17 follow-up) | see the READMEs | **measured**: $2.39/M on E16ds_v6 at 64 prompts ($4.56 at 16 on v7); W4A16 4–6× slower; 16 threads −50%; MTP-1 +16% at 64 sequences on GSM8K; W8A8 −2.75 MMLU points | – | ~14 VM-h, ~$20 | – | done | [E16](E16-task-quality/README.md), [E17](E17-vllm-scaling-mtp/README.md), [E18](E18-vllm-emerald-rapids/README.md) |
| R33 | P0 | **Decisions from the user** (not runnable by agents; ask before round 3 starts). (a) A sample of the real batch jobs, or their statistics: shared-instruction length, unique-input length, output length, task type (classification, extraction, other), share of exact-duplicate prompts (deduplication is free), monthly volume. (b) May the data go to Azure AI Foundry managed models (data control)? If yes, R08 runs at P0. (c) The quality bar. Proposed default for the user to confirm: on the task metric, ≤ 2 points below Qwen3.8-27B (reference: the W8A8 now in use; BF16 if the user prefers, which adds ~2.6 VM-h to R11) in a paired non-inferiority test (one-sided 95%); ≥ 98% JSON-schema validity; ≤ 3 points below on IFEval strict. (d) The monthly Azure credit amount and how much round 3 may spend. | – | Decides which rows matter: for a classification job R34/R35 dominate every engine-level lever; if (b) is yes, R08 may make CPU self-hosting moot (managed $0.06–0.30/M list). | – | 0 VM-h | Only the user can answer. Until then chains use public proxies (Banking77, `paraloq/json_data_extraction`, random and real-text 512/128) and the default bar. Unblocks R03, R14, R20, R22, R34–R36 (a); R08 (b); go/no-go for R09, R10, R04, R36, R39, R41 (c); PLAN.md budget (d). | blocked (user) | – |
| R27 | P0 | **Zero-VM quota probes** through ARM in the resource group: (a) an Azure Batch account in Batch-service mode: read `dedicatedCoreQuota`, the per-family quotas and `lowPriorityCoreQuota`; (b) a Container Apps workload-profiles environment: read its usages for Consumption NCA100/T4 GPUs; (c) `Microsoft.MachineLearningServices/locations/{region}/usages` (a subscription read, like the Compute usages that already work). | With ≥ 80% probability every GPU and Spot quota is 0 on this offer. Falsified by any nonzero A100/T4 GPU quota or Spot core quota. | If ACA A100 is allowed: ~$6.35/h per replica at 1.5–3k tok/s → $0.6–1.2/M (est.). If Batch Spot works on a D16s_v6-class size: ~$0.32/M (−87%). Otherwise a documented negative. | low (that anything is nonzero) | 0 VM-h (~30 min of orchestrator time; empty accounts cost nothing) | Microsoft.Batch, Microsoft.App and Microsoft.MachineLearningServices must already be registered (registration is a subscription write and fails here; check provider state first). Getting results out of a Batch or ACA node needs a channel readable through ARM. Moved from P1 to P0: free. | queued | – |
| R14 | P0 | **Output budget and structured output** on real inputs (the user's sample from R33(a), else `paraloq/json_data_extraction` and Banking77): (i) free text, max_tokens 128; (ii) JSON schema through xgrammar with short keys; (iii) a terse prompt; 64 sequences (128 on a 128 GiB VM). Confirm `enable_thinking=false` in every bench script (Qwen3.8 thinks by default). Report $/1k requests and $/M tokens. Moved from P1 to P0 (best expected value per VM-hour among the measurement rows). | JSON or terse output halves output tokens (128 → ~64) at equal task quality, and xgrammar adds < 5% per step even though it shares the 8 cores. Falsified if the grammar costs > 10%, task quality drops > 1 point, or MTP + grammar crashes. | Per request 1.95 + 1.18 = 3.13 s vs 4.31 s: −27% per request ($1.53 → ~$1.11 per 1k requests), −19% per M tokens on 576 tokens ($2.39 → ~$1.93). Leaving thinking on (~+1,000 output tokens) costs ~5× per request. | medium | ~0.5–0.7 VM-h ($0.84–1.18) | A labelled extraction or classification sample (R33(a); public proxy meanwhile). CPU GDN + MTP + structured output can kill the EngineCore ([#56419](https://github.com/vllm-project/vllm/issues/56419), fix PR #56518; *unverified*, checked by the `probe` step). | queued | E24 |
| R10 | P0 | **Qwen3.5-9B** (dense GDN hybrid, the same vLLM code path as the 27B): BF16 and `RedHatAI/Qwen3.5-9B-quantized.w8a8` (compressed-tensors W8A8 published for vLLM; snippet) first; a self-made RTN W8A8 (llm-compressor, per-channel weights, dynamic per-token activations, lm_head in BF16) only as a fallback. Prefill, decode and mixed at 64/128/256 prompts from one engine per variant (`bench/vllm_sweep.py`), plus one real-text 512/128 run; thinking off; 3-prompt correctness check at the largest batch. | Prefill ≥ 3.4× the 27B's (13.8 vs 54 GFLOP/token). On the E17 basis, mixed 512/128 at 128 prompts ≥ 250 tok/s in BF16 (central 306) and ≥ 380 tok/s in W8A8 (central 444). Falsified if W8A8 mixed is < 2× the 27B's on the same VM. | Est. on the E16ds_v7 basis (vs $3.47 measured): W8A8 $0.96–1.24 (mid 1.05), BF16 $1.39–1.80 (mid 1.53), i.e. 1.9–3.6× cheaper; × 0.69 on E16ds_v6. Quality (thinking mode): MMLU-Pro 82.5, GPQA 81.7, IFEval 91.5. | medium | ~1.5 VM-h ($2.52) | Same VM and chain as R09 (like-for-like); usable only if R11 passes; then R41 re-prices it on D16ls_v6. | queued | E19 |
| R26 | P0 | **Same CPU, cheaper size**, in two steps. (1) On chain C's 128 GiB E16ds_v6: W8A8 mixed 512/128 at 64 prompts with KV_GB=16 next to KV_GB=24 (E18's baseline setting), logging vLLM's block count, preemptions, min `MemAvailable` and the container's cgroup `memory.peak`. (2) On D16s_v6 (64 GiB, no local disk; D16ds_v6 has NVMe): E18's W8A8 chain at 16 and 64 prompts with KV_GB=16 only (the prologue references too), plus the llama-bench reference. Dsv6, Ddsv6 and Edsv6 all use the Xeon Platinum 8573C. | (1) 16 GiB of KV gives the same mixed throughput as 24 GiB (±3%) with zero preemptions: 16 GiB holds ~293 blocks (16 layers × ~3.67 MB per 896-token page; vLLM reports 66,446 tokens = blocks × 896 ÷ 4 layer groups) and 64 sequences × (1 attention + 3 GDN blocks) need ~256 (87%, no headroom for MTP or align-mode extra state). (2) Mixed 512/128 at 64 prompts is ≥ 0.9× E18 (154 tok/s) after reference normalization, with min MemAvailable ≥ 3 GiB while Defender and the monitoring agents run, and no preemption. Falsified by OOM, preemption, MemAvailable < 3 GiB, or < 0.85×. Evidence so far: the $2.39 baseline ran with 24 GiB of KV and its largest process peaked at 65.1–65.7 GB (~61 GiB, which would not fit); the 56–57 GB peaks were the 16-prompt runs at 16 GiB. The sampler records only the largest process (the Worker; vLLM reports 37.15 GiB for the model); EngineCore, the API front end, the 8 GB docker shm, dockerd and the agents come on top. | D16s_v6 $0.836/h → 0.836 / 0.554 = **$1.51 (−37%)**; D16ds_v6 $1.015/h → $1.83 (−24%). Spot-equivalent D16s_v6 ~$0.32. Rows that fit 64 GiB get cheaper by the same factor: the W8A8 27B at ≤ 64 sequences with ≤ 16 GiB KV, R13, R14, R21, R34, and the small models of R10 (and R09's 35 GiB W8A8). It does **not** apply to R12 (> ~20 GiB KV), MTP at 64 sequences with FP32 state (≥ 19.5 GiB KV; R15, unless R13 passes), R09 BF16 (~70 GB), R04/R17 BF16 (~50 GB) or R43; those need a 128 GiB size (E16s_v6, ~$1.088/h all-in). | medium | ~2 VM-h on D16s_v6 ($3.36; ~$1.7 at its price) + ~0.3 VM-h in chain C | `-p vmSize=Standard_D16s_v6 -p osDiskSizeGB=128` (both Bicep parameters exist; cloud-init keeps `/mnt/data` on the OS disk when there is no local NVMe). The whole-system memory sampler in `bench/vllm_bench.sh` (~30 min; today it samples only the largest process). Read the Dsv6/Ddsv6 usages and SKU restrictions in the region. Model load from a P10 disk is ~5 min slower. OOM fallback: 12 GiB KV or 48 prompts. Open: same memory bandwidth on D as on E hosts. | queued | E21 |
| R09 | P0 | **Qwen3.6-35B-A3B** (MoE GDN hybrid, ~3B active, 256 experts with 8 routed + 1 shared, Apache 2.0, official FP8): official BF16 with the default MoE prepack, then `VLLM_CPU_SGL_KERNEL=1` if it still exists, then AMD's W8A8 (`amd/Qwen3.6-35B-A3B-w8a8-llmcompressor-v0.12.0`: RTN, static INT8 per-channel weights, dynamic INT8 per-token activations; router, shared-expert gate, linear-attention modules, vision tower and lm_head in BF16; 67.0 → 35.1 GiB; snippet), then the official FP8 (W8A16) if it loads; 64/128/256 prompts from one engine per variant (`vllm_sweep.py`); `max-num-batched-tokens` 2048 → 8192 → 16384 (more tokens per expert GEMM). Throughput on **real-text prompts** truncated or padded to 512 tokens (E16's GSM8K/MMLU prompts or a named news set) with a fixed 128-token output budget, plus one random-prompt run for comparability with E17/E18: random token IDs give unrealistic expert routing, which dominates the estimate. | BF16 mixed 512/128 (real text) at 128 prompts ≥ 250 tok/s (central 330) and prefill ≥ 500 tok/s (central 750); W8A8 or FP8 ≥ 400 tok/s, with the startup log naming an INT8 MoE method for W8A8 (not a BF16 fallback). Falsified if BF16 mixed < 180 tok/s (CPU MoE kernel efficiency < 0.2). | Est. on the E16ds_v7 basis: BF16 $1.20–1.70 (mid 1.42), FP8-W8A16 $0.94–1.45, INT8 $0.76–1.11 (mid 0.92), i.e. 2.0–4.6× cheaper than $3.47; even at MoE efficiency 0.2, $1.47. × 0.69 on E16ds_v6; the AMD W8A8 (35 GiB) is the only variant that fits 64 GiB (× ~0.44 on D16s_v6). Quality: MMLU-Pro 85.2, GPQA 86.0 (thinking mode). | medium | ~3 VM-h ($5.04) | 128 GiB VM for BF16 (~67–70 GB; GDN state ~63 MB per sequence). AMD's W8A8 is validated only on EPYC (ZenDNN 6.1, zentorch 2.11.0.3, PyTorch 2.11, vLLM 0.26; snippet), so the Intel AMX path for INT8 MoE experts is *unverified*: check the selected MoE method in the startup log. Smoke-test that the vLLM 0.31 `qwen3_5_moe` CPU path loads first (*unverified*: the vLLM recipe snippet names Qwen3-30B-A3B, not Qwen3.5/3.6-35B-A3B, as validated on Xeon 6). ([llm-compressor #2787](https://github.com/vllm-project/llm-compressor/issues/2787) is about INT8 GPUs and missing examples, not a CPU blocker.) | queued | E19 |
| R11 | P0 | **Quality gate** for R09/R10 (and R04, R36, R39, R41): thinking off, greedy, paired item-level comparison against Qwen3.8-27B W8A8 (the deployed configuration; E16 has BF16 for the GSM8K/MMLU subsets). Named sets, revisions pinned in the README: GSM8K test (1,319), MMLU (≥ 2,800 items stratified by subject) or MMLU-Pro, IFEval (541 prompts, strict), Banking77 test (3,080; accuracy) and `paraloq/json_data_extraction` (text + JSON schema → JSON; field F1 and schema validity, with and without guided decoding). **Stage 1**: E16's 200/400 subsets plus IFEval, 1,000 Banking77 items and the JSON set; a candidate > 4 points behind on any set fails there. **Stage 2** (survivors only): the full sets, for a non-inferiority decision. Paired McNemar or bootstrap 95% intervals; margin from R33(c) (default 2 points). | Qwen3.6-35B-A3B is non-inferior within 2 points on GSM8K/MMLU and within 3 on IFEval; Qwen3.5-9B is within 4–5 (fails the default bar); both reach ≥ 98% JSON schema validity with guided decoding; the 27B's lead is largest on strict multi-constraint instructions. Falsified (per candidate) if the lower 95% bound of (candidate − 27B) is below −margin on any task set. Power: with ~18% discordant pairs the SE of a paired difference at n = 400 is ~2.1 points (E16: ±3–4 points), so 400 items can't show "within 2"; a one-sided test at a 2-point margin with 80% power needs n ≈ 2,800. | None directly; decides whether R09/R10's 2–4× is usable. | – | Stage 1: ~2.3 VM-h for the 27B W8A8 on the new sets (IFEval ~50 min, JSON ~45 min, Banking77 ~45 min; scaled from E16's 1,027 s per 200 GSM8K items at 61 output tok/s) + ~1 VM-h per candidate ≈ 4.5 VM-h ($7.6). Stage 2: ~2.5 VM-h for the 27B (GSM8K 1,319 ≈ 1.9 h, MMLU 2,800 ≈ 0.5 h) + ~1–1.5 per survivor. | `bench/quality_tasks.py` supports only gsm8k and mmlu: add ifeval, json and classify builders and scorers (~2–3 h; IFEval strict scoring needs its instruction checkers in the container). Datasets downloaded on the VM; `chat_template_kwargs={"enable_thinking": false}`. Candidate variants: the quantized ones R09/R10 would deploy; if B2 starts before B1, gate BF16 plus a W8A8-vs-BF16 agreement check on the E16 subsets. | queued | E20 |
| R12 | P0 | **Larger batches** (128 GiB sizes only): W8A8, max_num_seqs 64/128/192/256 through `EXTRA='--max-num-seqs N'` in `bench/vllm_bench.sh` (it has no MAX_NUM_SEQS knob) with NUM_PROMPTS ≥ 4× that (steady state), or `vllm_sweep.py --extra '{"max_num_seqs": 256}'` with batches 64–256 from one engine; KV 24/40/60 GiB; log vLLM's KV-concurrency line and preemptions; 3-prompt greedy agreement check at the largest batch (E04 saw batch-dependent corruption in llama.cpp). | The decode step is ≈ 0.29 s + 13.9 ms × n, so output tok/s rises 54 → ~62 at 128 (+14%) and ~67 at 256 (+22%), with prefill flat. Falsified if 128 sequences give < +5% decode, preempt, or fail the agreement check. | Per request 4.31 → 4.02 s (128) or 3.87 s (256), plus ~3% from amortizing ~9 s of fixed per-run overhead: $2.39 → ~$2.08–2.22 (−7 to −13%). On E16s_v6 ($1.088/h) at 166–185 tok/s: $1.63–1.82. Not on 64 GiB sizes (R26). | medium | ~1.5 VM-h ($2.52) | A 128 GiB VM: largest-process RSS ≈ 56.5 GB + 1.07 GB per GiB of KV above 16 (reproduces E17's 65.3 GB at 24 GiB), i.e. ~104 GB at 60 GiB with FP32 state, plus the processes the sampler misses: step up and stop at OOM. R13 cuts the KV need ~1.4–1.7×. | queued | E22 |
| R13 | P0 | **BF16 GDN state** (`--mamba-ssm-cache-dtype bfloat16`) with W8A8: decode at 64 and 128 sequences and mixed 512/128 (random prompts), then real text at 64 sequences: GSM8K with the E16 harness (512-token cap; `quality_tasks.py` logs output_tokens and n_truncated) and MMLU 400, against the FP32-state run. Log the "Setting attention block size" and "Padding mamba page size" lines. Report $/request, not only tok/s. | The FP32 state is ~154 MB per sequence (48 layers × 48 V-heads × 128 × 128 × 4 B plus conv state), ~20 GB of state traffic per decode step at 64. BF16 cuts the per-sequence cost by ≥ 1.5 ms (decode ≥ +4%, up to +16%). The attention block falls to ~448–512 tokens, not the GPU's ~400: the FP32 mamba page is 783 tokens × 4,096 B, which the CPU path rounds up to 896 (the GPU to 784); BF16 needs ~399, which the CPU likely rounds to 448 or 512. Sequences per GiB rise ~1.4× (512-token block) to ~1.7× (416), not 2×, because a 640-token sequence then spans 2 attention blocks. Quality: accuracy within ±1 point **and** mean output tokens per request within +3% with no more truncations. The documented failure of a lower-precision recurrent state is longer outputs, not lower accuracy: NVIDIA's Nemotron 3 Super report found a plain FP16 cast of the Mamba state raised verbosity by up to 37–40% (+0.4% with stochastic rounding, which NVIDIA ships), and BF16 has fewer mantissa bits than FP16; vLLM's CPU cast is presumably round-to-nearest. Falsified if decode gains < 4%, the override is ignored (Qwen3.8's config pins float32), MMLU or GSM8K drops > 1 point, or output tokens rise > 3% or truncations rise. | −4 to −8% per request ($2.39 → ~$2.2–2.3) if output length holds; +10% output length would erase it. Lets 256 sequences fit in ~33–40 GiB of KV instead of ~56, brings MTP at 64 sequences to est. ~14 GiB of KV (fits a 64 GiB VM; R37), and enables prefix hits for shared prefixes ≥ the logged block (R03). | medium | ~1.5 VM-h incl. quality ($2.52) | Flag present in 0.31 (`probe` step). A search summary says vLLM v0.29 added "FP16/BF16 persisted GDN state on AMX" (*unverified*). Qwen ships FP32 for a reason; no published ablation. | queued | E22 |
| R16 | P0 | **Profile one decode step** at 64 sequences (W8A8, decode 32/256): torch profiler (`VLLM_TORCH_PROFILER_DIR`) or `perf` + `py-spy` on the EngineCore; plus an AMX GEMM microbenchmark in the same container (vLLM's INT8 scaled-mm and a BF16 matmul at M = 16, 64, 128, 512, 4096 with the model's K/N shapes, 8 threads) and a dependent-add loop to estimate the core clock under AMX load (no PMU in these VMs). | INT8 GEMM is ≤ 55% of the 13.9 ms per-sequence cost, and GDN decode plus state copies ≥ 20%; GEMM efficiency at M = 64 is ≤ 60% of that at M = 4096; large-M INT8 GEMM reaches ≤ 40% of nominal peak; v7's large-GEMM TOPS are ≤ v6's (lower AMX clock). If GEMM is ≥ 80% of the step, drop the decode-overhead ideas and go straight to smaller models. If Python, scheduler or sampler work is ≥ 20%, run R42 and R43. | None directly. It decides how to get the est. 1.3–1.6× same-model headroom ($2.39 → ~$1.5–1.85): GEMM (3.4–3.8 ms), FP32 state traffic (3.2 ms) and attention (< 0.5 ms) explain only half of the 13.9 ms; halving the per-sequence term would give decode × 1.6 at 64 sequences (≈ −20%). | – | ~0.6–0.9 VM-h ($1.0–1.5), appended to R12/R13 | `perf` in cloud-init (E07 used it); profile output compressed under the 4 KiB fetch limit (top-N op table as text). | queued | E22 |
| R08 | P0 | **Managed-API quality per dollar.** Desk part done (answer (e)). Next: an Azure AI Foundry (AIServices) resource in the RG through ARM with `disableLocalAuth: true`; deploy GPT-6 Luna, GPT-5 mini (Batch where offered) and DeepSeek-V4-Flash (thinking off and on), plus GPT-5 nano as a low-cost control (Mercury 2.5 if Marketplace terms allow); run R11's task sets (IFEval, the JSON-extraction set, Banking77) and a hard knowledge set (an MMLU-Pro or GPQA Diamond subset) from a VM, and the 27B W8A8 on the hard set. **Keyless**: assign the VM's managed identity "Cognitive Services OpenAI User" (or "Cognitive Services User" for non-OpenAI Foundry models) on the Foundry resource, and fetch tokens from IMDS (audience `https://cognitiveservices.azure.com`) with stdlib `urllib`. Never call `listKeys`: a key passed through `deploy.py -e` lands in the Run Command record, the job script and the systemd unit environment. Log billed input, output and reasoning tokens per request. | GPT-6 Luna, GPT-5 mini or DeepSeek-V4-Flash is within 2 points of Qwen3.8-27B W8A8 on the task sets and the hard set at ≤ $0.30/M including billed reasoning tokens. gpt-oss-120b and Phi-4 are not candidates (Artificial Analysis Intelligence Index: gpt-oss-120b 24 vs Qwen3.8-27B 43; Phi-4 dominated, see (b)). Falsified if every candidate trails W8A8 by > 3 points on the task sets or the hard set. E16's GSM8K/MMLU subsets are near ceiling (~90%) and can't separate these models. | Decides the project's conclusion if the user allows managed APIs (R33(b)): managed $0.06–0.30/M at list price (before reasoning tokens) vs CPU ~$1.4–2.4/M on demand. Compared in $/1k requests (tokenizers differ). | medium | ~1–1.5 VM-h for the client ($1.7–2.5) + ~$1–3 of tokens | R33(b). Foundry deployment quota on a Visual Studio subscription (may be restricted); partner models may not be covered by credits; Marketplace terms acceptance is subscription-scope. The role assignment is an RG-scope write (Owner suffices). Moved from P1 to P0, conditional on (b). | blocked (user: R33(b)) | E27 |
| R30 | P1 | **Owner-level account change** (not runnable by agents): request GPU quota in the portal, and/or move the work to a pay-as-you-go (0003P) subscription, which Learn lists as Spot-eligible (with EA, Sponsored and CSP). MCA supports reservations and savings plans; Spot on MCA is unconfirmed. | Spot capacity for D16s_v6, E16ds_v6 or F16as_v7 exists in at least one US region; GPU quota (one RTX PRO 6000 1/2 or one A100) is granted after some billing history. | Spot D16s_v6 (~$0.18/h all-in) → ~$0.32/M (−87%): the best option for intermittent batch work. A Spot chain must survive eviction: eviction policy Deallocate, and the per-step markers (`/mnt/data/results/.done`) moved to the OS disk, since `/mnt/data` on NVMe sizes is wiped on deallocation. Reservations only at high utilization: 1-year D16s_v6 $0.50/h → ~$0.95/M at 100% utilization for 12 months ($4,380 committed; break-even vs on demand at 62% utilization); 3-year $0.315/h ($8,265; break-even 39%). Savings plans (EA, MCA or MPA only; not pay-as-you-go Web Direct) give a smaller discount but follow the work across sizes and regions (v6, v7, Turin). GPU: RTX PRO 6000 1/2 on demand ~$0.23–0.45/M (est., FP8), on Spot ($0.45/h) ~$0.05–0.10/M. | low | 0 VM-h; a payment method; Microsoft may still decline GPU quota for new accounts | The user's decision. Visual Studio offers get no Spot, no reservations and, per Q&A answers, no GPU quota increases. | blocked (owner decision) | – |
| R34 | P1 | **Constrained single-token classification**: for classification jobs, return the label as one constrained token (vLLM structured outputs with a choice list, i.e. guided choice, or score the label tokens' logprobs and take the max) instead of up to 128 free tokens; Banking77 and AG News (or the user's set), W8A8 27B, 64 sequences, against free-text output with the same prompt. Append to chain C when R33(a) says classification. | Accuracy within 1 point of free-text output (free text can also fail to parse), and cost per request falls to about prefill only: 512/1 at 263 tok/s = 1.95 s vs 4.15 s. Falsified if accuracy drops > 1 point or guided choice adds > 10% per request. | Classification only: −53% per request ($1.53 → ~$0.72 per 1k requests at $1.326/h). $/M tokens is the wrong metric here. | high | ~0.3 VM-h ($0.50) | R33(a) (a classification job; public proxy until then); the structured-outputs backend on CPU (xgrammar; shares the #56419 caveat with MTP). | queued | – |
| R03 | P1 | **Shared-prefix workloads** with prefix caching (`vllm bench throughput --dataset-name prefix_repetition`), W8A8, 128 sequences: (a) 384 shared + 128 unique tokens, defaults (negative control); (b) 896 shared + 384 unique; (c) 384 shared + 128 unique with `--enable-mamba-fine-grained-prefix-cache` (and MTP-1, `--mamba-cache-mode align`); (d) BF16 state with a shared prefix ≥ the block it logs (e.g. 640 shared + 128 unique). Each against `--no-enable-prefix-caching`, with the hit rate logged and outputs spot-checked. Moved from P0 to P1: 0% on the reference workload. | Hybrid models cache per attention block (896 tokens with FP32 state on our CPU; est. 448–512 with BF16 state, R13), so (a) gets 0% hits, (b) hits at the 896 boundary, (c) reaches sub-block prefixes only through fine-grained mode, and (d) hits once the shared prefix is ≥ the logged block. Falsified if (b) saves < 50% of the predicted prefill, or if (c) and (d) both get 0%. | Reference workload: 0. Case (b): per request 1280/263 + 2.36 = 7.23 s → 384/263 + 2.36 = 3.82 s (−47%). Case (d), 640 of 768 tokens shared: 768/263 + 2.36 = 5.28 s → 128/263 + 2.36 = 2.85 s (−46%). Case (c), 384 of 512 shared, if fine-grained mode works: 4.31 → 2.85 s (−34%). | medium | ~1.5 VM-h ($2.52) | Flag names in 0.31 (PR #46384 fine-grained, PR #36649 'all' mode; `probe` step); prefix-cache + MTP correctness bugs on hybrids (#47861, #53504); the user's real shared and unique lengths (R33(a)). | queued | E23 |
| R35 | P1 | **Distil the 27B's labels into a small classifier**: label ~5k–20k items of the job's distribution with the 27B (W8A8, R34's single-token mode), then train on the VM's CPU (i) a ModernBERT- or DeBERTa-v3-class encoder (~0.1–0.4B) for classification and (ii) sentence embeddings + logistic regression; evaluate on a held-out set against the 27B's labels and on gold labels (Banking77, AG News). | The encoder reaches ≥ 97% of the 27B's accuracy on gold labels and embeddings + a linear head ≥ 93%. Inference needs ~0.4/27 of the 27B's prefill FLOPs per request: est. 30–70× cheaper per request. Falsified if the encoder is < 95% of the 27B's accuracy on gold labels. | Classification only: est. $0.72 → ~$0.01–0.03 per 1k requests after a one-off labelling cost (~$7.7 for 5k items at 512/128 on the baseline, ~$3.6 with R34) and ~0.5–1 VM-h of CPU training (**low confidence**). | low | ~1–2 VM-h ($1.7–3.4) | R33(a) (a classification job); pip on the VM (transformers, scikit-learn); the user accepts a non-LLM classifier (its errors differ from the 27B's). | queued | – |
| R40 | P1 | **Instance-speed distribution and a keep-the-fast-one policy**: deploy 4–6 short-lived VMs of the chosen size (in sequence or across regions), run the 5-minute llama-bench reference and one 10-minute W8A8 mixed reference on each, then delete them. | Same-size instances differ by ≥ 10% (E08: 15–25% between two E16ds_v7; E18: this v6 instance 13–22% faster than E17's v7 on llama.cpp), so best-of-3 gives ≥ 8% for jobs longer than ~2 h after paying ~3 × 15 min of extra VM time. Falsified if the spread across ≥ 4 instances is < 5%. | −5 to −10% for long jobs (est.); also calibrates the 2×-CV rule. | medium | ~1 VM-h ($1.68) | None. If it holds, the policy goes into the chain prologue. | queued | – |
| R37 | P1 | **Measure the decode stack on the size R26 selects**, with the KV budget as an explicit variable: on D16s_v6 (if R26 passes) W8A8 at 64 sequences, FP32 vs BF16 state, MTP-1 at KV 12/16/20 GiB; on E16s_v6 or E16ds_v6 (if it fails) 128 sequences with BF16 state and MTP-1. Real text (the E16 harness and a 512/128 set), whole-system memory logged. | D16s_v6: BF16 state + MTP-1 at 64 sequences gives decode × ~1.22 (−10% per request) and fits ≤ 16 GiB of KV with min MemAvailable ≥ 3 GiB. 128 GiB sizes: decode × ~1.3 at 128 (−13%). Falsified if the measured stack gain is < half the prediction or the D16s_v6 run preempts. | D16s_v6: $1.51 → ~$1.36/M (est.); E16s_v6: ~$1.70/M; E16ds_v6: ~$2.07/M. | medium | ~1 VM-h ($1.68; ~$0.84 on D16s_v6) | R26, R13 and R15 results; the memory sampler. | queued | E21/E24 |
| R38 | P1 | **New vLLM CPU release A/B**: the pinned 0.31.0 digest vs the newest `vllm/vllm-openai-cpu` release on the same VM, W8A8 64-prompt mixed 512/128, interleaved A B A B, plus the `probe` step on the new image. Check for a release at the start of every round. | A newer release is ≥ 5% faster on mixed 512/128 (Intel reports +56% Offline from software alone between MLPerf v6.0 and v6.1 on Llama-3.1-8B; some of it should reach upstream vLLM). Falsified if < 5%; then stay on the pinned digest. | 0 to −20% (est.); also re-tests the unverified upstream fixes (#56419, #59107, #59829, BF16 GDN state). | low | ~0.5 VM-h ($0.84); append to any chain | A release newer than 0.31.0; record its digest. | queued (trigger: new release) | – |
| R04 | P1 | **Gemma 4 26B-A4B-it** (autoregressive MoE, 3.8B active) on vLLM CPU, BF16 (W8A8 if a checkpoint exists), 64/128/256 prompts on real-text prompts truncated or padded to 512 tokens (plus one random-prompt run for comparability); a cross-family hedge and the autoregressive control for R17 on the same VM. (In the first seed R04 was "CPU-friendlier models"; the Qwen candidates moved to R09/R10, and Phi-4 and the others were rejected in R32.) | BF16 mixed 512/128 (real text) at 128 prompts ≥ 270 tok/s (central 342), prefill ≥ 450 tok/s. Load risk: the CPU fused-MoE GELU_TANH gap ([vLLM #43326](https://github.com/vllm-project/vllm/issues/43326)) is *unverified* in 0.31; vLLM's fused-MoE activation docs list GELU_TANH (snippet), so it may already be fixed (the `probe` step prints the `cpu_moe` activation list). Falsified if BF16 mixed < 180 tok/s. | Est. on the E16ds_v7 basis: BF16 $1.11–1.69, INT8 $0.72–1.11 (other researchers: $0.75–1.7 on v6), i.e. 2–4.8× cheaper than $3.47, at ~3 points below Qwen3.6-35B-A3B (MMLU-Pro 82.6, GPQA 82.3). | low | ~1.5 VM-h ($2.52) | Run after R09 shows the CPU MoE path works; BF16 (~50 GB) needs a 128 GiB size; Hugging Face license acceptance for `google/` repos from the VM; R11 gate. | queued | E25 |
| R15 | P1 | **MTP k=1 on real 512/128 text** (e.g. 512-token news articles summarized in ≤ 128 tokens, thinking off) at 64, 128 and 256 sequences, MTP 0 and 1, KV sized for MTP (~29% fewer tokens per GiB: 24 GiB holds 70,485 tokens instead of 99,669; 16 GiB holds 46,933, i.e. 52 sequence slots of 896 tokens, so 64 sequences need ≥ 19.5 GiB with FP32 state). | MTP and larger batches compete for the same per-sequence budget, so the gain shrinks with the batch: the step model (0.29 s + 13.9 ms × n) plus E17's measured cost of the extra position (~12.7 ms per sequence) predicts +16% at 64, ~+10% at 128 and ~+6% at 256 (~+5% at 128 on top of a BF16 state), with ≥ 85% acceptance. Falsified (worse than predicted) if < +5% at 128 or < +10% at 64. | At 64 sequences 0.55 × (1 − 1/1.16) = −8%; at 128 ~−5%; at 256 ~−3%. In a stack: ~$1.36/M on D16s_v6 at 64 sequences with BF16 state (needs R13 to fit); ~$2.07/M on E16ds_v6 and ~$1.70/M on E16s_v6 at 128 with R12 and R13 (est., see (c)). The 128-sequence gain needs a 128 GiB size. | medium | ~1 VM-h ($1.68) | A real dataset on the VM; R12 for the KV size; #56419 if combined with structured output. | queued | E24 |
| R05 | P1 | **AMD Turin F16as_v7** (16 full Zen 5 cores, no SMT, AVX-512 VNNI/BF16, no AMX, 64 GiB; F16ams_v7 if 128 GiB is needed) in Central India ($0.706/h, 35% below eastus2). Plan on AMD's validated stack (vLLM 0.26 + zentorch 2.11.0.3 + PyTorch 2.11, ZenDNN 6.1; snippet) with AMD's Qwen3.6-35B-A3B W8A8 as a known-good control, or first confirm a zentorch build for torch 2.13 (our 0.31 image). Qwen3.8-27B W8A8 and BF16 at 16 and 64 prompts, 16 threads (one per core); before the throughput counts, the E16 paired subset (GSM8K 200 + MMLU 400, ~0.5 VM-h) against the AMX W8A8 answers. | Mixed 512/128 is 0.7–1.05× E16ds_v6: prefill 0.55–0.75× (VNNI peak ≈ 16 cores × 256 ops/cycle × ~4 GHz ≈ 16.4 TOPS; typical VNNI GEMM efficiency 50–65%; matching v6's measured 14.4 TOPS of prefill would need ~88%, while our AMX prefill reaches 29% of its own peak), decode 1–1.5× (batch-64 decode uses ~7% of AMX peak on Emerald Rapids). Falsified if < 0.55×, if the GDN or W8A8 paths don't run without AMX, or if MMLU differs from the AMX W8A8 run by > 1.5 points. | At ~$0.74/h all-in, 108–157 tok/s → est. **$1.3–1.9/M** (mid ~$1.6; **low confidence**). In eastus2 ($1.093/h) it beats D16s_v6 only at ≥ 1.35× E16ds_v6. | low | ~3–3.5 VM-h incl. the quality subset ($5.0–5.9; ~$2.5 at F16as_v7 prices) | Fasv7 usage in centralindia (20 vCPU per region per the capacity check) and whether the Visual Studio offer allows that geography; zentorch vs torch 2.13; vLLM's SGLang-derived GDN kernels may assume AMX; on 64 GiB the 27B needs R26's memory result. | queued | E26 |
| R02 | P1 | **SGLang CPU backend vs vLLM** on the same VM: W8A8 (`--quantization w8a8_int8`), BF16 and the official Qwen3.8-27B-FP8; `sglang.bench_offline_throughput` 512/128, 512/1 and 32/256 at 16 and 64 prompts; `--device cpu --attention-backend intel_amx --disable-radix-cache --disable-overlap-schedule`, `SGLANG_USE_CPU_ENGINE=1`, `SGLANG_CPU_OMP_THREADS_BIND` on the 8 physical cores; plus a vLLM W8A8 reference; before the throughput counts, the E16 paired subset (GSM8K 200 + MMLU 400, ~0.5 VM-h) for SGLang W8A8 against vLLM W8A8. | If both stacks run SGLang's AMX INT8 GEMM and GDN kernels for W8A8 (vLLM vendoring them, PR #50387/#50801, is *unverified*: snippets only; the `probe` step checks for sgl-kernel sources and symbols), then SGLang W8A8 lands at 0.85–1.3× vLLM; in BF16 and FP8 (vLLM uses oneDNN) SGLang is ≥ 1.15× faster. Falsified if W8A8 < 0.9× and BF16 within ±10%. | $2.39 → $1.84–2.81 (× 1/1.3 to × 1/0.85); a larger effect on the BF16/FP8 quality fallback. | low | ~3–3.5 VM-h incl. the quality subset ($5.0–5.9) | Without the workaround flags SGLang crashes at startup on hybrids ("extra_buffer needs CUDA/MUSA/NPU/ROCm/XPU", fix [PR #37705](https://github.com/sgl-project/sglang/pull/37705) open). The compressed-tensors W8A8 checkpoint may need PR #41330. These SGLang PR states are *unverified* (snippets). The newest `-xeon` image tag may predate Qwen3.5 (then build `xeon.Dockerfile`). No MTP on SGLang CPU yet (#35835). | queued | E14 |
| R39 | P1 | **Own W8A8 for Qwen3.8-27B**: llm-compressor with calibrated SmoothQuant + RTN (GPTQ only if time allows: slow on CPU), the most sensitive layers kept in BF16 (lm_head, first and last blocks, or the layers with the largest activation outliers); made on a 128 GiB VM; then the E16 paired test (GSM8K 200 + MMLU 400; R11 stage 2 if close) and the E18 mixed run. Replaces the unaudited community checkpoint (`Avesed/…`). | MMLU gap to BF16 ≤ 1 point (Avesed's W8A8: −2.75, p = 0.035) at ≤ 5% throughput loss. Falsified if the gap stays > 2 points or throughput drops > 5%. | 0 on $/M; avoids R18's +20–40% (FP8) if the MMLU loss matters for a job. | medium | ~2–3 VM-h incl. the paired test ($3.4–5.0) | pip llm-compressor on the VM; the ~54 GB BF16 download; calibration of a 27B on CPU (est. 0.5–1.5 h). | queued | – |
| R17 | P1 | **DiffusionGemma-26B-A4B vs its autoregressive twin** (R04) on the same VM: BF16, concurrency 1/4/16/64, canvas 256 and 128, `--max-num-seqs` raised above the recipe's 4 (~268 MB sampler state per sequence), on real-text prompts (E16 prompts; confidence-based unmasking makes the number of denoising steps content-dependent, so random prompts make steps per canvas meaningless), mean denoising steps per canvas logged, plus one random-prompt run; a 200-item E16 subset for both. | DiffusionGemma is ≥ 3× faster than the twin at concurrency 1; at 64 the twin's blended throughput is ≥ 1.3× DiffusionGemma's; DiffusionGemma is ≥ 3 points lower on the E16 subset. Falsified if DiffusionGemma's blended tok/s at 64 is ≥ the twin's with quality within 1 point. | From diffusion itself ~0 (best case parity): est. $1.1–7.7/M vs $0.7–1.7/M for the twin (**low confidence**). Answers the user's question with data. | low | ~1–1.5 VM-h appended to R04 ($1.7–2.5); 2.5–3.5 standalone | The 0.31.0 image may lack the CPU diffusion fixes (PR #59107, #59829; both *unverified*, `probe` step): then a nightly image or a source build (+30 min). Smoke test on 4 prompts first. The recipe mentions ≥ 4 NUMA nodes (ours: 1). | queued | E25 |
| R28 | P2 | **W4A8 (INT4 weights, dynamic INT8 activations on AMX) for the dense 27B.** First a desk check (the `probe` step of any chain): is there an x86 AMX W4A8 linear kernel reachable from compressed-tensors checkpoints? Run only if yes. | If the kernel exists, the 0.29 s weight term halves, the 64-sequence decode step falls ≥ 10%, and prefill stays within ±5% of W8A8. If the startup log selects a WNA16/W4A16 kernel, throughput drops to E17's W4A16 level and the idea is dead. | −6 to −8% at 64 sequences (more if the ~13 GB it frees allows larger batches). | low | 0 VM-h desk check; ~2 VM-h run + quality gate ($3.36) | vLLM has W4A8 for CPU MoE experts only (`fused_experts_cpu`); dense DA8W4 exists for AMD Zen (PR #54024) and in torchao; #38064 reports a silent W4A16 fallback. A W4A8 checkpoint has to be made on a VM. | queued | – |
| R41 | P2 | **The winning small model on D16ls_v6** (Dlsv6: 16 vCPU, 8 cores, 32 GiB, the same Xeon Platinum 8573C, $0.714/h in eastus2/westus2/centralindia, ~$0.744/h all-in with a P10 disk): re-run R10's (or R09's W8A8, if it fits) mixed runs at 64 sequences with whole-system memory logging. The 27B (37 GiB loaded) doesn't fit. | Qwen3.5-9B W8A8 at 64 sequences fits 32 GiB with min MemAvailable ≥ 3 GiB and is within 10% of its E16ds_v6 throughput. Falsified by OOM or < 0.85×. | × ~0.39 vs the v7 basis (0.744/1.681 × the measured v6 speed-up): Qwen3.5-9B W8A8 est. ~$0.41/M (mid 1.05 × 0.39). | medium | ~0.5 VM-h ($0.84; ~$0.37 at its price) | R10 passes R11 and the user accepts the model; Dlsv6 quota in the region (listed in the capacity checks). | blocked (trigger: R10 + R11) | – |
| R22 | P2 | **Prompt compression** (LLMLingua-2, XLM-RoBERTa-large, ~0.56B) at 2× on the unique part of the prompt, on a labelled extraction or classification set. | Accuracy within 1–2 points for classification (less reliable for exact-span extraction); the compressor costs ~1–2% of the 27B's prefill. | Prefill 1.95 → ~1.05 s: −22% per request ($1.53 → ~$1.19 per 1k requests; $2.39 → ~$1.86 per million original tokens); lossy. | low | ~1 VM-h ($1.68) | pip on the VM; a labelled dataset; the job owner accepts lossy inputs. | queued | – |
| R36 | P2 | **Sequence-level distillation (LoRA SFT)** of Qwen3.5-4B (or 2B) on ~5k Qwen3.8-27B extraction outputs (the user's job or the paraloq set), trained with LoRA on a VM, served with vLLM W8A8. | The tuned 4B is within the R33(c) bar of the 27B on a held-out extraction set (field F1, schema validity), at Qwen3.5-4B's cost (est. $0.66/M on the v7 basis, ~$0.46 on E16ds_v6). Falsified if field F1 stays > 3 points below the 27B after tuning. | Extraction only: est. −80% vs $2.39 (**low confidence**; CPU training time is the main unknown: ~6M training tokens of a 4B model ≈ 1e17 FLOP, est. 2–4 h on 8 AMX cores). | low | ~3–5 VM-h ($5.0–8.4) | R33(a) (an extraction job), R11's extraction set and gate; pip (peft, transformers) on the VM; R10 shows the Qwen3.5 small-model path works. | queued | – |
| R21 | P2 | **Runtime knob bundle** (absorbs R06): `--no-enable-prefix-caching` when nothing is shared (drops mamba 'align' checkpointing), `max-num-batched-tokens` 8192/16384 (logs show 4096), THP `always` vs `madvise`, check that tcmalloc and libiomp5 are loaded in the running process, an AOT compile-cache workaround (`VLLM_USE_AOT_COMPILE=0` or a matching torch), and `VLLM_CPU_SGL_KERNEL=1` for BF16 if it still exists. | Each knob gives 0–5%, combined 3–10%; a missing allocator preload, if found, ≥ 10%. Falsified if all are within 2× the CV measured in the chain prologue (never measured so far; every E17/E18 configuration was a single run). | 0 to −10% ($2.39 → ~$2.15–2.39); ~2.5–3 min less startup per engine start (matters for many short jobs). | medium | ~0.7 VM-h ($1.18), appended to chain A or C | Same VM as a reference run; root access to `/sys/kernel/mm/transparent_hugepage` (the container already runs `--privileged`). | queued | E21/E22 |
| R42 | P2 | **Async scheduling** (`--async-scheduling`: overlaps CPU-side scheduling with execution) if the 0.31 CPU backend supports it (`probe` step). | If R16 shows Python, scheduler or sampler work ≥ 20% of the decode step, overlap gives ≥ +8% decode at 64 sequences. Falsified if < 3%. | −3 to −8% (est.). | low | ~0.5 VM-h ($0.84) | R16's result; flag support on CPU. | blocked (trigger: R16) | – |
| R43 | P2 | **Two engines per VM**: 2 × 4-core vLLM engines (split `VLLM_CPU_OMP_THREADS_BIND`, 32 sequences each, half the KV each; the weights load twice, ~2 × 37 GiB, so 128 GiB sizes only) against 1 × 8-core engine with 64 sequences. | If R16 shows single-threaded CPU-side overhead ≥ 20% of the step, two engines give ≥ +8% total throughput. Falsified if < 3%, or if memory-bandwidth contention erases it. | −3 to −8% on 128 GiB sizes (est.); doesn't fit 64 GiB with the 27B. | low | ~0.5 VM-h ($0.84) | R16's result. | blocked (trigger: R16) | – |
| R20 | P2 | **n-gram and suffix speculation** (vLLM `ngram`; Arctic Inference `suffix`) on an extraction task whose outputs quote input spans, 64 sequences. | On copy-heavy outputs suffix decoding accepts ≥ 2 tokens per step and decode gains 15–40%; ~0 on free text. Falsified if acceptance stays below ~1 token per step, or the hybrid model rejects non-MTP drafters. | 0 to −16% for extraction jobs only (0.55 × 0–29%); can't be stacked with MTP. | low | ~1 VM-h ($1.68) | `arctic-inference` installed into the 0.31 CPU image; hybrid state rollback for these methods unverified; an extraction dataset. | queued | – |
| R29 | P2 | **MLPerf-style calibration**: one 778-in/73-out random run on Qwen3.5-9B W8A8 (or RedHatAI Llama-3.1-8B W8A8) during R10, to compare TOPS-equivalent per core with Intel's Xeon 6 MLPerf results. | ≥ 0.9 TOPS-equivalent per core (MLPerf v6.0 Xeon 6: ~0.84; our 27B mixed run: 1.05). Below 0.7 means our image lacks upstreamed Intel optimizations. | None directly; says whether a newer vLLM is worth chasing (R38; Intel claims +56% Offline from software in MLPerf v6.1, ~1.3 TOPS-equivalent per core). | medium | +0.1–0.2 VM-h ($0.17–0.34) on top of R10 | R10 scheduled. | queued | E19 |
| R18 | P2 | **Official Qwen/Qwen3.8-27B-FP8** (block 128 × 128) through vLLM's `CPUFp8BlockScaledMMKernel` (FP8 W8A16 on AMX-BF16), 512/128 at 16 and 64 prompts, plus MMLU 400 and GSM8K 200. | It loads on CPU; prefill runs at ~BF16 speed (~0.7× W8A8), decode between BF16 and W8A8; MMLU within 1 point of BF16 (W8A8 lost 2.75). | Cost up 20–40% vs W8A8 (~$2.9–3.3/M on v6), but ~25% below BF16 (~$3.8/M). Only worth it if W8A8's MMLU loss matters for a job. | medium | ~1.5 VM-h ($2.52) | Try R39 first (cheaper if it closes the MMLU gap). Kernel merged since May 2026 (PR #41186); confirm in 0.31 (`probe` step); 28 GB download; confirm the checkpoint is Qwen's own. | queued | – |
| R19 | P3 | **DFlash2 block-diffusion speculative decoding** (`z-lab/Qwen3.8-27B-DFlash2`, lossless, 7 draft tokens per block) in vLLM CPU from a PR branch; real-text GSM8K at 16 and 64 sequences, 3 and 7 draft tokens, vs MTP-1. Moved from P2 to P3 after E17. | At 64 sequences DFlash2 ≤ MTP-1 (+16%), because each extra verified position costs ~13–17 ms on this stack (E17: MTP-2 +14%, MTP-3 +6%); at 16 sequences it beats MTP-1 by 0–15%. Falsified if ≥ +30% at 64 sequences. | ~0 at our batch sizes; ≤ −5–10% for decode-heavy small-batch jobs. Re-score if R16 removes the per-position overhead. | low | ~2 VM-h + a source build from a PR branch (+30–45 min) ($3.36+) | vLLM DFlash support merged or buildable for CPU (the drafter's non-causal attention may lack a CPU path); R16 first. | queued | – |
| R07 | P3 | **Ollama sanity run** (optional): v0.35.1, `qwen3.8:27b` Q4_K_M, `OLLAMA_NUM_PARALLEL=16`, `num_thread=16`, `OLLAMA_KEEP_ALIVE=-1`, 16 requests of 512/128. | The log shows "model architecture does not currently support parallel requests" and llama-server runs with `-np 1`; throughput is within ±15% of llama.cpp single-sequence. | None: confirms est. $20–29/M on E16ds_v6 (9–12× the baseline). | high | ~0.5 VM-h ($0.84); can be appended to any chain | None; skipping it is reasonable, since the source code is conclusive. | queued | – |
| R31 | P3 | **Cobalt 100 (Arm)**: D16ps_v6 or E16ps_v6 in Central India ($0.37 / $0.485/h, 16 full cores, SVE2 with I8MM/BF16), vLLM aarch64 CPU build with W8A8 (oneDNN/ACL/KleidiAI), mimalloc, 16 threads. | Mixed throughput is 0.35–0.6× E16ds_v6 (SMMLA INT8 peak ~7 TOPS). Falsified if the GDN layers have no aarch64 CPU path. | At $0.40–0.52/h all-in, 54–92 tok/s → $1.2–2.7/M; beats D16s_v6 ($1.51) only at the optimistic end. | low | ~3 VM-h ($5.04; ~$1.5 at its price) + an arm64 image or a source build (30–60 min) | Dpsv6/Epsv6 quota and Central India availability. Cobalt 200 is early-access only. | queued | – |
| R23 | P3 | OpenVINO GenAI nightly or next release with the PagedGatedDeltaNetFusion fix ([PR #38488](https://github.com/openvinotoolkit/openvino/pull/38488), 2026-09-30) and an NNCF full-INT8 IR; re-run E10's prefill batch scaling (1/4/16 × 512/1). | If the paged GDN fusion now fires, prefill scales with batch to ≥ 100 tok/s and INT8 AMX brings it to 150–200; still ≤ 1× vLLM W8A8 on 512/128. | < 20% chance of beating vLLM; realistic $19–34 → ~$8–12/M. | low | ~2 VM-h ($3.36) + ~1 h INT8 export | Run only if release notes mention batched hybrid-model prefill. | blocked (trigger) | – |
| R24 | P3 | ik_llama.cpp refresh (no AMX, faster AVX-512 kernels, MTP, the hybrid multi-sequence fix PR #2260): Q8_0_R8 and IQ4_XS at np 1/16/32; optionally Ternary Bonsai 2 27B (a ternary Qwen3.8-27B derivative, ~6 GB) for a decode-heavy 32/256 job. | Prefill on 8 cores 40–70 tok/s (1.5–2.5× mainline llama.cpp, ~3× below vLLM W8A8); ternary decode ≥ 2× Q4_K_M at batch 16, prefill not faster. | ~$10–15/M on 512/128, no win expected; might win decode-heavy jobs only. | low | ~1–1.5 VM-h ($1.7–2.5) | Only if a decode-heavy workload enters scope. | blocked (trigger) | – |
| R44 | P3 | **Bake a VM image** (Azure Compute Gallery or a managed image in the RG) with Docker and the pinned vLLM image, to cut per-VM setup. Not the weights: a P10/P15 OS disk reads ~100–125 MB/s, slower than the ~1 GB/s Hugging Face download. | Boot to first benchmark falls by ≥ 3 min (cloud-init Docker install and the image pull). Falsified if the saving is < 2 min. | Fixed cost per VM falls by ~$0.05–0.1; matters only for many short jobs (see "Job-level cost"). | medium | ~0.5 VM-h ($0.84) + image storage (est. a few $/month) | An RG-scoped Compute Gallery; trigger: production jobs shorter than ~2 h. | blocked (trigger: short production jobs) | – |
| R25 | P3 | **Watch list**: SGLang CPU MTP/DFlash for hybrid GDN (#35835, #36782, #37851) and INT4 AMX routing (#41331); a dense DA8W4 path for x86 in vLLM or SGLang; Qwen3.8-Flash-Next CPU support (at 4.0–4.85 bits per weight with the 51B n-gram table quantized too it is 88–107 GB and would fit a 128 GiB VM; with the table in BF16, ~170 GB, it fits neither 128 GiB nor an E20ds_v7's 160 GiB); a Qwen3.8 MoE in the 30–40B range (community request only); gpt-oss-20b and Nemotron 3 Nano only if every Qwen/Gemma MoE run fails. | – | Speculative: MTP on SGLang CPU +10–30% decode; DA8W4 −15–25%; a Qwen3.8 ~35B-A3B would give R09's cost with Qwen3.8's instruction following. | – | 0 now; ~1.5 VM-h per re-test | Upstream releases. Re-check at the start of each round (desk, 15 min) and record the date checked here (last: 2026-10-04). | blocked (upstream) | – |
| R32 | P3 | **Rejected on desk evidence or measurement** (record only): patched Ollama (= llama.cpp no-AMX batched, $19–24/M); W4A16 on vLLM CPU (**measured** 4–6× slower, E17/E18); 16 threads per VM (**measured** −50%, E17); FP8/INT8 KV cache (~5% of decode traffic at 512/128, no memory gain under hybrid page alignment); activation sparsity (TEAL, Deja Vu, SparAMX: ≥ 75% dense at batch ≥ 32); LUT low-bit kernels (T-MAC, bitnet.cpp) and PowerInfer; EAGLE-3 heads and separate draft models (the MTP head already accepts 96%); xFasterTransformer, IPEX (end of life 2026-03), IPEX-LLM and TGI (archived), DeepSpeed CPU inference (IPEX-based); KTransformers (MoE with a GPU), CTranslate2 (no GDN), llamafile (= llama.cpp), ORT GenAI (no evidence of AMX-class prefill); HBv4/HBv5 (quota 0, 176/368 vCPU, $7.20/$19.80/h, not bandwidth-bound); Cobalt 200 (early access); Intel v7 D-series (no better per dollar than v6); Phi-4, Phi-4-mini, Phi-4-mini-flash-reasoning, BitNet b1.58 2B4T, gpt-oss-20b, Nemotron 3 Nano, Granite 4.x, LFM2 and Mistral Small 4 (quality, reasoning-token overhead or size, see (b)); small-block diffusion LMs (SDAR, LLaDA2.x, Fast-dLLM v2: below Qwen3.8's quality, SGLang dLLM CPU path unverified, autoregressive wins at batch). | – | ≤ 5% each, or negative. | – | 0 | Re-open only if R16 shows weight streaming dominating, or if the batch can't grow. | rejected | – |
Retired IDs: **R06** (vLLM runtime knobs: tcmalloc, Intel OpenMP, AOT compile cache) was merged into R21.
Priority changes on 2026-10-05: R27 P1 → P0 (free), R08 P1 → P0 (conditional on R33(b)), R14 P1 → P0
(best expected value per VM-hour among the measured levers).

## Hypotheses register

The key falsifiable hypotheses behind the queue. Each is tested by the rows named.

| ID | hypothesis | tested by | falsified if |
| --- | --- | --- | --- |
| H1 | The 64-prompt W8A8 job runs with 16 GiB of KV at the 24 GiB baseline's throughput, and on a 64 GiB D16s_v6 at ≥ 0.9× E16ds_v6 throughput (same CPU) with min MemAvailable ≥ 3 GiB. | R26 | OOM, preemption, MemAvailable < 3 GiB, or < 0.85× after reference normalization |
| H2 | On vLLM CPU, cost per token tracks active-parameter FLOPs: a 3B-active MoE and a 9B dense model are ≥ 2× cheaper per token than the 27B. | R09, R10, R04 | Qwen3.6-35B-A3B BF16 mixed (real-text prompts) < 180 tok/s, or Qwen3.5-9B W8A8 < 2× the 27B |
| H3 | In non-thinking mode Qwen3.6-35B-A3B is non-inferior to Qwen3.8-27B within the R33(c) margin (default 2 points) on GSM8K/MMLU and 3 on IFEval (Qwen3.5-9B within 4–5). | R11 | lower 95% bound of the paired difference below −margin (needs n ≈ 2,800 for a 2-point margin) |
| H4 | The decode step is ≈ 0.29 s + 13.9 ms per sequence, and at most ~55% of the per-sequence term is GEMM. | R12, R16 | < +5% decode from 64 to 128 sequences; GEMM ≥ 80% of the step |
| H5 | A BF16 GDN state cuts decode cost ≥ 4% and costs ≤ 1 point on MMLU and GSM8K and ≤ +3% mean output tokens. | R13 | < 4% gain, override ignored, > 1 point lost, output tokens +3% or more, or more truncations |
| H6 | MTP-1's gain shrinks with the batch on summarization-style text: ~+16% at 64, ~+10% at 128, ~+6% at 256 sequences. | R15 | < +5% at 128 (worse than predicted) |
| H7 | On this CPU, block diffusion loses to its autoregressive twin at ≥ 16 sequences. | R17 | DiffusionGemma ≥ the twin at 64 sequences on real-text prompts with quality within 1 point |
| H8 | A 16-core Turin without AMX reaches 0.7–1.05× E16ds_v6 mixed throughput with vLLM + zentorch (prefill 0.55–0.75×). | R05 | < 0.55×, or the GDN/W8A8 paths fail without AMX |
| H9 | SGLang W8A8 is within 0.85–1.3× vLLM because both run the same kernels (*unverified* premise; `probe` step). | R02 | outside that range |
| H10 | A managed model within 2 points of Qwen3.8-27B W8A8 on the task sets and a hard knowledge set costs ≤ 1/5 of the best CPU option per 1,000 requests, including reasoning tokens. | R08 | every candidate trails W8A8 by > 3 points on the task sets or the hard set |
| H11 | Every GPU and Spot quota reachable from this subscription is 0. | R27 | any nonzero A100/T4 GPU or Spot core quota |
| H12 | Prefix caching hits only for shared prefixes ≥ the attention block: 896 tokens with an FP32 GDN state; with a BF16 state, ≥ the logged block (est. 448–512). Fine-grained mode is the only route below the block. | R03, R13 | hits on shorter prefixes in default mode, or < 50% of the predicted saving at 896 |
| H13 | Multi-position verification (DFlash, MTP ≥ 2) loses to MTP-1 at 64 sequences on this stack. | R19 (E17 already supports it) | DFlash2 ≥ +30% at 64 sequences |
| H14 | For classification, one constrained label token keeps accuracy within 1 point of free text and halves cost per request; a distilled encoder reaches ≥ 97% of the 27B's accuracy at < 1/30 of its cost per request. | R34, R35 | > 1 point lost; encoder < 95% of the 27B's accuracy |
| H15 | Same-size instances differ by ≥ 10% in throughput, so best-of-3 pays for long jobs. | R40 | spread < 5% across ≥ 4 instances |
| H16 | Run-to-run CV of the vLLM reference on one VM is ≤ 2%, so 4–8% effects are measurable with interleaved A/B pairs. | every chain prologue | CV > 3% (then R13/R15/R21 need more repeats) |

## Sources

URLs the researchers cited, grouped. The per-claim mapping, confidence levels and snippet/blocked
notes are in [research/findings-2026-10-04.json](research/findings-2026-10-04.json). Pages on
huggingface.co, arxiv.org, vllm.ai, docs.vllm.ai, lmsys.org, docs.sglang.io, intel.com, amd.com,
ai.google.dev and several aggregators were blocked from the sandbox, so their figures come from search
snippets; learn.microsoft.com, azure.microsoft.com and prices.azure.com were read directly.

Repository measurements: [E03](E03-llamacpp-single-stream/), [E04](E04-llamacpp-amx-multiseq-bug/),
[E05](E05-llamacpp-batched/), [E07](E07-prefill-profile/), [E08](E08-threads-smt/), [E09](E09-vllm-cpu/),
[E10](E10-openvino-genai/), [E12](E12-speculative-decoding/), [E13](E13-emerald-vs-granite/),
[E15](E15-summary/README.md), [E16](E16-task-quality/README.md), [E17](E17-vllm-scaling-mtp/README.md),
[E18](E18-vllm-emerald-rapids/README.md), [COST_MODEL.md](COST_MODEL.md).

**Ollama**
- https://github.com/ollama/ollama/commit/9db4bdbad6a4981ad761aa2b603e69e8fb83212c
- https://github.com/ollama/ollama/commit/e36f389e8281da2532074ae3af53d3a7b1ca6ce5
- https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/server/sched.go#L502-L515
- https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/LLAMA_CPP_VERSION
- https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/llama/README.md
- https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/llama/server/CMakePresets.json
- https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/Dockerfile
- https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/create/metadata.go
- https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/llm/llama_server.go
- https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/api/types.go#L1140-L1152
- https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/envconfig/config.go#L276-L281
- https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/server/routes.go#L2217-L2227
- https://github.com/ollama/ollama/blob/42e911bc3d05798cad729cb474bf62f378cb2e26/docs/faq.mdx
- https://github.com/ollama/ollama/releases/tag/v0.30.0
- https://github.com/ollama/ollama/issues/14510, https://github.com/ollama/ollama/issues/14879,
  https://github.com/ollama/ollama/issues/16415, https://github.com/ollama/ollama/issues/18038
- https://github.com/Chrisma-98/ollama-qwen3.5-GPU-parallel
- https://github.com/ggml-org/llama.cpp/blob/b11232/ggml/src/CMakeLists.txt,
  https://github.com/ggml-org/llama.cpp/blob/b11232/common/common.cpp
- https://www.yottalabs.ai/post/how-to-run-qwen-3-8-with-ollama-2026, https://www.orcarouter.ai/blog/qwen-3-8-27b-ollama
- https://developers.redhat.com/articles/2025/08/08/ollama-vs-vllm-deep-dive-performance-benchmarking,
  https://www.sitepoint.com/ollama-vs-vllm-performance-benchmark-2026/,
  https://markaicode.com/benchmarks/tool-throughput-benchmark/,
  https://www.glukhov.org/llm-hosting/comparisons/llama-cpp-vs-ollama/

**vLLM CPU backend**
- PRs: https://github.com/vllm-project/vllm/pull/41025, https://github.com/vllm-project/vllm/pull/41186,
  https://github.com/vllm-project/vllm/pull/49942, https://github.com/vllm-project/vllm/pull/50387,
  https://github.com/vllm-project/vllm/pull/50801, https://github.com/vllm-project/vllm/pull/58932,
  https://github.com/vllm-project/vllm/pull/36649, https://github.com/vllm-project/vllm/pull/46384,
  https://github.com/vllm-project/vllm/pull/47861, https://github.com/vllm-project/vllm/pull/53896,
  https://github.com/vllm-project/vllm/pull/54024
- Issues: https://github.com/vllm-project/vllm/issues/38064, https://github.com/vllm-project/vllm/issues/40696,
  https://github.com/vllm-project/vllm/issues/43326, https://github.com/vllm-project/vllm/issues/45238,
  https://github.com/vllm-project/vllm/issues/53504, https://github.com/vllm-project/vllm/issues/55196,
  https://github.com/vllm-project/vllm/issues/56419
- Releases: https://github.com/vllm-project/vllm/releases/tag/v0.29.0, https://github.com/vllm-project/vllm/releases/tag/v0.30.0
- Docs: https://docs.vllm.ai/en/latest/getting_started/installation/cpu/,
  https://docs.vllm.ai/en/v0.17.1/getting_started/installation/cpu/,
  https://docs.vllm.ai/en/v0.10.2/getting_started/installation/cpu.html,
  https://docs.vllm.ai/en/v0.9.2/getting_started/installation/cpu.html,
  https://docs.vllm.ai/en/v0.6.0/getting_started/cpu-installation.html,
  https://docs.vllm.ai/en/stable/api/vllm/platforms/cpu/, https://docs.vllm.ai/en/v0.10.1/api/vllm/platforms/cpu.html,
  https://docs.vllm.ai/en/latest/api/vllm/config/cache/,
  https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/scaled_mm/,
  https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/experts/cpu_moe/,
  https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int8_w4a8/,
  https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/,
  https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/,
  https://docs.vllm.ai/en/latest/cli/bench/throughput/, https://docs.vllm.ai/en/v0.12.0/design/debug_vllm_compile/,
  https://openi.pcl.ac.cn/JeffDing/vllm/src/tag/v0.14.0/docker/Dockerfile.cpu
- Blogs and recipes: https://vllm.ai/blog/2026-04-22-fp8-kvcache, https://vllm.ai/blog/2026-05-28-speculators-v050,
  https://vllm.ai/blog/2026-07-29-optimizing-vllm-on-arm-cpus, https://recipes.vllm.ai/Qwen/Qwen3.5-35B-A3B,
  https://recipes.vllm.ai/Qwen/Qwen3-30B-A3B, https://recipes.vllm.ai/openai/gpt-oss-20b,
  https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
- Hybrid-model state and prefix caching: https://huggingface.co/Qwen/Qwen3.5-397B-A17B/discussions/2,
  https://docs.modular.com/max/api/python/pipelines.architectures.qwen3_5/, https://docs.lmcache.ai/mp/hybrid_models.html
- https://github.com/vllm-project/llm-compressor/issues/2787

**SGLang and other CPU stacks**
- SGLang: https://www.lmsys.org/blog/2025-07-14-intel-xeon-optimization/, https://docs.sglang.io/platforms/cpu_server.html,
  https://docs.sglang.ai/references/cpu.html, https://docs.sglang.ai/basic_usage/qwen3.html,
  https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5,
  https://github.com/sgl-project/sglang/pull/12525, https://github.com/sgl-project/sglang/pull/22498,
  https://github.com/sgl-project/sglang/pull/32959, https://github.com/sgl-project/sglang/pull/35835,
  https://github.com/sgl-project/sglang/pull/36782, https://github.com/sgl-project/sglang/pull/37705,
  https://github.com/sgl-project/sglang/pull/37851, https://github.com/sgl-project/sglang/pull/41329,
  https://github.com/sgl-project/sglang/pull/41330, https://github.com/sgl-project/sglang/pull/41331
- Intel: https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Cost-Effective-Deployment-of-DeepSeek-R1-with-Intel-Xeon-6-CPU/post/1704597,
  https://github.com/intel/xFasterTransformer, https://pytorch-extension.intel.com/,
  https://github.com/intel/intel-extension-for-pytorch, https://github.com/intel/intel-extension-for-pytorch/issues/581,
  https://github.com/intel/ipex-llm, https://phoronix.com/news/Intel-Ending-BigDL,
  https://github.com/huggingface/text-generation-inference
- OpenVINO: https://github.com/openvinotoolkit/openvino/pull/35070, https://github.com/openvinotoolkit/openvino/pull/35532,
  https://github.com/openvinotoolkit/openvino/pull/38488, https://github.com/openvinotoolkit/openvino/releases,
  https://github.com/openvinotoolkit/openvino.genai/pull/4065
- ik_llama.cpp: https://github.com/ikawrakow/ik_llama.cpp, https://github.com/ikawrakow/ik_llama.cpp/issues/2409,
  https://github.com/ikawrakow/ik_llama.cpp/issues/437, https://github.com/ikawrakow/ik_llama.cpp/pull/2260,
  https://github.com/ikawrakow/ik_llama.cpp/pull/2550, https://datanorth.ai/news/prismml-releases-ternary-bonsai-2-27b
- Others: https://github.com/mozilla-ai/llamafile, https://www.helpnetsecurity.com/2026/03/20/llamafile-0-10-0-released/,
  https://github.com/kvcache-ai/ktransformers, https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md,
  https://www.lmsys.org/blog/2025-10-22-KTransformers,
  https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf,
  https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.30.0,
  https://ai.azure.com/catalog/models/qwen3.5-2b-generic-cpu, https://github.com/OpenNMT/CTranslate2,
  https://pytorch.org/blog/high-performance-quantized-llm-inference-on-intel-cpus-with-native-pytorch/,
  https://docs.pytorch.org/ao/main/generated/torchao.quantization.Int8DynamicActivationInt4WeightConfig.html,
  https://github.com/microsoft/BitNet, https://arxiv.org/abs/2407.00088

**Models**
- Qwen3.8: https://huggingface.co/Qwen/Qwen3.8-27B-FP8, https://huggingface.co/Qwen/Qwen3.8-27B/discussions/120,
  https://friendli.ai/models/Qwen/Qwen3.8-27B, https://friendli.ai/models/Qwen/Qwen3.8-27B-FP8,
  https://www.qubrid.com/blog-news/qwen38-27b-api-the-complete-developer-guide,
  https://benchlm.ai/models/qwen3-8-27b, https://llm-stats.com/models/compare/qwen3.6-27b-vs-qwen3.8-27b,
  https://emergent.sh/learn/qwen-3-8-benchmarks, https://artificialanalysis.ai/models/qwen3-8-27b,
  https://quesma.com/blog/qwen38-27b-quantizations-benchmarked/,
  https://codersera.com/blog/qwen-3-8-model-lineup-2026/, https://kie.ai/blog/what-is-qwen3-8-35b,
  https://qwen.ai/blog?id=qwen3.8-flash-next, https://github.com/QwenLM/Qwen3.8-Flash-Next,
  https://datanorth.ai/news/alibaba-releases-qwen3-8-flash-next,
  https://intuitionlabs.ai/articles/qwen3-8-flash-next-architecture-memory
- Qwen3.6 and Qwen3.5: https://qwen.ai/blog?id=qwen3.6-35b-a3b, https://qwen.ai/blog?id=qwen3.6-27b,
  https://huggingface.co/Qwen/Qwen3.6-35B-A3B, https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8,
  https://huggingface.co/blog/EXDai/qwen36-35b-a3b-architecture-overview,
  https://llm-stats.com/models/compare/qwen3.6-27b-vs-qwen3.6-35b-a3b, https://apxml.com/models/qwen36-35b-a3b,
  https://friendli.ai/models/88plug/Qwen3.6-35B-A3B-W8A16, https://huggingface.co/Qwen/Qwen3.5-27B,
  https://huggingface.co/Qwen/Qwen3.5-9B, https://apxml.com/models/qwen35-9b,
  https://artificialanalysis.ai/articles/qwen3-5-small-models,
  https://rits.shanghai.nyu.edu/ai/qwen-3-5-medium-series-frontier-ai-that-fits-on-your-gpu/,
  https://www.xda-developers.com/qwen-3-5-9b-tops-ai-benchmarks-not-how-pick-model/
- Gemma 4: https://huggingface.co/blog/gemma4, https://sebastianraschka.com/blog/2026/gemma-4-release-notes.html,
  https://arxiv.org/pdf/2607.02770, https://docs.nvidia.com/nemo/megatron-bridge/latest/models/gemma/gemma4-vl.html,
  https://aurigait.com/blog/gemma-4-features-benchmarks-guide/,
  https://danilchenko.dev/posts/2026-04-07-run-gemma-4-locally-ollama-llama-cpp-vllm/,
  https://apxml.com/models/gemma-4-26b-a4b, https://www.jetson-ai-lab.com/models/gemma4-26b-a4b,
  https://benchlm.ai/compare/gemma-4-26b-a4b-vs-qwen3-8-27b
- Phi: https://arxiv.org/pdf/2412.08905, https://ai.azure.com/catalog/models/Phi-4,
  https://azure.microsoft.com/en-us/blog/reasoning-reimagined-introducing-phi-4-mini-flash-reasoning/,
  https://www.forbes.com/sites/janakirammsv/2026/03/06/microsoft-builds-a-compact-ai-model-that-decides-when-to-think/,
  https://www.spheron.network/blog/deploy-phi-5-gpu-cloud/
- Others: https://huggingface.co/openai/gpt-oss-20b, https://build.nvidia.com/openai/gpt-oss-20b/modelcard,
  https://docs.nvidia.com/nemo/megatron-bridge/latest/models/gpt_oss/gpt-oss.html,
  https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf,
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8, https://huggingface.co/blog/ibm-granite/granite-4-1,
  https://huggingface.co/ibm-granite/granite-4.0-h-small, https://huggingface.co/LiquidAI/LFM2-8B-A1B,
  https://www.liquid.ai/blog/lfm2-24b-a2b, https://huggingface.co/mistralai/Mistral-Small-4-119B-2603,
  https://langdb.ai/app/models/openrouter/ernie-4.5-21b-a3b
- MoE routing at batch: https://arxiv.org/abs/2511.02237, https://icml.cc/virtual/2026/poster/62958

**Diffusion LMs**
- DiffusionGemma: https://huggingface.co/google/diffusiongemma-26B-A4B-it, https://arxiv.org/abs/2608.00146,
  https://www.alphaxiv.org/overview/2608.00146, https://ai.google.dev/gemma/docs/diffusiongemma,
  https://ai.google.dev/gemma/docs/diffusiongemma/model_card,
  https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/,
  https://vllm.ai/blog/2026-06-10-diffusion-gemma, https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it,
  https://github.com/vllm-project/recipes/blob/cbf76c0/models/Google/diffusiongemma-26B-A4B-it.yaml,
  https://github.com/vllm-project/vllm/blob/155488d/vllm/config/diffusion.py,
  https://github.com/vllm-project/vllm/blob/155488d/vllm/model_executor/models/diffusion_gemma_sampler.py,
  https://github.com/vllm-project/vllm/blob/155488d/tests/evals/gsm8k/configs/DiffusionGemma-26B-A4B-it-FP8-dynamic.yaml,
  https://github.com/vllm-project/vllm/pull/59107, https://github.com/vllm-project/vllm/pull/59829,
  https://github.com/vllm-project/vllm/issues/59828, https://github.com/vllm-project/vllm/pull/57250,
  https://github.com/mmastrac/djev, https://explainx.ai/blog/diffusiongemma-jev-vllm-open-source-2026,
  https://build.nvidia.com/google/diffusiongemma-26b-a4b-it/modelcard,
  https://huggingface.co/docs/diffusers/main/api/pipelines/diffusion_gemma, https://unsloth.ai/docs/models/diffusiongemma,
  https://developers.redhat.com/articles/2026/09/28/run-decision-model-vllm-and-red-hat-ai,
  https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/deploying_models/deploying_diffusiongemma_models,
  https://docs.sglang.io/cookbook/autoregressive/Google/DiffusionGemma,
  https://llm-stats.com/models/compare/diffusiongemma-26b-a4b-it-vs-gemma-4-26b-a4b-it,
  https://www.digitalapplied.com/blog/google-diffusiongemma-open-weight-text-diffusion-model-guide,
  https://toolhalla.ai/blog/diffusiongemma-26b-local-inference-speed-tradeoffs, https://arxiv.org/abs/2606.14620,
  https://gigazine.net/gsc_news/en/20260611-google-ai-diffusiongemma/,
  https://the-decoder.com/googles-new-open-model-diffusiongemma-generates-text-from-noise-instead-of-word-by-word/,
  https://the-decoder.com/googles-diffusiongemma-proves-you-dont-need-to-train-from-scratch-to-build-a-text-diffusion-model/,
  https://www.spheron.network/blog/deploy-diffusiongemma-gpu-cloud/,
  https://www.spheron.network/tools/gpu-recommender/nvidia/diffusiongemma-26B-A4B-it-NVFP4/,
  https://dev.to/reidmarlow/diffusiongemma-is-fast-because-it-stops-pretending-text-has-to-be-written-left-to-right-2h2n
- llama.cpp and Ollama support: https://github.com/ggml-org/llama.cpp/blob/7f2dd88/src/llama-arch.cpp,
  https://github.com/ggml-org/llama.cpp/blob/7f2dd88/examples/diffusion/README.md, https://diffusiongemma.dev/llama-cpp/,
  https://avenchat.com/blog/does-diffusiongemma-work-with-llama-cpp,
  https://dev.to/vigoss_luke_3604c1d0e9b4a/diffusiongemma-in-july-2026-what-works-what-doesnt-and-when-ollama-gets-it-3945
- Other diffusion LMs and engines: https://arxiv.org/abs/2512.15745, https://arxiv.org/abs/2602.08676,
  https://huggingface.co/inclusionAI/LLaDA2.1-flash, https://huggingface.co/inclusionAI/LLaDA2.2-flash,
  https://www.orcarouter.ai/de/blog/llada-2-2-mini-release, https://alphaxiv.org/abs/2509.24389,
  https://arxiv.org/abs/2510.06303, https://arxiv.org/pdf/2602.09555, https://arxiv.org/abs/2509.26328,
  https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B, https://github.com/vz415/RND1,
  https://arxiv.org/abs/2512.22737, https://huggingface.co/tencent/WeDLM-8B-Base, https://arxiv.org/abs/2512.14067,
  https://arxiv.org/abs/2609.20751, https://github.com/AntonXue/dQwen, https://arxiv.org/html/2508.15487v1,
  https://arxiv.org/abs/2510.08666, https://github.com/inclusionAI/dInfer,
  https://docs.sglang.io/supported_models/text_generation/diffusion_language_models.html,
  https://www.lmsys.org/blog/2025-12-19-diffusion-llm,
  https://huggingface.co/diffuse-cpp/LLaDA-8B-Instruct-GGUF/blob/main/README.md,
  https://huggingface.co/diffuse-cpp/Dream-v0-Instruct-7B-GGUF/blob/main/README.md,
  https://huggingface.co/naranor/Dream-Coder-7B-ov-int8, https://openclawdc.com/blog/open-text-diffusion-models-local/
- Efficiency at batch: https://arxiv.org/abs/2510.18480, https://arxiv.org/html/2505.22618v3,
  https://arxiv.org/pdf/2511.00606, https://arxiv.org/abs/2605.24832, https://arxiv.org/abs/2601.23278,
  https://arxiv.org/abs/2608.23807, https://arxiv.org/abs/2512.17077,
  https://www.spheron.network/blog/deploy-diffusion-language-models-dllm-gpu-cloud-2026/,
  https://medium.com/data-science-collective/diffusion-llms-claim-1-000-tokens-a-second-i-got-1-05-7af472e586f8
- Closed models: https://deepmind.google/models/gemini-diffusion/, https://www.cometapi.com/tag/gemini-diffusion,
  https://www.inceptionlabs.ai/blog/mercury-2-on-azure-foundry, https://ai.azure.com/catalog/models/Mercury-2,
  https://llm-stats.com/models/mercury-2, https://openrouter.ai/inception/mercury-2.5-preview,
  https://runtimewire.com/article/inception-mercury-2-5-diffusion-language-model-launch
- DFlash: https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2, https://github.com/z-lab/dflash, https://arxiv.org/abs/2602.06036,
  https://www.mindstudio.ai/blog/qwen3-8-27b-dflash2-speculative-decoding,
  https://www.mindstudio.ai/blog/run-qwen3-8-27b-dflash2-vllm-sglang/, https://interfaze.ai/models/z-labqwen38-27b-dflash2,
  https://rits.shanghai.nyu.edu/ai/luce-dflash-brings-2x-speculative-decoding-to-qwen3-6-27b-on-a-single-rtx-3090/

**Techniques**
- Speculative decoding at batch: https://arxiv.org/pdf/2408.11049, https://arxiv.org/abs/2411.04975,
  https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/
- Sparsity and low-bit kernels: https://arxiv.org/pdf/2408.14690, https://arxiv.org/pdf/2310.17157,
  https://arxiv.org/pdf/2505.14884, https://arxiv.org/abs/2502.12444, https://arxiv.org/html/2407.00088v2
- Prompt compression and structured output: https://arxiv.org/abs/2403.12968v1, https://github.com/microsoft/LLMLingua,
  https://arxiv.org/pdf/2411.15100
- THP: https://www.phoronix.com/review/thp-madvise-always/4, https://docs.amd.com/r/en-US/57300-ZenDNN-user-guide/Transparent-Huge-Pages
- FP8 hardware: https://phoronix.com/news/Intel-AMX-FP8-In-LLVM,
  https://aiweekly.co/alerts/intel-bumps-xeon-7-diamond-rapids-to-256-cores-slips-to-2027

**AMX and CPU-inference literature**
- https://arxiv.org/pdf/2507.03522, https://arxiv.org/pdf/2609.04663,
  https://elijah-ye.github.io/assets/pdf/Exploiting_Intel_Advanced_Matrix_Extensions_AMX_for_Large_Language_Model_Inference.pdf,
  https://arxiv.org/abs/2407.07304, https://arxiv.org/abs/2407.00029, https://arxiv.org/abs/2311.00502,
  https://arxiv.org/abs/2505.19349, https://arxiv.org/abs/2505.06461, https://arxiv.org/abs/2507.18454,
  https://arxiv.org/abs/2406.07553, https://ieeexplore.ieee.org/document/10763564,
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/8cd1ce03ea58b3d7dfd809e4d42f08ea-Abstract-Conference.html,
  https://proceedings.mlsys.org/paper_files/paper/2025/hash/66a026c0d17040889b50f0dfa650e5e0-Abstract-Conference.html
- MLPerf: https://lenovopress.lenovo.com/lp2414-sr650-v4-scalable-enterprise-ai-performance-in-mlperf-60,
  https://www.intel.com/content/www/us/en/newsroom/news/data-center/intel-software-optimizations-boost-ai-inference-in-mlperf-v6-1.html,
  https://hyperframeresearch.com/2026/09/18/did-mlperf-v6-1-just-make-the-case-for-the-xeon-you-already-own/,
  https://electronics-usa.com/news/114887-intel-demonstrates-ai-inference-gains-in-mlperf-inference-v6-1-benchmarks,
  https://arxiv.org/pdf/2512.11588
- AMX clocks: https://el.whatpsu.com/articles/2553-Intel-Reveals-Granite-Rapids-WS-Xeon-600-Turbo-Frequencies-AVX-512-and-AMX-Significantly-Reduce-Boost-Speeds,
  https://cdrdv2-public.intel.com/873642/Intel%20Xeon%20600%20Processors%20for%20Workstation_GNR-W_PressPresentation_Rev1.pdf,
  https://www.phoronix.com/review/intel-xeon-amx/3

**Azure hardware, prices, offers and quotas**
- Prices: https://prices.azure.com/api/retail/prices (filters on `serviceName eq 'Virtual Machines'`, `'Storage'`,
  `'Foundry Models'`, `'Azure Container Apps'`; re-checked 2026-10-04)
- VM sizes: https://learn.microsoft.com/azure/virtual-machines/sizes/general-purpose/dsv6-series,
  https://learn.microsoft.com/azure/virtual-machines/sizes/general-purpose/ddsv6-series,
  https://learn.microsoft.com/azure/virtual-machines/sizes/memory-optimized/edsv6-series,
  https://learn.microsoft.com/azure/virtual-machines/sizes/general-purpose/dsv7-series,
  https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/memory-optimized/edsv7-series,
  https://learn.microsoft.com/azure/virtual-machines/sizes/compute-optimized/fasv7-series,
  https://learn.microsoft.com/azure/virtual-machines/sizes/compute-optimized/famsv7-series,
  https://learn.microsoft.com/azure/virtual-machines/sizes/general-purpose/dasv7-series,
  https://techcommunity.microsoft.com/blog/azurecompute/-/4448360, https://en.wikipedia.org/wiki/Zen_5,
  https://learn.microsoft.com/azure/virtual-machines/sizes/cobalt-overview,
  https://azure.microsoft.com/en-us/blog/new-azure-cobalt-200-vms-deliver-50-performance-improvement-fully-optimized-for-modern-agentic-ai-workloads/,
  https://learn.microsoft.com/azure/virtual-machines/sizes/high-performance-compute/hb-family,
  https://learn.microsoft.com/azure/virtual-machines/sizes/high-performance-compute/hbv5-series,
  https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/nc-rtxpro6000-bse-v6-series
- AMD software: https://docs.amd.com/r/en-US/57300-ZenDNN-user-guide/vLLM-In-Tree-Platform,
  https://www.amd.com/en/developer/resources/technical-articles/2026/amd-pace-integrates-with-vllm.html,
  https://www.amd.com/en/developer/resources/technical-articles/2026/zendnn-5-2-1-on-amd-epyc-cpus.html
- Offers and quotas: https://learn.microsoft.com/visualstudio/subscriptions/faq/subscriber/azure/,
  https://learn.microsoft.com/azure/virtual-machines/spot-vms#limitations,
  https://learn.microsoft.com/azure/cost-management-billing/reservations/prepare-buy-reservation,
  https://learn.microsoft.com/answers/a/12302701, https://learn.microsoft.com/answers/a/12957599,
  https://learn.microsoft.com/answers/a/12252412, https://learn.microsoft.com/azure/batch/batch-quota-limit,
  https://learn.microsoft.com/azure/batch/batch-spot-vms,
  https://learn.microsoft.com/azure/container-apps/gpu-serverless-overview, https://learn.microsoft.com/azure/container-apps/quotas,
  https://learn.microsoft.com/azure/container-apps/billing, https://learn.microsoft.com/azure/container-apps/workload-profiles-overview,
  https://learn.microsoft.com/azure/machine-learning/how-to-manage-quotas?view=azureml-api-2,
  https://learn.microsoft.com/azure/machine-learning/how-to-manage-optimize-cost?view=azureml-api-2

**GPU throughput and managed-API prices**
- https://docs.gpustack.ai/2.0/performance-lab/qwen3-32b/h100/, https://www.databasemart.com/blog/vllm-gpu-benchmark-pro6000,
  https://databasemart.com/blog/vllm-gpu-benchmark-h100
- https://datanorth.ai/news/openai-launches-gpt-6-sol-and-luna, https://artificialanalysis.ai/models/qwen3-8-27b/providers,
  https://computeprices.com/providers/deep-infra/models/qwen3-8-27b, https://llmgateway.io/models/qwen3.8-27b,
  https://cloudprice.net/models/openrouter/qwen/qwen3.5-27b

**Review pass (2026-10-05)** (snippets unless noted)
- Model and checkpoint facts: https://friendli.ai/models/amd/Qwen3.6-35B-A3B-w8a8-llmcompressor-v0.12.0,
  https://docs.vllm.ai/projects/llm-compressor/en/latest/key-models/qwen3.5/ (RedHatAI/Qwen3.5-9B-quantized.w8a8),
  https://arxiv.org/pdf/2503.01743 (Phi-4-mini), https://arxiv.org/pdf/2507.06607 (Phi-4-mini-flash-reasoning),
  https://arxiv.org/pdf/2504.12285 (BitNet b1.58 2B4T),
  https://www.yottalabs.ai/post/qwen-3-8-flash-next-specs-qwen-4-preview-2026
- Quality comparisons: https://artificialanalysis.ai/models/comparisons/qwen3-8-27b-low-vs-gpt-oss-120b,
  https://emelia.io/hub/qwen-35-9b-review, https://aicybr.com/blog/deepseek-v4-flash-complete-guide,
  https://benchlm.ai/md/models/qwen3-8-27b.md, https://huggingface.co/datasets/paraloq/json_data_extraction
- Recurrent-state precision: https://arxiv.org/pdf/2604.12374 (Nemotron 3 Super),
  https://docs.nvidia.com/nemotron/nightly/nemotron/super3/quantization.html
- vLLM: https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/activation/
- Ollama release date: https://insiderllm.com/guides/ollama-0-30-0-whats-new/
- Azure (read directly): https://learn.microsoft.com/azure/virtual-machines/sizes/memory-optimized/edsv6-series,
  https://learn.microsoft.com/azure/cost-management-billing/savings-plan/savings-plan-overview,
  https://learn.microsoft.com/azure/foundry-classic/openai/how-to/managed-identity; Retail Prices API
  for D16ls_v6 (2026-10-05)

## Open questions

Grouped by what answers them. Questions that a queued row answers name the row.

**Inside the vLLM 0.31.0 image** (`vllm/vllm-openai-cpu@sha256:8024248339dc6878daa5349344ed29d49c8a6732f6bdf7400fda32ce33e4b30b`;
the chain `probe` step answers most of these)
- Does it include PR #50801 (W8A8 always on SGLang's AMX INT8 kernel) and the 2026-07-30 sgl-kernels
  sync? If so, E09's prefill is already SGLang-kernel performance and R02's upside shrinks. Does
  `VLLM_CPU_SGL_KERNEL` still exist?
- Does `--mamba-ssm-cache-dtype bfloat16` work for Qwen3.8 (its config pins float32), and does the CPU
  GDN decode kernel accept a BF16 state? (R13)
- Are `--enable-mamba-fine-grained-prefix-cache` (PR #46384) and the 'all' mamba cache mode (PR #36649)
  present and supported on CPU? Does 'align' mode cost anything when no prefix can hit? (R03, R21)
- Are the DiffusionGemma CPU fixes (PR #59107, #59829) and model runner v2 on CPU in it? (R17)
- Is `CPUFp8BlockScaledMMKernel` in it, and is there a dense x86 W4A8 linear kernel? (R18, R28)
- Is the Gemma 4 GELU_TANH CPU fused-MoE gap (#43326) fixed? vLLM's fused-MoE activation docs list
  GELU_TANH (snippet), so probably. (R04)
- Does `--async-scheduling` exist and work on CPU? (R42)
- Does the Intel CPU path load INT8 MoE experts (AMD's Qwen3.6-35B-A3B W8A8), and which MoE method
  does it log? (R09)

**Performance unknowns**
- Where do the ~13.9 ms per sequence per decode step go: decode-sized INT8 GEMMs, FP32 GDN state
  reads/writes or gather/scatter copies, the sampler over 64 × 248k logits, or Python scheduling? (R16)
- What clock does AMX actually run at on the 6973P-C (v7) and the 8573C (v6) under load? There is no
  PMU in these VMs, and Azure documents only non-AVX turbo. Why does the v7 instance prefill slower than
  v6 despite a higher nominal clock: AMX frequency, instance variance, or memory/LLC? (R16)
- Do D-series hosts give the same memory bandwidth and AMX clocks as E-series hosts with the same CPU,
  and does the 64-prompt job fit in 64 GiB with the Defender and monitoring agents running? Unmeasured:
  the baseline used 24 GiB of KV (largest process ~65 GB), and the RSS sampler misses every other
  process. (R26)
- What is the run-to-run CV of a vLLM run on one VM? Never measured; every E17/E18 configuration was a
  single run. (chain prologue)
- Does a lower-precision GDN state lengthen outputs (Nemotron 3 Super's FP16-cast failure mode)? (R13)
- How efficient is vLLM CPU's MoE path for `qwen3_5_moe` and Gemma 4 (prepack vs SGL kernel)? Efficiency
  0.2–0.7 moves Qwen3.6-35B-A3B between ~$1.5 and ~$0.8/M on the v7 basis, so it dominates the
  estimates. How skewed is real expert routing at 64–256 sequences? (R09, R04)
- Does Qwen3.8 keep its accuracy with a BF16 recurrent state? No published ablation. (R13)
- How many denoising steps does DiffusionGemma need for 128-token answers, and does a 128-token canvas
  halve the compute without hurting quality? Does its vLLM CPU path run on a single-NUMA 8-core VM? (R17)
- Does the vLLM CPU backend run Qwen3.5/3.8 GDN layers on non-AMX x86 (Zen 5) and on aarch64, and does
  zentorch install against torch 2.13? (R05, R31)
- How fast is Qwen3.8-27B on GPUs at 512/128 with vLLM FP8/NVFP4? No public benchmark; the GPU $/M
  figures are extrapolated from Qwen3-32B.

**Quality**
- Every published score for the candidate models is thinking mode. How large is the gap at 128-token
  non-thinking answers, especially on IFEval/IFBench and strict JSON extraction? (R11)
- Is a switch from Qwen3.8-27B to a ~3–4B-active MoE or a 9B dense model within the user's
  "acceptable quality"? Only the user can decide (R33(c)), with R11's numbers.
- Do the managed candidates (GPT-6 Luna, GPT-5 mini, DeepSeek V4 Flash) match Qwen3.8-27B on sets
  that aren't near ceiling, and at how many billed reasoning tokens? (R08)
- Is `Qwen/Qwen3.8-27B-FP8` Qwen's own checkpoint (Hugging Face was blocked from the sandbox)? (R18)

**Decisions from the user (R33)**
- What do the real batch jobs look like: shared instruction length, unique input length, output
  length, classification or extraction, copy-heavy or not, share of exact-duplicate prompts, monthly
  volume? This decides R03 (prefix caching needs a shared prefix ≥ the attention block), R14, R20,
  R22 and R34–R36.
- May the data go to Azure AI Foundry managed models? (R08)
- What is the quality bar? Proposed default in R33(c).
- What is the monthly Azure credit, and how much of it may round 3 spend? (PLAN.md budget)

**Account and quota (orchestrator or account owner)**
- What are the Container Apps GPU, Batch-account and Azure ML quotas for this subscription, and are the
  needed resource providers already registered? (R27)
- Can this Visual Studio subscription deploy Foundry models (GPT-6 Luna, GPT-5 mini/nano, DeepSeek V4
  Flash, Mercury), and do its credits cover partner models? (R08)
- Is Central India (and Jio India West) open to this subscription for Fasv7/Famsv7 and Dpsv6, and are
  the AMD v7 quotas 20 vCPU there? Is Esv6 (E16s_v6) quota available? (R05, R12, R26, R41)
- Can a Container Apps GPU replica request less than the profile maximum (24 vCPU / 220 GiB for A100),
  which would bring its hourly cost from ~$6.35 toward ~$2.6? (R27)

**Upstream and other**
- Does Ollama's `qwen3.8:27b` tag include the MTP tensors (moves the Ollama estimate between ~$29 and
  ~$37/M), and why does Ollama force parallel=1 for qwen35 (memory accounting, a correctness bug, or MTP
  interaction)? Only matters for R07.
- SGLang: the newest `-xeon` image tag with Qwen3.5 CPU support; whether `--quantization w8a8_int8`
  loads the Avesed compressed-tensors checkpoint (or PR #41330 is needed); the accepted
  `SGLANG_CPU_OMP_THREADS_BIND` format. (R02)
- Does the OpenVINO 2026.4.1 IR from E10 fail to trigger PagedGatedDeltaNetFusion, and does PR #38488 or
  a re-export fix it? (R23)
- Will Qwen ship a Qwen3.8 MoE in the 30–40B range, or a diffusion conversion of Qwen3.6/3.8-27B? Does a
  "Phi-5" exist? (R25)
- Snippet-sourced figures that should be re-verified from a VM before they drive a decision:
  third-party Qwen3.8-27B API prices, GPU throughput figures, MLPerf absolute numbers, DiffusionGemma
  benchmark scores, Gemini Diffusion availability, the Artificial Analysis index values in (e), the
  AMD and Red Hat W8A8 checkpoint details, Phi-4-mini and BitNet scores, Qwen3.8-Flash-Next's GPQA.
- Upstream PR, issue and recipe claims marked *unverified* in the queue (R02, R04, R09, R13, R14,
  R17): the chain `probe` step, or a fetch from a VM (github.com and docs.vllm.ai are blocked from the
  sandbox, not from the VMs).
