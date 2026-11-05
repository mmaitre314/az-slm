# E28: Decision models (Jev, OpenJev & co) on Azure CPUs: brainstorm, survey and plan

| | |
| --- | --- |
| Status | planned (2026-11-05); measurements in E29 (llama.cpp `/v1/systemone`), E30 (vLLM/Python readouts), E31 (cheapest size, cost table) |
| Owner | orchestrator |
| VMs | `dm-a` (E16ds_v6, westus2), `dm-b` (E16ds_v6, eastus2); `dm-c` (D16ls_v6) later |
| Sources | scout bundle fetched from a VM on 2026-11-05 (`bench/dm_scout.sh`: Hugging Face API search, model cards, GitHub READMEs, JevBench boards, papers); web search. Vendor and community numbers below are **as reported**, not verified, unless marked measured |

## What a decision model is

TypeSafe released **Jev** on 2026-09-15 as a "System One" model: a decision model, not a text
generator. One endpoint, `POST /v1/systemone`, takes a **state** (text or JSON, up to 32k tokens
for Jev 1.13) and named, typed **questions**:

- `choice`: pick one of 1–255 named options (`criteria`: option name → optional description);
- `noul`: yes/no, answered with one probability;
- `score`: a level on an ordered scale.

It returns a probability for every allowed answer, computed in one forward pass. Nothing is
generated, so there is nothing to parse and no off-schema answer. Jev lists at **$0.042 per million
input tokens, output free** (OpenRouter), at a reported 0.2–0.3 s median latency.

Within three weeks the idea had dozens of open reimplementations. Wire compatibility (`/v1/systemone`)
is the norm, and llama.cpp added native support for decision GGUFs
([PR #29818](https://github.com/ggml-org/llama.cpp/pull/29818); `llama serve -hf ggml-org/Kev-4B-GGUF`).

## Survey (as of 2026-11-05)

Three designs, all reading probabilities out of one prefill:

1. **Pointer or letter readout on an LLM backbone (most entries).** A Qwen3.5/3.8 or Gemma base,
   usually with a LoRA adapter, scores each option, either by option-letter logits at the last
   prompt token or by a small pointer head over option tokens. Several entries encode the state
   once and give each question its own branch, so extra questions cost only their own tokens.
2. **Encoder classifiers.** ModernBERT/mmBERT/e5 encoders with a decision head (Laya, Verdict),
   0.1–0.4B parameters, 512–8,192-token context.
3. **Training-free readouts.** An unchanged chat LLM prompted with lettered options, read at the
   next-token position (LLM2Jev, the frozen baselines in several cards). This is what
   `bench/dm_readout.py` does on vLLM CPU.

| Family | Base and size | Runtime on CPU | Reported quality (source) |
| --- | --- | --- | --- |
| **Jev 1.13** (TypeSafe, closed API) | undisclosed | API only | JevBench public 86.6% (200/231); Praveenrajus jev-bench 0.733 acc, ECE 0.113 |
| **Kev** 0.8B/4B/9B, 27B (Jared Palmer, Apache-2.0) | Qwen3.5-*-Base + LoRA r16 + pointer head; Kev-27B a full fine-tune of Qwen3.8-27B | llama.cpp GGUF (`ggml-org/Kev-4B-GGUF`) | Kev-4B: 0.838 on its out-of-domain transfer test; 98% on an 8-route BANKING77 test (TDS article) |
| **Laya** (Convai, Apache-2.0) | ModernBERT-large 421M (512 ctx); multilingual mmBERT-base 322M | `pip install laya` (ONNX extra), llama.cpp GGUF | 0.766 after fine-tuning on a typed-decisions train split; 0.362 zero-shot (near random) |
| **decider-2b / 4b** (Mapika, Apache-2.0) | Qwen3.5-2B/4B-Base + trained readout | `decider-ai` (transformers), GGUF builds | decider-2b v12: JevBench public standard tier 0.917, hard tier 0.595 |
| **decision-2b / 4b** (FlyMy.AI) | MiniCPM5-2B / Qwen3.5-4B + LoRA, letter readout | transformers + peft, own `server.py` | decision-4b-v1.2: **88.3%** on JevBench public 231 (frozen Qwen3.5-4B readout: 80.1%; Jev 86.6%); ECE 0.070 on the hard tier |
| **Strands Decider 2B** (Amazon Strands Labs, Apache-2.0, data and scripts open) | Qwen3.5-2B-Base + pointer head, 1.9B | `pip install strands-decider` (cpu extra) | v19: 0.723 on the JevBench v1.4.2 board |
| **Clef / Clef-flash** (Cloudflare, Apache-2.0) | Qwen3.8-27B / Qwen3.5-9B + joint schema head | vLLM (GPU, NVFP4) | vendor-run comparison, Jev leads on 2 of 6 rows |
| **pplx-decider v1/v1.1/v1.2** (Perplexity) | 27B (+ flash) | GGUF exists (`Livesport/pplx-decider-v1.1-27b-GGUF`, `perplexity-ai/pplx-decider-flash-gguf-v1.2`) | 0.840 on jev-bench held-out sources |
| **OpenJev-4B** (MIT) / **OpenJev-8B** (Apache-2.0) | Qwen3.5-4B + option-set interactor; Qwen3-8B-Base + pointer head | own packages (`wev-ai`), GGUF (`ggml-org/OpenJev-GGUF`) | 79% on 8-route BANKING77 (TDS) |
| **JEV-9B / 27B / 35B** (autotrust, Apache-2.0) | Qwen3.5-9B, Qwen3.8-27B, 35B; distilled from Jev 1.13 outputs | transformers, vLLM | KL 0.021 to Jev's distributions on its own test set |
| **LLM2Jev** (training-free, Apache-2.0) | any local LLM | SGLang, Transformers, MLX | frozen Qwen3.5-4B listwise 81.4% on public JevBench |
| Many more | StartLux-Decision v1.1 (4B/9B/27B GGUF), Jev-Style 0.8B/2B, JevK5, Intern-Decision, Verdict (118–151M), domain guards (1.5B) | mostly GGUF or transformers | – |

Benchmarks: **JevBench public** (231 tasks, 18 families; official runner
[fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench), MIT, with a `typesafe` adapter that
targets `/v1/systemone`; the Benchmark Heaven board ranks 235 open systems, method v1.8) and
**Praveenrajus/jev-bench** (22 configs, 22,773 test records, human label distributions; scores
accuracy and ECE). Their numbers are not interchangeable.

## What it could achieve here (brainstorm)

1. **Decision jobs are prefill-only, which is the half of inference our CPUs do well.** Generative
   batch inference spends ~55% of its time in decode (E17/E18), where a CPU is weakest. A decision
   reads the state once and stops. Same model, same VM (Qwen3.8-27B W8A8 on E16ds_v6, E18): a
   512-token classification request costs **$1.53 per 1k** generated as 512/128, but only about
   **$0.72 per 1k** as a decision (512 tokens at 263 tok/s, $1.326/h). Typed probabilities also
   replace output parsing.
2. **Small dedicated decision models change the cost by 10–100×.** Prefill cost scales with active
   parameters. Estimates from our measured 27B prefill rate, on the cheapest 8-core AMX size that
   fits (D16ls_v6, 32 GiB, $0.714/h + $0.018):

   | model size | est. prefill tok/s | est. $/M input | est. $ per 1k decisions (300 tokens) |
   | --- | ---: | ---: | ---: |
   | 27B (Kev-27B, Clef, JEV-27B) | ~260 (needs a 128 GiB size: $1.33/h) | ~1.40 | ~0.42 |
   | 9B (JEV-9B, Clef-flash) | ~800 (64 GiB size) | ~0.29 | ~0.09 |
   | 4B (Kev-4B, decision-4b, decider-4b) | ~1,500 | ~0.14 | ~0.04 |
   | 2B (decider-2b, Strands Decider, decision-2b) | ~3,000 | ~0.07 | ~0.02 |
   | 0.8B (Kev-0.8B, Jev-Style-0.8B) | ~7,000 | ~0.03 | ~0.009 |
   | 0.4B encoder (Laya) | ~12,000 | ~0.017 | ~0.005 |
   | Jev 1.13 API | – | 0.042 | ~0.013 (≈ $32 per 1M decisions reported) |

   So a self-hosted ≤ 2B decision model could match or beat Jev's list price on a Visual Studio
   subscription with no Spot, if CPU efficiency holds at these sizes (E29–E31 measure it).
3. **Many questions per state for little extra.** Pointer/branch designs (Kev, OpenJev-8B, decider)
   encode the state once; each extra question costs only its own tokens. Extracting 10 typed fields
   from one 1,000-token document then costs ~1.1 prefills instead of 10 generations.
4. **Calibrated cascades.** With calibrated probabilities, a 2B/4B model can answer the confident
   bulk and send only low-confidence items to Qwen3.8-27B (or Kev-27B). If 85–90% stay on the small
   model, the blend costs ~15% of the 27B price at close to 27B accuracy (E31 simulates it from
   per-item probabilities).
5. **Distill to the job.** Label in-domain data with the 27B, then fine-tune a 0.4B encoder (Laya
   fits on labels; ~0.766 after fine-tuning vs 0.362 zero-shot) or a 2B LoRA. The 27B then only
   handles new categories and audits.
6. **Drop-in, private Jev.** A `/v1/systemone` server inside the resource group keeps data in the
   tenant. Jev SDKs and OpenAI's `client.decisions.create` work unchanged against several open
   servers.

Limits: decision models answer closed questions only (no free-text extraction or summaries);
context limits (Laya 512 tokens; most LLM-based ones 4–16k); zero-shot quality varies widely (Laya,
Open-Jev-2B); most vendor numbers are self-reported; and llama.cpp's AMX path corrupts
multi-sequence batches for Qwen3.5-family models (E04), which may also affect batched decision
serving.

## Hypotheses

_Written before the results were known._

- **H1 (economics)**: on the same VM and model, a decision readout costs ≤ 50% of the generative answer per request.
- **H2 (size)**: dedicated 2–4B decision models come within 5 points of Jev 1.13 on JevBench public
  (86.6%), and the best 4B (FlyMy decision-4b-v1.2, reported 88.3%) matches it.
- **H3 (CPU cost)**: on an 8-core AMX VM, a ≤ 2B decision model reaches ≥ 2,500 prompt tok/s
  (≤ $0.08/M input on D16ls_v6), and ≤ 1B models beat Jev's $0.042/M list price.
- **H4 (runtime)**: vLLM CPU (oneDNN AMX) prefills 3–10× faster than llama.cpp for the same 2–4B
  backbone (E07: llama.cpp prefill is GEMM-bound at ~3% of AMX peak), so llama.cpp's native
  `/v1/systemone` is the easy path but not the cheap one.
- **H5 (calibration)**: trained decision models have ECE ≤ 0.1 out of the box; training-free
  readouts need temperature scaling to get there.
- **H6 (cascade)**: a calibrated small-model → 27B cascade recovers ≥ 80% of the accuracy gap at
  ≤ 30% of the 27B cost.

## Method

- **E29** (`dm-a`): llama.cpp master's `llama-server` `/v1/systemone` with GGUF decision models
  (Kev-4B, Laya, OpenJev, others as available). Correctness: the same items at 1 and N parallel
  slots must give the same probabilities (E04), AMX vs no-AMX builds. Quality: the official
  JevBench public runner (231 tasks) plus BANKING77 (77-way and an 8-option changing menu) through
  `bench/dm_load.py`. Throughput: `dm_load.py` at concurrency 1–32.
- **E30** (`dm-b`): Python and vLLM paths. Training-free `bench/dm_readout.py` on vLLM CPU
  (Qwen3.5-0.8B/2B/4B/9B and Qwen3.8-27B W8A8) for throughput and BANKING77; dedicated models with
  their own CPU-capable code (FlyMy decision-4b-v1.2, Mapika decider-2b, Laya, Strands Decider 2B)
  through their `/v1/systemone` servers, the official JevBench runner and `dm_load.py`.
- **E31**: the 2–3 best candidates on D16ls_v6 (the cheapest AMX size per core) to confirm $/M, plus
  the cost table and a cascade simulation from the per-item probabilities.
- Scoring: `bench/dm_score.py` (accuracy, macro-F1, ECE, Brier, NLL, and ECE after temperature
  scaling fitted on a held-out half). Datasets: `bench/dm_data.py` (BANKING77 from PolyAI's CSVs).
- Cost: `$/M input = $/h ÷ (prompt tok/s × 3600) × 1e6`, `$ per 1k decisions = $/h ÷ (decisions/s × 3600) × 1000`,
  VM busy 100% ([COST_MODEL.md](../COST_MODEL.md)); E16ds_v6 $1.326/h all-in, D16ls_v6 $0.732/h.
- Budget: ~$12 (dm-a and dm-b ~3 h each, dm-c ~1.5 h).
