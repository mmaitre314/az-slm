# E28: Decision models (Jev, OpenJev & co) on Azure CPUs: brainstorm, survey and plan

| | |
| --- | --- |
| Status | **done** (2026-11-05): plan, then results from E29 (llama.cpp `/v1/systemone`), E30 (vLLM training-free readouts), E32 (trained letter-readout models on vLLM) and E31 (cheapest size, cost vs Jev, cascades); see [Results](#results-e29e32) |
| Owner | orchestrator |
| VMs | `dm-a` (E16ds_v6, westus2), `dm-b` (E16ds_v6, eastus2), `dm-c` (D16ls_v6, centralus); all deleted |
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
  VM busy 100% ([COST_MODEL.md](../COST_MODEL.md)); E16ds_v6 $1.326/h all-in, D16ls_v6 $0.744/h (with the P10 disk it
  needs; $0.732 in the estimates above assumed a P6).
- Budget: ~$12 (dm-a and dm-b ~3 h each, dm-c ~1.5 h). Spent: about $13 (dm-a ~4 h and dm-b 3.9 h at $1.326/h,
  dm-c 3.2 h at $0.837/h in centralus). About $4 of it was idle time between the chains' end and the harvest
  (a watcher timed out, and the first harvest tried to fetch MBs through Run Command); `bench/dm_pack.sh` now
  makes the harvest take minutes.

## Results (E29–E32)

Measured on 2026-11-05 on 8-core Emerald Rapids VMs: BANKING77-8 (an 8-option intent question per
customer message, 3,080 test items) and six jev-bench sets (200 items each). Details, tables and caveats:
[E29](../E29-llamacpp-decision-models/README.md), [E30](../E30-vllm-decision-readout/README.md),
[E32](../E32-trained-readout-vllm/README.md), and the cost table and cascades in
[E31](../E31-decision-cost-cascade/README.md).

| hypothesis | verdict | evidence |
| --- | --- | --- |
| H1 economics: a readout costs ≤ 50% of the generative answer | holds at the margin | 27B W8A8: a 512-token decision costs $0.72–0.79 per 1k vs $1.53 for 512/128 (47–52%). A generation constrained to one label would cost about the same as the readout; its gain is the full distribution in one pass (E30) |
| H2 size: 2–4B within 5 points of Jev on JevBench public | untested | the JevBench runner is third-party code we did not run, and Jev was not called. On our sets the fine-tunes beat their own base by only 0.4 (FlyMy) and 2.8 (JevK5) points (E32) |
| H3 CPU cost: ≤ 2B at ≥ 2,500 tok/s, ≤ 1B below Jev's $0.042/M | rejected | Qwen3.5-0.8B 3,056 tok/s = $0.068/M on D16ls_v6 (1.6× Jev); 2B 1,585 tok/s. Measured prefill is 41–56% of the estimates in the table above (E30, E31) |
| H4 runtime: vLLM 3–10× faster than llama.cpp | holds, 7–9× | Kev-0.8B 402 vs Qwen3.5-0.8B 2,873 tok/s; Kev-4B 91 vs 841; llama.cpp's decision server gains nothing from concurrency (E29) |
| H5 calibration: trained models ECE ≤ 0.1, readouts need T | rejected as stated | Calibration depends on the task more than on training: every list readout has ECE ≤ 0.06 on BANKING77-8 untouched, every model is overconfident on go_emotions and sst5 (0.18–0.39; a fitted T fixes most); trained models range from 0.016 (JevK5) to 0.67 (Laya off-domain) |
| H6 cascade: ≥ 80% of the gap at ≤ 30% of the 27B cost | holds within noise | Qwen3.5-4B → 27B: 75% of the gap at 25% of the cost, 87% at 34%, all of it at 38% (308 items) (E31) |

Also measured: the E04 AMX multi-sequence corruption does not reach llama.cpp's decision server (Kev's
probabilities are identical at 1 and 8 slots, E29).

Brainstorm items revisited: (1) prefill-only pays, though less against a one-label generation than against
free text; (2) smaller models are cheaper, but untrained ones below 4B are not accurate enough, and trained
small ones (Kev-0.8B) are accurate only in their domain; (3) many questions per state was not tested (R49);
(4) calibrated cascades work (E31); (5) distillation was not tested; (6) a private drop-in `/v1/systemone`
works today with llama.cpp and Kev, at ~5.7 decisions/s (~20k per hour) per 8-core VM for Kev-0.8B.

### Best options

| need | option | BANKING77-8 | $ per 1k decisions (D16ls_v6) |
| --- | --- | ---: | ---: |
| general decisions, self-hosted | Qwen3.5-4B training-free list readout, vLLM CPU | 91.5% | 0.029 |
| +1.5 points | the same, escalating items below 0.9 confidence (13%) to JevK5 v0.3 or Qwen3.5-9B | 93.0–93.1% | 0.036 |
| intent routing behind a Jev-compatible API | Kev-0.8B, llama.cpp `/v1/systemone` (→ Kev-4B below 0.8 confidence: 97.3%, $0.061) | 95.9% | ≈ 0.036 (est.) |
| highest accuracy | Qwen3.5-4B escalating its least confident 23% to Qwen3.8-27B W8A8 | 94.8% (308) | 0.071 |
| lowest price, if data may leave the tenant | Jev 1.13 API at list price (accuracy not checked) | – | 0.003–0.005 |

On this subscription, self-hosting is 6–10× Jev's list price per decision for short items: it buys data
control, no rate limits and the choice of model, not a lower price. Two levers could bring CPU below Jev's
price: Kev's pointer head on vLLM's prefill (est. $0.005 per 1k at Kev-0.8B's routing accuracy, R45) plus a
3-year reservation (× 0.41, R30).
