# E16: Task-level quality of the vLLM formats (BF16, W8A8, W4A16)

| | |
| --- | --- |
| Status | done (chain 13:02–15:24 UTC, exit 0; VM deallocated by the watchdog with results kept, harvested and deleted 17:20 UTC) |
| VM | `b2-qual` (Standard_E16ds_v7, centralus, Regular) |
| Stack | vLLM CPU Docker image `vllm/vllm-openai-cpu:latest-x86_64` = vLLM 0.31.0, torch 2.13.0+cpu, digest `sha256:8024248339dc6878daa5349344ed29d49c8a6732f6bdf7400fda32ce33e4b30b` (image created 2026-10-03) |
| Model | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff` (BF16, reference), `Avesed/Qwen3.8-27B-INT8-W8A8` @ `86b8427a5e`, `Avesed/Qwen3.8-27B-INT4-W4A16` @ `135ecac28b` (full SHAs in `raw/model-revisions.txt`) |
| Dates | 2026-10-04 12:28–15:24 UTC (setup and smoke test 12:28–12:50; chain 13:02–15:24) |
| Raw data | `raw/` (per-question answers for each model, scores) |

## Question

The cheapest stack so far (vLLM W8A8, E09) uses a community quantization that was checked on
only 3 prompts. Do W8A8 and W4A16 lose task accuracy compared with BF16? This decides which
format the recommendation can use.

## Hypotheses

_Written before the results were known._

- **H1**: W8A8 (per-channel INT8 weights, dynamic per-token INT8 activations) is within 1 point
  of BF16 on both benchmarks and gives the same final answer as BF16 on ≥ 95% of questions.
- **H2**: W4A16 (INT4 weights, group-wise) loses 1–3 points and agrees with BF16 on 90–95% of
  questions, comparable to llama.cpp Q4_K_M's 93.6% top-1 token agreement (E06).
- **H3**: disagreements cluster on questions BF16 itself gets wrong or answers with low
  confidence, so accuracy drops less than agreement does.

## Setup

- VM `b2-qual`: Standard_E16ds_v7, centralus, Regular priority, 8 cores / 16 threads, 128 GiB. Container run with
  `--privileged --shm-size 8g`, `VLLM_CPU_KVCACHE_SPACE=16` (GiB; 66k KV tokens, enough for 64 sequences of ~700 tokens),
  `VLLM_CPU_OMP_THREADS_BIND` = one thread per physical core (8 threads), as in E09.
- `vllm.LLM(dtype=bfloat16, max_model_len=2048, max_num_seqs=64, limit_mm_per_prompt={image:0, video:0})`, torch.compile
  on (the benchmark default; the 3-prompt correctness check of `vllm_correct.py` runs eager). If the compile run fails
  the chain retries once with `--enforce-eager` for the benchmarks that have no result yet.
- Datasets from Hugging Face on the VM (`hf download --repo-type dataset`): `openai/gsm8k` @ `740312add88f` (test split, 1319 rows,
  first 200 used) and `cais/mmlu` @ `c30699e8356d` (`all/test`, 14042 rows; 400 sampled with `random.Random(16).sample`,
  indices sorted). Converted from parquet to JSONL inside the container (`bench/e16_convert.py`; the container has pyarrow and pandas).
- Prompts (chat template, `enable_thinking=False`, greedy, one `llm.chat` batch per benchmark):
  GSM8K "Solve the following math problem step by step, keeping the reasoning concise. Finish with a final line of the
  form '#### <number>' ..." (512 new tokens); MMLU "The following is a multiple choice question about <subject>. Answer with
  only the letter (A, B, C or D) ... Answer:" (8 new tokens). "Keeping the reasoning concise" was added after the first smoke
  test showed ~280-token Markdown solutions with a tail near the 512 limit; `n_truncated` in the summary counts the rest.
- Scoring (`bench/quality_tasks.py`): GSM8K number after the last `####` (else the last number), commas stripped, numeric
  compare; MMLU first standalone A-D letter. Per-question rows go to `quality-tasks.jsonl` (answer, reference, correct,
  method, output tokens, finish reason, first 300 characters), one summary row per (model, benchmark) to `quality-summary.jsonl`
  (accuracy, n, Wilson 95% CI, wall seconds, output tokens, truncated/no-answer counts).
- Chain `bench/e16_chain.sh` (job `e16`): setup (3 attempts) -> datasets (3 attempts) -> for BF16, W8A8, W4A16: correctness
  check (2 attempts), quality run (2 attempts). Failures are logged and the chain continues. Progress is in
  `chain.log`; results are saved to the OS disk after every step. 

- Smoke test (not part of the results): W8A8 (5 questions per benchmark) and W4A16 (8 per benchmark) answered
  every question correctly with `marker`/`letter` extraction (e.g. GSM8K ids 0-4 -> 18, 3, 70000, 540, 20 =
  references; MMLU ids 3, 95, 101, 149, 165 -> B, C, B, C, B = references). Timings from the smoke tests: model load with
  torch.compile 145 s (W8A8) / 346 s (W4A16); W4A16 prefill is about 4x slower than W8A8 (MMLU: 948 prompt tokens in 25.7 s vs
  658 in 4.2 s). Expected chain duration 2-3 h, W4A16 the longest.

## Method

Script `bench/quality_tasks.py` (runs inside the vLLM container, offline `vllm.LLM`, greedy,
thinking disabled through the chat template, one batch per benchmark), driven by the
self-contained chain `bench/e16_chain.sh`, which saves after every model:

1. **GSM8K**: the first 200 questions of the test split. The prompt asks for step-by-step
   reasoning ending with `#### <number>`; max 512 new tokens. Score: exact match of the last
   number with the reference.
2. **MMLU**: 400 questions from the test split, sampled with a fixed seed across all subjects.
   The prompt asks for the letter only; max 8 new tokens. Score: first A–D letter in the output.
3. Per model, record accuracy, and per question the extracted answer, so agreement with BF16 is
   computed afterwards. Report a 95% confidence interval for accuracy (binomial) and for the
   paired difference (McNemar).
4. Datasets come from Hugging Face on the VM (`openai/gsm8k`, `cais/mmlu`); record the revisions.

The same 3-prompt correctness check as E09 runs first for each model. BF16 runs first, then
W8A8, then W4A16.

## Measurements

Run `e16-20261004T1302`, greedy, thinking off, 64 sequences in flight, torch.compile (no eager
fallback was needed). Raw: [`raw/quality-summary.jsonl`](raw/quality-summary.jsonl) (per model and
benchmark), [`raw/quality-tasks.jsonl`](raw/quality-tasks.jsonl) (per question).

| model | GSM8K (n=200) | 95% CI | MMLU (n=400) | 95% CI | GSM8K outputs truncated at 512 tokens |
| --- | ---: | --- | ---: | --- | ---: |
| BF16 | 92.5% | 88.0–95.4 | **90.0%** | 86.7–92.6 | 15 |
| W8A8 | 92.5% | 88.0–95.4 | 87.25% | 83.6–90.2 | 18 |
| W4A16 | 92.5% | 88.0–95.4 | 87.0% | 83.4–90.0 | 16 |

Paired comparison with BF16 (same questions; McNemar exact test on the discordant pairs):

| benchmark | model | same answer as BF16 | only BF16 correct | only this model correct | McNemar p | disagreements where BF16 was wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | W8A8 | 90.5% | 6 | 6 | 1.00 | 13 of 19 |
| GSM8K | W4A16 | 89.0% | 8 | 8 | 1.00 | 14 of 22 |
| MMLU | W8A8 | 93.5% | 17 | 6 | **0.035** | 9 of 26 |
| MMLU | W4A16 | 94.0% | 16 | 4 | **0.012** | 8 of 24 |

Every GSM8K disagreement involves at least one output truncated at 512 tokens: when both models
finish their reasoning, they agree on the answer. W8A8 and W4A16 agree with each other on 94.5% of MMLU answers.

Answer extraction: GSM8K used the `####` marker for 183–185 of 200 outputs and the last number
otherwise (the truncated ones). No MMLU output lacked a letter.

## Cost per token

Not a throughput experiment, but the GSM8K runs are a useful real-text throughput check (200
requests, ~121 input and ~315 output tokens each, 64 in flight; centralus E16ds_v7, $1.615/h all-in):

| model | GSM8K wall (s) | output tok/s | total tok/s | $/M tokens (this workload) |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 1660 | 38.5 | 53.0 | 8.46 |
| W8A8 | 1027 | 61.2 | 84.8 | 5.29 |
| W4A16 | 2891 | 22.5 | 30.9 | 14.52 |

The ranking matches E17's random-prompt runs (W8A8 fastest, W4A16 slowest). E16's VM time for all
three models was ~2.4 h, about $3.90.

## Analysis

_Orchestrator, 2026-10-04._

- **H1 (W8A8 within 1 point, ≥ 95% agreement): rejected for MMLU, holds for GSM8K.** On math
  reasoning (GSM8K) W8A8 matches BF16 exactly (92.5%, 6 vs 6 discordant pairs). On knowledge
  questions (MMLU) it loses 2.75 points (90.0 → 87.25%), with 93.5% answer agreement, and the paired
  test says the loss is real (p = 0.035). Single-letter answers expose small shifts in the
  logits that a long reasoning chain averages out.
- **H2 (W4A16 loses 1–3 points, 90–95% agreement): confirmed.** −3.0 points on MMLU (p = 0.012),
  equal on GSM8K, 89–94% agreement. W4A16 and W8A8 are statistically indistinguishable from each other.
  Since W4A16 is also 3–5× slower on this stack (E17), it has no use here.
- **H3 (disagreements cluster on hard questions): confirmed.** On GSM8K, 13 of 19 disagreements are
  questions BF16 got wrong (BF16's error rate is 7.5%). On MMLU, 9 of 26 (BF16's error rate is 10%), so
  disagreements are 3.5× more likely where BF16 itself fails.
- **What it means for the recommendation**: W8A8 costs ~1.6× less than BF16 on vLLM (E09, E16
  throughput) and gives the same answers on reasoning tasks, but drops ~3 points on knowledge-style
  multiple choice. For classification or extraction jobs, run a task-specific check (a few hundred
  labelled items, BF16 vs W8A8) before switching. If the job can't afford a 3-point loss, BF16 is
  the fallback, at ~1.6× the cost.

## Threats to validity

- 200 GSM8K and 400 MMLU questions: the 95% intervals are ±3–4 points. Only the paired test
  separates the models.
- 512-token cap: 15–18 GSM8K outputs per model were truncated. That lowers absolute GSM8K scores
  equally for all models, and the truncated outputs are where the models disagree.
- Greedy decoding, thinking disabled. With thinking on, accuracy would be higher and outputs far
  longer (more decode cost).
- One community quantization each (`Avesed/*`): another W8A8 recipe (e.g. with SmoothQuant or
  GPTQ calibration) could lose less.

## Next steps

1. A task-specific quality check for the user's actual batch task (classification/extraction) when it is known.
2. If ~3 MMLU points matter: try another INT8 recipe (calibrated SmoothQuant W8A8, or W8A8 that
   keeps sensitive layers in BF16) and re-run this paired test.
