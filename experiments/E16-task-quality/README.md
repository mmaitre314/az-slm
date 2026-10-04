# E16: Task-level quality of the vLLM formats (BF16, W8A8, W4A16)

| | |
| --- | --- |
| Status | planned |
| VM | `b2-qual` (Standard_E16ds_v7, centralus, Regular) |
| Stack | same vLLM image as E17 (record the digest) |
| Model | `Qwen/Qwen3.8-27B` (BF16, reference), `Avesed/Qwen3.8-27B-INT8-W8A8`, `Avesed/Qwen3.8-27B-INT4-W4A16` (record revisions) |
| Dates | |
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

## Cost per token

Not a throughput experiment. Record the wall time per model for context only.

## Analysis

## Threats to validity

## Next steps
