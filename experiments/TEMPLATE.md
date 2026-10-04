# EXX: <title>

| | |
| --- | --- |
| Status | planned / running / done / blocked |
| VM | `<vm>` (`<size>`, `<region>`, Regular/Spot) |
| Stack | e.g. llama.cpp `<commit>`, vLLM `<version>` |
| Model | e.g. `bartowski/Qwen3.8-27B-GGUF` Q4_K_M @ `<revision>` |
| Dates | started / finished (UTC) |
| Raw data | files in this directory |

## Question

What decision does this experiment inform?

## Hypotheses

- H1: … (with the reasoning and the expected magnitude)
- H2: …

## Setup

VM, software versions, model files, relevant settings (threads, batch sizes, context).

## Method

Exact commands or scripts (`bench/...`) with parameters, repetitions, and the correctness check used.

## Measurements

Tables with units. Mark anything invalid (for example, wrong output) and why.

## Cost per token

USD per million input (prefill) and output (decode) tokens, plus a blended figure for the
workload mix, using [COST_MODEL.md](COST_MODEL.md). List the assumptions used.

## Analysis

Did the hypotheses hold? What explains the numbers (bottleneck: compute, memory bandwidth,
software)? Compare with theoretical limits.

## Threats to validity

Noise, warm-up, configuration differences, anything that would change the conclusion.

## Next steps

Follow-up experiments (add them to PLAN.md) and open questions.
