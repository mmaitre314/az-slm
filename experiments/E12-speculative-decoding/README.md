# E12: Speculative decoding: draft model and MTP head

| | |
| --- | --- |
| Status | planned |
| VM | `bench-v6` (llama.cpp); OpenVINO and vLLM parts on their E09/E10 VMs if supported |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-noamx` = explicit AVX-512 flag list without AMX; `build-native-noamx` = -march=native minus AMX (clean control); OpenVINO GenAI; vLLM |
| Model | `bartowski/Qwen3.8-27B-GGUF` @ `0c92138c51` (target Q4_K_M; draft: smallest Qwen3.8 sibling with the same tokenizer, or the model's own MTP (nextn) layer) |
| Raw data | files in this directory |

## Question

Can speculative decoding (a small draft model, or Qwen3.8's built-in multi-token-prediction layer) raise decode throughput on CPU, for one sequence and for batches?

## Hypotheses

_Written before the results were known._

- H1: For one sequence (memory-bound decode), a good draft gives 1.5–2.5× decode speed when acceptance is high (≥60%).
- H2: For batches of 16+ sequences, decode is closer to compute-bound and speculative decoding gives little or negative gain, because verifying draft tokens costs extra compute.
- H3: Hybrid recurrent layers complicate speculation (state rollback on rejection); support may be missing or slow.

## Method

Check support first: llama.cpp (`llama-speculative`, `llama-server --spec-*`/MTP options, the `blk.64.nextn` MTP tensors that load as unused), OpenVINO GenAI (MTP/EAGLE for Qwen3.5 family), vLLM (`--speculative-config` method for Qwen3.5 MTP on CPU). Then measure single-sequence decode tok/s with and without speculation and acceptance rate on a fixed set of 10 prompts (64–256 output tokens), and batched throughput at 4 and 16 sequences if supported.

## Measurements

_To be filled by the reporter._

## Cost per token

_To be filled (see [COST_MODEL.md](../COST_MODEL.md))._

## Analysis

## Threats to validity

## Next steps
