# E11: OpenVINO Model Server (OVMS) with continuous batching

| | |
| --- | --- |
| Status | deprioritized (2026-10-04): E10 found OpenVINO GenAI's continuous-batching engine prefills one request at a time at ~34 tok/s for this model; OVMS uses the same engine. Not run. The OVMS image `openvino/model_server:2026.4.0` was pulled and `bench/ovms_*.sh` written, untested |
| VM | `bench-lc2` (Standard_E16ds_v7, northcentralus, Regular) |
| Stack | `openvino/model_server` Docker image (record tag/digest) with the LLM calculator (OpenAI-compatible API) |
| Model | The best OpenVINO IR from E10 |
| Raw data | `ovms.jsonl`, short logs |

## Question

Does a production serving layer (OVMS: OpenAI-compatible API, continuous batching, prefix caching)
keep E10's throughput, so batch jobs can be submitted as plain HTTP requests?

## Hypotheses

- **H1**: OVMS uses the same OpenVINO GenAI continuous-batching engine, so aggregate throughput at
  the same concurrency is within 10% of E10.
- **H2**: throughput rises with concurrency up to ~16–32 parallel requests, then flattens as
  decode turns compute-bound. The best concurrency is where tokens/s per $ peaks.
- **H3**: prefix caching (a shared system prompt across a batch) raises effective prefill
  throughput in proportion to the shared fraction. Measure with a 256-token shared prefix.

## Method

1. Serve the E10 model with OVMS (`--rest_port`, `--target_device CPU`, `--cache_size`,
   `--max_num_seqs`, `--enable_prefix_caching`), container blocked from IMDS.
2. **Correctness**: the same 3 fixed prompts through `/v3/chat/completions`, greedy.
3. **Load test**: `bench/` client (Python standard library, threads) sending N concurrent requests
   (N = 1, 4, 16, 32) with 512-token prompts and `max_tokens=128`. Record requests/s, input and
   output tokens/s, and per-request latency percentiles (for reference only).
4. Repeat N=16 with a shared 256-token prefix, prefix caching on.

## Measurements

## Cost per token

## Analysis

## Threats to validity

## Next steps
