# E10: OpenVINO GenAI for Qwen3.8-27B (INT4/INT8 weights, continuous batching)

| | |
| --- | --- |
| Status | queued |
| VM | `bench-ov` (Standard_E16ds_v7, centralus, Regular): Xeon 6 6973P-C, 8 cores / 16 threads, 128 GiB |
| Stack | `openvino`, `openvino-genai`, `optimum-intel` from PyPI in a venv (record versions) |
| Model | `Qwen/Qwen3.8-27B` exported to OpenVINO IR (or a pre-converted IR from the `OpenVINO` HF org, if one exists) |
| Raw data | `openvino.jsonl`, short logs |

## Question

Intel's own CPU inference stack is designed around AMX (INT8 dynamic quantization of activations,
INT4/INT8 weight compression, u8 KV cache, continuous batching). Is it the cheapest way to run
Qwen3.8-27B batch jobs on Azure Intel VMs?

## Hypotheses

- **H1 (prefill)**: with INT4 or INT8 weights and dynamic INT8 activation quantization (on by
  default on CPU, group size 32), prefill GEMMs run on AMX-INT8. Expect 150–400 tok/s on 8 cores,
  well above llama.cpp (21–30 t/s, E03).
- **H2 (decode)**: INT4 weights (~15 GB) make single-sequence decode memory-bound at ~5 tok/s
  (like llama.cpp Q4). Continuous batching should reach 20–40 tok/s aggregate at 16+ sequences.
- **H3 (support)**: OpenVINO supports the Qwen3.5 architecture (2026.x release notes mention
  Qwen3.5/3.6 MTP), so export works. Risk: exporting a 27B model needs ~2× BF16 size in RAM
  (~110 GB of 128 GiB), so use `--weight-format int4` directly and watch memory. Fallback:
  pre-converted IR on Hugging Face.
- **H4 (quality)**: INT4 with AWQ or scale estimation, and symmetric group size 128, is close to
  llama.cpp Q4_K_M quality. INT8 weights are near lossless.

## Method

1. venv with `openvino-genai` and `optimum-intel[openvino]` (plus `nncf`). Check for pre-converted
   IR first (`OpenVINO/Qwen3.8-27B-*-ov`); otherwise
   `optimum-cli export openvino --model Qwen/Qwen3.8-27B --task text-generation-with-past --weight-format int4 ...`,
   then an `int8` variant. Record export time and peak RAM.
2. **Correctness**: greedy generation for the same 3 fixed prompts as E09; coherent output.
3. **AMX check**: `ONEDNN_VERBOSE=1` (or OpenVINO's verbose/profiling) shows AMX kernels;
   alternatively, compare against `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` once.
4. **Throughput** with `openvino_genai.LLMPipeline(model, "CPU", scheduler_config=...)` (continuous
   batching), with a script added to `bench/`: batches of 1, 4, 16 and 32 prompts, for the
   prefill-only (512/1), decode-heavy (32/256) and mixed (512/128) workloads, the same as E09.
   Also single-sequence prefill and decode tokens/s via OpenVINO GenAI's `llm_bench` or perf metrics.
5. Record versions, export settings (group size, ratio, sym/asym, AWQ), and KV cache precision.

## Measurements

_To be filled by the runner/reporter._

## Cost per token

_To be filled (COST_MODEL.md; centralus E16ds_v7 at $1.615/h all-in, Spot $0.313/h all-in)._

## Analysis

## Threats to validity

## Next steps
