# E10: OpenVINO GenAI for Qwen3.8-27B (INT4/INT8 weights, continuous batching)

| | |
| --- | --- |
| Status | partial (2026-10-04 07:00–08:50 UTC); INT4 measured (prefill, decode, knobs, profile), INT8 and the 32-sequence runs not done. Numbers recovered from the runner's transcript; raw files lost (VM deleted by the idle watchdog) |
| VM | `bench-lc2` (Standard_E16ds_v7, northcentralus, Regular): Xeon 6 6973P-C, 8 cores / 16 threads, 128 GiB. (First assigned `bench-ov` in centralus; deleted idle by the watchdog after its runner hit a usage limit.) |
| Stack | `openvino` 2026.4.1 (`2026.4.1-22982-e213a147257`), `openvino-genai` 2026.4.1.0, `openvino-tokenizers` 2026.4.1.0 in a venv (`bench/openvino_setup.sh`) |
| Model | pre-converted `OpenVINO/Qwen3.8-27B-int4-ov` (no export needed; INT4 weights, bundled MTP head `openvino_mtp_model.xml`). `OpenVINO/Qwen3.8-27B-int8-ov` exists but wasn't run |
| Raw data | lost; values below are the summaries printed by `bench/openvino_bench.py`, `openvino_knobs.sh` and `openvino_profile.py` |

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

> **Provenance.** Same outage as E09: the runner stopped at 08:51 UTC and the VM deleted itself an
> hour later. The numbers below were printed by the bench scripts in the runner's transcript and
> copied by the orchestrator. Single runs; the jsonl files are lost.

**Setup and correctness.**

- `LLMPipeline` (stateful) fails to load this IR; the continuous-batching pipeline (`LLMPipeline`
  with a `SchedulerConfig`) and `VLMPipeline` load it in ~5 s, peak RSS 29.9 GB.
- Correctness, 3 fixed prompts, greedy, thinking off: coherent and correct answers.
- `ONEDNN_VERBOSE`: ISA "Intel AVX10.1 and Intel AMX with bfloat16, float16 and 8-bit integer
  support". The profile shows prefill matmuls in `brgemm_avx512_amx_bf16` and decode matmuls in
  `brgemm_avx512_bf16` (INT4 weights decompressed to BF16, no INT8 activations by default). KV cache u8.
- The IR has 48 `Loop` ops, one per Gated DeltaNet layer (the recurrent scan).

**Batch scaling (default properties, 8 cores).** `bench/openvino_bench.py`: random prompts,
continuous batching, all requests submitted at once.

| workload (in/out) | requests | wall (s) | input tok/s | decode tok/s, all sequences | TTFT max (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill (512/1) | 1 | 15.0 | 34.0 | – | 15.0 |
| prefill (512/1) | 4 | 61.6 | 33.2 | – | 15.4 |
| prefill (512/1) | 8 | 120.1 | 34.1 | – | 15.6 |
| prefill (512/1) | 16 | 240.3 | 34.1 | – | – |
| decode (32/32) | 1 | 11.8 | 2.7 | 2.95 | 1.3 |
| decode (32/32) | 8 | 30.7 | 8.3 | 10.7 | – |
| mixed (512/128) | 1 | 58.3 | – | 2.93 | – |
| mixed (512/128) | 4 | 186.9 | – | – | – |

Prefill stays at ~34 tok/s from 1 to 16 requests, and the TTFT of *every* request is ~15 s: the
scheduler prefills one request at a time. Decode tok/s excludes the prefill time ("est_decode").
The script's `out_tokens` field over-counts by the number of requests when n > 1 (e.g. 64 output
tokens reported for 8 × 1), so total tok/s for n > 1 is recomputed here from the request shape:
mixed n=4 = 4 × 640 / 186.9 = **13.7 tok/s**.

**Knobs** (`bench/openvino_knobs.sh`; prefill n=8 input tok/s, decode = est. decode tok/s):

| configuration | prefill n=1 | prefill n=8 | decode n=1 | decode n=8 |
| --- | ---: | ---: | ---: | ---: |
| default | 33.4 | 34.0 | 2.95 | 10.7 |
| `INFERENCE_NUM_THREADS=8`, no hyper-threading | 32.7 | 34.1 | 2.79 | 10.9 |
| `INFERENCE_NUM_THREADS=16` | 32.2 | 33.5 | 2.88 | 11.0 |
| `DYNAMIC_QUANTIZATION_GROUP_SIZE=0` | 33.8 | 32.9 | 2.89 | 10.7 |
| **`INFERENCE_PRECISION_HINT=f32`, DQ group 32** | 28.3 | 27.9 | **4.50** | **17.5** |
| `INFERENCE_PRECISION_HINT=f16` | 26.6 | 26.8 | 1.63 | 9.4 |
| `--max-batched-tokens 1024` | 30.4 | 30.6 | 2.95 | 10.7 |
| AMX off (`ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16`) | 12.5 | 12.5 | 2.93 | 7.3 |

**Profile** (`bench/openvino_profile.py`, per-node counters, one 512-token prefill and decode steps):

- Prefill 512 tokens: 6.16 s wall in a bare infer request, 3.3 s summed node time.
  `FullyConnectedCompressed` 2.59 s (78%, AMX-BF16 brgemm), `GatedDeltaNet` 0.41 s, `Transpose` 0.14 s.
- Decode step: 0.37 s wall (2.7 tok/s); `FullyConnectedCompressed` 81% of node time (AVX-512 BF16, not AMX).
- `perf`: 64% in `libopenvino_intel_cpu_plugin.so`, 21% in JIT code, 4% kernel.

## Cost per token

[COST_MODEL.md](../COST_MODEL.md), northcentralus E16ds_v7: $1.681/h all-in on demand, $0.325/h at Spot.

| measure | rate (tok/s) | $/M on demand | $/M Spot |
| --- | ---: | ---: | ---: |
| input (prefill, any batch) | 34.1 | 13.7 | 2.65 |
| output (decode, 8 sequences, default) | 10.7 | 43.6 | 8.44 |
| output (decode, 8 sequences, f32 hint) | 17.5 | 26.7 | 5.17 |
| blended 512/128, measured, 4 requests | 13.7 | **34.1** | 6.6 |
| blended 512/128, phase model, 8+ requests (f32 hint) | – | ~18.5 | ~3.6 |

The phase model assumes prefill and decode don't slow each other down (512/28.3 + 128/17.5 = 25.4 s
per request). The measured n=4 mixed run is ~35% slower than that model predicts, so the real
cost at 8+ requests is likely between $19 and $34 per million tokens.

## Analysis

_Orchestrator, 2026-10-04._

- **H1 (prefill) rejected.** Prefill is 34 tok/s, not 150–400, and it doesn't scale with batch
  size. It's only ~1.4× llama.cpp (E05: 21.5 tok/s) and 6× slower than vLLM W8A8 (E09: 198 tok/s).
  A bare infer request does 512 tokens in 6.2 s (83 tok/s), so the continuous-batching pipeline
  loses more than half the time on top of the graph. Also, only ~3.3 s of the 6.2 s is in profiled nodes.
  Likely causes: the 48 GDN `Loop` ops scan the sequence serially, and the scheduler
  prefills one request at a time (TTFT ≈ 15 s for every request even when 16 are queued).
- **H2 (decode) partly confirmed.** Single-sequence decode is 2.9 tok/s (predicted ~5), but
  it rises to 10.7 tok/s at 8 sequences (17.5 with the f32 hint). By default, decode matmuls run
  in AVX-512 BF16 after INT4 decompression, not AMX. The f32 precision hint (with DQ group 32) likely
  switches the INT4 matmuls to dynamic INT8 activation quantization (not verified in the verbose log); it makes decode 1.5–1.6× faster but prefill 17% slower.
  A per-phase precision setting would combine both, but GenAI exposes one property set per pipeline.
- **H3 (support) confirmed**, using the pre-converted IR (no export needed). The stateful
  `LLMPipeline` doesn't load it; the continuous-batching and VLM pipelines do.
- **AMX matters for prefill** (×2.7: 12.5 → 34 tok/s) and somewhat for batched decode (×1.5 at 8 sequences).
- **Verdict**: on this model and version, OpenVINO GenAI costs about the same as llama.cpp
  ($19–34/M blended) and 4–7× more than vLLM W8A8 ($4.6/M). It is not the stack to optimize further
  for this model unless a release fixes batched prefill for hybrid (GDN) models.

## Threats to validity

- Single runs, numbers copied from a transcript. The `out_tokens` over-count was corrected by hand from the request shape.
- The INT4 IR's quantization settings (group size, AWQ, ratio) were not recorded, and quality wasn't measured.
- Random-token prompts; thinking disabled through the chat template.

## Next steps

1. Not a priority: re-run only if a later OpenVINO release changes GDN/batched prefill. E11 (OVMS) uses
   the same engine and would inherit the 34 tok/s prefill, so it is deprioritized.
2. If revisited: INT8 IR, f32 hint at 16/32 sequences, `SchedulerConfig` (`max_num_batched_tokens`,
   `dynamic_split_fuse`), and OVMS MTP (`num_assistant_tokens` 1–3) for decode-heavy jobs.
