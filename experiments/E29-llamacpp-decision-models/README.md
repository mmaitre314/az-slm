# E29: llama.cpp `/v1/systemone` with ggml-org decision GGUFs

| | |
| --- | --- |
| Status | running (started 2026-11-05 07:15 UTC, background job `e29` on `dm-a`; chain script `bench/dm_chain_e29.sh`) |
| VM | `dm-a` (`Standard_E16ds_v6`, westus2, Regular), Emerald Rapids, 8 cores / 16 threads, 128 GiB, local NVMe |
| Stack | llama.cpp `a01dff908b9c9311d7af316bba3c918063749bd0` (cloud-init native AMX build, `/opt/llama.cpp/build`) and a no-AMX control build of the same tree (`bench/build_native_noamx.sh` plus `--target llama-server`) |
| Model | `ggml-org/Kev-4B-GGUF` Q8_0, `ggml-org/Laya-GGUF` BF16, `ggml-org/OpenJev-GGUF` Q8_0 (a 64-layer, 5120-wide Qwen3.5-arch model, about 27B, 28.6 GB, CC BY-NC 4.0, not 4B), `ggml-org/Kev-0.8B-GGUF` Q8_0, `ggml-org/Kev-9B-GGUF` Q8_0; revisions in `versions.txt` (raw data) |
| Dates | started 2026-11-05 07:15 UTC; finished: _pending_ |
| Raw data | `raw/` (harvested from `/var/lib/azslm/results`): `chain.log`, `corr_*.jsonl`, `q_summary.jsonl`, `score_*.txt`, `tp_summary.jsonl`, `versions.txt` |

Part of [E28](../E28-decision-models/README.md).

## Question

Can llama.cpp's built-in decision server serve the ggml-org decision GGUFs correctly under batching (E04
AMX multi-sequence bug), how accurate and calibrated are they, and what do they cost per decision on an
8-core AMX VM?

## Hypotheses

E28 H3/H4: llama.cpp is the easy path but not the cheap one (prefill GEMM-bound). Batched AMX output may be
corrupted for the Qwen3.5-family models (Kev, OpenJev); the chain checks this before any measurement.

## Setup

- Models and data as above; data prepared by `bench/dm_data.py` (BANKING77 `banking77_8`: gold plus 7 random
  options; `banking77_77_small`: 308 items, 77 options) and `bench/dm_jevbench_e29.py` (Praveenrajus/jev-bench
  test files read as data: banking77, clinc150, massive, go_emotions, mmlu, ledgar, boolq (noul),
  helpsteer2_helpfulness (score); 200 items each, sorted by sha256 of the id).
- Server: `llama-server -m <gguf> -np N -t T -tb T -c N*ctx --no-webui` (ctx per slot 4096; Laya 512).
  Response of `POST /v1/systemone`: `{"model", "answers": {"q": ...}, "usage": {"input_tokens", "output_tokens": 0}}` with
  `choice`: `{"type","choice","probabilities":{option: p},"confidence"}`, `noul`: `{"type","noul": p(yes)}`,
  `score`: `{"type","score": expected level,"legend","probabilities","confidence"}`.
- `bench/dm_load.py` parses the choice shape. `bench/dm_load_e29.py` wraps it (dm_load.py is shared): it sends
  the original typed question for jev-bench items (descriptions, noul, score) and parses `noul`.
- Throughput prompt tokens come from `usage.input_tokens`.

## Method

`bench/dm_chain_e29.sh`: setup; per model (1) correctness on 100 banking77_8 items at 1 vs 8 slots on the AMX and
no-AMX builds (tolerance 1e-3 on probabilities, `bench/dm_compare_e29.py`), which picks the build and the maximum
parallelism for that model; (2) quality (`dm_score.py`); (3) throughput at concurrency 1, 4, 16 with `-np` =
concurrency and `-t 8`, plus `-t 16` at concurrency 16. Item counts per model are in the script (reduced for
the slow 9B and 27B-class models).

## Measurements

_Pending; the chain is running._

## Cost per token

_Pending._ $1.326/h all-in ([COST_MODEL.md](../COST_MODEL.md)).

## Analysis

_Pending._

## Threats to validity

- Subsets of banking77_8 and jev-bench are used for the slower models.
- Identical VM sizes vary by 9-25% (E08); Defender agents run in the background.

## Next steps

See E28 (E30 vLLM/Python paths, E31 cost table).
