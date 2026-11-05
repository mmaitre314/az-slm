# E29: llama.cpp `/v1/systemone` with ggml-org decision GGUFs

| | |
| --- | --- |
| Status | **done** (2026-11-05). Chain 07:20–09:27 UTC, background job `e29b` on `dm-a`, chain script `bench/dm_chain_e29.sh`; VM deleted after the harvest |
| VM | `dm-a` (`Standard_E16ds_v6`, westus2, Regular), Emerald Rapids, 8 cores / 16 threads, 128 GiB, local NVMe |
| Stack | llama.cpp `a01dff908b9c9311d7af316bba3c918063749bd0` (cloud-init native AMX build, `/opt/llama.cpp/build`) and a no-AMX control build of the same tree (`bench/build_native_noamx.sh` plus `--target llama-server`) |
| Model | `ggml-org/Kev-4B-GGUF` Q8_0, `ggml-org/Laya-GGUF` BF16, `ggml-org/OpenJev-GGUF` Q8_0 (a 64-layer, 5120-wide Qwen3.5-arch model, about 27B, 28.6 GB, CC BY-NC 4.0, not 4B), `ggml-org/Kev-0.8B-GGUF` Q8_0, `ggml-org/Kev-9B-GGUF` Q8_0; revisions in `raw/versions.txt` |
| Raw data | `raw/`, packed on the VM by `bench/dm_pack.sh`: `score-all.md`/`.jsonl` (every metric from the full probabilities, `bench/dm_score.py`), `compact-preds.jsonl` (per item: top-1 option, its probability; `bench/dm_compact.py`), `tp_summary.jsonl` and `q_summary.jsonl` (throughput and per-set load runs), `corr_*.jsonl` (batched vs single-slot, AMX vs no-AMX), `ok_*.txt` (build and parallelism chosen), `chain.log`, `versions.txt`. The full prediction files were not fetched (Run Command returns 4 KiB per call) and were deleted with the VM |

Part of [E28](../E28-decision-models/README.md). Cost comparison and cascades across E29, E30 and E32: [E31](../E31-decision-cost-cascade/README.md).

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
  helpsteer2_helpfulness (score); items sorted by sha256 of the id). The chain scored the first 100 items per
  set for Kev-0.8B and Laya, 40 for Kev-4B, 25 for Kev-9B and 8 for OpenJev; on banking77_8, all 3,080 for
  Kev-0.8B and Laya and the first 800, 300 and 60 items for Kev-4B, Kev-9B and OpenJev.
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
concurrency and `-t 8`, plus `-t 16` at concurrency 16 (skipped: no model was cleared for 16 slots).

## Measurements

### Correctness: batching and AMX

100 banking77_8 items (Kev-9B 50, OpenJev 24), maximum absolute difference between the probability vectors:

| model | 8 slots vs 1, AMX | 8 slots vs 1, no-AMX | AMX vs no-AMX, 1 slot | top-1 agreement, all pairs | cleared for |
| --- | ---: | ---: | ---: | ---: | --- |
| Kev-0.8B | 0.0 | 0.0 | 0.022 (32 items > 1e-3) | 100% | AMX, 8 slots |
| Kev-4B | 0.0 | 0.0 | 0.014 (35) | 100% | AMX, 8 slots |
| Kev-9B | 0.0 | 0.0 | 0.010 (20) | 100% | AMX, 8 slots |
| Laya | 0.0085 (33) | 0.011 (32) | 0.0 | 99–100% | AMX, 1 slot |
| OpenJev | 0.0035 (3) | 0.0052 (2) | 0.011 (5) | 100% | AMX, 1 slot |

The Kev models are exactly batch-invariant. Laya and OpenJev move by up to 0.011 with the batch composition on
both builds (numeric, not the E04 corruption: top-1 agrees), and the chain's 1e-3 rule then limited them to
one slot.

### Quality

Accuracy in % (n items scored), ECE (15 bins) before → after temperature scaling fitted on a hash half and
evaluated on the other (`raw/score-all.md`):

| set (options) | Kev-0.8B | Kev-4B | Kev-9B | Laya | OpenJev |
| --- | ---: | ---: | ---: | ---: | ---: |
| banking77_8 (8) | **95.9** (3,080), ECE 0.056 → 0.011 | **96.8** (800), 0.057 → 0.018 | **97.3** (300), 0.037 → 0.024 | 85.7 (3,080), 0.051 → 0.042 | 100 (60), 0.039 |
| banking77_77 (77) | 83.0 (100), 0.197 → 0.132 | 88.3 (60), 0.195 → 0.166 | 85.0 (40) | 31.0 (100), 0.541 → 0.099 | HTTP 400 (15) |
| jb_boolq (yes/no) | 81.0 (100), 0.085 | 90.0 (40), 0.085 | 92.0 (25) | 82.8 (99, 1 error), 0.097 | 100 (8) |
| jb_mmlu (4) | 41.0 (100), 0.115 | 75.0 (40), 0.157 | 80.0 (25) | 37.0 (100), 0.144 | 100 (8) |
| jb_go_emotions (28) | 23.0 (100), 0.098 | 42.5 (40), 0.149 | 36.0 (25) | 40.0 (100), 0.588 | 37.5 (8) |
| jb_ledgar (100) | 55.0 (100), 0.239 | 60.0 (40), 0.226 | 56.0 (25) | 27.7 (83, 17 errors), 0.670 | HTTP 400 |
| jb_helpsteer2_helpfulness (score 0–4) | 45.5 (66), 0.202 | 42.3 (26) | 37.5 (16) | 34.6 (26, 40 errors) | 60.0 (5) |
| jb_banking77 (77), jb_clinc150 (150), jb_massive (60) | HTTP 400 | HTTP 400 | HTTP 400 | 39.0, HTTP 500, 44.0 | HTTP 400 |

Errors: the Kev and OpenJev servers answered HTTP 400 to every jev-bench banking77, clinc150 and massive
question (and OpenJev to banking77_77 and ledgar), while Kev handled our own 77-option banking77_77. Not
diagnosed: the server logs were not kept. Laya answered HTTP 500 to inputs beyond its 512-token context:
all of clinc150 and the long ledgar, helpsteer2 and boolq items.

Matched items (from `raw/compact-preds.jsonl` and E30/E32's, same item ids and gold labels): on Kev-4B's 40
jb_boolq items Kev-4B scores 90.0% vs Qwen3.5-4B's training-free readout 95.0% (E30) and JevK5 95.0% (E32);
on its 40 jb_mmlu items 75.0% vs 77.5% and 77.5%; on go_emotions 42.5% vs 25.0%. Kev-0.8B on its 100 mmlu
items: 41% vs Qwen3.5-0.8B 40% and Qwen3.5-4B 76%. On banking77_8, Kev-4B's first 800 items: Kev-4B 96.8%,
Kev-0.8B 95.4%, Qwen3.5-4B 94.6%, JevK5 95.0% (these first items are easier than the full set: Qwen3.5-4B
scores 91.5% on all 3,080).

### Throughput

banking77_8, AMX build, `-t 8`, decisions per second and prompt (input) tokens per second as counted by the
server (`usage.input_tokens`):

| model | items | concurrency 1 | concurrency 4 | 8 slots, concurrency 8 (correctness run) | tok/item | p50 latency, c1 | no-AMX build, c1 |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| Kev-0.8B | 600 | **5.69/s, 402 tok/s** | 5.68/s | 5.40/s | 71.7 | 0.18 s | 4.16/s (−26%) |
| Kev-4B | 160 | **1.30/s, 91 tok/s** | 1.28/s | 1.29/s | 71.4 | 0.81 s | 0.92/s (−29%) |
| Kev-9B | 60 | **0.72/s, 50.5 tok/s** | 0.72/s | 0.72/s | 71.4 | 1.47 s | 0.52/s (−28%) |
| Laya | 600 | **6.96/s, 468 tok/s** | – (1 slot) | 8.64/s, 568 tok/s | 68.3 | 0.14 s | 6.98/s (±0) |
| OpenJev | 12 | **0.26/s, 29 tok/s** | – (1 slot) | 0.27/s | 113.9 | 3.80 s | 0.11/s (−58%) |

More parallel requests do not raise throughput: the decision server processes them one at a time (only Laya
gains 25% at 8 slots). Longer jev-bench inputs (140–614 tokens per item, 8 slots) run at 450–500 tok/s on
Kev-0.8B (0.8–3.3 decisions/s). The chain did not record `top`; E30's snapshots on the same image show the
Defender and Azure agents idle (< 1% CPU).

## Cost per token

$1.326/h all-in for E16ds_v6 ([COST_MODEL.md](../COST_MODEL.md)), the VM busy 100%, best rate above. The
D16ls_v6 column ($0.744/h, same CPU) assumes the same per-core speed, which E31 measured for vLLM (within 4–6%).

| model | banking77_8 accuracy | $ per 1k decisions, E16ds_v6 | $ per M input tokens | $ per 1k decisions, D16ls_v6 (est.) |
| --- | ---: | ---: | ---: | ---: |
| Kev-0.8B | 95.9% | 0.065 | 0.92 | **0.036** |
| Laya | 85.7% | 0.053 | 0.79 | 0.030 |
| Kev-4B | 96.8% (800) | 0.284 | 4.05 | 0.159 |
| Kev-9B | 97.3% (300) | 0.512 | 7.29 | 0.287 |
| OpenJev (27B class) | 100% (60) | 1.40 | 12.6 | – (needs ≥ 64 GiB) |
| Jev 1.13 API, list price | – | $0.042 per M input tokens: ≈ 0.003 per 1k decisions at 71 tokens | 0.042 | – |

## Analysis

1. **Correct, but serial.** The E04 AMX corruption does not reach the decision server: the Qwen3.5-based Kev
   models give bit-identical probabilities at 1 and 8 slots. The no-AMX build is 26–58% slower (no difference
   for the BF16 ModernBERT Laya). But concurrency adds nothing, so the CPU is poorly used: Kev-0.8B prefills 402 tok/s, while vLLM runs
   the same-size Qwen3.5-0.8B backbone at 2,873 tok/s (E30), 7× faster; Kev-4B 91 vs 841 tok/s (9×), Kev-9B 51
   vs 427 (8.5×). E28 H4 (vLLM 3–10× faster than llama.cpp) holds, at its upper end.
2. **Kev-0.8B is the find for intent routing.** 95.9% on all 3,080 BANKING77-8 items, ECE 0.011 after
   temperature scaling, beats every vLLM readout in E30/E32, including Qwen3.5-9B (93.0%) and JevK5 (93.1%), at
   $0.065 per 1k decisions (est. $0.036 on D16ls_v6). It is not a general model: 41% on MMLU (chance is 25%),
   81% on BoolQ (Qwen3.5-4B's readout: 93% on the same items). Kev's training data is not documented in what
   we read; the gap between routing and MMLU suggests a routing-heavy mix, so treat 95.9% as in-domain accuracy.
3. **Bigger Kev models help less than their cost.** On the 800 items both scored, Kev-4B is 1.4 points above
   Kev-0.8B at 4.4× the cost per decision; Kev-9B adds nothing measurable on 300 items.
4. **Laya zero-shot is cheap and narrow.** 85.7% on BANKING77-8 at 7 decisions/s, but 31–44% on the 60–77-way
   sets with ECE 0.5–0.6, and its 512-token context rejects long inputs. Useful only fine-tuned, as its card says.
5. **OpenJev is a 27B-class model under a non-commercial license.** Accurate on the few items it ran, but
   0.26 decisions/s ($1.40 per 1k) and CC BY-NC 4.0: not a candidate here.
6. **Versus Jev's list price** ($0.042/M input), every llama.cpp option costs 19–300× more per token on
   E16ds_v6 (about 10–100× on D16ls_v6). The native endpoint is the easy, Jev-compatible way to serve Kev, not
   the cheap one.

## Threats to validity

- Subsets: the slower models ran the first 800/300/60 banking77_8 items and 40/25/8 jev-bench items, and the
  first items are easier (Qwen3.5-4B: 94.6% on the first 800, 91.5% on all). The matched comparisons above use
  the same items; 40-item sets have ±7-point 95% intervals.
- In-domain data: BANKING77 and the jev-bench configs are public and probably in the trained models' data.
- Throughput came from one VM, banking77_8 only (~70 tokens per item); identical VM sizes vary by 9–25% (E08).
- The HTTP 400 rejections were not diagnosed, so Kev's results on 60–150-option questions are unknown.

## Next steps

- Run Kev's pointer head on vLLM's prefill (pooling output plus our own head code): Kev-0.8B's routing accuracy
  at vLLM speed would be about 7× cheaper ([RESEARCH.md](../RESEARCH.md) R45).
- Keep the llama-server log in the chain and find the option-count or context limit behind the HTTP 400s.
- E31 has the cross-stack costs and cascades (Kev-0.8B → Kev-4B: 97.3% on 800 items at 38% of Kev-4B's cost).
