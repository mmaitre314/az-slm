# E12: Speculative decoding: draft model and MTP head

| | |
| --- | --- |
| Status | partial (2026-10-04 06:55–08:50 UTC): single-sequence sweep done, batched runs at 4 slots done, 16 slots baseline only. Numbers recovered from the runner's transcript; raw files lost (VM deleted by the idle watchdog) |
| VM | `bench-v6` (Standard_E16ds_v6, westus2, Regular): Xeon Platinum 8573C (Emerald Rapids), 8 cores / 16 threads, 128 GiB; 8 threads used |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-native-noamx` = -march=native minus AMX (clean control; used whenever more than one sequence is decoded, E04). `llama-server` for every measurement. OpenVINO GenAI/OVMS and vLLM: documentation only (not run) |
| Model | target `bartowski/Qwen3.8-27B-GGUF` Q4_K_M @ `0c92138c51`; MTP file `unsloth/Qwen3.8-27B-GGUF` @ `4ca720788d` (`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`); drafts `unsloth/Qwen3.5-0.8B-GGUF` @ `6ab461498e` (Q4_0), `unsloth/Qwen3.5-2B-GGUF` @ `f6d5376be1` (Q4_0) |
| Raw data | lost; values below are `bench/spec_analyze.py` and `spec_tiecheck.py` output copied from the runner's transcript |

## Question

Can speculative decoding (a small draft model, Qwen3.8's built-in multi-token-prediction layer, or n-gram lookup) raise decode throughput on CPU, for one sequence and for batches?

## Hypotheses

_Written before the results were known._

- H1: For one sequence (memory-bound decode), a good draft gives 1.5–2.5× decode speed when acceptance is high (≥60%).
- H2: For batches of 16+ sequences, decode is closer to compute-bound and speculative decoding gives little or negative gain, because verifying draft tokens costs extra compute.
- H3: Hybrid recurrent layers complicate speculation (state rollback on rejection); support may be missing or slow.

## Setup

Software and model files:

- llama.cpp `11fe02151f79`, same builds as E03/E08/E13 (`/opt/llama.cpp/build`, `build-native-noamx`); `bench/spec_build.sh` added `llama-server`, `llama-speculative` and `llama-speculative-simple` to both. Everything below uses `llama-server` (it implements every `--spec-type`; the older `llama-speculative*` examples were not needed).
- Target: `Qwen3.8-27B-Q4_K_M.gguf` (17.4 GB), the quantization E03/E05 used as the reference. 8 threads (`-t 8 -tb 8`), `-np` slots, `-c` = slots × 1024, default batch sizes.
- MTP head: either the `blk.64.nextn.*` tensors already inside the bartowski GGUF (`--spec-type draft-mtp`, no extra file) or the separate unsloth `mtp-Qwen3.8-27B-Q4_0.gguf` (1.37 GB) passed with `--spec-draft-model`.
- Small draft models: Qwen3.8 has **no smaller dense sibling** (see support matrix), so the drafts are the Qwen3.5 0.8B and 2B models, whose `vocab.json` and `merges.txt` are byte-identical to Qwen3.8-27B's. `tokenizer.json` differs only by 7 audio/TTS special tokens (ids 248070–248076) that sit in unused slots of the padded 248,320-token vocabulary; llama.cpp accepted the pairs without a vocabulary warning.
- Requests: `/v1/chat/completions`, greedy (`temperature 0, top_k 1`), `enable_thinking: false`, prompt cache off. Per-request numbers come from the server's own `timings` (decode tokens/s excludes prompt processing; `draft_n` / `draft_n_accepted` give acceptance).
- Prompts (`bench/spec_client.py`): `std3` = the 3 fixed correctness prompts shared with E09/E10/E11 (64 tokens max); `ten` = 10 varied prompts (explanation, code, spelling fix, JSON extraction, story, translation, step-by-step math, SQL, summary, list) with 64–256 max output tokens; `sixteen` = those plus 6 more.

### Support matrix

| Method | llama.cpp (`11fe02151f79`) | Result with Qwen3.8-27B |
| --- | --- | --- |
| MTP head of the target (`--spec-type draft-mtp`) | merged upstream 2026-05 (PR #22673); the `qwen35` hybrid architecture has an MTP context type with its own plain-attention KV cache | **works** with the bartowski GGUF's own `blk.64.nextn` tensors; also works with the separate unsloth MTP GGUF |
| Separate draft model (`--spec-type draft-simple`, `--spec-draft-model`) | supported | **works** with Qwen3.5-0.8B and 2B GGUFs (same vocabulary); no smaller Qwen3.8 exists: Hugging Face lists only Qwen3.8-27B, Qwen3.8-Flash-Next (180B MoE, 512 experts) and Qwen3.8-2.4T-A95B |
| n-gram lookup (`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`) | supported, no draft model | **works** (hybrid state rollback handled); gains only where the output repeats earlier text |
| EAGLE-3, DFlash, DSpark (`draft-eagle3`, `draft-dflash`, `draft-dspark`) | listed in `--spec-type` | not tried: no EAGLE-3/DFlash draft exists for Qwen3.8 |
| OpenVINO GenAI / OVMS 2026.4 | MTP, EAGLE-3, DFlash and fast-draft (separate draft model) strategies; the OVMS speculative-decoding page uses `OpenVINO/Qwen3.8-27B-int4-ov` as its MTP example (bundled `openvino_mtp_model.xml`, 263 MB) | **not run here**. Docs note: needs OVMS 2026.4 or a weekly build, prefix caching unsupported in MTP mode, `max_tokens` required, `num_assistant_tokens` default 5. The page's benchmark is a GPU run, not CPU. Candidate for E10/E11 |
| vLLM | `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'` | **not run here**. A vLLM forum report (Qwen3.5-27B-FP8) says only N=1 worked and N=2 errored, attributed to conv/recurrent states that can't be partially accepted; draft-model speculation failed there with a tensor-shape mismatch. Unverified for the CPU backend and current vLLM |

## Method

Scripts, all in `bench/`:

- `spec_configs.sh`: the table of configurations (flags per name).
- `spec_run.sh`: starts `llama-server` for one configuration, runs `spec_client.py` for each prompt set, stops the server. Results append to `/mnt/data/results/spec*.jsonl`, server logs to `/mnt/data/results/spec-logs/`.
- `spec_single.sh`: one slot, AMX build, loops over configurations. `spec_batch.sh`: 4 and 16 slots on `build-native-noamx`, clients keep that many requests in flight (continuous batching), `max_tokens` 128.
- `spec_analyze.py`: per-configuration table and comparison with the baseline's text. `spec_tiecheck.py` / `spec_tiecheck.sh`: teacher-forced check of the first divergent token.

Correctness: (1) the 3 standard prompts, greedy, every configuration vs the non-speculative run; (2) the same comparison on the 10 varied prompts; (3) for the divergent outputs, the baseline server's own log-probabilities for the two competing tokens.

## Measurements

> **Provenance.** The runner stopped at a usage limit at 08:51 UTC and the VM deleted itself an
> hour later. The tables are `bench/spec_analyze.py` output from the runner's transcript, copied by
> the orchestrator. Single runs.

Configuration names: `base` = no speculation; `mtpN` = `--spec-type draft-mtp` (the GGUF's own
`blk.64.nextn` head) with N draft tokens; `d08q4-4` / `d2q4-4` = Qwen3.5-0.8B / 2B Q4_0 draft, 4
draft tokens; `ngsimple-s`, `ngmapk`, `ngmod` = n-gram methods.

**One sequence**, AMX build, `ten` prompt set (10 requests, ~1,300 output tokens per configuration):

| configuration | decode tok/s | vs base | acceptance | tokens / step | s / step | text identical to base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 3.43 | 1.00× | – | 1.00 | 0.291 | 10/10 |
| mtp1 | 3.95 | 1.15× | 0.89 | 1.88 | 0.475 | 4/10 |
| mtp2 | 4.87 | 1.42× | 0.79 | 2.56 | 0.526 | 4/10 |
| mtp3 | 5.22 | 1.52× | 0.70 | 3.04 | 0.582 | 4/10 |
| **mtp4** | **5.33** | **1.55×** | 0.61 | 3.37 | 0.632 | 4/10 |
| d08q4-4 (0.8B draft) | 3.92 | 1.14× | 0.52 | 2.27 | 0.578 | 4/10 |
| d2q4-4 (2B draft) | 4.00 | 1.17× | 0.60 | 2.56 | 0.640 | 4/10 |
| ngsimple-s | 3.43 | 1.00× | 0.25 | 1.08 | 0.316 | 4/10 |
| ngmapk | 3.46 | 1.01× | 0.00 | 1.00 | 0.289 | 10/10 |
| ngmod | 3.46 | 1.01× | – | 1.00 | 0.289 | 10/10 |

On the `std3` set (3 short prompts), the MTP head and the 2B draft both reached ~76% acceptance
and ~5.0 tok/s mean decode, vs ~3.5 for base.

**Correctness.** Every configuration that actually drafted diverges from the non-speculative text on
6 of the 10 prompts. The MTP and draft-model configurations all diverge at the same character positions
(179, 16, 32, 473, 261, 747); `ngsimple-s` diverges on a different set of prompts and positions. `spec_tiecheck.py` teacher-forced the baseline at each first divergent token: the baseline's own
choice had log-probability −0.70 to −0.89 (probability 0.41–0.50) in all 6 cases, i.e. the two
candidates were near-ties. The divergence comes from numerics: the verification batch evaluates
several tokens at once, which changes matmul shapes and rounding. It is not a speculation bug.
The same happens without speculation when the batch changes: the base run with 4 slots and the
base run with 16 slots produced identical text on only 9 of 12 shared prompts.

**Several sequences** (`build-native-noamx`, `sixteen` set, continuous batching with 4 or 16 requests
in flight, `max_tokens` 128; mtp3 output matched base on 7 of 12 requests, d08q4-4 on 8 of 12). Throughput = output tokens / wall-clock time, including prefill:

| configuration | slots | requests | output tokens | tok/s (wall) | vs base | acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4 | 12 | 1,215 | 5.66 | 1.00× | – |
| mtp3 | 4 | 12 | 1,215 | **6.74** | **1.19×** | 0.75 |
| d08q4-4 | 4 | 12 | 1,215 | 3.83 | 0.68× | 0.54 |
| base | 16 | 32 | 3,448 | 7.18 | – | – |

The 16-slot speculative runs had not started when the VM was lost.

## Cost per token

[COST_MODEL.md](../COST_MODEL.md), westus2 E16ds_v6: $1.326/h all-in on demand, $0.260/h at Spot.
Output tokens only (these runs are decode-dominated); wall-clock rates from the tables above.

| configuration | tok/s | $/M output, on demand | $/M output, Spot |
| --- | ---: | ---: | ---: |
| base, 1 sequence | 3.43 | 107 | 21.0 |
| mtp4, 1 sequence | 5.33 | 69 | 13.5 |
| base, 4 slots | 5.66 | 65 | 12.7 |
| mtp3, 4 slots | 6.74 | 55 | 10.7 |
| base, 16 slots | 7.18 | 51 | 10.0 |

For reference, vLLM W8A8 decodes at 32 output tok/s with 16 sequences on an E16ds_v7 (E09): $14.5/M on demand.

## Analysis

_Orchestrator, 2026-10-04._

- **H1 partly confirmed.** For one sequence, the model's own MTP head gives 1.52–1.55× decode
  (3 to 4 draft tokens). Separate draft models give only 1.14–1.17×, even at 52–60% acceptance:
  on CPU a 0.8B or 2B draft costs a large share of the target's step (s/step doubles from 0.29 to
  0.58–0.64), while the MTP head is a single extra layer. n-gram lookup finds almost nothing to
  reuse in these prompts.
- **H2 partly confirmed.** With 4 slots, MTP still adds 19%, but a separate draft model loses 32%.
  Verification is cheap only while decode is memory-bound. The 16-slot result is missing; with
  aggregate decode already ~2× higher at 16 slots, the MTP gain should shrink further.
- **H3 rejected for llama.cpp.** The hybrid architecture's state rollback works for MTP, draft
  models and n-gram methods. The vLLM report of MTP being limited to one draft token for Qwen3.5
  hybrids is unverified here.
- **Outputs are not bit-identical** with speculation, but the differences are near-ties caused by
  batch-shape numerics, of the same kind as changing the slot count. That is acceptable for batch
  jobs that already accept quantization.
- **Cost impact**: MTP lowers llama.cpp's output-token cost by 15–35%, but llama.cpp's best
  ($51/M at 16 slots) is still ~3.5× vLLM W8A8's ($14.5/M). Speculation only matters for the
  recommendation if it works inside vLLM.

## Threats to validity

- Single runs, numbers copied from a transcript. Emerald Rapids VM (E13: within noise of Granite Rapids for llama.cpp).
- Single-sequence runs use the AMX build and batched runs use the no-AMX build (E04), so the two
  tables are not directly comparable.
- Short, varied chat prompts; acceptance depends strongly on the task (structured output and code
  usually accept more).

## Next steps

1. vLLM MTP (`num_speculative_tokens` 1–3) on W8A8 at 16–64 prompts, as part of the E09 re-run.
2. llama.cpp MTP at 16 slots, only if llama.cpp stays in contention (it doesn't, at present).
