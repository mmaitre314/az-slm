# E12: Speculative decoding: draft model and MTP head

| | |
| --- | --- |
| Status | running (llama.cpp part; started 2026-10-04 ~06:55 UTC) |
| VM | `bench-v6` (Standard_E16ds_v6, westus2, Regular): Xeon Platinum 8573C (Emerald Rapids), 8 cores / 16 threads, 128 GiB; 8 threads used |
| Stack | llama.cpp `11fe02151f79` (2026-10-04): `build` = native (-march=native, AMX); `build-native-noamx` = -march=native minus AMX (clean control; used whenever more than one sequence is decoded, E04). `llama-server` for every measurement. OpenVINO GenAI/OVMS and vLLM: documentation only (not run) |
| Model | target `bartowski/Qwen3.8-27B-GGUF` Q4_K_M @ `0c92138c51`; MTP file `unsloth/Qwen3.8-27B-GGUF` @ `4ca720788d` (`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`); drafts `unsloth/Qwen3.5-0.8B-GGUF` @ `6ab461498e` (Q4_0), `unsloth/Qwen3.5-2B-GGUF` @ `f6d5376be1` (Q4_0) |
| Raw data | files in this directory |

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

_To be filled._

## Cost per token

_To be filled (see [COST_MODEL.md](../COST_MODEL.md))._

## Analysis

_To be filled._

## Threats to validity

## Next steps
