# E17: vLLM CPU batch scaling, W4A16, threads and MTP (E09 continued)

| | |
| --- | --- |
| Status | done (chain 12:49–16:56 UTC, exit 0, 22 runs, no failed rows); harvested and VM deleted 17:10 UTC. MTP on real text: see the follow-up (run on `b2-v6`) |
| VM | `b2-vllm` (Standard_E16ds_v7, eastus2, Regular): Xeon 6 6973P-C, 8 cores / 16 threads, 128 GiB |
| Stack | vLLM 0.31.0, torch 2.13.0+cpu, image `vllm/vllm-openai-cpu@sha256:8024248339dc…` (built 2026-10-03); llama.cpp `dd266785` (cloud-init build, AMX) for the reference run |
| Model | `Avesed/Qwen3.8-27B-INT8-W8A8` @ `86b8427a5e`, `Avesed/Qwen3.8-27B-INT4-W4A16` @ `135ecac28b`; `bartowski/Qwen3.8-27B-GGUF` Q4_K_M for the reference run |
| Dates | 2026-10-04 12:49–16:56 UTC (smoke test 12:32–12:47 UTC) |
| Raw data | `raw/` (harvested with `deploy.py harvest`) |

## Question

E09 found vLLM W8A8 the cheapest stack so far ($4.6/M blended at 16 prompts, 5× cheaper than
llama.cpp), but its batch scaling, the W4A16 variant, thread count and speculative decoding
were lost with the VM. How low does the cost go with larger batches, and which knobs matter?

## Hypotheses

_Written before the results were known._

- **H1 (batch scaling)**: at 16 sequences, W8A8 decode reads weights at only ~56 GB/s (E09), so
  per-step cost is not all weight traffic, but weight reads are still shared across sequences.
  Expect mixed 512/128 throughput to rise 30–60% from 16 to 64 prompts, mostly from decode;
  prefill tok/s stays flat (compute-bound already).
- **H2 (W4A16)**: weights are half of W8A8's, so decode at 16 sequences is faster (up to 1.5×),
  but prefill dequantizes INT4 to BF16 and runs AMX-BF16 or slower (≤ BF16's 133 tok/s; E09 saw
  "millions of tiny ukernel calls"). Blended 512/128 cost ends up equal to or worse than W8A8's.
- **H3 (threads)**: AMX has one tile unit per core, so 16 threads (SMT) gives ≤ 5% on prefill
  and possibly a loss, unlike llama.cpp's +14–25% (E08).
- **H4 (MTP)**: Qwen3.8's MTP head accepted 89% of single draft tokens in llama.cpp (E12). With
  `num_speculative_tokens` 1, decode-heavy throughput at 16 prompts rises 20–40%, less at 64.
  Risks: the CPU backend or the hybrid model may not support MTP; the W8A8 checkpoint may lack
  the MTP weights.

## Setup

- Chain: `bench/vllm_chain.sh`, run as background job `chain` (a transient systemd unit) with no agent in
  the loop. A failing step is logged and the chain goes on; a model whose 3-prompt correctness check
  fails twice is not benchmarked. Progress log: `/mnt/data/results/chain.log` (copied to the OS disk
  after every step). One `vllm_bench.sh` call per workload (own 2 h timeout, own save). Each run is a
  fresh container, so the model is loaded again every time; torch.compile artefacts are cached in
  `/mnt/data/vllm-cache`.
- Image `vllm/vllm-openai-cpu:latest-x86_64` = vLLM 0.31.0, torch 2.13.0+cpu, digest
  `sha256:8024248339dc6878daa5349344ed29d49c8a6732f6bdf7400fda32ce33e4b30b` (same as E09's tag; the
  digest was not recorded there). Model revisions: W8A8 `86b8427a5e621c18f203bbce795dd83122496412`,
  W4A16 `135ecac28b03e7f3e0d3458df40eea8dc10dc973`.
- Only the two quantized repos are downloaded (no BF16). KV cache (`VLLM_CPU_KVCACHE_SPACE`) 16 GiB for
  16 and 32 prompts, 24 GiB for 64 prompts; `--max-model-len 2048`; random prompts, all requests
  submitted at once (`vllm bench throughput`, as in E09). Runs are labelled in `vllm.jsonl` by `run`:
  `t8` (8 OpenMP threads, one per physical core), `t16` (all 16 logical CPUs), `mtp1`, `mtp2`
  (`num_speculative_tokens`).
- MTP: `--speculative-config {"method":"mtp","num_speculative_tokens":N}` goes through `EXTRA` and
  `vllm_bench.sh` expands it unquoted, so the JSON has no spaces and no wrapping quotes. The chain
  scans the safetensors headers for `mtp` tensors first (`mtp-check.jsonl`) and skips the MTP steps if
  there are none, or if the first MTP step fails on every run.
- Smoke test before the real chain: `PROFILE=smoke` (setup, MTP check, one correctness run, a 4-prompt
  mixed 128/16 W8A8 run and a 4-prompt MTP run), results moved to `/mnt/data/results-smoke` on the VM.

## Method

One self-contained background chain, `bench/vllm_chain.sh` (`PROFILE=e17`, the default), that calls
`bench/save_results.sh` after every step (and after every single `vllm bench throughput` run):

0. **Reference**: `llama-bench` Q4_K_M pp512/tg128, 8 threads, AMX build (comparable with E03/E13
   to place this VM instance relative to earlier ones).
1. `bench/vllm_setup.sh`, then the 3-prompt correctness check for each model.
2. W8A8: prefill (512/1), decode (32/256) and mixed (512/128) at 16, 32 and 64 prompts, 8
   threads (16 repeats E09 on this instance). KV cache sized for 64 × 768 tokens plus the GDN state.
3. W4A16: the same three workloads at 16 and 64 prompts.
4. W8A8 with 16 threads (all logical CPUs): prefill and mixed at 64 prompts.
5. MTP: check whether the W8A8 checkpoint contains the MTP weights (`mtp` tensors in the
   safetensors index). If so, W8A8 with `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`,
   N = 1 for decode and mixed at 16 and 64 prompts and N = 2 for decode at 16. If not, or if the
   CPU backend rejects it, record the error and move on.
6. Save `top` snapshots, versions and digests.

### Follow-up: MTP on real text (added 2026-10-04 13:30 UTC by the orchestrator)

`vllm bench throughput --dataset-name random` feeds random-token prompts, so the MTP head drafts
for nonsense continuations and acceptance is understated (the smoke run accepted 0 of 3 drafts).
`bench/mtp_real_chain.sh` runs on `b2-v6` after its E18 chain (job `mtpreal`, waits on the chain lock):
the first 200 GSM8K test questions (E16's prompt, greedy, thinking off, ≤ 512 output tokens,
64 sequences, KV cache 24 GiB) with W8A8 and `num_speculative_tokens` 0 (baseline), 1, 2 and 3.
Per run, `quality-summary.jsonl` holds accuracy (must match the baseline closely), wall time, output
tokens, and vLLM's spec-decode counters (acceptance rate, mean acceptance length).
**H5**: on real text, acceptance for 1 draft token is ≥ 80% (llama.cpp: 89%, E12), and MTP with 1–2
tokens cuts GSM8K wall time by 15–35% at 64 sequences.

## Measurements

All rows: `vllm bench throughput`, random-token prompts, all requests submitted at once, one run
each, 8 threads (one per physical core) unless marked t16. Output tok/s = prompts × output length / wall.
Raw: [`raw/vllm.jsonl`](raw/vllm.jsonl), log [`raw/chain.log`](raw/chain.log).

**Reference run** (`llama-bench` Q4_K_M, AMX build, 8 threads, 3 repetitions): pp512 25.7 ± 0.4 tok/s,
tg128 3.13 ± 0.06 tok/s. E03 measured 28.7 / 3.88 on `bench-e16v7` with an older llama.cpp commit
(`11fe021`), so this instance is ~10% slower for prefill and ~19% slower for decode, though part of the gap may be the commit.

**W8A8, batch scaling (t8):**

| workload (in/out) | prompts | wall (s) | total tok/s | output tok/s | peak RSS (GB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill (512/1) | 16 | 49.9 | 164.6 | – | 57.3 |
| prefill (512/1) | 32 | 91.8 | 178.9 | – | 56.4 |
| prefill (512/1) | 64 | 167.1 | 196.4 | – | 65.3 |
| decode (32/256) | 16 | 151.8 | 30.3 | 27.0 | 56.3 |
| decode (32/256) | 32 | 185.5 | 49.7 | 44.2 | 57.1 |
| decode (32/256) | 64 | 296.8 | 62.1 | 55.2 | 65.3 |
| mixed (512/128) | 16 | 117.9 | 86.8 | 17.4 | 56.5 |
| mixed (512/128) | 32 | 176.3 | 116.1 | 23.2 | 56.9 |
| mixed (512/128) | 64 | 304.8 | 134.4 | 26.9 | 65.1 |

E09 measured W8A8 at 16 prompts on another E16ds_v7 instance: prefill 198.1, decode 36.2, mixed 102.3
total tok/s. This instance is 17–18% slower on the same runs, consistent with the reference run.

**W4A16 (t8):**

| workload | prompts | wall (s) | total tok/s | output tok/s |
| --- | ---: | ---: | ---: | ---: |
| prefill (512/1) | 16 | 259.0 | 31.7 | – |
| prefill (512/1) | 64 | 1008.1 | 32.6 | – |
| decode (32/256) | 16 | 356.3 | 12.9 | 11.5 |
| decode (32/256) | 64 | 725.6 | 25.4 | 22.6 |
| mixed (512/128) | 16 | 422.5 | 24.2 | 4.9 |
| mixed (512/128) | 64 | 1367.0 | 30.0 | 6.0 |

**W8A8 with 16 threads** (all logical CPUs, 64 prompts): prefill 79.8 total tok/s (vs 196.4 with 8),
mixed 66.4 (vs 134.4).

**W8A8 with MTP** (`--speculative-config {"method":"mtp","num_speculative_tokens":N}`; the checkpoint
keeps the 15 `mtp.*` tensors in BF16 and vLLM loads them as `Qwen3_5MTP`):

| run | workload | prompts | total tok/s | vs no MTP |
| --- | --- | ---: | ---: | ---: |
| mtp1 | decode | 16 | 21.8 | −28% |
| mtp1 | mixed | 16 | 58.5 | −33% |
| mtp1 | decode | 64 | 34.8 | −44% |
| mtp1 | mixed | 64 | 84.6 | −37% |
| mtp2 | decode | 16 | 17.6 | −42% |

vLLM's spec-decode log lines during these runs show acceptance of 5–13% of drafted tokens (mean
acceptance length 1.05–1.13): random-token prompts make the continuation unpredictable. MTP also
shrinks the KV cache (16 GiB holds 46,933 tokens instead of 66,446).

### MTP on real text (follow-up, `b2-v6`)

`bench/mtp_real_chain.sh` on `b2-v6` (E16ds_v6, westus2) after E18, 14:13–15:20 UTC: the first 200
GSM8K test questions (E16's prompt; ~121 input and ~315 output tokens per request), W8A8, 64
sequences in flight, KV cache 24 GiB, greedy. Raw: [`raw-mtp-real/quality-summary.jsonl`](raw-mtp-real/quality-summary.jsonl),
per question in [`raw-mtp-real/quality-tasks.jsonl`](raw-mtp-real/quality-tasks.jsonl).

| spec tokens | GSM8K accuracy | wall (s) | output tok/s | vs base | acceptance rate | per-position acceptance | mean acceptance length |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0 (base) | 92.5% | 1318.9 | 47.7 | – | – | – | – |
| **1** | 92.0% | **1142.2** | **55.5** | **+16%** | 95.9% | 95.9% | 1.96 |
| 2 | 92.5% | 1169.8 | 54.2 | +14% | 92.0% | 95.5%, 88.6% | 2.84 |
| 3 | 92.5% | 1257.4 | 50.4 | +6% | 87.2% | 94.9%, 87.6%, 78.9% | 3.62 |

On real text the MTP head is excellent at predicting the next token (96% accepted). Accuracy doesn't change (one
question flips with 1 token; greedy outputs differ slightly because verification batches change the numerics, see E12).

## Cost per token

[COST_MODEL.md](../COST_MODEL.md): eastus2 E16ds_v7, $1.681/h all-in on demand, $0.325/h at Spot.
Assumptions 1–4 apply; each run's wall time is charged entirely to its workload.

| configuration | $/M input (prefill run) | $/M output (decode run) | $/M blended 512/128 (mixed run) | blended at Spot |
| --- | ---: | ---: | ---: | ---: |
| W8A8, 16 prompts | 2.84 | 17.3 | 5.38 | 1.04 |
| W8A8, 32 prompts | 2.61 | 10.6 | 4.02 | 0.78 |
| **W8A8, 64 prompts** | **2.38** | **8.46** | **3.47** | **0.67** |
| W8A8, 64 prompts, 16 threads | 5.85 | – | 7.04 | 1.36 |
| W8A8, 64 prompts, MTP 1 (random prompts) | – | 15.1 | 5.52 | 1.07 |
| W8A8, GSM8K 200, 64 in flight, base (v6, $1.326/h) | – | – | 5.58 (this workload) | 1.09 |
| W8A8, GSM8K 200, 64 in flight, MTP 1 (v6) | – | – | **4.80** (this workload) | 0.94 |
| W4A16, 16 prompts | 14.7 | 40.6 | 19.3 | 3.73 |
| W4A16, 64 prompts | 14.3 | 20.7 | 15.6 | 3.02 |

On E09's faster instance, the same scaling (×1.55 from 16 to 64 prompts on mixed) would give about
$2.9/M blended.

## Analysis

_Orchestrator, 2026-10-04._

- **H1 (batch scaling) confirmed, at the top of the predicted range.** Mixed 512/128 throughput rises 55% from 16 to 64
  prompts (86.8 → 134.4 tok/s), and blended cost falls from $5.38 to $3.47 per million tokens. Decode does
  most of it: output tok/s doubles (27.0 → 55.2) while prefill rises 19% (164.6 → 196.4).
  At 64 decode sequences a step takes 64/55.2 ≈ 1.16 s, so per-sequence work (GDN state updates,
  attention, sampling) now dominates the ~0.3 s of weight reads. The curve is flattening
  (+34% from 16 to 32, +16% from 32 to 64), so 128 prompts might add another ~10%.
- **H2 (W4A16) rejected, in both phases.** W4A16 is slower everywhere: prefill 5–6× slower
  (32 tok/s; the INT4 path doesn't use AMX-INT8 and runs as many tiny dequantize-and-multiply
  kernels), and decode is 2.3× slower at 16 sequences despite reading half the bytes. vLLM's CPU INT4
  (compressed-tensors W4A16) path is not optimized; W8A8 is the format to use on this stack.
- **H3 (threads) confirmed, much more strongly than predicted.** 16 threads halve throughput (prefill −59%, mixed −51%).
  Two threads per core contend for the one AMX unit, and the OpenMP barriers wait on the slowest
  sibling. Always bind one thread per physical core for vLLM. For llama.cpp, E08 found the opposite.
- **H4 (MTP) rejected for random prompts, by design.** With random-token prompts the MTP head accepts only 5–13% of
  drafts, so every drafted token is wasted work: −28% to −44%. Benchmarks of speculative decoding must use real text.
- **H5 (MTP on real text) confirmed at the low end.** Acceptance is 96% for one draft token (predicted ≥ 80%),
  but the gain at 64 sequences is only +16% (predicted 15–35%). At this batch size a decode step is
  dominated by per-sequence compute (E17 above: ~1.16 s per 64-token step vs ~0.3 s of weight reads),
  and verification makes each sequence process 2 tokens per step. So 1.96 tokens per step cost ~1.7×
  the step time. More draft tokens add more compute than they save (+14% with 2, +6% with 3).
  For the 512/128 reference workload, where decode is ~53% of the wall time at 64 prompts, the
  expected gain is ~7–8%: about $2.39 → ~$2.2/M on v6 (estimate, not measured). Use
  `num_speculative_tokens=1` for decode-heavy jobs. It matters more at smaller batches (llama.cpp single sequence: 1.5×, E12).

## Threats to validity

- One instance, ~10–19% slower than the E03/E09 instances by the reference run. Compare configurations within this experiment;
  absolute costs carry instance variance (9–25%, E08).
- Random-token prompts with fixed output lengths (`ignore_eos`): fine for throughput, wrong for
  speculative decoding (no predictable continuation).
- Each run pays ~2.5–3 min of startup (vLLM 0.31's AOT compile cache fails to load), excluded from
  `elapsed_time`; it matters for small jobs, not for the steady-state cost.

## Next steps

1. Measure MTP 1 on a real-text 512/128-like workload (e.g. summarization) at 64–128 prompts, to replace the ~7–8% estimate.
2. 128 prompts with a larger KV cache, to see where the scaling curve flattens.
3. vLLM runtime knobs (tcmalloc, the AOT compile-cache error) and a shared-prefix workload (prefix
   caching), as in [RESEARCH.md](../RESEARCH.md).
