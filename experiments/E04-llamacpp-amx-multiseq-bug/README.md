# E04: llama.cpp AMX output corruption with several sequences

| | |
| --- | --- |
| Status | done: finding (2026-10-04) |
| VM | `bench-e16v7` (Standard_E16ds_v7, eastus2) |
| Stack | llama.cpp `11fe02151f79` (master, 2026-10-04), native build (AMX) and a build with AMX compiled out |
| Model | `bartowski/Qwen3.8-27B-GGUF` Q4_K_M |

## Question

Before trusting throughput numbers: does llama.cpp produce correct output on the AMX path for this
model, for one sequence and for several sequences decoded together, which batch processing relies on?

## Hypotheses

- H1: Single-sequence output is identical with and without AMX (greedy decoding).
- H2: Multi-sequence decoding may be broken on the AMX path. Upstream issue #29670 reports
  garbage output and crashes with Qwen3.5/3.6 and parallel sequences.

## Method

`bench/sanity.sh`: greedy completion of a fixed prompt with `llama-completion` (one sequence),
then `llama-parallel -np 4 -ns 4` (4 independent prompts decoded together), on three configurations:
native build (AMX), native build with `--no-repack` (AMX buffers disabled), and a build with AMX
compiled out.

## Measurements

| Configuration | 1 sequence | 4 sequences decoded together |
| --- | --- | --- |
| native build, AMX | correct (identical to no-AMX) | **corrupt**: `333333…` for client 0, binary junk for others |
| native build, `--no-repack` | not run | correct, coherent answers |
| build with AMX compiled out | correct | correct, coherent answers |

Side finding: `llama-batched` (one prompt shared by several sequences) fails for this model on
every build with `split_equal: sequential split is not supported when there are coupled sequences`.
Hybrid recurrent models (Gated DeltaNet) don't support shared-prompt batches in llama.cpp.
Independent prompts, the batch-processing case, work.

## Analysis

H1 and H2 confirmed. The corruption is specific to the AMX weight buffer/repack path: disabling
repacking or compiling AMX out fixes it, so it's not the hybrid architecture by itself. The bug is
still present on master as of 2026-10-04 (the research notes pointed at fix PR #29671, which
apparently isn't merged or doesn't cover this model).

Consequences for this study:

- Single-sequence numbers (E03) on the AMX build are valid.
- **Batched llama.cpp numbers on the AMX build are invalid.** E05 measures throughput on a build
  without AMX and reports the AMX build's speed only as a reference ("what you'd get once fixed").
- llama-perplexity evaluates several sequences per batch by default, so quality (E06) uses a build
  without AMX.
- Production advice: until fixed, don't run llama.cpp/Ollama/LM Studio with parallel slots on AMX
  hosts for Qwen3.5-family models without `--no-repack`.

## Threats to validity

One prompt set and one quantization (Q4_K_M) tested; Q4_0 and Q8_0 are likely affected the same
way (same AMX kernels), not verified.

## Next steps

- Re-test on new llama.cpp releases; check upstream PR #29671's status from a VM (github.com is
  blocked from the sandbox).
- Reporting upstream (an issue comment with this reproduction) is outward-facing: ask the user first.
