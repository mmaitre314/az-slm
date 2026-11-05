# E32: Trained letter-readout decision models on vLLM CPU (FlyMy decision-4b, JevK5) + E31 cheapest-size costs

| | |
| --- | --- |
| Status | running (2026-11-05): smoke job `e32smoke`, then chain `bench/dm_chain_e32.sh` |
| VM | `dm-c` (Standard_D16ls_v6, centralus, Regular): Xeon Platinum 8573C (Emerald Rapids), 8 cores / 16 vCPU, **32 GiB**, no local disk (128 GB OS disk) — the cheapest 8-core AMX size ($0.714/h + $0.030 disk and IP) |
| Stack | vLLM 0.31.0 CPU image `vllm/vllm-openai-cpu@sha256:8024248339dc…` (as E16–E18, E30); our `bench/dm_readout.py` (`--format semif`) and `bench/dm_lora_merge.py`; no model-repository code is executed |
| Model | `Qwen/Qwen3.5-4B` @ `851bf6e806` (the base FlyMy trained on); `flymy-ai/decision-4b-v1.2` (LoRA r16/α32 adapter + `model.json`, merged by us); `alibiserikbay/JevK5` v0.3 (LoRA already merged); `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B` |
| Raw data | `raw/` (harvested) |

## Question

E29 found llama.cpp's native `/v1/systemone` slow on CPU (Kev-4B: ~89 prompt tok/s, 1.3 decisions/s),
while vLLM's prefill is ~30× faster (E30). The best reported open 4B decision models, FlyMy
decision-4b-v1.2 (88.3% on JevBench public vs Jev 1.13's 86.6%) and JevK5, are plain Qwen3.5-4B
fine-tunes read at the option letters of the SemIf prompt. So they can run on vLLM's fast path once
the LoRA is merged. Does that give Jev-class accuracy at vLLM speed, on the cheapest AMX VM?

## Hypotheses

_Written before the results were known._

- **H1 (training helps)**: FlyMy decision-4b and JevK5 beat the same base (Qwen3.5-4B, SemIf format,
  T = 1) by ≥ 5 points on the jev-bench sets and BANKING77-8, and are better calibrated (ECE ≤ 0.08).
- **H2 (speed)**: the merged models run at the base model's prefill speed (same architecture), within ±5%.
- **H3 (cheapest size)**: per-core throughput on D16ls_v6 equals E16ds_v6's within instance noise
  (same CPU), so $ per 1k decisions falls by the price ratio (0.744 / 1.326 = 0.56).
- **H4 (format)**: the SemIf prompt (JSON, system prompt) is longer than our list prompt, so
  per-decision cost is 20–50% higher for the same model; trained models need it.

## Method

`bench/dm_chain_e32.sh`: setup (image, weights, BANKING77 via `dm_data.py`, jev-bench sets via
`dm_jevbench_e30.py`, the same items as E30), merge, then readout steps, each with `--repeat 2`
where throughput matters. The SemIf rendering in `dm_readout.py` follows FlyMy's published
`model.py` (read as reference, not run): system prompt "Apply the supplied criterion to the supplied
evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no
explanation or reasoning."; user message `{"evidence", "criterion", "options": [{"letter",
"description": "<key>: <text or key>"}]}`; yes/no as `true`/`false` ("The proposition is …");
readout = softmax over the bare uppercase letters' logits at the next position, divided by the
model's temperature (FlyMy 1.45 from its `model.json`; JevK5 1.22 from its card/config). vLLM computes
the logits in bf16 (FlyMy's runtime projects in fp32), and JevK5's multi-pass readout for > 16
options is not reproduced (our sets have ≤ 8 options, except go_emotions: 28).

## Measurements

## Cost per token

## Analysis

## Threats to validity

## Next steps
