# E32: Trained letter-readout decision models on vLLM CPU (FlyMy decision-4b, JevK5) + E31 cheapest-size costs

| | |
| --- | --- |
| Status | **done** (2026-11-05). Chain `bench/dm_chain_e32.sh`, job `e32` on `dm-c`, 07:54–10:18 UTC (after smoke job `e32smoke`); VM deleted after the harvest |
| VM | `dm-c` (Standard_D16ls_v6, centralus, Regular): Xeon Platinum 8573C (Emerald Rapids), 8 cores / 16 vCPU, **32 GiB**, no local disk (128 GB OS disk) — the cheapest 8-core AMX size: $0.714/h in eastus2/westus2/westus3/eastus/northcentralus (+ $0.030 disk and IP), but $0.807/h in centralus, where quota put it |
| Stack | vLLM 0.31.0 CPU image `vllm/vllm-openai-cpu@sha256:8024248339dc…` (as E16–E18, E30); our `bench/dm_readout.py` (`--format semif`) and `bench/dm_lora_merge.py`; no model-repository code is executed |
| Model | `Qwen/Qwen3.5-4B` @ `851bf6e806` (the base FlyMy trained on); `flymy-ai/decision-4b-v1.2` @ `79792bfb` (LoRA r16/α32 adapter + `model.json`, merged by us: 152 modules); `alibiserikbay/JevK5` v0.3 @ `c4f7fdb3` (LoRA already merged); `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B` (revisions in `raw/model-revisions.txt`) |
| Raw data | `raw/`, packed on the VM by `bench/dm_pack.sh`: `score-all.md`/`.jsonl`, `compact-preds.jsonl`, `summary-dm-e32.jsonl`, `chain.log`, `merge.log`, `dm-e32-top.log`, `vm-info.txt`, step-log digests. The full prediction files were deleted with the VM |

The cost table across all decision experiments, the comparison with Jev's price and the cascade
simulations are in [E31](../E31-decision-cost-cascade/README.md), which uses this chain's list-format runs.

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
options is not reproduced (our sets have ≤ 8 options, except go_emotions: 28, which SemIf's 26 letters
cannot label, so it is skipped for the SemIf runs). `top` before every step: idle.

## Measurements

### Accuracy (%) and calibration

BANKING77-8: all 3,080 test items; jev-bench sets: 200 items each, the same items as E30. "jev mean" is over
the 5 sets SemIf can label (go_emotions excluded); Qwen3.5-4B's list readout scores 79.5 on those 5.

| model, format, temperature | BANKING77-8 | arc | boolq | mmlu | sms_spam | sst5 | jev mean (5) | BANKING77-8 ECE → after T | jev mean ECE → after T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B, SemIf, T = 1 (control) | 89.9 | 93.0 | 88.5 | 73.0 | 93.0 | 51.0 | 79.7 | 0.033 → 0.023 | 0.074 → 0.072 |
| FlyMy decision-4b-v1.2, SemIf, T = 1.45 | 91.8 | 95.0 | 89.0 | **79.5** | 86.5 | 50.5 | 80.1 | 0.048 → 0.016 | 0.097 → 0.073 |
| **JevK5 v0.3, SemIf, T = 1.22** | **93.1** | **95.5** | **91.0** | 79.0 | **96.0** | 51.0 | **82.5** | **0.016** → 0.019 | 0.085 → 0.068 |
| Qwen3.5-2B, SemIf, T = 1 | 80.8 | 72.0 | 80.0 | 51.5 | 25.5 | 27.5 | 51.3 | 0.068 → 0.024 | 0.203 → 0.138 |

Differences from the control: FlyMy +1.9 on BANKING77-8, +2.0 arc, +0.5 boolq, +6.5 mmlu, −6.5 sms_spam,
−0.5 sst5 (+0.4 on the jev mean); JevK5 +3.2, +2.5, +2.5, +6.0, +3.0, 0 (+2.8). Per-set ECE is in
`raw/score-all.md` (JevK5 0.016–0.128, FlyMy 0.048–0.158). FlyMy's published T = 1.45 leaves it underconfident
on BANKING77-8 (mean top-1 probability 0.87 at 91.8% accuracy; the fitted extra factor is 0.74). JevK5's
1.22 fits (0.92 at 93.1%). Label mass ≥ 0.99 everywhere.

### Throughput (D16ls_v6)

BANKING77-8, 3,080 items, `--max-num-seqs 64`, best timed pass:

| model, format | tok/item | prompt tok/s | decisions/s | load s |
|---|---:|---:|---:|---:|
| Qwen3.5-4B, SemIf (control) | 234.6 | 860 | 3.67 | 197 |
| FlyMy decision-4b, SemIf | 234.6 | 858 | 3.66 | 126 |
| JevK5 v0.3, SemIf | 234.6 | 871 | 3.71 | 190 |
| Qwen3.5-2B, SemIf | 234.6 | 1,589 | 6.77 | 93 |
| Qwen3.5-0.8B, list (E31) | 121.4 | 3,056 | 25.16 | 77 |
| Qwen3.5-2B, list (E31) | 121.4 | 1,585 | 13.05 | 109 |
| Qwen3.5-4B, list (E31) | 121.4 | 874 | 7.20 | 131 |

E31 against E30 (E16ds_v6, same image, same items): the D16ls_v6 runs are 6.3% (0.8B), 5.9% (2B) and 4.0% (4B)
faster in tok/s, and every per-item prediction is identical.

## Cost per token

$0.744/h all-in for D16ls_v6 at the eastus2/westus2 price ($0.714 VM + $0.0245 P10 128 GiB OS disk + $0.005 IP;
Retail Prices API, 2026-11-05), VM busy 100%. The same CPU runs at the same speed in any region; in centralus,
where this VM ran ($0.807/h, $0.837 all-in), every cost below is 12.5% higher.

| model, format | $ per 1k decisions (BANKING77-8) | $ per M input tokens | vs Jev's $0.042/M |
|---|---:|---:|---:|
| Qwen3.5-0.8B, list (E31) | 0.0082 | 0.068 | 1.6× |
| Qwen3.5-2B, list (E31) | 0.0158 | 0.130 | 3.1× |
| Qwen3.5-4B, list (E31) | 0.0287 | 0.236 | 5.6× |
| Qwen3.5-2B, SemIf | 0.0305 | 0.130 | 3.1× |
| JevK5 v0.3, SemIf | 0.0557 | 0.237 | 5.6× |
| Qwen3.5-4B, SemIf | 0.0563 | 0.240 | 5.7× |
| FlyMy decision-4b, SemIf | 0.0565 | 0.241 | 5.7× |

## Analysis

1. **H1 rejected: training buys 0.4–2.8 points, not ≥ 5.** The gains sit on MMLU (+6) and BANKING77 (+2–3);
   FlyMy loses 6.5 points on sms_spam. The reported JevBench gap (FlyMy 88.3% vs 80.1% for its frozen base)
   does not carry over to these sets. JevK5 is the better fine-tune here: best or tied-best 4B on every set but
   MMLU (79.0 vs FlyMy's 79.5) and calibrated out of the box on BANKING77-8 (ECE 0.016); ECE ≤ 0.08 holds on 4
   of 6 sets for JevK5 and 3 of 6 for FlyMy.
2. **H2 holds: a merged LoRA costs nothing at inference** (858–871 tok/s vs the base's 860).
3. **H3 holds: D16ls_v6 matches E16ds_v6 per core** (4–6% faster here), so a decision costs 0.53–0.54× as much.
   The 32 GiB fit every model up to 4B in BF16 with an 8 GiB KV cache.
4. **H4 rejected on size: SemIf costs +93% per decision**, not 20–50%: 235 vs 121 tokens on BANKING77-8 (JSON
   wrapping, a system prompt, and each option written as "key: key"). For the untrained base, SemIf is no
   better than the list prompt (−1.6 on BANKING77-8, +0.2 on the 5-set jev mean). So per decision the plain 4B
   list readout (91.5%, $0.029 per 1k) matches FlyMy (91.8%, $0.057), and JevK5 buys +1.6 points on
   BANKING77-8 and +3.0 on the jev mean for 1.9× the cost.
5. **Best use of JevK5 is as a cascade's second stage.** Escalating the 4B list readout's items below 0.9
   top-1 probability (13%) to JevK5 gives 93.0% at $0.036 per 1k (E31).

## Threats to validity

- vLLM computes the letter logits in bf16; FlyMy's runtime projects in fp32. Probabilities can differ slightly;
  the readout otherwise follows the published `model.py` as read. JevK5's multi-pass readout (> 16 options) is
  not reproduced; our sets have ≤ 8 options.
- Our SemIf rendering is a reimplementation; a mismatch would understate the trained models. The control (base
  model, SemIf ≈ list) shows the rendering is sane, not that it is exact.
- BANKING77 and the jev-bench configs are public and probably in FlyMy's and JevK5's training data (both
  target JevBench), so these are in-domain numbers. The official JevBench public runner (231 tasks) was not run:
  it is third-party code (see E28), so FlyMy's 88.3% claim is untested here.
- 200-item sets: ±3–7 points (95% interval). One VM; one timed pass for the SemIf controls and JevK5.

## Next steps

- A compact SemIf rendering (no "key: key" duplication, shorter system prompt) to cut its 1.9× token overhead
  for the trained models.
- JevK5 as the escalation target behind the 4B list readout ([E31](../E31-decision-cost-cascade/README.md)).
- Evaluate on a held-out domain (the user's own decisions, RESEARCH.md R33) before trusting the fine-tunes' gains.
