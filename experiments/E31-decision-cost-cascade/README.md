# E31: Decision models on the cheapest AMX size: cost per decision vs Jev, and calibrated cascades

| | |
| --- | --- |
| Status | **done** (2026-11-05) |
| VM | `dm-c` (`Standard_D16ls_v6`, centralus, Regular; Xeon Platinum 8573C, 8 cores / 16 vCPU, 32 GiB), the list-format steps of E32's chain (`bench/dm_chain_e32.sh`: `q08_list`, `q2_list`, `q4_list`); the rest is analysis of E29, E30 and E32 data |
| Stack | vLLM 0.31.0 CPU (pinned image, as E30) and llama.cpp `a01dff9` (E29); analysis with `bench/dm_cascade.py`, `bench/dm_compact.py`, `bench/dm_report.py` |
| Model | see E29, E30, E32 |
| Dates | measured 09:18–10:06 UTC (E32 chain); analysis the same day |
| Raw data | [E32 raw](../E32-trained-readout-vllm/raw/) (`summary-dm-e32.jsonl`, `score-all.md`, `compact-preds.jsonl`), plus [E29](../E29-llamacpp-decision-models/raw/) and [E30](../E30-vllm-decision-readout/raw/) raw data |

Part of [E28](../E28-decision-models/README.md).

## Question

Which decision setup gives the best accuracy per dollar on the cheapest 8-core AMX VM, how does it compare
with Jev 1.13's list price, and can a calibrated small → big cascade buy big-model accuracy cheaply?

## Hypotheses

From E28, written before the results:

- **H3 (CPU cost)**: a ≤ 2B decision model reaches ≥ 2,500 prompt tok/s (≤ $0.08/M input on D16ls_v6), and
  ≤ 1B models beat Jev's $0.042/M list price.
- **H6 (cascade)**: a calibrated small-model → 27B cascade recovers ≥ 80% of the accuracy gap at ≤ 30% of the
  27B cost.
- E32 H3: D16ls_v6 matches E16ds_v6 per core, so costs fall by the price ratio (0.56).

## Setup

- Prices (Retail Prices API, 2026-11-05, Linux pay-as-you-go): D16ls_v6 $0.714/h in eastus2, westus2,
  westus3, eastus and northcentralus ($0.807/h in centralus, where `dm-c` ran because the regional vCPU quota
  of eastus2 and westus2 was in use), **$0.744/h all-in** with a P10 OS disk and an IP; E16ds_v6 $1.308/h
  ($1.326 all-in). Reservations for D16ls_v6: $0.443/h (1 year), $0.279/h (3 years); Spot $0.132/h. This
  Visual Studio subscription has no Spot (and reservations are an owner decision, RESEARCH.md R30).
- Jev 1.13: $0.042 per M input tokens, output free. Its token count for our items is unknown; we use
  llama.cpp's `/v1/systemone` rendering of the same requests (71 tokens per BANKING77-8 item, E29) and our
  list prompt (121) as a range: **$0.0030–0.0051 per 1k decisions**. Jev's accuracy on these sets was not
  measured (no API access from the sandbox, and sending the data out is the user's call).
- Workload: BANKING77-8 (an 8-option intent question per customer message, 3,080 test items), the set every
  model ran; jev-bench sets for generality.

## Method

1. Measure the Qwen3.5 0.8B/2B/4B list readouts on D16ls_v6 (E32 chain) and compare with E30 on E16ds_v6.
2. Cost per decision = $/h ÷ (decisions/s × 3600). Models that did not run on D16ls_v6 are priced at their
   E16ds_v6 rate (E29: llama.cpp; E30: 9B), assuming equal per-core speed (measured: D16ls_v6 4–6% faster).
3. Matched-item accuracy from the compact per-item predictions (same item ids and gold labels across runs).
4. Cascades: `bench/dm_cascade.py` sends every item to the small model and escalates items whose top-1
   probability is below a threshold to the big model; cost = small + escalated share × big, e.g.

   ```
   E32=experiments/E32-trained-readout-vllm/raw/compact-preds.jsonl
   python3 bench/dm_cascade.py $E32 --small qwen3.5-4b --big jevk5-v0.3 --dataset banking77_8 \
       --small-cost 0.0287 --big-cost 0.0557 --md
   ```

## Measurements

### Same CPU, cheaper size

| model (list readout) | E16ds_v6 tok/s (E30) | D16ls_v6 tok/s | ratio | $ per 1k decisions: E16ds_v6 → D16ls_v6 |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-0.8B | 2,873 | 3,056 | 1.06 | 0.0156 → **0.0082** |
| Qwen3.5-2B | 1,496 | 1,585 | 1.06 | 0.0299 → **0.0158** |
| Qwen3.5-4B | 841 | 874 | 1.04 | 0.0532 → **0.0287** |

Per-item predictions are identical on both VMs.

### Cost table (BANKING77-8)

Accuracy on all 3,080 items unless noted; mmlu and boolq on E30's 200-item sets (Kev and Laya on E29's
subsets of them). m: measured on D16ls_v6; e: E16ds_v6 rate at the D16ls_v6 price.

| option | stack | BANKING77-8 | mmlu | boolq | tok/item | $ per M input | $ per 1k decisions | × Jev per decision (71 tok) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev 1.13 API, list price | API | not measured | – | – | 71–121 | 0.042 | 0.0030–0.0051 | 1 |
| Qwen3.5-0.8B, list | vLLM | 67.3 | 43.0 | 69.0 | 121 | 0.068 m | 0.0082 m | 2.8 |
| Qwen3.5-2B, list | vLLM | 82.1 | 58.5 | 66.5 | 121 | 0.130 m | 0.0158 m | 5.3 |
| **Qwen3.5-4B, list** | vLLM | 91.5 | 73.0 | 89.0 | 121 | 0.236 m | **0.0287 m** | 9.6 |
| Laya (zero-shot) | llama.cpp | 85.7 | 37 (100) | 83 (99) | 68 | 0.44 e | 0.030 e | 10 |
| **Kev-0.8B** | llama.cpp | **95.9** | 41 (100) | 81 (100) | 72 | 0.51 e | **0.036 e** | 12 |
| JevK5 v0.3, SemIf | vLLM | 93.1 | 79.0 | 91.0 | 235 | 0.237 m | 0.0557 m | 19 |
| FlyMy decision-4b, SemIf | vLLM | 91.8 | 79.5 | 89.0 | 235 | 0.241 m | 0.0565 m | 19 |
| Qwen3.5-9B, list | vLLM | 93.0 | 80.5 | 90.5 | 121 | 0.48 e | 0.059 e (fit in 32 GiB untested) | 20 |
| Kev-4B | llama.cpp | 96.8 (800) | 75 (40) | 90 (40) | 71 | 2.27 e | 0.159 e | 53 |
| Qwen3.8-27B W8A8, list | vLLM | 94.8 (308) | 82.0 | 87.0 | 121 | 1.54 (E16ds_v6) | 0.187 (E16ds_v6; needs > 32 GiB) | 63 |

On matched items: the first 800 (Kev-4B's): Kev-4B 96.8, Kev-0.8B 95.4, Qwen3.5-9B 95.2, JevK5 95.0, Qwen3.5-4B
94.6, FlyMy 94.4. The stratified 308 (the 27B's): Kev-0.8B 97.1, Qwen3.8-27B 94.8, JevK5 94.5, Qwen3.5-4B 92.2.
Kev-0.8B's mmlu items: Qwen3.5-4B 76; its boolq items: Qwen3.5-4B 93.

### Cascades (BANKING77-8)

Escalate when the small model's top-1 probability is below the threshold; costs per 1k decisions on the
D16ls_v6 basis (the 27B at its E16ds_v6 cost):

| small → big | items | threshold | escalated | accuracy | $ per 1k | big alone | gap closed | cost vs big alone |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3.5-4B → JevK5 | 3,080 | 0.9 | 12.9% | 93.0 | 0.036 | 93.1, $0.056 | 92% | 64% |
| Qwen3.5-4B → JevK5 | 3,080 | 0.98 | 25.0% | 93.3 | 0.043 | | 110% | 76% |
| Qwen3.5-4B → Qwen3.5-9B | 3,080 | 0.9 | 12.9% | 93.1 | 0.036 | 93.0, $0.059 | 107% | 62% |
| Qwen3.5-4B → Qwen3.8-27B | 308 | 0.8 | 9.7% | 94.2 | 0.047 | 94.8, $0.187 | 75% | 25% |
| Qwen3.5-4B → Qwen3.8-27B | 308 | 0.95 | 18.2% | 94.5 | 0.063 | | 87% | 34% |
| Qwen3.5-4B → Qwen3.8-27B | 308 | 0.98 | 22.7% | 94.8 | 0.071 | | 100% | 38% |
| Qwen3.5-4B → Kev-4B | 800 | 0.95 | 15.8% | 96.8 | 0.054 | 96.8, $0.159 | 100% | 34% |
| Kev-0.8B → Kev-4B | 800 | 0.8 | 15.2% | 97.3 | 0.061 | 96.8, $0.159 | 136% | 38% |
| Qwen3.5-0.8B → Qwen3.5-4B | 3,080 | 0.9 | 83.3% | 91.3 | 0.032 | 91.5, $0.029 | 99% | 112% |
| Qwen3.5-2B → Qwen3.5-4B | 3,080 | 0.9 | 49.7% | 91.4 | 0.030 | 91.5, $0.029 | 99% | 105% |

"Gap closed" above 100% means the cascade beats the big model alone: the two models err on different items.

## Cost per token

See the cost table: on D16ls_v6 the cheapest measured readout is $0.068 per M input tokens (Qwen3.5-0.8B,
1.6× Jev's list price) and the 4B is $0.236/M (5.6×). With a 3-year reservation ($0.308/h all-in) these fall to
$0.028/M and $0.098/M; on Spot ($0.161/h all-in, not available on this subscription) to $0.015/M and $0.051/M.

## Analysis

1. **H3 rejected.** Only the 0.8B reaches 2,500 prompt tok/s (3,056; $0.068/M), and it misses Jev's $0.042/M
   list price by 1.6×; the 2B runs at 1,585. Per decision, the cheapest CPU option is 2.8× Jev's price for these
   short items, and the cheapest accurate one (Qwen3.5-4B, 91.5%) is 6–10×.
2. **E32 H3 holds:** the cheapest AMX size gives the same speed per core, so each decision costs 0.53–0.54× its
   E16ds_v6 price.
3. **H6 holds within noise:** behind the 4B readout, the 27B recovers 75% of the gap at 25% of its own cost and
   87% at 34%, and matches it at 38% (308 items: the gap is 8 items). Cascades work best when the second stage
   is a different kind of model (4B → Kev-4B, Kev-0.8B → Kev-4B, 4B → JevK5): they exceed the big model alone at
   38–76% of its cost. Small → 4B cascades don't pay: the 0.8B and 2B are unsure on half or more
   of the items.
4. **Calibrated confidence is what makes cascades work.** The 4B readout's errors concentrate where it is unsure:
   its 12.9% of items below 0.9 confidence hold 64% of its 263 errors (57% accurate), and the 25% below 0.98 hold
   85%. One threshold is the only setting.

## Best options (as measured on BANKING77-8 and six jev-bench sets)

1. **Default for self-hosting: Qwen3.5-4B, training-free list readout on vLLM CPU, D16ls_v6.** $0.029 per 1k
   decisions ($0.24/M input), 91.5% on BANKING77-8, 73–93% on mmlu/boolq/arc/sms_spam, calibrated out of the box
   on clear questions. No training, any question with up to 52 options, the official vLLM image, our own script.
2. **+1.5 points for +25%: the 4B readout with escalation to JevK5 v0.3 (or Qwen3.5-9B)** below 0.9 confidence:
   93.0–93.1% at $0.036 per 1k.
3. **Intent routing behind a Jev-compatible API: Kev-0.8B on llama.cpp's `/v1/systemone`.** 95.9% at ≈ $0.036
   per 1k (est.), drop-in for Jev clients; narrow (mmlu 41%). Kev-0.8B → Kev-4B below 0.8 confidence reaches
   97.3% at ≈ $0.061 per 1k.
4. **Highest accuracy: escalate the 4B's least confident 23% to Qwen3.8-27B W8A8**: the 27B's 94.8% at 38% of its
   cost ($0.071 per 1k).
5. **If the data may leave the tenant, Jev's API is the cheapest per decision** at list price ($0.003–0.005 per
   1k for these short items, 6–10× below option 1), with accuracy we could not check. Self-hosting buys data
   control, no rate limits and the choice of model, not a lower price, on this subscription.

To undercut Jev on CPU, two changes would compound: Kev's pointer head on vLLM's prefill (Kev-0.8B's ~72-token
requests at the 0.8B's 3,056 tok/s: ~42 decisions/s, ≈ $0.0049 per 1k, 1.6× Jev) and a 3-year reservation
(× 0.41: ≈ $0.0020 per 1k, about 0.7× Jev). Both are estimates (RESEARCH.md R45, R30).

## Threats to validity

- One workload (BANKING77-8, ~70–120 tokens per item); costs scale with tokens per decision, so long documents
  change the ranking little but raise every cost. jev-bench samples are 200 items (±3–7 points).
- Estimated rows assume equal per-core speed across sizes (measured only for vLLM) and the 9B fitting in 32 GiB.
- Kev and Laya results come from E29's item subsets; cascade results on 308–800 items have wide intervals.
- In-domain data: BANKING77 and jev-bench are public; trained models may have seen them, and Qwen's pretraining
  may have too.
- Jev's per-decision price assumes its token count is 71–121 per item; its accuracy is unknown here.

## Next steps

- RESEARCH.md R45–R51: Kev's head on vLLM; a lean list prompt (−30% tokens); menu-first prefix caching;
  W8A8 4B/9B readouts; many questions per state; a held-out evaluation on the user's own decisions; a paired
  Jev API comparison if the user allows sending data out.
