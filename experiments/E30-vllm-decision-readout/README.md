# E30: Training-free decision readout with vLLM CPU (Qwen3.5 0.8B-9B, Qwen3.8-27B W8A8)

| | |
| --- | --- |
| Status | **done** (2026-11-05). Background job `e30` on `dm-b`, chain 07:13–08:59 UTC; VM deleted after the harvest |
| VM | `dm-b` (`Standard_E16ds_v6`, eastus2, Regular; 8 cores / 16 threads, 128 GiB, NVMe `/mnt/data`) |
| Stack | official vLLM CPU image `vllm/vllm-openai-cpu@sha256:8024248339dc6878daa5349344ed29d49c8a6732f6bdf7400fda32ce33e4b30b` (vLLM 0.31.0), `bench/dm_readout.py`, no trust_remote_code |
| Model | `Qwen/Qwen3.5-0.8B` @ 2fc06364, `Qwen/Qwen3.5-2B` @ 15852e8c, `Qwen/Qwen3.5-4B` @ 851bf6e8, `Qwen/Qwen3.5-9B` @ c2022362, `Avesed/Qwen3.8-27B-INT8-W8A8` @ 86b8427a (revisions in `raw/model-revisions.txt`) |
| Raw data | `raw/`, packed on the VM by `bench/dm_pack.sh`: `score-all.md`/`.jsonl` (metrics from the full probabilities), `compact-preds.jsonl` (per item: top-1 option, its probability, label mass), `summary-dm-e30.jsonl` (throughput per step, dataset and pass), `chain.log`, `dm-e30-top.log`, step-log digests. The full prediction files were deleted with the VM |

## Question

What does a training-free decision readout (one prefill, next-token probabilities over lettered
options, no generation) cost and how accurate and calibrated is it with unchanged chat models on
vLLM CPU? Part of [E28](../E28-decision-models/README.md) (H1, H3, H4, H5). Cost comparison and
cascades across experiments: [E31](../E31-decision-cost-cascade/README.md).

## Hypotheses

See E28 (H1 economics, H3 CPU cost: >= 2,500 prompt tok/s for <= 2B, H4 vLLM faster than llama.cpp
prefill, H5 calibration: readouts need temperature scaling).

## Setup

- `bench/vllm_setup.sh` with `IMAGE` = the pinned digest and `REPOS` = the five models above;
  Docker data root and IMDS block as in E16-E18. Data by `bench/dm_data.py` (BANKING77 test: 77 and
  8 options, plus `_small` 308-item subsets, 4 items per intent) and `bench/dm_jevbench_e30.py` (converts
  `Praveenrajus/jev-bench` test.jsonl files, 200 items per config, deterministic by sha256 of id:
  go_emotions, mmlu, sst5 (score levels as ordered options), sms_spam (noul -> yes/no), boolq (noul),
  arc_challenge). banking77, clinc150, massive and ledgar exceed 52 options and are not used.
- Container flags as `bench/vllm_bench.sh`: `--privileged`, `/mnt/data` mounted, one OpenMP thread
  per physical core (`VLLM_CPU_OMP_THREADS_BIND`), `VLLM_CPU_KVCACHE_SPACE=16`.
- `bench/dm_readout.py` changes made for E30: prefix caching off (so repeated passes redo the
  prefill), an untimed warm-up batch (first-dataset throughput was 87 tok/s with compile cost, ~1,900
  after), state truncated at 8,000 characters. Chat template with thinking disabled; labels A-Z, a-z
  (52), probabilities from the top-128 logprobs of the first generated token, summed over "X" and " X".
- Smoke test (Qwen3.5-0.8B, 8 items per set): label_mass 0.997-0.999, probabilities sane.

## Method

One chain, `bench/dm_chain_e30.sh` (log `chain.log`, scores `dm-e30-score.txt`, throughput
`summary-dm-e30.jsonl`, predictions `dm-e30-p-<step>.jsonl` on the VM). Steps, smallest first: 0.8B, 2B, 4B on
banking77_8 (3,080) + the six jev sets, 2 timed passes at `--max-num-seqs 64`; 0.8B on
banking77_77_small (expected error rows, >52 options); 4B at `--max-num-seqs` 16 and 128 (first 1,000
items); 9B (1 pass); 27B W8A8 on banking77_8_small + jev sets. Scoring: `bench/dm_score.py`.
`top` before every step: 91–100% idle (Defender and the Azure agents quiet).

## Measurements

### Accuracy (%)

BANKING77-8 is the full test set (3,080 items; 27B: the stratified 308-item subset); jev-bench sets have 200
items each (`raw/score-all.md`):

| model | BANKING77-8 | arc | boolq | go_emotions | mmlu | sms_spam | sst5 | jev mean, 6 sets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 67.3 | 59.0 | 69.0 | 23.0 | 43.0 | 85.0 | 30.5 | 51.6 |
| Qwen3.5-2B | 82.1 | 82.5 | 66.5 | 19.5 | 58.5 | 16.5 | 32.0 | 45.9 |
| **Qwen3.5-4B** | **91.5** | 91.5 | 89.0 | 26.0 | 73.0 | 93.0 | 51.0 | **70.6** |
| Qwen3.5-9B | 93.0 | 96.0 | 90.5 | 22.5 | 80.5 | 94.0 | 53.5 | 72.8 |
| Qwen3.8-27B W8A8 | 94.8 (308) | 96.5 | 87.0 | 34.0 | 82.0 | 93.5 | 52.0 | 74.2 |

On the same 308 stratified BANKING77-8 items: 0.8B 70.1, 2B 84.7, 4B 92.2, 9B 91.9, 27B 94.8. 0.8B on the
77-option set: all 308 items rejected by design (> 52 labels). Label mass (probability on any valid letter)
is 0.98–1.00 for every model and set, so the format is followed; the 2B's 16.5% on sms_spam is a bias (it
answers "yes, spam" to 197 of 200 messages, gold "no" for 170), not a format failure.

### Calibration

ECE (15 bins) of the raw readout → after temperature scaling (T fitted on a hash half, ECE on the other half):

| model | BANKING77-8 (fitted T) | arc | boolq | go_emotions | mmlu | sms_spam | sst5 | jev mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 0.057 → 0.062 (1.01) | 0.104 → 0.099 | 0.065 → 0.078 | 0.314 → 0.048 | 0.176 → 0.063 | 0.110 → 0.082 | 0.280 → 0.066 | 0.175 → 0.073 |
| Qwen3.5-2B | 0.035 → 0.052 (1.01) | 0.056 → 0.088 | 0.122 → 0.141 | 0.318 → 0.019 | 0.103 → 0.072 | 0.655 → 0.468 | 0.369 → 0.099 | 0.270 → 0.148 |
| Qwen3.5-4B | 0.036 → 0.011 (1.37) | 0.048 → 0.061 | 0.064 → 0.065 | 0.358 → 0.153 | 0.124 → 0.119 | 0.030 → 0.039 | 0.215 → 0.098 | 0.140 → 0.089 |
| Qwen3.5-9B | 0.027 → 0.018 (1.17) | 0.035 → 0.036 | 0.054 → 0.074 | 0.391 → 0.044 | 0.045 → 0.096 | 0.051 → 0.052 | 0.177 → 0.146 | 0.125 → 0.075 |
| Qwen3.8-27B W8A8 | 0.027 → 0.034 (1.09) | 0.019 → 0.037 | 0.063 → 0.078 | 0.308 → 0.093 | 0.069 → 0.094 | 0.036 → 0.032 | 0.260 → 0.093 | 0.126 → 0.071 |

### Throughput

BANKING77-8, 3,080 items (27B: 308), `--max-num-seqs 64`, best of two timed passes (9B, 27B: one), prompt
tokens counted by vLLM (chat template included):

| model | tok/item | prompt tok/s | decisions/s | load s |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 121.4 | 2,873 | 23.66 | 78.5 |
| Qwen3.5-2B | 121.4 | 1,496 | 12.32 | 94.3 |
| Qwen3.5-4B | 121.4 | 841 | 6.92 | 105.0 |
| Qwen3.5-9B | 121.4 | 427 | 3.52 | 111.8 |
| Qwen3.8-27B W8A8 | 121.1 | 239 | 1.97 | 168.6 |

- Passes agree within about 1%. Prompt length barely matters between 121 and 221 tokens per item: boolq (221)
  and mmlu (176) run within 0–4% of BANKING77-8's tok/s for every model.
- Batch size (Qwen3.5-4B, first 1,000 items): 697 tok/s at 16 sequences, 797 at 128; 841 at 64 on all 3,080.
- Against E28's estimates (scaled from the 27B's prefill): 41% of the estimate for 0.8B (7,000), 50% for 2B,
  56% for 4B, 53% for 9B. Small models use the cores less efficiently (smaller GEMMs, fixed per-step costs).

## Cost per token

$1.326/h all-in for E16ds_v6 ([COST_MODEL.md](../COST_MODEL.md)), VM busy 100%, rates above. E31 repeats the
0.8B, 2B and 4B on D16ls_v6 at 0.53–0.54× these costs.

| model | $ per 1k decisions (BANKING77-8) | $ per M input tokens | vs Jev's $0.042/M |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 0.0156 | 0.128 | 3.0× |
| Qwen3.5-2B | 0.0299 | 0.246 | 5.9× |
| Qwen3.5-4B | 0.0532 | 0.438 | 10× |
| Qwen3.5-9B | 0.105 | 0.863 | 21× |
| Qwen3.8-27B W8A8 | 0.187 | 1.54 | 37× |

## Analysis

1. **4B is the knee.** From 2B to 4B, BANKING77-8 accuracy rises 9.4 points and the jev mean 24.7; 9B adds
   1.5 and 2.2 at twice the cost; the 27B adds 2.6 points over the 4B on the stratified 308 and 3.6 on the jev
   mean at 3.5× the 4B's cost. Below 4B the readout is unreliable: the 0.8B scores 67% and the 2B answers "yes"
   to almost every spam question.
2. **Calibration depends on the task more than on the model.** On BANKING77-8 every size is calibrated out of
   the box (ECE 0.027–0.057), and so is the 4B on arc, boolq and sms_spam. On ambiguous label sets (go_emotions,
   sst5: several plausible answers, one gold label) all sizes are overconfident (ECE 0.18–0.39); a temperature
   fitted on 100 items brings most cells to ≤ 0.1. E28 H5 ("readouts need temperature scaling") holds only for
   those sets.
3. **Prefill speed is what we pay for, and it is lower than E28 assumed.** H3 (≥ 2,500 prompt tok/s for
   ≤ 2B models): only the 0.8B (2,873 here, 3,056 on D16ls_v6); the 2B runs at 1,496. The 0.8B's $0.128/M is
   3× Jev's list price here and 1.6× on D16ls_v6 (E31), so H3's "≤ 1B beats Jev's list price" fails.
4. **vLLM beats llama.cpp's decision server by 7–9× per token** at the same sizes (E29: Kev-0.8B 402 tok/s,
   Kev-4B 91, Kev-9B 51), confirming E28 H4.
5. **Decision vs generation (H1).** With the same model and VM as E18 (27B W8A8, E16ds_v6), a 512-token decision
   costs $0.72–0.79 per 1k (512 tokens at 239–263 tok/s), 47–52% of a 512/128 generated answer ($1.53 per 1k at
   E15's $2.39/M blended). H1 (≤ 50%) holds at the margin. A generation constrained to a one-token label would
   cost about the same as the readout; the readout's gain is the full probability distribution in one pass,
   with no parsing.
6. **The prompt template is a third of the bill.** Our list prompt is 121 tokens per BANKING77-8 item, of which
   ~50 are instructions and chat-template tokens (llama.cpp renders the same question in ~71 tokens, E29).

## Threats to validity

Identical-size VMs vary by 9-25%; one VM, one run per configuration (two passes for the small models).
jev-bench samples are 200 items per config (±3 to ±7 points). The prompt is one fixed zero-shot template.
BANKING77 and the jev-bench test sets are public and may be in Qwen3.5's pretraining data. The 27B ran on the
308-item subset only. Logprobs come from vLLM's top-128 at the first output position: an option letter outside
the top 128 gets probability 0 (label mass ≥ 0.98 shows this is rare).

## Next steps

- Shorter prompt (instructions trimmed to ~15 tokens): about −30% cost per decision for the same model.
- Put a fixed option menu first so prefix caching reuses it (the hybrid Qwen3.5 caches only 896-token blocks
  in vLLM; a pure-attention backbone or a long shared menu is needed).
- W8A8 (INT8) 4B and 9B: E18's W8A8 gain on the 27B suggests a faster prefill at the same accuracy.
- Cascades and the cost table against Jev: [E31](../E31-decision-cost-cascade/README.md).
