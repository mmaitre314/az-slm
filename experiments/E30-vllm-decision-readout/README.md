# E30: Training-free decision readout with vLLM CPU (Qwen3.5 0.8B-9B, Qwen3.8-27B W8A8)

| | |
| --- | --- |
| Status | running; background job `e30` on `dm-b`, started 2026-11-05 07:13 UTC (chain target <= 3 h) |
| VM | `dm-b` (`Standard_E16ds_v6`, eastus2, Regular; 8 cores / 16 threads, 128 GiB, NVMe `/mnt/data`) |
| Stack | official vLLM CPU image `vllm/vllm-openai-cpu@sha256:8024248339dc6878daa5349344ed29d49c8a6732f6bdf7400fda32ce33e4b30b` (vLLM 0.31.0), `bench/dm_readout.py`, no trust_remote_code |
| Model | `Qwen/Qwen3.5-0.8B` @ 2fc06364, `Qwen/Qwen3.5-2B` @ 15852e8c, `Qwen/Qwen3.5-4B` @ 851bf6e8, `Qwen/Qwen3.5-9B` @ c2022362, `Avesed/Qwen3.8-27B-INT8-W8A8` @ 86b8427a (revisions in `model-revisions.txt`) |
| Dates | started 2026-11-05 |
| Raw data | `raw/` (harvest from `dm-b`) |

## Question

What does a training-free decision readout (one prefill, next-token probabilities over lettered
options, no generation) cost and how accurate and calibrated is it with unchanged chat models on
vLLM CPU? Part of [E28](../E28-decision-models/README.md) (H1, H3, H4, H5).

## Hypotheses

See E28 (H1 economics, H3 CPU cost: >= 2,500 prompt tok/s for <= 2B, H4 vLLM faster than llama.cpp
prefill, H5 calibration: readouts need temperature scaling).

## Setup

- `bench/vllm_setup.sh` with `IMAGE` = the pinned digest and `REPOS` = the five models above;
  Docker data root and IMDS block as in E16-E18. Data by `bench/dm_data.py` (BANKING77 test: 77 and
  8 options, plus `_small` 308-item subsets) and `bench/dm_jevbench_e30.py` (converts
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
`summary-dm-e30.jsonl`, predictions `dm-e30-p-<step>.jsonl`). Steps, smallest first: 0.8B, 2B, 4B on
banking77_8 (3,080) + the six jev sets, 2 timed passes at `--max-num-seqs 64`; 0.8B on
banking77_77_small (expected error rows, >52 options); 4B at `--max-num-seqs` 16 and 128 (1,000
items); 9B (1 pass); 27B W8A8 on banking77_8_small + jev sets. Scoring: `bench/dm_score.py`.
Poll: `python3 scripts/deploy.py job dm-b e30`; fetch: `deploy.py harvest dm-b experiments/E30-vllm-decision-readout/raw`.

## Measurements

_Pending._

## Cost per token

_Pending._ ($1.326/h all-in for E16ds_v6; see [COST_MODEL.md](../COST_MODEL.md).)

## Analysis

_Pending._

## Threats to validity

Identical-size VMs vary by 9-25%; Defender agents run on the VM; jev-bench samples are 200 items
per config (+-3 points); the prompt is one fixed zero-shot template.

## Next steps

_Pending._
