# E15: Cross-stack cost and quality summary

| | |
| --- | --- |
| Status | draft (round 1 numbers; to be finalized after E16–E18) |
| Owner | orchestrator |
| Sources | E03, E05, E06, E09, E10, E12, E13 (and E16–E18 when done) |

## Question

What is the cheapest way to run Qwen3.8-27B batch jobs (latency irrelevant) on Azure Intel AMX
VMs, at acceptable quality, and what does it cost per million tokens?

## Reference workload and assumptions

512 input + 128 output tokens per request, the VM busy 100% of the time, list prices
([COST_MODEL.md](../COST_MODEL.md)): E16ds_v7 $1.681/h all-in on demand ($0.325/h Spot),
E16ds_v6 $1.326/h ($0.260/h Spot). The subscription can't use Spot (MSDN offer); Spot columns
are what the same throughput would cost on an EA or pay-as-you-go subscription.

## Best configuration per stack (round 1)

| stack | configuration | VM | blended $/M, 512/128, on demand | Spot | quality evidence | source |
| --- | --- | --- | ---: | ---: | --- | --- |
| **vLLM 0.31.0 CPU** | W8A8 (community), 16 prompts, 8 threads | E16ds_v7 | **4.56** | **0.88** | 3 prompts correct; E16 pending | E09 |
| vLLM 0.31.0 CPU | BF16, 16 prompts | E16ds_v7 | 7.45 | 1.44 | reference precision | E09 |
| llama.cpp `11fe021` | Q4_K_M, 16 sequences, no-AMX build | E16ds_v6 | 19.3 | 3.8 | KLD 0.0135, 93.6% top-1 (E06) | E13 |
| llama.cpp `11fe021` | Q4_K_M, 32 sequences, no-AMX build | E16ds_v7 | 23.9 | 4.6 | same | E05 |
| OpenVINO GenAI 2026.4.1 | INT4 IR, 4 requests (measured) / 8+ (model) | E16ds_v7 | 34.1 / ~18.5 | 6.6 / ~3.6 | 3 prompts correct | E10 |
| llama.cpp, one sequence | Q4_0, AMX build | E16ds_v7 | 35.4 | 6.8 | KLD 0.0252 | E03 |

## Findings so far

1. **The serving stack matters more than the quantization or the CPU generation.** vLLM's oneDNN
   path reaches ~25% of AMX-BF16 peak in prefill. llama.cpp's AMX kernel reaches ~3% (E07), and
   OpenVINO GenAI prefills one request at a time through the hybrid model's 48 recurrent `Loop` ops (E10).
2. **AMX is required but not sufficient.** It gives vLLM's 133–198 tok/s prefill and OpenVINO's ×2.7,
   but llama.cpp's AMX path both underperforms and corrupts multi-sequence output for this model (E04).
3. **Batching**: decode is memory-bound per sequence (~100 GB/s per 8-core slice, E13), so
   aggregate decode rises 4–5× from 1 to 16 sequences in llama.cpp (E05); vLLM batch scaling beyond 16 is E17.
4. **Speculative decoding** with the built-in MTP head gives 1.5× single-sequence decode in llama.cpp,
   +19% at 4 slots (E12). Draft models don't pay off on CPU. vLLM MTP is E17.
5. **Spot** would cut every number by ~81% (E01) but isn't available on this subscription.

## Recommendation (provisional)

vLLM CPU with INT8 W8A8 weights on an E16ds_v7 (or v6, pending E18), with requests submitted in
large offline batches: ~$4.6 per million tokens on demand at 16 prompts. It is provisional until
E16 confirms the community W8A8 checkpoint's accuracy against BF16; if it doesn't, BF16 at $7.45/M
is the fallback. That fallback is still 2.6× cheaper than the best llama.cpp configuration.

## Open questions

- Cost at 32–64 prompts, W4A16, 16 threads, vLLM MTP (E17).
- vLLM on E16ds_v6 (E18).
- W8A8/W4A16 task accuracy vs BF16 (E16).
- SGLang's CPU backend (E14, candidate) has AMX kernels for this model family and could beat vLLM.
