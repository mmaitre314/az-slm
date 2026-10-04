# Cost model

How experiments convert measured throughput into USD per million tokens. Every experiment report
states which of these assumptions it uses and any it changes.

## Hourly prices (Linux, Azure Retail Prices API, retrieved 2026-10-04)

| Resource | Region | Pay-as-you-go | Spot |
| --- | --- | ---: | ---: |
| Standard_E16ds_v7 (Granite Rapids, 16 vCPU / 8 cores, 128 GiB) | eastus2, westus3 | $1.663/h | $0.3073/h |
| Standard_E16ds_v7 | centralus | $1.597/h | $0.2951/h |
| Standard_E16ds_v6 (Emerald Rapids, 16 vCPU / 8 cores, 128 GiB) | westus2 | $1.308/h | $0.2417/h |
| OS disk, Premium SSD P6 (64 GiB) | any | $9.28/month = $0.0127/h | same |
| Standard static public IPv4 | any | $0.005/h | same |

Overhead per VM (disk + IP) is $0.018/h, about 1% of the pay-as-you-go VM price and 6% of the
Spot price. Reports include it: **all-in hourly price = VM + $0.018**.

This subscription (Visual Studio/MSDN offer) cannot run Spot VMs, so experiments pay the
pay-as-you-go price. Reports show both: what we paid, and what the same throughput would cost on
Spot from an EA or pay-as-you-go subscription.

## Formulas

For a VM at all-in price `P` USD/hour:

- **Input (prefill) tokens**: `$/M input = P / (prefill_tok_per_s × 3600) × 1e6`
- **Output (decode) tokens**: `$/M output = P / (decode_tok_per_s × 3600) × 1e6`, using the
  aggregate decode rate across all sequences processed together (the batch size used must be
  stated: 1 sequence and N sequences give very different costs).
- **Blended, for a workload of N requests with I input and O output tokens each**, when prefill
  and decode run as separate phases (llama-batched-bench):
  `T = N·I / prefill_rate + N·O / decode_rate`, `$/request = P · T / 3600`,
  `$/M tokens = $/request / (I + O) × 1e6`.
  For engines that overlap phases (vLLM, OVMS continuous batching), use the measured wall time of
  the whole batch instead of `T`.

## Standard assumptions

1. **100% utilization**: the VM is busy for every billed second. Idle time, VM boot and setup
   (~5 min), model download (~3 min for 170 GB at ~1 GB/s) and model load are excluded and stated
   separately when material.
2. **Steady-state throughput** from the benchmark (warm-up excluded) applies to the whole job.
3. **Prices are list prices**: no reservations, savings plans or Hybrid Benefit. Spot prices
   change over time and Spot VMs can be evicted; evictions cost redone work, not modeled here.
4. **Data transfer**: inbound is free, and outbound for results is negligible.
5. **Quality**: costs are per token for a given quantization. Compare across quantizations
   together with the quality numbers (E06), not alone.
6. **Prompt/output lengths**: unless stated, the reference workload is 512 input and 128 output
   tokens per request (a typical extraction/classification batch job).
