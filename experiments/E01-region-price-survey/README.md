# E01: Which Azure VMs have Intel AMX, and where is Spot capacity cheapest?

| | |
| --- | --- |
| Status | done (2026-10-04) |
| VM | none (ARM and pricing APIs only) |
| Stack | `scripts/spot_amx.py`, classification workflow (Microsoft Learn sources, adversarial verification) |
| Raw data | [`data/amx_families.json`](../../data/amx_families.json), [`results/spot_amx_2026-10-04.csv`](../../results/spot_amx_2026-10-04.csv) |

## Question

Which VM families guarantee Intel AMX, and which regions give the most AMX cores at the lowest Spot
price, so later experiments run on representative, cheap hardware?

## Hypotheses

- H1: Only the newest Intel families (v6 Emerald Rapids, v7 Granite Rapids) guarantee AMX; older
  families may land on AMX hosts but can't be relied on.
- H2: Spot prices vary by region enough (>20%) to make region choice matter.
- H3: Regional Spot quota, not price, limits "most cores".

## Method

1. Listed every x64 VM family visible to the subscription (ARM `Microsoft.Compute/skus`, 79
   regions). Excluded AMD and GPU families up front.
2. Classified the 93 Intel families from their Microsoft Learn series pages: **yes** (every
   documented CPU has AMX), **partial** (mixed hardware, AMX only on some hosts), **no**,
   **unknown**. Three classifier agents and three adversarial verifier agents; every claim cites
   its Learn page.
3. Joined AMX-yes families with Spot capability, per-subscription restrictions, Linux Spot and
   pay-as-you-go prices (Retail Prices API), regional Spot quota (`lowPriorityCores`) and Spot
   eviction rates (Resource Graph `SpotResources`, tenant scope).

## Measurements

**AMX classification** (93 Intel families): 33 yes, 28 partial, 25 no, 7 unknown.

- **Yes**: Dsv6/Ddsv6/Dlsv6/Dldsv6/Dnsv6 family and Esv6/Edsv6/Ensv6/Endsv6 (Xeon Platinum 8573C,
  Emerald Rapids); Dsv7/Ddsv7/Dlsv7/Dldsv7 and Esv7/Edsv7 (Xeon 6 6973PC, Granite Rapids);
  FXmsv2/FXmdsv2; Lsv4; DCesv6/ECesv6 (confidential); M-series v3 (Sapphire Rapids).
- **Partial** (mixed hosts, no guarantee): most v2–v5 Intel families, for example Dsv5/Esv5, which
  list Emerald/Sapphire Rapids alongside Ice Lake or older.

**Spot price per vCPU-hour, cheapest AMX series** (Linux, sizes ≥ 8 vCPUs):

| Region | Cheapest AMX series | Spot $/vCPU-h | Eviction | Spot quota (vCPUs) | Largest AMX VM |
| --- | --- | ---: | --- | ---: | --- |
| westus3 | Dlsv6 | 0.00825 | 0–5% | 20 | M832s_12_v3 |
| eastus2 | Dlsv6 | 0.00825 | 0–5% | 20 | D372s_v7 |
| westus2 | Dlsv6 | 0.00825 | 0–5% | 20 | D372s_v7 |
| centralindia | Dlsv6 | 0.00825 | 0–5% | 20 | D192s_v6 |
| northcentralus | Dlsv6 | 0.00825 | 0–5% | 20 | D192s_v6 |
| uksouth | Dlsv6 | 0.00846 | 0–5% | 20 | D192s_v6 |
| swedencentral | Dlsv6 | 0.00882 | 0–5% | 20 | M832s_12_v3 |

**By series** (minimum across regions): Dlsv6 $0.00825, Dsv6 $0.00930, Dldsv6 $0.01016,
Dlsv7 $0.01035, Ddsv6 $0.01150, Dsv7 $0.01173, Esv6 $0.01220, Edsv6 $0.01511, Edsv7 $0.01843
per vCPU-hour. Every Intel v6/v7 size has 2 vCPUs per physical core, so $/core-hour is twice these.

**Subscription limits**: the regional Spot quota is 20 vCPUs everywhere, so the largest Spot VM is
16 vCPUs. **Spot deployments fail on this subscription for every size and region**: it's a
Visual Studio (MSDN) offer, and Azure only allows Spot on EA, pay-as-you-go (0003P) and Sponsored
offers. The error is a misleading `SkuNotAvailable` "Capacity Restrictions", while Regular
priority passes.

## Cost per token

Not applicable (no inference). Prices feed [COST_MODEL.md](../COST_MODEL.md).

## Analysis

- H1 confirmed: only v6/v7 (plus FXv2, Lsv4 and M v3) guarantee AMX. Older series are a lottery.
- H2 rejected: the Spot discount is a flat ~81.5% of pay-as-you-go across series and regions, so
  Spot ranks regions exactly like list prices. The US regions eastus2/westus2/westus3/northcentralus
  and centralindia tie for cheapest.
- H3 confirmed, and moot here: quota caps Spot at 16 vCPUs per region, and this subscription can't
  run Spot at all.
- Eviction data covers only ~17% of SKU-region pairs, mostly in the 0–5% bucket.

## Threats to validity

Prices and eviction rates change over time; this is a snapshot. The AMX classification relies on
Microsoft's documentation of host CPUs; E02 verifies it on one size.

## Next steps

- E02: verify AMX inside a VM.
- On an EA or pay-as-you-go subscription: request Spot quota above 20 vCPUs to reach the 96–372
  vCPU sizes.
