#!/usr/bin/env python3
"""Rank Azure regions for Spot VMs with Intel AMX: the most vCPUs at the lowest Spot price.

Input: data/amx_families.json, which maps ARM VM families to an AMX classification taken from
Microsoft Learn (see the file for sources). Data pulled live:
- Compute SKU catalog: sizes, Spot capability, per-subscription restrictions (cached in .cache/)
- Spot and pay-as-you-go Linux prices from the Azure Retail Prices API (prices.azure.com)
- Regional Spot vCPU quota (Microsoft.Compute usages, "lowPriorityCores")
- Spot eviction rates (Azure Resource Graph SpotResources table)

Writes results/spot_amx_<date>.csv (one row per region x SKU) and prints a region ranking.
"""

import argparse
import concurrent.futures
import csv
import datetime
import json
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

import arm

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRICES = "https://prices.azure.com/api/retail/prices"


def subscription_path():
    return f"/subscriptions/{arm.env('AZURE_SUBSCRIPTION_ID')}"


def cap(sku, name, default=None):
    return next((c["value"] for c in sku["capabilities"] if c["name"] == name), default)


def load_skus(refresh):
    """All VM SKU entries (one per SKU x region) visible to the subscription, cached on disk."""
    cache = ROOT / ".cache" / "compute_skus.json"
    if refresh or not cache.exists():
        status, body = arm.request("GET", subscription_path() + "/providers/Microsoft.Compute/skus",
                                   "2021-07-01", timeout=600)
        if status != 200:
            raise SystemExit(f"SKU catalog GET returned HTTP {status}")
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps([s for s in body["value"] if s["resourceType"] == "virtualMachines"]))
    return json.loads(cache.read_text())


def get_json(url, attempts=5):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
        time.sleep(2 ** attempt)


def linux_prices(sku_name):
    """{region: {"spot": usd_per_hour, "payg": usd_per_hour}} for one SKU, Linux only."""
    flt = f"serviceName eq 'Virtual Machines' and armSkuName eq '{sku_name}' and priceType eq 'Consumption'"
    url, items = f"{PRICES}?{urllib.parse.urlencode({'$filter': flt})}", []
    while url:
        page = get_json(url)
        items += page["Items"]
        url = page.get("NextPageLink")
    latest = {}  # (region, kind) -> item with the latest effectiveStartDate
    for it in items:
        if "Windows" in it["productName"] or "Low Priority" in it["skuName"] or it["retailPrice"] <= 0:
            continue
        key = (it["armRegionName"].lower(), "spot" if it["skuName"].endswith(" Spot") else "payg")
        if key not in latest or it["effectiveStartDate"] > latest[key]["effectiveStartDate"]:
            latest[key] = it
    out = {}
    for (region, kind), it in latest.items():
        out.setdefault(region, {})[kind] = it["retailPrice"]
    return out


def spot_quota(region):
    status, body = arm.request("GET", f"{subscription_path()}/providers/Microsoft.Compute/locations/{region}/usages",
                               "2024-07-01")
    if status != 200:
        return None
    u = next((u for u in body["value"] if u["name"]["value"] == "lowPriorityCores"), None)
    return u and {"limit": u["limit"], "used": u["currentValue"]}


def eviction_rates(sku_names):
    names = ", ".join(f"'{n.lower()}'" for n in sorted(sku_names))
    rows = arm.resource_graph(
        "SpotResources"
        " | where type =~ 'microsoft.compute/skuspotevictionrate/location'"
        f" | where tolower(tostring(sku.name)) in ({names})"
        " | project sku = tolower(tostring(sku.name)), location = tolower(location),"
        " rate = tostring(properties.evictionRate)")
    return {(r["sku"], r["location"]): r["rate"] for r in rows}


def region_metadata():
    status, body = arm.request("GET", subscription_path() + "/locations", "2022-12-01")
    if status != 200:
        return {}
    return {r["name"].lower(): {"display": r["displayName"], "geo": r.get("metadata", {}).get("geographyGroup", "")}
            for r in body["value"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--amx", default="yes", help="comma-separated AMX levels to include: yes,partial")
    ap.add_argument("--min-vcpus", type=int, default=8, help="ignore smaller sizes in the ranking")
    ap.add_argument("--top", type=int, default=20, help="regions to print")
    ap.add_argument("--refresh", action="store_true", help="re-download the SKU catalog")
    args = ap.parse_args()
    levels = set(args.amx.split(","))

    families = json.loads((ROOT / "data" / "amx_families.json").read_text())["families"]
    wanted = {f: v for f, v in families.items() if v["amx"] in levels}

    # SKU x region candidates: AMX family, x64, Spot-capable, not restricted for this subscription.
    cands = []
    for s in load_skus(args.refresh):
        if s.get("family") not in wanted or cap(s, "LowPriorityCapable") != "True":
            continue
        region = s["locations"][0].lower()
        if any(r["type"] == "Location" for r in s.get("restrictions", [])):
            continue
        vcpus, per_core = int(cap(s, "vCPUs")), int(cap(s, "vCPUsPerCore", "1"))
        cands.append({"region": region, "sku": s["name"], "family": s["family"],
                      "series": wanted[s["family"]]["series"], "amx": wanted[s["family"]]["amx"],
                      "vcpus": vcpus, "vcpus_per_core": per_core, "cores": vcpus // per_core,
                      "memory_gb": float(cap(s, "MemoryGB"))})
    sku_names = sorted({c["sku"] for c in cands})
    regions = sorted({c["region"] for c in cands})
    print(f"{len(cands)} Spot-capable AMX SKU x region pairs: {len(sku_names)} SKUs in {len(regions)} regions")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        prices = dict(zip(sku_names, pool.map(linux_prices, sku_names)))
        quotas = dict(zip(regions, pool.map(spot_quota, regions)))
    evictions = eviction_rates(sku_names)
    meta = region_metadata()

    rows = []
    for c in cands:
        p = prices[c["sku"]].get(c["region"], {})
        if "spot" not in p:
            continue  # no Spot meter in this region
        q = quotas.get(c["region"]) or {}
        rows.append({
            **c,
            "region_name": meta.get(c["region"], {}).get("display", c["region"]),
            "geography": meta.get(c["region"], {}).get("geo", ""),
            "spot_usd_h": p["spot"],
            "payg_usd_h": p.get("payg"),
            "spot_discount_pct": round(100 * (1 - p["spot"] / p["payg"]), 1) if p.get("payg") else None,
            "spot_usd_per_vcpu_h": round(p["spot"] / c["vcpus"], 6),
            "spot_usd_per_core_h": round(p["spot"] / c["cores"], 6),
            "eviction_rate": evictions.get((c["sku"].lower(), c["region"]), ""),
            "spot_quota_vcpus": q.get("limit"),
            "spot_quota_used": q.get("used"),
        })
    rows.sort(key=lambda r: (r["spot_usd_per_vcpu_h"], r["region"], r["sku"]))

    out = ROOT / "results" / f"spot_amx_{datetime.date.today().isoformat()}.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out.relative_to(ROOT)}")

    print_ranking(rows, args)


def print_ranking(rows, args):
    """Per region: cheapest AMX series by Spot $/vCPU-h, what fits the Spot quota, and the largest VM."""
    by_region = {}
    for r in rows:
        by_region.setdefault(r["region"], []).append(r)
    ranking = []
    for region, rs in by_region.items():
        sized = [r for r in rs if r["vcpus"] >= args.min_vcpus] or rs
        best = min(sized, key=lambda r: (r["spot_usd_per_vcpu_h"], -r["vcpus"]))
        series_rows = [r for r in rs if r["series"] == best["series"]]
        quota = best["spot_quota_vcpus"]
        fits = [r for r in series_rows if quota is None or r["vcpus"] <= quota]
        fit = max(fits, key=lambda r: r["vcpus"]) if fits else None
        largest = max(rs, key=lambda r: (r["vcpus"], -r["spot_usd_h"]))
        evict = sorted({r["eviction_rate"] for r in series_rows if r["eviction_rate"]})
        ranking.append((best["spot_usd_per_vcpu_h"], -largest["vcpus"], region,
                        best, fit, largest, "/".join(evict) or "?", len({r["series"] for r in rs})))
    ranking.sort(key=lambda t: t[:3])

    print(f"\nRegions by cheapest Spot Linux $/vCPU-hour (AMX: {args.amx}; sizes >= {args.min_vcpus} vCPUs)."
          " 'fits quota' = largest VM of that series within the region's Spot vCPU quota.")
    print(f"{'region':<19} {'geo':<13} {'series':<8} {'$/vCPU-h':>8} {'evict%':>7} {'quota':>5}"
          f" {'fits quota':<19} {'$/h':>6}  {'largest AMX VM':<19} {'$/h':>6} {'#ser':>4}")
    for _, _, region, best, fit, largest, evict, nseries in ranking[:args.top]:
        quota = best["spot_quota_vcpus"]
        print(f"{region:<19} {best['geography'][:13]:<13} {best['series']:<8} {best['spot_usd_per_vcpu_h']:>8.5f}"
              f" {evict:>7} {quota if quota is not None else '?':>5}"
              f" {fit['sku'] if fit else '-':<19} {fit['spot_usd_h'] if fit else 0:>6.3f}"
              f"  {largest['sku']:<19} {largest['spot_usd_h']:>6.3f} {nseries:>4}")


if __name__ == "__main__":
    main()
