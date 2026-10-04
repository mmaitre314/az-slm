# Agent instructions

This repo benchmarks Small Language Models (SLMs) on Azure VMs. It is a playground.

## Workflow

- Commit and push directly to `main`. Do not open pull requests or create feature branches.
- Write scripts in Python 3 (3.11 in the sandbox) using **only the standard library**. PyPI and npm
  are blocked in the agent sandbox (see [Network access](#network-access)), so third-party packages
  can't be installed. Use `urllib.request` for HTTP. It honors `HTTPS_PROXY` and the proxy CA bundle
  (`SSL_CERT_FILE`) with no extra setup. Ad-hoc `curl` is fine for one-off exploration.
- Put scripts in `scripts/` and call ARM through `scripts/arm.py`. Infrastructure is Bicep in `infra/`
  (no Azure Verified Modules: `mcr.microsoft.com` is blocked), deployed with `scripts/deploy.py`.
  Benchmark scripts that run on the VMs live in `bench/`.
- The Bicep CLI is installed by the environment setup script (`/usr/local/bin/bicep`, pinned release
  from GitHub). Run `bicep lint` and a `deploy.py ... --what-if-only` before deploying.

## Secrets and identifiers

Never commit identifying or secret values, including in code, docs, logs, notebooks, or benchmark
results:

- subscription ID, tenant ID, resource group name
- user names, emails, principal/object/client IDs
- public IPs, FQDNs of created resources, SSH keys, passwords, tokens

ARM resource IDs embed the subscription ID (`/subscriptions/<id>/...`). Redact them, or keep only
the resource name, before writing anything to the repo. Check `git diff --cached` before every commit.

## Azure configuration

Read all Azure context from environment variables and fail fast if any are unset:

| Variable                | Meaning                                 |
| ----------------------- | --------------------------------------- |
| `AZURE_TENANT_ID`       | Entra ID tenant                         |
| `AZURE_SUBSCRIPTION_ID` | Subscription that holds the RG          |
| `AZURE_RESOURCE_GROUP`  | Resource group where all work happens   |

### Authentication

Call Azure Resource Manager directly over HTTPS at `https://management.azure.com`, **without** an
`Authorization` header. The Claude environment's egress proxy attaches a Bearer token to requests
for that host.

- Do not run `az login`, request tokens, or use `azure-identity` / `DefaultAzureCredential`;
  `login.microsoftonline.com` is blocked and no credentials are present in the sandbox.
- Token injection covers `management.azure.com` only. Data-plane endpoints (Storage blobs,
  Key Vault, and so on) get no token, so prefer control-plane (ARM) operations.

```python
import arm  # scripts/arm.py

status, rg = arm.request("GET", arm.rg_path(), "2021-04-01")
```

ARM conventions:

- Every request needs an `api-version` query parameter.
- Long-running PUT/DELETE/POST calls return `201`/`202` with an `Azure-AsyncOperation` or `Location`
  header. Poll that URL, which is also on `management.azure.com`, until it reaches a terminal state.
- Browse API versions and schemas on Microsoft Learn (`learn.microsoft.com/rest/api/...`).

### Permissions

- **Owner** on `$AZURE_RESOURCE_GROUP` only. Create every resource inside that resource group.
- No role at subscription scope. Some subscription-level *reads* still work, as observed: the VM
  size/SKU catalog, regional compute quotas (`Microsoft.Compute/locations/{region}/usages`),
  marketplace images, and resource provider state. Expect subscription-level *writes* (new resource
  groups, provider registration, quota requests) to fail.
- Regional vCPU quotas are small (20 total and 20 per VM family per region). Check `usages` before
  picking VM sizes and counts.
- The subscription is a Visual Studio (MSDN) offer: **Spot VMs are not available** (Azure only
  allows Spot on EA, pay-as-you-go and Sponsored offers) and fail with a misleading
  `SkuNotAvailable` "Capacity Restrictions" error for every size and region. Deploy with
  `-p priority=Regular`. A spending limit caps total cost.
- `python3 scripts/check_azure_access.py` re-validates access and prints no identifiers.

## Working with VMs

- The sandbox can reach only HTTPS port 443 through the proxy, so SSH to VMs is not possible from here.
  Drive VMs through ARM: cloud-init (`infra/cloud-init.yaml`) at creation, then Run Command via
  `scripts/deploy.py`:
  - `deploy.py deploy infra/main.bicep --name <vm> -p vmName=<vm> -p priority=Regular`: one VM per
    deployment; use a new VM name per run (SSH key and customData can't change on an existing VM).
  - `deploy.py run <vm> bench/x.sh [-e K=V]`: runs as root under bash; returns the script's exit code.
    Run Command keeps only the **last 4 KiB** of output and times out after 90 minutes, so use
    `--background <job>` for anything long and `deploy.py job <vm> <job>` to poll it.
  - `deploy.py fetch <vm> <remote> <local>` copies small files (4 KiB per call: compress first).
  - `bench/save_results.sh <files>` (on the VM) copies results to `/var/lib/azslm/results` on the OS
    disk; `deploy.py harvest <vm> <local-dir> [--release]` fetches and unpacks them. Call it after
    every step of a job chain.
  - `deploy.py start <vm>` restarts a VM the watchdog deallocated.
  - `deploy.py teardown --run <vm>` deletes everything tagged `azslm-run=<vm>`.
- VMs have Internet access through their Standard public IP (no inbound rules), so they download
  models from Hugging Face directly (~1 GB/s observed) even though the sandbox cannot.
  `/mnt/data` is the local NVMe disk(s), striped: fast, but wiped on deallocation.
- Auto-shutdown: an in-VM watchdog deletes (default) or deallocates the VM after `idleHours` (1)
  without load or Run Command activity, using the VM's managed identity with a custom
  least-privilege role. While `/var/lib/azslm/keep` exists (written by `save_results.sh` and removed by
  `harvest --release`) it deallocates instead of deleting, so saved results survive. Azure's daily auto-shutdown deallocates it `fallbackHours` (12) after the
  deployment hour as a fallback. Touch `/var/lib/azslm/watchdog-disabled` on the VM to pause the
  watchdog. Delete VMs when a run finishes anyway.
- llama.cpp gotcha (checked 2026-10-04 on master): the AMX matmul path returns garbage for Qwen3.5-family
  models (Qwen3.8 included) when several sequences are decoded together (`llama-parallel`, server
  `-np`, `llama-perplexity` default batching). Single-sequence output is correct. Check output, not
  just tokens/s, and compare against a build with AMX compiled out (`bench/build_noamx.sh`).

## Benchmarking notes

- Experiments live in `experiments/` (see its README for the process, roles and cost model). Keep
  PLAN.md current.
- Subagents share a 5-hour usage limit and can stop mid-task (twice on 2026-10-04, losing results).
  Run each experiment as one self-contained background chain that needs no agent to proceed, save
  results after every step, and commit fetched results right away.
- Identical VM sizes vary: two E16ds_v7 instances differed by 9–25% on the same llama.cpp runs
  (E08). Compare configurations on the same VM, or include a reference run on each VM.
- Azure Policy installs Microsoft Defender for Endpoint and monitoring agents on every VM. Check
  `top` before measuring, and note it.
- llama.cpp on these 8-core/16-vCPU sizes: use 16 threads for prefill-heavy work (+14–25%,
  E08); pinning doesn't help. Its prefill is GEMM-bound at ~3% of AMX peak (E07), so prefer
  oneDNN-based stacks when prefill dominates.

## Docs

- Use the Microsoft Learn MCP tools (`microsoft_docs_search`, `microsoft_docs_fetch`,
  `microsoft_code_sample_search`) or fetch `https://learn.microsoft.com/...` directly.
- Prices: Retail Prices API at `prices.azure.com` (no auth). `docs.vllm.ai`, `recipes.vllm.ai`,
  `huggingface.co` and most of `github.com` are blocked from the sandbox; fetch them from a VM if needed.

## Network access

Observed from the agent sandbox on 2026-10-03, updated 2026-10-04. Re-probe if something fails, because the sandbox
configuration may have changed. `curl -sS "$HTTPS_PROXY/__agentproxy/status"` lists recent proxy
denials.

| Host                                   | Status                                          |
| -------------------------------------- | ----------------------------------------------- |
| `management.azure.com`                 | allowed, Bearer token injected                  |
| `learn.microsoft.com`                  | allowed                                         |
| `github.com` / `api.github.com`        | this repo, plus release asset downloads          |
| `login.microsoftonline.com`            | blocked                                         |
| `graph.microsoft.com`                  | blocked                                         |
| `azure.microsoft.com`, `prices.azure.com` | allowed                                      |
| `aka.ms`                               | blocked                                         |
| `raw.githubusercontent.com`            | blocked                                         |
| `huggingface.co`, `cdn-lfs.huggingface.co` | blocked                                     |
| `ollama.com`                           | blocked                                         |
| `docs.vllm.ai`, `recipes.vllm.ai`, `hub.docker.com` | blocked                            |
| `pypi.org`, `files.pythonhosted.org`   | blocked (`x-deny-reason: host_not_allowed`)     |
| `registry.npmjs.org`                   | blocked (`x-deny-reason: host_not_allowed`)     |
| `packages.microsoft.com`, `mcr.microsoft.com`, `archive.ubuntu.com`, `download.pytorch.org` | blocked |
