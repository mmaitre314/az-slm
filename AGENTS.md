# Agent instructions

This repo benchmarks Small Language Models (SLMs) on Azure VMs. It is a playground.

## Workflow

- Commit and push directly to `main`. Do not open pull requests or create feature branches.
- Keep tooling lightweight: `bash`, `curl`, and the Python 3 standard library are always available.
  PyPI and npm are blocked in the agent sandbox (see [Network access](#network-access)), so do not
  depend on third-party packages on the agent side.

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

```bash
ARM=https://management.azure.com
curl -sS "$ARM/subscriptions/$AZURE_SUBSCRIPTION_ID/resourceGroups/$AZURE_RESOURCE_GROUP?api-version=2021-04-01"
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
- Regional vCPU quotas are small. Check `usages` before picking VM sizes and counts.
- `scripts/check-azure-access.sh` re-validates access and prints no identifiers.

## Working with VMs

- The sandbox can reach only HTTPS port 443 through the proxy, so SSH to VMs is not possible from here.
  Drive VMs through ARM instead: `customData` / cloud-init at creation time, then
  VM Run Command (`Microsoft.Compute/virtualMachines/runCommands`) to execute benchmarks and
  collect their output.
- VMs egress through Azure, not through the sandbox proxy, so they should be able to download
  models (for example from Hugging Face) even though the sandbox cannot. This is not yet verified.
  Subnets in new VNets created with recent `Microsoft.Network` API versions are private by default
  (`defaultOutboundAccess: false`), so give VMs an explicit outbound method (a NAT Gateway or a
  public IP) when they need the internet.
- Control cost: deallocate or delete VMs when a run finishes, and tag resources with the benchmark
  they belong to.

## Docs

- Use the Microsoft Learn MCP tools (`microsoft_docs_search`, `microsoft_docs_fetch`,
  `microsoft_code_sample_search`) or fetch `https://learn.microsoft.com/...` directly.
- `azure.microsoft.com` (pricing pages) and `prices.azure.com` (Retail Prices API) are blocked.

## Network access

Observed from the agent sandbox on 2026-10-03. Re-probe if something fails, because the sandbox
configuration may have changed. `curl -sS "$HTTPS_PROXY/__agentproxy/status"` lists recent proxy
denials.

| Host                                   | Status                                          |
| -------------------------------------- | ----------------------------------------------- |
| `management.azure.com`                 | allowed, Bearer token injected                  |
| `learn.microsoft.com`                  | allowed                                         |
| `github.com` / `api.github.com`        | allowed for this repo only                      |
| `login.microsoftonline.com`            | blocked                                         |
| `graph.microsoft.com`                  | blocked                                         |
| `azure.microsoft.com`, `prices.azure.com` | blocked                                      |
| `aka.ms`                               | blocked                                         |
| `raw.githubusercontent.com`            | blocked                                         |
| `huggingface.co`, `cdn-lfs.huggingface.co` | blocked                                     |
| `ollama.com`                           | blocked                                         |
| `pypi.org`, `files.pythonhosted.org`   | blocked (`x-deny-reason: host_not_allowed`)     |
| `registry.npmjs.org`                   | blocked (`x-deny-reason: host_not_allowed`)     |
| `packages.microsoft.com`, `mcr.microsoft.com`, `archive.ubuntu.com`, `download.pytorch.org` | blocked |
