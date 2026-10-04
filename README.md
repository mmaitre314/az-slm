# SLM batch inference on Azure CPU VMs

Benchmarks of language models on Azure CPU VMs. The current question: what is the cheapest way to
run offline batch inference of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on Azure VMs
with Intel AMX, at acceptable quality? Latency doesn't matter. The answer is reported as USD per
million input and output tokens for each serving stack, quantization and VM size (so far
`Standard_E16ds_v7`, Granite Rapids, and `Standard_E16ds_v6`, Emerald Rapids). The repo is a
playground: small standard-library Python and bash scripts, driven mostly by AI agents under the
rules in [AGENTS.md](AGENTS.md).

## Results so far

Best configuration per stack for the reference workload: 512 input + 128 output tokens per request,
VM busy 100% of the time, list prices plus $0.018/h for the OS disk and public IP
([COST_MODEL.md](experiments/COST_MODEL.md)). Costs are blended USD per million tokens.

| Stack | Configuration | VM | On demand | Spot-equivalent | Source |
| --- | --- | --- | ---: | ---: | --- |
| **vLLM 0.31.0 CPU** | INT8 W8A8 (community quant), 16 prompts, 8 threads | E16ds_v7 | **4.56** | **0.88** | [E09](experiments/E09-vllm-cpu/) (partial) |
| vLLM 0.31.0 CPU | BF16, 16 prompts | E16ds_v7 | 7.45 | 1.44 | [E09](experiments/E09-vllm-cpu/) (partial) |
| llama.cpp `11fe021` | Q4_K_M, 32 sequences, no-AMX build | E16ds_v6 | 19.3 | 3.8 | [E13](experiments/E13-emerald-vs-granite/) |
| llama.cpp `11fe021` | Q4_K_M, 32 sequences, no-AMX build | E16ds_v7 | 23.9 | 4.6 | [E05](experiments/E05-llamacpp-batched/) |
| OpenVINO GenAI 2026.4.1 | INT4 IR, 4 requests measured; 8+ estimated | E16ds_v7 | 34.1 (est. 19–34 at 8+) | 6.6 (est. 3.6–6.6) | [E10](experiments/E10-openvino-genai/) (partial) |
| llama.cpp `11fe021`, one sequence | Q4_0, AMX build | E16ds_v7 | 35.4 | 6.8 | [E03](experiments/E03-llamacpp-single-stream/) |

Read these numbers with care:

- They are **round-1 numbers as of 2026-10-04**. E09, E10 and E12 are partial: their VMs deleted
  themselves before the raw files were fetched, and the numbers were recovered from the runners'
  transcripts.
- This subscription can't deploy Spot VMs (E01). The Spot column is what the same throughput would
  cost on a subscription that can.
- Batched llama.cpp rows use a build with AMX compiled out, because llama.cpp's AMX path corrupts
  output when several sequences are decoded together for this model
  ([E04](experiments/E04-llamacpp-amx-multiseq-bug/)).
- The recommendation in [E15](experiments/E15-summary/) is **provisional**: vLLM CPU with INT8 W8A8
  weights, ~$4.6 per million tokens on demand at 16 prompts. It holds only if E16 confirms that the
  community W8A8 checkpoint keeps BF16's accuracy; BF16 at $7.45/M is the fallback. E16 (task
  quality), E17 (vLLM batch scaling, W4A16, MTP) and E18 (vLLM on E16ds_v6) are running.

Other round-1 findings: llama.cpp prefill is GEMM-bound and its AMX kernel reaches ~3% of AMX peak
([E07](experiments/E07-prefill-profile/)); the model's MTP head gives 1.5× single-sequence decode in
llama.cpp but only +19% at 4 slots ([E12](experiments/E12-speculative-decoding/), partial). See
[experiments/PLAN.md](experiments/PLAN.md) for current status.

## How it works

Nothing connects to the VMs over SSH. Everything goes through Azure Resource Manager (ARM):

1. **ARM over HTTPS.** Scripts call `https://management.azure.com` with no `Authorization` header.
   The Claude environment's egress proxy attaches a Bearer token for that host.
   [`scripts/arm.py`](scripts/arm.py) is the client: `api-version` handling, long-running
   operation polling, Resource Graph queries, and redaction of identifiers in output.
2. **One VM per deployment.** `scripts/deploy.py deploy` compiles
   [`infra/main.bicep`](infra/main.bicep) with the Bicep CLI, validates it, prints a what-if and
   deploys: Ubuntu 24.04, `Standard_E16ds_v7` in `eastus2` by default, in a shared per-region VNet
   whose NSG has no inbound rules. Azure requires an SSH key, so `deploy.py` generates a throwaway
   one and never stores the private half.
3. **cloud-init** ([`infra/cloud-init.yaml`](infra/cloud-init.yaml)) mounts the local NVMe disks at
   `/mnt/data` (RAID 0 when there are several), installs the Hugging Face CLI, builds llama.cpp
   natively (AMX kernels included) at `/opt/llama.cpp/build`, and installs the idle watchdog. VMs
   download models from Hugging Face themselves through their public IP.
4. **Run Command.** `deploy.py run` sends a `bench/` script to the VM and runs it as root. Run
   Command keeps only the last 4 KiB of output and times out after 90 minutes, so long work runs
   with `--background <job>` as a systemd unit (log in `/mnt/data/jobs/<job>.log`), polled with
   `deploy.py job`.
5. **Results.** Benchmarks append JSON lines under `/mnt/data/results`. `/mnt/data` is wiped when
   the VM is deallocated, so [`bench/save_results.sh`](bench/save_results.sh) copies results to
   `/var/lib/azslm/results` on the OS disk after every step, and `deploy.py harvest` fetches and
   unpacks them locally.

## Prerequisites

| Requirement | Detail |
| --- | --- |
| Environment variables | `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`. Scripts exit if any is unset. |
| ARM access | Owner on `$AZURE_RESOURCE_GROUP`. The scripts never log in or request tokens; they rely on the egress proxy that injects one for `management.azure.com`. |
| Python | Python 3, standard library only (3.11 in the sandbox). PyPI is blocked there. |
| Tools | Bicep CLI on `PATH` (`/usr/local/bin/bicep` in the sandbox) and `openssl` (throwaway SSH key). |
| Subscription limits | Visual Studio (MSDN) offer: Spot VMs fail with a misleading `SkuNotAvailable` error, so always pass `-p priority=Regular` (the template default is `Spot`). Regional vCPU quota is 20 total and 20 per VM family, so one 16-vCPU VM per region. A spending limit caps total cost. |

Details, including the sandbox network allow list, are in [AGENTS.md](AGENTS.md).

## Quick start

Use a new VM name for every run: the SSH key and cloud-init data can't change on an existing VM.

```bash
python3 scripts/check_azure_access.py          # env vars, RG access, Owner permissions; no identifiers printed

bicep lint infra/main.bicep
python3 scripts/deploy.py deploy infra/main.bicep --name bench1 -p vmName=bench1 -p priority=Regular --what-if-only
python3 scripts/deploy.py deploy infra/main.bicep --name bench1 -p vmName=bench1 -p priority=Regular

python3 scripts/deploy.py run bench1 bench/check_amx.sh   # AMX probe; "setup-done" shows when cloud-init finished
python3 scripts/deploy.py push bench1                     # copy bench/ to /opt/azslm/bench on the VM
python3 scripts/deploy.py run bench1 bench/download.sh -e REPO=bartowski/Qwen3.8-27B-GGUF -e INCLUDE='*Q4_K_M*'
python3 scripts/deploy.py run bench1 bench/llama_bench.sh --background llama -e QUANTS=Q4_K_M -e BUILDS=build
python3 scripts/deploy.py job bench1 llama                # job state and log tail

python3 scripts/deploy.py run bench1 -c 'bash /opt/azslm/bench/save_results.sh llama-bench.jsonl'
python3 scripts/deploy.py harvest bench1 experiments/EXX-<slug>/raw --release
python3 scripts/deploy.py teardown --run bench1
```

Other subcommands: `deploy.py fetch <vm> <remote> <local>` copies one small file (compress it first),
`deploy.py status` lists the VMs in the resource group, and `deploy.py start <vm>` restarts a
deallocated VM. `python3 scripts/summarize_bench.py <dir> --price <all-in $/h, e.g. 1.681> [--spot-price <all-in $/h, e.g. 0.325>]`
turns harvested JSONL into markdown cost tables (all-in = VM list price + $0.018/h, see
[COST_MODEL.md](experiments/COST_MODEL.md)).

## Cost safety

VMs cost money while they exist, so several layers clean up after them:

- **Idle watchdog.** A systemd timer on the VM checks every 5 minutes. Activity is a 15-minute load
  average of 1.0 or more, a `deploy.py` Run Command (each one touches `/var/lib/azslm/heartbeat`),
  or a boot. After `idleHours` (default 1) without activity, the VM deletes itself (`idleAction`
  `delete`, the default) or deallocates (`deallocate`). It uses its managed identity with a custom
  role limited to reading, deallocating and deleting its own VM, OS disk, NIC and public IP.
- **Keep marker.** While `/var/lib/azslm/keep` exists, the watchdog deallocates instead of deleting,
  so saved results survive. `save_results.sh` creates it and `harvest --release` removes it.
- **Pause.** Touch `/var/lib/azslm/watchdog-disabled` on the VM to stop the watchdog from acting.
- **Daily fallback.** An Azure auto-shutdown schedule deallocates the VM daily, `fallbackHours`
  (default 12) after the deployment hour, or at `dailyShutdownTime` (HHmm, UTC) if set.
- **Teardown.** `deploy.py teardown --run <vm>` deletes the resources tagged `azslm-run=<vm>` and
  the role assignments on them. The shared network is tagged `azslm-shared=network` and stays;
  `--all` deletes every resource in the resource group. A deallocated VM still holds vCPU quota
  and pays for its disk and IP: `deploy.py status` flags it.

Round 1 (2026-10-04, five VMs) cost about $50, estimated from activity-log VM lifetimes × list
price, of which ~$20 was VMs sitting idle after their agents stopped. Later rounds' budgets and
spend are in [PLAN.md](experiments/PLAN.md).

## Repository layout

| Path | Contents |
| --- | --- |
| [`AGENTS.md`](AGENTS.md) | Rules for agents and contributors: workflow, secrets, Azure auth and permissions, VMs, network access |
| [`CLAUDE.md`](CLAUDE.md) | Imports `AGENTS.md` for Claude Code |
| [`.claude/agents/`](.claude/agents/) | Subagent role files: experiment-runner, experiment-reporter, progress-tracker |
| [`scripts/arm.py`](scripts/arm.py) | Standard-library ARM client |
| [`scripts/check_azure_access.py`](scripts/check_azure_access.py) | Re-validates env vars, resource group access and Owner permissions |
| [`scripts/deploy.py`](scripts/deploy.py) | `deploy`, `run`, `push`, `job`, `fetch`, `harvest`, `start`, `status`, `teardown` |
| [`scripts/spot_amx.py`](scripts/spot_amx.py) | Ranks regions by Spot price for AMX VM sizes; writes `results/spot_amx_<date>.csv` |
| [`scripts/summarize_bench.py`](scripts/summarize_bench.py) | Markdown tables with USD per million tokens from benchmark JSONL |
| [`infra/main.bicep`](infra/main.bicep) | One benchmark VM: region, size, priority, watchdog and shutdown parameters |
| [`infra/modules/`](infra/modules/) | `network.bicep` (shared VNet and NSG per region), `spot-vm.bicep` (VM, identity, self-shutdown role, daily schedule) |
| [`infra/cloud-init.yaml`](infra/cloud-init.yaml) | NVMe mount, idle watchdog, llama.cpp build |
| [`bench/`](bench/) | Scripts that run on the VM (see below) |
| [`data/amx_families.json`](data/amx_families.json) | AMX support of 170 ARM VM families, classified from Microsoft Learn (E01) |
| [`results/spot_amx_2026-10-04.csv`](results/spot_amx_2026-10-04.csv) | Spot and pay-as-you-go prices, quotas and eviction rates per region and AMX size (E01) |
| [`experiments/`](experiments/) | One directory per experiment, plus the plan, research queue, cost model and template |

`bench/` groups:

| Group | Scripts |
| --- | --- |
| Setup and checks | `check_amx.sh`, `download.sh`, `list_models.sh`, `build_noamx.sh`, `build_native_noamx.sh`, `sanity.sh` |
| llama.cpp | `llama_bench.sh`, `llama_batched.sh`, `llama_threads.sh`, `profile_prefill.sh`, `quality.sh` |
| vLLM CPU (Docker) | `vllm_setup.sh`, `vllm_bench.sh`, `vllm_correct.sh`/`.py`, `vllm_queue.sh`, `vllm_sweep.sh`/`.py` |
| OpenVINO GenAI and OVMS | `openvino_*.sh`/`.py`, `ovms_serve.sh`, `ovms_bench.sh`, `ovms_client.py` |
| Speculative decoding (E12) | `spec_*.sh`, `spec_*.py` |
| Task quality (E16) | `quality_tasks.py`, `e16_convert.py` |
| Unattended chains | `e16_chain.sh`, `vllm_chain.sh`, `mtp_real_chain.sh`, `save_results.sh` |

Scripts that call other `bench/` scripts expect them at `/opt/azslm/bench`, so run
`deploy.py push <vm>` first. Each script's header lists its environment variables.

## Experiments

Each experiment is a directory `experiments/EXX-<slug>/` with a write-up based on
[TEMPLATE.md](experiments/TEMPLATE.md) (question, hypotheses, method, measurements, cost per token,
analysis, threats to validity) and small raw data files. [experiments/README.md](experiments/README.md)
describes the process, [PLAN.md](experiments/PLAN.md) is the backlog and status board,
[RESEARCH.md](experiments/RESEARCH.md) holds the capacity checks (no GPU or HBM quota on this
subscription) and the queue of ideas and candidate experiments, and
[COST_MODEL.md](experiments/COST_MODEL.md) defines prices, formulas and assumptions.

Snapshot of PLAN.md on 2026-10-04 17:30 UTC (current status is in [PLAN.md](experiments/PLAN.md)):

| ID | Title | Status |
| --- | --- | --- |
| [E01](experiments/E01-region-price-survey/) | AMX VM families, Spot prices and regions | done |
| [E02](experiments/E02-amx-enablement/) | Is AMX usable inside an Azure VM? | done |
| [E03](experiments/E03-llamacpp-single-stream/) | llama.cpp single-sequence prefill/decode per quant, AMX vs no-AMX | done |
| [E04](experiments/E04-llamacpp-amx-multiseq-bug/) | llama.cpp AMX output corruption with several sequences | done (finding) |
| [E05](experiments/E05-llamacpp-batched/) | llama.cpp batch throughput, 1–32 sequences | done |
| [E06](experiments/E06-quant-quality/) | Quantization quality: KL divergence vs BF16 | done |
| [E07](experiments/E07-prefill-profile/) | Why is prefill slow? CPU profile | done |
| [E08](experiments/E08-threads-smt/) | Threads, SMT and pinning; native no-AMX control build | done |
| [E09](experiments/E09-vllm-cpu/) | vLLM CPU backend: BF16, INT8 W8A8, INT4 W4A16 | done (partial; rest in E17) |
| [E10](experiments/E10-openvino-genai/) | OpenVINO GenAI: INT4/INT8 weights, continuous batching | done (partial; INT4 only) |
| [E11](experiments/E11-ovms/) | OpenVINO Model Server with continuous batching | deprioritized (E10) |
| [E12](experiments/E12-speculative-decoding/) | Speculative decoding in llama.cpp: draft model, MTP head, n-gram | done (partial; vLLM MTP in E17) |
| [E13](experiments/E13-emerald-vs-granite/) | Emerald Rapids (v6) vs Granite Rapids (v7), same llama.cpp runs | done |
| E14 | SGLang CPU backend (Intel AMX kernels) | candidate |
| [E15](experiments/E15-summary/) | Cross-stack cost and quality summary, recommendation | draft |
| [E16](experiments/E16-task-quality/) | Task-level quality of vLLM BF16, W8A8, W4A16 (GSM8K, MMLU) | done |
| [E17](experiments/E17-vllm-scaling-mtp/) | vLLM batch scaling, W4A16, threads, MTP | done |
| [E18](experiments/E18-vllm-emerald-rapids/) | vLLM on E16ds_v6 vs E16ds_v7 | running |

## Working with agents

[AGENTS.md](AGENTS.md) holds the rules for any agent or contributor; [CLAUDE.md](CLAUDE.md) only
imports it. Experiments are split across four roles ([experiments/README.md](experiments/README.md)):

| Role | Model | Owns |
| --- | --- | --- |
| Orchestrator | the most capable model (interactive session) | PLAN.md, hypotheses, experiment design, anomalies, conclusions, reviews |
| [experiment-runner](.claude/agents/experiment-runner.md) | Sonnet | `bench/` scripts, deploying VMs, starting background jobs, raw measurements |
| [experiment-reporter](.claude/agents/experiment-reporter.md) | Sonnet | harvesting results, measurement and cost sections, a draft analysis |
| [progress-tracker](.claude/agents/progress-tracker.md) | Haiku | polling jobs, the status column in PLAN.md, flagging failures and idle VMs |

The lifecycle is plan, run, track, report, conclude, clean up. One lesson shaped it: subagents share
a 5-hour usage limit, and on 2026-10-04 it stopped the runners twice. With no Run Commands arriving,
the VMs went idle and deleted themselves, and the raw results of E09, E10 and E12 were lost with
them. Since then each experiment runs as one self-contained background chain that needs no agent to
proceed, saves results to the OS disk after every step, and gets its results committed as soon as
they are harvested.

## Rules

- Never commit identifying or secret values: subscription and tenant IDs, the resource group name,
  user names, emails, principal IDs, public IPs, FQDNs, SSH keys, passwords or tokens. ARM resource
  IDs embed the subscription ID, so redact them. Check `git diff --cached` before every commit.
  `deploy.py` and `check_azure_access.py` print through `arm.redact()`, which masks the subscription
  and tenant IDs, and the resource group only inside resource IDs. Files fetched by `deploy.py
  fetch`/`harvest` are not redacted: grep them for the three environment values before committing.
- Check output, not just speed: a correctness check comes before any throughput number (E04).
- Record versions (llama.cpp commit, vLLM and OpenVINO versions, image digests, model revisions) and
  state every cost assumption.

The full rules are in [AGENTS.md](AGENTS.md) and [experiments/README.md](experiments/README.md).

## License

MIT. See [LICENSE](LICENSE).
