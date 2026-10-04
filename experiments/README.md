# Experiments

Each benchmark is an experiment with its own directory, `EXX-<slug>/`, containing:

- `README.md`: the write-up (see [TEMPLATE.md](TEMPLATE.md)): hypotheses, setup, method,
  measurements, cost per token, analysis, threats to validity, next steps.
- Raw data copied from the VM (`*.jsonl`, short logs). Keep files small, and never include
  subscription/tenant IDs, the resource group name, principal IDs, IPs or keys.

[PLAN.md](PLAN.md) is the backlog and status board. [COST_MODEL.md](COST_MODEL.md) defines how cost
per token is computed and which assumptions every report must state.

## Roles

| Role | Model | Owns |
| --- | --- | --- |
| Orchestrator | the most capable model (interactive session) | PLAN.md, hypotheses, experiment design, investigation of anomalies, analysis and conclusions, guidance in this directory and in `AGENTS.md`, reviewing agents' work |
| [experiment-runner](../.claude/agents/experiment-runner.md) | Sonnet | writing and fixing `bench/` scripts, deploying VMs, running experiments as background jobs, debugging installs, recording setup and raw measurements |
| [experiment-reporter](../.claude/agents/experiment-reporter.md) | Sonnet | fetching results, tables via `scripts/summarize_bench.py`, cost-per-token sections, a draft analysis flagged for orchestrator review |
| [progress-tracker](../.claude/agents/progress-tracker.md) | Haiku | polling background jobs on all VMs, updating the status column in PLAN.md, flagging failures and idle VMs |

The orchestrator delegates with the Workflow or Agent tool, passing the agent's role file and the
experiment directory. Agents get the plain facts they need in their prompt: VM name, region,
experiment ID, the scripts to use, and the stop conditions. They report back a short structured
result; the experiment README is the durable record.

## Lifecycle

1. **Plan** (orchestrator): add a row to PLAN.md and create `EXX-<slug>/README.md` from the template
   with Question, Hypotheses and Method filled in. Pick a VM, keeping one stack per VM so runs
   don't contend for CPU.
2. **Run** (runner): deploy the VM if needed, write or adapt `bench/` scripts, start them with
   `scripts/deploy.py run <vm> <script> --background <job>`, and confirm they're running. Long work
   always runs as a background job, written as one self-contained chain that runs every step
   without an agent in the loop and calls `bench/save_results.sh` after each step. The runner
   returns instead of waiting.
3. **Track** (tracker, or a shell monitor): poll `scripts/deploy.py job <vm> <job>` and update
   PLAN.md.
4. **Report** (reporter): `deploy.py harvest <vm> EXX-<slug>/raw` the saved results, write
   Measurements and Cost, draft Analysis. Pass `--release` once nothing else is needed from the VM.
5. **Conclude** (orchestrator): review the numbers, write Analysis and Next steps, and update
   PLAN.md, adding new experiments the results call for.
6. **Clean up**: `scripts/deploy.py teardown --run <vm>` once no planned experiment needs the VM.

## Rules for everyone

- One experiment per directory; append, don't overwrite, raw data across reruns (tag rows with
  the run).
- Check outputs, not just speed: a correctness check (fixed prompt, perplexity/KLD) comes before
  trusting a throughput number (see E04 for why).
- Record versions: llama.cpp commit, vLLM/OpenVINO versions, image digests, model repo revisions.
- State every assumption behind a cost number (see COST_MODEL.md).
- Mind the budget in PLAN.md: tear down VMs that have nothing queued.
- Assume any agent can stop mid-task (agents share a 5-hour usage limit; on 2026-10-04 it stopped
  every runner twice). Results must never live only in an agent's context or only on `/mnt/data`
  (wiped on deallocation, lost when the VM deletes itself). `bench/save_results.sh` copies them to
  the OS disk and makes the watchdog deallocate instead of delete; commit fetched results right away.
