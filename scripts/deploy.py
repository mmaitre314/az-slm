#!/usr/bin/env python3
"""Deploy Bicep templates to $AZURE_RESOURCE_GROUP and drive the VMs they create, over ARM only.

  deploy.py deploy infra/main.bicep --name bench1 -p vmName=bench1 [--what-if-only]
  deploy.py run bench1 bench/check_amx.sh [-e KEY=VALUE ...]    # RunShellScript, prints output
  deploy.py run bench1 -c 'uptime'
  deploy.py run bench1 bench/llama_bench.sh --background llama -e QUANTS=Q4_K_M   # long jobs
  deploy.py job bench1 llama                                     # job state + log tail
  deploy.py fetch bench1 /mnt/data/results.tar.gz results/x.tar.gz
  deploy.py status
  deploy.py teardown --run bench1 [--all]

`deploy` compiles with the Bicep CLI, then validates, runs what-if, deploys and waits. A template
parameter named sshPublicKey that is not passed gets a throwaway Ed25519 key whose private half is
never written anywhere: VMs are driven through Run Command, not SSH.
"""

import argparse
import base64
import io
import json
import pathlib
import shlex
import struct
import subprocess
import sys
import tarfile

import arm

API = {
    "deployments": "2024-03-01",
    "Microsoft.Compute/virtualMachines": "2024-07-01",
    "Microsoft.Compute/disks": "2024-03-02",
    "Microsoft.Network/networkInterfaces": "2024-05-01",
    "Microsoft.Network/publicIPAddresses": "2024-05-01",
    "Microsoft.Network/networkSecurityGroups": "2024-05-01",
    "Microsoft.Network/virtualNetworks": "2024-05-01",
    "Microsoft.DevTestLab/schedules": "2018-09-15",
}
# Deletion order for teardown: dependents before what they depend on.
DELETE_ORDER = [
    "Microsoft.Compute/virtualMachines",
    "Microsoft.DevTestLab/schedules",
    "Microsoft.Network/networkInterfaces",
    "Microsoft.Network/publicIPAddresses",
    "Microsoft.Compute/disks",
    "Microsoft.Network/virtualNetworks",
    "Microsoft.Network/networkSecurityGroups",
]
RUN_TAG = "azslm-run"
HEARTBEAT = "mkdir -p /var/lib/azslm && touch /var/lib/azslm/heartbeat"  # keeps the idle watchdog at bay
RUN_OUTPUT_LIMIT = 4096  # Run Command returns only the last 4 KiB of stdout/stderr


def out(text):
    print(arm.redact(str(text)), flush=True)


def fail(text):
    raise SystemExit(arm.redact(str(text)))


def throwaway_ssh_key():
    """OpenSSH-format Ed25519 public key; the private key only ever exists in an openssl pipe."""
    priv = subprocess.run(["openssl", "genpkey", "-algorithm", "ed25519"], check=True, capture_output=True).stdout
    der = subprocess.run(["openssl", "pkey", "-pubout", "-outform", "DER"], input=priv, check=True,
                         capture_output=True).stdout
    raw = der[-32:]  # SubjectPublicKeyInfo for Ed25519 ends with the 32-byte public key

    def field(b):
        return struct.pack(">I", len(b)) + b
    blob = field(b"ssh-ed25519") + field(raw)
    return "ssh-ed25519 " + base64.b64encode(blob).decode() + " azslm-throwaway"


def parse_value(text, param_type):
    """Convert a -p value to the template parameter's declared type ('1900' stays a string)."""
    t = (param_type or "string").lower()
    if t in ("string", "securestring"):
        return text
    if t == "int":
        return int(text)
    if t == "bool":
        return text.lower() in ("true", "1", "yes")
    return json.loads(text)  # object / array


def deployment_path(name):
    return f"{arm.rg_path()}/providers/Microsoft.Resources/deployments/{name}"


def short_id(resource_id):
    """'Namespace/type/name' of the innermost resource, e.g. Microsoft.Authorization/roleAssignments/<guid>."""
    return resource_id.rsplit("/providers/", 1)[-1]


def cmd_deploy(args):
    template = json.loads(subprocess.run(["bicep", "build", args.template, "--stdout"], check=True,
                                         capture_output=True, text=True).stdout)
    declared = template.get("parameters", {})
    params = {}
    for k, v in (p.split("=", 1) for p in args.param):
        if k not in declared:
            fail(f"unknown template parameter: {k}")
        params[k] = {"value": parse_value(v, declared[k].get("type"))}
    if "sshPublicKey" in template.get("parameters", {}) and "sshPublicKey" not in params:
        params["sshPublicKey"] = {"value": throwaway_ssh_key()}
    body = {"properties": {"mode": "Incremental", "template": template, "parameters": params}}
    path = deployment_path(args.name)

    status, result = arm.lro("POST", path + "/validate", API["deployments"], body=body)
    if status != 200 or (result or {}).get("error"):
        fail(f"validate failed (HTTP {status}): {json.dumps(result, indent=1)}")
    out("validate: ok")

    status, result = arm.lro("POST", path + "/whatIf", API["deployments"], body=body)
    if status != 200 or result.get("status") not in (None, "Succeeded"):
        fail(f"what-if failed (HTTP {status}): {json.dumps(result, indent=1)}")
    changes = result.get("properties", result).get("changes", [])
    out(f"what-if: {len(changes)} change(s)")
    for c in changes:
        out(f"  {c['changeType']:<10} {short_id(c['resourceId'])}")
    if args.what_if_only:
        return

    status, result = arm.lro("PUT", path, API["deployments"], body=body, poll_seconds=10)
    if status not in (200, 201):
        fail(f"deployment request failed (HTTP {status}): {json.dumps(result, indent=1)}")
    status, dep = arm.request("GET", path, API["deployments"])
    state = dep["properties"]["provisioningState"]
    out(f"deployment {args.name}: {state} ({dep['properties'].get('duration', '?')})")
    if state != "Succeeded":
        _, ops = arm.request("GET", path + "/operations", API["deployments"])
        for op in ops.get("value", []):
            p = op["properties"]
            if p.get("provisioningState") == "Failed":
                target = p.get("targetResource", {}).get("id", "?")
                out(f"  FAILED {short_id(target)}: {json.dumps(p.get('statusMessage'))[:1500]}")
        raise SystemExit(1)
    for k, v in (dep["properties"].get("outputs") or {}).items():
        out(f"  output {k} = {v['value']}")


def vm_path(vm):
    return f"{arm.rg_path()}/providers/Microsoft.Compute/virtualMachines/{vm}"


EXIT_MARKER = "[azslm-exit "


def run_script(vm, script, env=(), timeout=5400):
    """Run a bash script as root on the VM via Run Command.

    Returns (stdout, stderr, exit_code); stdout and stderr are each limited to their last 4 KiB.
    The script is written to a temp file and run with bash, whatever shell Run Command uses.
    """
    lines = [HEARTBEAT] + [f"export {k}={shlex.quote(v)}" for k, v in env] + [
        'f=$(mktemp /tmp/azslm-XXXXXX.sh)',
        "cat > \"$f\" <<'AZSLM_SCRIPT_EOF'",
        *script.splitlines(),
        "AZSLM_SCRIPT_EOF",
        'bash "$f"; rc=$?; rm -f "$f"',
        f'echo "{EXIT_MARKER}$rc]"; exit $rc',
    ]
    body = {"commandId": "RunShellScript", "script": lines}
    status, result = arm.lro("POST", vm_path(vm) + "/runCommand", API["Microsoft.Compute/virtualMachines"],
                             body=body, poll_seconds=5, max_wait=timeout)
    if status != 200 or result.get("status") not in (None, "Succeeded"):
        fail(f"run command failed (HTTP {status}): {json.dumps(result)[:2000]}")
    value = result.get("properties", result).get("output", result).get("value", [])
    message = value[0]["message"] if value else ""
    stdout, _, stderr = message.partition("[stderr]")
    stdout = stdout.split("[stdout]", 1)[-1].strip("\n")
    body_text, marker, rest = stdout.rpartition(EXIT_MARKER)
    code = int(rest.split("]", 1)[0]) if marker and rest.split("]", 1)[0].isdigit() else -1
    return (body_text.rstrip("\n") if marker else stdout), stderr.strip("\n"), code


JOBS = "/mnt/data/jobs"


def background_wrapper(name, script, env):
    """Script that stores `script` on the VM and starts it as a transient systemd unit, so it can
    outlive Run Command's 90-minute limit. Output goes to JOBS/<name>.log, exit code to <name>.exit."""
    setenv = " ".join(shlex.quote(f"--setenv={k}={v}") for k, v in env)
    return f"""set -e
mkdir -p {JOBS}
cat > {JOBS}/{name}.sh <<'AZSLM_JOB_EOF'
{script}
AZSLM_JOB_EOF
rm -f {JOBS}/{name}.exit
systemctl reset-failed azslm-job-{name} 2>/dev/null || true
systemd-run --unit=azslm-job-{name} --collect {setenv} \\
  -p StandardOutput=append:{JOBS}/{name}.log -p StandardError=append:{JOBS}/{name}.log \\
  /bin/bash -c 'bash {JOBS}/{name}.sh; echo $? > {JOBS}/{name}.exit'
echo "started azslm-job-{name}"
"""


def cmd_run(args):
    script = args.command if args.command else pathlib.Path(args.script).read_text()
    env = [e.split("=", 1) for e in args.env]
    if args.background:
        script, env = background_wrapper(args.background, script, env), []
    stdout, stderr, code = run_script(args.vm, script, env)
    out(stdout)
    if stderr:
        print(arm.redact(stderr), file=sys.stderr)
    if code != 0:
        raise SystemExit(f"script exited with {code}")


def cmd_push(args):
    """Copy a small local directory (e.g. bench/) to the VM, for scripts that call each other."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in sorted(pathlib.Path(args.src).iterdir()):
            if f.is_file():
                tar.add(f, arcname=f.name)
    b64 = base64.encodebytes(buf.getvalue()).decode()
    if len(b64) > 200_000:
        fail(f"{args.src} is too large to push via Run Command ({len(b64)} bytes base64)")
    dest = shlex.quote(args.dest)
    stdout, _, code = run_script(args.vm, f"""mkdir -p {dest}
base64 -d > /tmp/azslm-push.tgz <<'AZSLM_PUSH_EOF'
{b64}AZSLM_PUSH_EOF
tar xzf /tmp/azslm-push.tgz -C {dest} && rm -f /tmp/azslm-push.tgz && chmod +x {dest}/*.sh 2>/dev/null
ls {dest} | tr '\\n' ' '""")
    out(stdout)
    if code != 0:
        raise SystemExit(code)


def cmd_job(args):
    """Status of a background job started with `run --background NAME`."""
    n = args.name
    # State goes last: Run Command keeps only the last 4 KiB of output.
    stdout, _, _ = run_script(args.vm, f"""tail -n {args.lines} {JOBS}/{n}.log 2>/dev/null | cut -c1-400
if [ -f {JOBS}/{n}.exit ]; then echo "state: finished, exit $(cat {JOBS}/{n}.exit)"
elif systemctl is-active -q azslm-job-{n}; then echo "state: running since $(systemctl show -p ActiveEnterTimestamp --value azslm-job-{n})"
else echo "state: not running (no exit file)"; fi""")
    out(stdout)


def cmd_fetch(args):
    """Copy a (small) file off the VM in base64 chunks that fit Run Command's 4 KiB output limit."""
    q = shlex.quote(args.remote)
    stdout, _, code = run_script(args.vm, f"stat -c %s {q}")
    if code != 0:
        fail(f"cannot stat {args.remote} on {args.vm}")
    size = int(stdout.strip().splitlines()[-1])
    chunk = (RUN_OUTPUT_LIMIT - 200) // 4 * 3
    data = b""
    while len(data) < size:
        stdout, _, _ = run_script(args.vm, f"tail -c +{len(data) + 1} {q} | head -c {chunk} | base64 -w0")
        piece = base64.b64decode(stdout.strip().splitlines()[-1])
        if not piece:
            fail(f"fetch stalled at {len(data)}/{size} bytes")
        data += piece
        out(f"  {len(data)}/{size} bytes")
    pathlib.Path(args.local).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.local).write_bytes(data)
    out(f"fetched {args.remote} -> {args.local} ({size} bytes)")


def cmd_status(args):
    _, vms = arm.request("GET", f"{arm.rg_path()}/providers/Microsoft.Compute/virtualMachines",
                         API["Microsoft.Compute/virtualMachines"])
    if not vms["value"]:
        out("no VMs")
    for vm in vms["value"]:
        _, iv = arm.request("GET", vm["id"] + "/instanceView", API["Microsoft.Compute/virtualMachines"])
        power = [s["code"].split("/")[-1] for s in iv.get("statuses", []) if s["code"].startswith("PowerState/")]
        p = vm["properties"]
        state = (power or ["?"])[0]
        note = "  <- still holds vCPU quota and disk/IP costs; teardown when done" if state == "deallocated" else ""
        out(f"{vm['name']:<16} {vm['location']:<12} {p['hardwareProfile']['vmSize']:<20} {p.get('priority', 'Regular'):<8}"
            f" {state:<12} {p.get('provisioningState')}{note}")


def cmd_teardown(args):
    flt = f"tagName eq '{RUN_TAG}' and tagValue eq '{args.run}'"
    _, res = arm.request("GET", f"{arm.rg_path()}/resources?$filter={flt.replace(' ', '%20')}", "2021-04-01")
    targets = res["value"]
    if args.all:
        _, everything = arm.request("GET", f"{arm.rg_path()}/resources", "2021-04-01")
        targets = everything["value"]
    order = {t: i for i, t in enumerate(DELETE_ORDER)}
    targets.sort(key=lambda r: order.get(r["type"], len(order)))
    # Role assignments scoped to the run's resources are not listed by /resources and are not deleted
    # with the VM identity, so remove them explicitly (by scope, before the resources go away).
    _, ras = arm.request("GET", f"{arm.rg_path()}/providers/Microsoft.Authorization/roleAssignments", "2022-04-01")
    run_scopes = (f"/virtualMachines/{args.run}", f"/disks/osdisk-{args.run}",
                  f"/networkInterfaces/nic-{args.run}", f"/publicIPAddresses/pip-{args.run}")
    for ra in ras.get("value", []):
        scope = ra["properties"]["scope"].lower()
        if (args.all and "/providers/" in scope.split("/resourcegroups/", 1)[-1]) or scope.endswith(tuple(s.lower() for s in run_scopes)):
            status, _ = arm.request("DELETE", ra["id"], "2022-04-01")
            out(f"  {'deleted' if status in (200, 204) else f'FAILED (HTTP {status})'} role assignment on {scope.rsplit('/', 2)[-2]}/{scope.rsplit('/', 1)[-1]}")
    if not targets:
        out("nothing to delete")
    for r in targets:
        api = API.get(r["type"])
        if not api:
            out(f"  skip (unknown type) {short_id(r['id'])}")
            continue
        status, result = arm.lro("DELETE", r["id"], api, poll_seconds=10)
        ok = status in (200, 202, 204) and (not isinstance(result, dict) or result.get("status") in (None, "Succeeded"))
        out(f"  {'deleted' if ok else f'FAILED (HTTP {status})'} {short_id(r['id'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("deploy", help="build + validate + what-if + deploy a Bicep template")
    d.add_argument("template")
    d.add_argument("--name", required=True, help="deployment name")
    d.add_argument("-p", "--param", action="append", default=[], help="template parameter key=value")
    d.add_argument("--what-if-only", action="store_true")
    d.set_defaults(fn=cmd_deploy)
    r = sub.add_parser("run", help="run a shell script on a VM via Run Command")
    r.add_argument("vm")
    r.add_argument("script", nargs="?")
    r.add_argument("-c", "--command", help="inline script instead of a file")
    r.add_argument("-e", "--env", action="append", default=[], help="KEY=VALUE exported before the script")
    r.add_argument("--background", metavar="NAME", help="run as a background job (see `job`)")
    r.set_defaults(fn=cmd_run)
    pu = sub.add_parser("push", help="copy a small local directory (default bench/) to the VM")
    pu.add_argument("vm")
    pu.add_argument("src", nargs="?", default="bench")
    pu.add_argument("--dest", default="/opt/azslm/bench")
    pu.set_defaults(fn=cmd_push)
    j = sub.add_parser("job", help="status and log tail of a background job")
    j.add_argument("vm")
    j.add_argument("name")
    j.add_argument("-n", "--lines", type=int, default=40)
    j.set_defaults(fn=cmd_job)
    f = sub.add_parser("fetch", help="copy a small file off a VM")
    f.add_argument("vm")
    f.add_argument("remote")
    f.add_argument("local")
    f.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("status", help="list VMs in the resource group")
    s.set_defaults(fn=cmd_status)
    t = sub.add_parser("teardown", help=f"delete resources tagged {RUN_TAG}=<run>")
    t.add_argument("--run", required=True)
    t.add_argument("--all", action="store_true", help="delete every resource in the resource group")
    t.set_defaults(fn=cmd_teardown)
    args = ap.parse_args()
    if args.cmd == "run" and not (args.script or args.command):
        ap.error("run needs a script file or -c")
    args.fn(args)


if __name__ == "__main__":
    main()
