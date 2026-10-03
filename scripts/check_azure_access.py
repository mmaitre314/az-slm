#!/usr/bin/env python3
"""Validate Azure access from the agent sandbox: env vars set, resource group readable,
caller has Owner-equivalent permissions on it, and Azure docs reachable.

Prints no subscription/tenant IDs. Exits non-zero on any failure.
"""

import os
import sys
import urllib.request

import arm

DOCS_URL = "https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview"


def check_resource_group():
    status, body = arm.request("GET", arm.rg_path(), "2021-04-01")
    if status != 200:
        return False, f"resource group GET returned HTTP {status}"
    return True, f"resource group readable ({body['location']})"


def check_owner():
    # Effective permissions of the caller on the RG. Owner == actions ["*"] with no notActions.
    path = arm.rg_path() + "/providers/Microsoft.Authorization/permissions"
    status, body = arm.request("GET", path, "2022-04-01")
    if status != 200:
        return False, f"permissions GET returned HTTP {status}"
    actions = {a for p in body["value"] for a in p["actions"]}
    not_actions = {a for p in body["value"] for a in p["notActions"]}
    if "*" in actions and not not_actions:
        return True, "caller has Owner-equivalent permissions on the resource group (actions: *, notActions: none)"
    return False, f"caller lacks Owner-equivalent permissions (actions: {sorted(actions)}, notActions: {sorted(not_actions)})"


def check_docs():
    with urllib.request.urlopen(DOCS_URL, timeout=30) as resp:
        if resp.status != 200:
            return False, f"learn.microsoft.com returned HTTP {resp.status}"
    return True, "learn.microsoft.com reachable"


def main():
    failed = False
    for name in arm.ENV_VARS:
        is_set = bool(os.environ.get(name))
        print(f"{'PASS' if is_set else 'FAIL'}  {name} is {'set' if is_set else 'not set'}")
        failed |= not is_set
    if failed:
        return 1

    for check in (check_resource_group, check_owner, check_docs):
        try:
            ok, msg = check()
        except Exception as e:  # e.g. proxy denial (URLError) or timeout
            ok, msg = False, f"{check.__name__}: {e}"
        print(f"{'PASS' if ok else 'FAIL'}  {arm.redact(msg)}")
        failed |= not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
