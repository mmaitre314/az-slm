#!/usr/bin/env bash
# Validate Azure access from the agent sandbox: env vars set, resource group readable,
# caller has Owner-equivalent permissions on it, and Azure docs reachable.
# Output never prints subscription/tenant IDs. Exits non-zero on any failure.
set -uo pipefail

ARM=https://management.azure.com
fail=0

ok()  { echo "PASS  $*"; }
bad() { echo "FAIL  $*"; fail=1; }

for v in AZURE_TENANT_ID AZURE_SUBSCRIPTION_ID AZURE_RESOURCE_GROUP; do
  if [[ -n "${!v:-}" ]]; then ok "$v is set"; else bad "$v is not set"; fi
done
[[ $fail -eq 0 ]] || exit 1

RG_URL="$ARM/subscriptions/$AZURE_SUBSCRIPTION_ID/resourceGroups/$AZURE_RESOURCE_GROUP"

# GET a URL; print the HTTP status code and leave the body in $body.
body=$(mktemp); trap 'rm -f "$body"' EXIT
get() { curl -sS -m 30 -o "$body" -w '%{http_code}' "$1" 2>/dev/null || echo 000; }

code=$(get "$RG_URL?api-version=2021-04-01")
if [[ $code == 200 ]]; then
  ok "resource group readable ($(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["location"])' "$body"))"
else
  bad "resource group GET returned HTTP $code"
fi

# Effective permissions of the caller on the RG. Owner == actions ["*"] with no notActions.
code=$(get "$RG_URL/providers/Microsoft.Authorization/permissions?api-version=2022-04-01")
if [[ $code == 200 ]] && python3 - "$body" <<'EOF'
import json, sys
perms = json.load(open(sys.argv[1]))["value"]
actions = {a for p in perms for a in p["actions"]}
not_actions = {a for p in perms for a in p["notActions"]}
sys.exit(0 if "*" in actions and not not_actions else 1)
EOF
then
  ok "caller has Owner-equivalent permissions on the resource group (actions: *, notActions: none)"
else
  bad "caller lacks Owner-equivalent permissions on the resource group (HTTP $code)"
fi

code=$(get "https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview")
if [[ $code == 200 ]]; then ok "learn.microsoft.com reachable"; else bad "learn.microsoft.com returned HTTP $code"; fi

exit $fail
