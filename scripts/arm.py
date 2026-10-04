"""Minimal Azure Resource Manager (ARM) client using only the Python standard library.

The sandbox egress proxy attaches a Bearer token to requests for management.azure.com, so requests
carry no Authorization header. Azure context comes from environment variables (see ENV_VARS).
"""

import json
import os
import time
import urllib.error
import urllib.request

ARM = "https://management.azure.com"
ENV_VARS = ("AZURE_TENANT_ID", "AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP")


def env(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


def rg_path():
    """ARM path of the resource group, e.g. for rg_path() + '/providers/Microsoft.Compute/...'."""
    return f"/subscriptions/{env('AZURE_SUBSCRIPTION_ID')}/resourceGroups/{env('AZURE_RESOURCE_GROUP')}"


def send(method, path, api_version=None, body=None, timeout=60):
    """Send an ARM request and return (status, parsed JSON body or None, response headers).

    `path` is an ARM path ('/subscriptions/...') or a full URL (e.g. an Azure-AsyncOperation URL,
    which already carries its api-version). HTTP error statuses are returned, not raised.
    """
    url = path if path.startswith("https://") else ARM + path
    if api_version:
        url += ("&" if "?" in url else "?") + f"api-version={api_version}"
    data = None if body is None else json.dumps(body).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw, hdrs = resp.status, resp.read(), resp.headers
    except urllib.error.HTTPError as e:
        status, raw, hdrs = e.code, e.read(), e.headers
    return status, json.loads(raw) if raw else None, hdrs


def request(method, path, api_version=None, body=None, timeout=60):
    """Like send(), without the headers: returns (status, parsed JSON body or None)."""
    status, body, _ = send(method, path, api_version, body, timeout)
    return status, body


def lro(method, path, api_version, body=None, poll_seconds=5, max_wait=3600):
    """Send a request and wait for a long-running operation to finish.

    Returns (status, body) of the final poll: for Azure-AsyncOperation that is the operation status
    document ({"status": "Succeeded"|"Failed"|..., "error": ..., "properties": ...}); for Location
    polling it is the final resource or result. Requests that complete synchronously return as is.
    """
    status, result, hdrs = send(method, path, api_version, body)
    poll = hdrs.get("Azure-AsyncOperation") or hdrs.get("Location")
    if status not in (201, 202) or not poll:
        return status, result
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        time.sleep(int(hdrs.get("Retry-After") or poll_seconds))
        status, result, hdrs = send("GET", poll)
        if status == 202:
            continue
        if status == 200 and isinstance(result, dict) and result.get("status") in ("InProgress", "Accepted", "Running"):
            continue
        return status, result
    raise TimeoutError(f"{method} operation did not finish within {max_wait}s")


def resource_graph(query, subscriptions=None):
    """Run an Azure Resource Graph query and return all rows, following $skipToken paging.

    Leave `subscriptions` unset for tenant-wide tables such as SpotResources, which return nothing
    when scoped to a subscription where the caller has only resource-group access.
    """
    rows, options = [], {"$top": 1000}
    while True:
        body = {"query": query, "options": options}
        if subscriptions:
            body["subscriptions"] = subscriptions
        status, page = request("POST", "/providers/Microsoft.ResourceGraph/resources", "2022-10-01", body=body)
        if status != 200:
            raise RuntimeError(f"Resource Graph query failed: HTTP {status}: {redact(json.dumps(page))[:500]}")
        rows += page["data"]
        if not page.get("$skipToken"):
            return rows
        options = {"$top": 1000, "$skipToken": page["$skipToken"]}


def redact(text):
    """Mask subscription and tenant IDs before printing or saving text."""
    for name in ("AZURE_SUBSCRIPTION_ID", "AZURE_TENANT_ID"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, f"<{name}>")
    return text
