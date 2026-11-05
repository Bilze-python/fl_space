"""Create and verify the SpaceFL public demo on Render (free Web Service).

Usage:
    $env:RENDER_API_KEY = "rnd_..."
    venv\\Scripts\\python.exe deploy\\render\\create_service.py --proxy http://127.0.0.1:10808

The script creates a Docker web service from the public GitHub repository
(https://github.com/Bilze-python/fl_space, branch deploy-demo), waits for the
deploy to go live, and checks /api/health.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests

API = "https://api.render.com/v1"
DEFAULT_REPO = "https://github.com/Bilze-python/fl_space"
DEFAULT_BRANCH = "deploy-demo"
DEFAULT_NAME = "spacefl-research-agent"


def api_request(method: str, path: str, token: str, session: requests.Session, **kwargs):
    response = session.request(method, f"{API}{path}", headers={"Authorization": f"Bearer {token}"}, timeout=60, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {response.status_code}: {response.text[:500]}")
    if not response.text:
        return None
    return response.json()


def pick_owner(token: str, session: requests.Session, wanted: str) -> dict:
    owners = api_request("GET", "/owners?limit=20", token, session)
    if not owners:
        raise RuntimeError("no Render workspace found for this API key")
    candidates = []
    for entry in owners:
        owner = entry.get("owner", entry)
        candidates.append(owner)
    if wanted:
        for owner in candidates:
            if wanted in {owner.get("id"), owner.get("name")}:
                return owner
        raise RuntimeError(f"workspace not found: {wanted}; available: {[o.get('name') for o in candidates]}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create SpaceFL demo web service on Render")
    parser.add_argument("--token", default="", help="Render API key (or set RENDER_API_KEY)")
    parser.add_argument("--owner", default="", help="workspace id or name (default: first workspace)")
    parser.add_argument("--name", default=DEFAULT_NAME, help="service name (must be unique)")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="public GitHub repository URL")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--region", default="oregon", choices=["oregon", "frankfurt", "ohio", "singapore", "virginia"])
    parser.add_argument("--proxy", default="", help="HTTP proxy, e.g. http://127.0.0.1:10808")
    parser.add_argument("--no-wait", action="store_true", help="create only, do not wait for the deploy")
    args = parser.parse_args()

    token = args.token or os.environ.get("RENDER_API_KEY") or os.environ.get("RENDER_TOKEN") or ""
    if not token:
        print("ERROR: missing Render API key. Create one in Render Dashboard -> Account Settings -> API Keys,", file=sys.stderr)
        print("       then set RENDER_API_KEY (or pass --token).", file=sys.stderr)
        return 2

    proxy = args.proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""
    session = requests.Session()
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})

    owner = pick_owner(token, session, args.owner)
    print(f"Render workspace: {owner.get('name')} ({owner.get('id')})")

    existing = api_request("GET", f"/services?name={args.name}&limit=5", token, session) or []
    if existing:
        service = existing[0].get("service", existing[0])
        print(f"Service already exists: {service.get('id')}")
        service_id = service.get("id")
    else:
        payload = {
            "type": "web_service",
            "name": args.name,
            "ownerId": owner.get("id"),
            "repo": args.repo,
            "branch": args.branch,
            "autoDeployTrigger": "commit",
            "serviceDetails": {
                "runtime": "docker",
                "plan": "free",
                "region": args.region,
                "healthCheckPath": "/api/health",
                "envSpecificDetails": {
                    "dockerfilePath": "./Dockerfile",
                    "dockerContext": ".",
                },
            },
            "envVars": [
                {"key": "SPACEFL_PUBLIC_DEMO", "value": "1"},
                {"key": "SPACEFL_OFFLINE_ONLY", "value": "1"},
            ],
        }
        created = api_request("POST", "/services", token, session, json=payload)
        service = created.get("service", created)
        service_id = service.get("id")
        print(f"Created service: {service_id}")

    url = f"https://{args.name}.onrender.com"
    print(f"Service URL: {url}")
    if args.no_wait:
        return 0

    print("Waiting for deploy (up to 20 minutes)...")
    deadline = time.time() + 20 * 60
    last = ""
    while time.time() < deadline:
        try:
            deploys = api_request("GET", f"/services/{service_id}/deploys?limit=1", token, session) or []
            status = ""
            if deploys:
                deploy = deploys[0].get("deploy", deploys[0])
                status = str(deploy.get("status") or "")
            if status != last:
                print(f"  deploy status: {status}")
                last = status
            if status in {"live", "build_failed", "canceled", "deactivated"}:
                if status != "live":
                    print(f"ERROR: deploy ended with status={status}")
                    return 4
                break
        except Exception as exc:  # noqa: BLE001
            print(f"  (waiting, api note: {exc})")
        time.sleep(15)

    for attempt in range(1, 13):
        try:
            health = session.get(f"{url}/api/health", timeout=30)
            print(f"health check {attempt}: {health.status_code} {health.text[:120]}")
            if health.status_code == 200:
                print(f"PUBLIC_URL={url}")
                return 0
        except Exception as exc:  # noqa: BLE001
            print(f"  health check {attempt} failed: {exc}")
        time.sleep(10)
    print(f"Service created but health is not ready yet: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
