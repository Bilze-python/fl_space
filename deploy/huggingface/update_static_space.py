"""Update the free HF static Space (bilzepython/spacefl-agent-demo).

Uploads index.html and live.json (current public tunnel URL) through the HF commit API.
Authentication: cookie stored in deploy/huggingface/cookie.txt (one line) or
the HF_COOKIE environment variable. This is only a fallback for the free static
entry while the live app is hosted elsewhere.
"""

import base64
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "deploy" / "huggingface" / "static"
URL_FILE = ROOT / "deploy" / "public_url.txt"
RENDER_URL_FILE = ROOT / "deploy" / "render_url.txt"
COOKIE_FILE = ROOT / "deploy" / "huggingface" / "cookie.txt"
REPO = "bilzepython/spacefl-agent-demo"
PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""


def read_cookie() -> str:
    cookie = os.environ.get("HF_COOKIE", "").strip()
    if not cookie and COOKIE_FILE.exists():
        cookie = COOKIE_FILE.read_text(encoding="utf-8").strip()
    return cookie


def read_live_url() -> str:
    # Prefer the stable Render URL when present; fall back to the live tunnel.
    for candidate in (RENDER_URL_FILE, URL_FILE):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip().strip("\ufeff")
            if line.startswith("https://"):
                return line.rstrip("/")
    raise SystemExit("no URL in render_url.txt / public_url.txt")


def main() -> int:
    cookie = read_cookie()
    if not cookie:
        print("ERROR: missing HF cookie (deploy/huggingface/cookie.txt or HF_COOKIE)", file=sys.stderr)
        return 2
    url = read_live_url()
    files = {
        "index.html": (STATIC / "index.html").read_bytes(),
        "live.json": json.dumps(
            {"url": url, "updated": __import__("datetime").datetime.now().isoformat(timespec="seconds")},
            ensure_ascii=False,
        ).encode("utf-8"),
    }
    lines = [json.dumps({"key": "header", "value": {"summary": f"Update live demo URL: {url}", "description": ""}})]
    for path, content in files.items():
        lines.append(
            json.dumps(
                {
                    "key": "file",
                    "value": {
                        "content": base64.b64encode(content).decode("ascii"),
                        "path": path,
                        "encoding": "base64",
                    },
                }
            )
        )
    body = "\n".join(lines).encode("utf-8")
    request = urllib.request.Request(
        f"https://huggingface.co/api/spaces/{REPO}/commit/main",
        data=body,
        headers={
            "Content-Type": "application/x-ndjson",
            "Cookie": f"token={cookie}",
            "Origin": "https://huggingface.co",
            "Referer": "https://huggingface.co/",
            "User-Agent": "Mozilla/5.0",
        },
        method="POST",
    )
    handlers = [urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})] if PROXY else []
    opener = urllib.request.build_opener(*handlers)
    try:
        with opener.open(request, timeout=120) as response:
            print("status:", response.status)
            print(response.read().decode("utf-8", errors="replace")[:500])
            print("static entry: https://bilzepython-spacefl-agent-demo.static.hf.space")
            return 0
    except urllib.error.HTTPError as exc:
        print("status:", exc.code, exc.read().decode("utf-8", errors="replace")[:500], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
