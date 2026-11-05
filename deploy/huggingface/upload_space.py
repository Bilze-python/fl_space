"""Publish the public demo to a Hugging Face Docker Space.

Usage (PowerShell):
    $env:HF_TOKEN = "hf_..."          # required, write permission
    $env:HTTP_PROXY = "http://127.0.0.1:10808"   # optional, local proxy
    venv\\Scripts\\python.exe deploy\\huggingface\\upload_space.py --repo-id <user>/<space>

The script stages only what the Dockerfile needs (no venv/node_modules/local literature),
writes a Space README with sdk: docker and app_port: 8720, then uploads and waits for the build.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "deploy" / "huggingface" / "_staging"
SPACE_PORT = 8720

COPY_DIRS = ["fl_space", "web", "deploy/demo_data"]
COPY_FILES = [
    "Dockerfile",
    "requirements-deploy.txt",
    "package.json",
    "package-lock.json",
]
OPTIONAL_FILES = [
    ".spacefl_intel_cache.json",
]

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", "node_modules"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def copy_tree(src: Path, dst: Path) -> None:
    """Copy a tree, skipping caches and local state."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in SKIP_DIRS:
            continue
        if item.is_dir():
            copy_tree(item, dst / item.name)
            continue
        if item.suffix in SKIP_SUFFIXES:
            continue
        shutil.copy2(item, dst / item.name)


def build_readme(repo_id: str) -> str:
    return f"""---
title: SpaceFL Research Agent
emoji: 🛰️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: {SPACE_PORT}
pinned: false
license: mit
short_description: 太空联邦学习科研智能体：文献检索、前沿情报、实验规划与结果解读
---

# SpaceFL 科研智能平台 · 在线演示

面向 LEO 卫星星座联邦学习的科研智能体演示实例（只读演示模式）：

- 期刊文献知识库：全文检索、AI 精读、知识库问答（带页码引用）；
- 前沿情报雷达：arXiv / 航天资讯、按研究目标相关度重排、情报简报；
- 研究智能体：自然语言 → 结构化实验计划（假设/变量/对照/指标/风险）；
- 实验档案解读：指标对照、证据链、文献对照、尚不能支持的结论。

公网实例仅读取 `deploy/demo_data/` 的演示数据，禁用训练/导入/配置写入，并使用离线规则引擎。
完整功能请在本地运行 `start_spacefl.bat`（8700 端口）。

- Docker image runs at port {SPACE_PORT}
- Source: https://github.com/Bilze-python/fl_space
"""


def stage() -> Path:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    for name in COPY_DIRS:
        source = ROOT / name
        if not source.exists():
            raise SystemExit(f"missing required path: {source}")
        copy_tree(source, STAGING / name)
    for name in COPY_FILES:
        source = ROOT / name
        if not source.exists():
            raise SystemExit(f"missing required file: {source}")
        shutil.copy2(source, STAGING / name)
    for name in OPTIONAL_FILES:
        source = ROOT / name
        if source.exists():
            shutil.copy2(source, STAGING / name)
    (STAGING / "README.md").write_text(build_readme("space"), encoding="utf-8")
    return STAGING


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload SpaceFL demo to Hugging Face Space")
    parser.add_argument("--repo-id", default="", help="<user>/<space-name>; default <whoami>/spacefl-research-agent")
    parser.add_argument("--token", default="", help="HF write token (or set HF_TOKEN)")
    parser.add_argument("--proxy", default="", help="HTTP proxy, e.g. http://127.0.0.1:10808")
    parser.add_argument("--private", action="store_true", help="create a private Space")
    parser.add_argument("--no-wait", action="store_true", help="do not wait for build")
    parser.add_argument("--stage-only", action="store_true", help="only build the upload staging folder")
    args = parser.parse_args()

    if args.stage_only:
        staging = stage()
        total_bytes = sum(item.stat().st_size for item in staging.rglob("*") if item.is_file())
        print(f"Staged {total_bytes / 1024 / 1024:.1f} MB in {staging}")
        return 0

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: missing HF token (set HF_TOKEN or pass --token)", file=sys.stderr)
        return 2
    proxy = args.proxy or os.environ.get("HTTPS_PROXY") or ""
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        who = api.whoami()
    except Exception as exc:
        print(f"ERROR: token invalid or network blocked: {exc}", file=sys.stderr)
        return 3
    username = str(who.get("name") or "").strip()
    if not username:
        print("ERROR: cannot resolve HF username from token", file=sys.stderr)
        return 3
    repo_id = args.repo_id or f"{username}/spacefl-research-agent"
    if "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"
    print(f"HF user: {username}")
    print(f"Space repo: {repo_id}")

    staging = stage()
    total_bytes = sum(item.stat().st_size for item in staging.rglob("*") if item.is_file())
    print(f"Staged {total_bytes / 1024 / 1024:.1f} MB in {staging}")

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=args.private)
    print("Uploading files...")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy SpaceFL public research-agent demo",
    )
    url = f"https://{username.lower()}-{repo_id.split('/')[-1].lower()}.hf.space"
    print(f"Space URL: {url}")
    print(f"Space page: https://huggingface.co/spaces/{repo_id}")

    if args.no_wait:
        return 0

    print("Waiting for Space build to finish (up to 25 minutes)...")
    deadline = time.time() + 25 * 60
    last_stage = ""
    while time.time() < deadline:
        try:
            runtime = api.get_space_runtime(repo_id)
            stage_value = str(getattr(runtime, "stage", "") or "")
            if stage_value != last_stage:
                print(f"  stage: {stage_value}")
                last_stage = stage_value
            if stage_value == "RUNNING":
                print("Space is running.")
                break
            if stage_value in {"BUILD_ERROR", "CONFIG_ERROR", "RUNTIME_ERROR"}:
                print(f"ERROR: Space stage={stage_value}")
                return 4
        except Exception as exc:
            print(f"  (waiting, api note: {exc})")
        time.sleep(15)

    import urllib.request

    for attempt in range(1, 13):
        try:
            opener = urllib.request.build_opener()
            request = urllib.request.Request(f"{url}/api/health")
            with opener.open(request, timeout=20) as response:
                print(f"health check: {response.status}")
                print(f"PUBLIC_URL={url}")
                return 0
        except Exception as exc:
            print(f"  health check {attempt}/12 failed: {exc}")
            time.sleep(10)
    print(f"Uploaded, but health check not ready yet: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
