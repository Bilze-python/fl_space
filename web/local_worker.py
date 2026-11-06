"""Opt-in outbound worker. Run with: python -m web.local_worker --help."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from web.dataset_store import import_imagefolder, list_datasets
from web.local_compute import PREFIX, public_datasets, validate_local_session

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROGRESS = re.compile(r"(?:Round|FedBuff update)\s+(\d+)\s*/\s*(\d+).*accuracy=([\d.]+)")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a pairing credential to a redirected destination.
        return None


class RelayClient:
    def __init__(self, server: str):
        parsed = urlsplit(server)
        if (parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"} or not parsed.hostname
                or (parsed.scheme != "https" and not (
                    parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}))):
            raise ValueError("公网连接需要 HTTPS；仅本机调试可使用 http://127.0.0.1")
        self.server = server.rstrip("/")
        self.token = ""
        self.worker_id = ""
        self.opener = build_opener(NoRedirect())

    def post(self, suffix: str, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(self.server + PREFIX + suffix,
                          data=json.dumps(payload, allow_nan=False).encode(), headers=headers)
        with self.opener.open(request, timeout=15) as response:
            raw = response.read(256 * 1024 + 1)
            if len(raw) > 256 * 1024:
                raise ValueError("平台响应过大")
            return json.loads(raw)


def dataset_metadata(root: Path) -> list[dict]:
    return public_datasets(list_datasets(root / "datasets"))


def choose_device(requested: str) -> str:
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("未检测到可用 CUDA，请安装对应 PyTorch 或使用 --device cpu")
    return "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"


def execute_job(client, job: dict, root: Path, device: str, threads: int,
                max_seconds: float) -> dict:
    """Only a fixed repository CLI runs; config, paths and resources are local decisions."""
    job_id = job.get("id", "")
    if not re.fullmatch(r"[a-f0-9]{24}", job_id):
        raise ValueError("无效任务 ID")
    output = root / "experiments" / job_id
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "FAILED", "progress_round": 0, "accuracy": None}
    process = None
    timer = None
    started = time.monotonic()
    lines = deque(maxlen=120)
    progress_lock = threading.Lock()
    last_report = 0.0
    disconnected = False
    try:
        session = validate_local_session(job.get("session"), job.get("dataset_id"),
                                         dataset_metadata(root), root, device)
        (output / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
        env = dict(os.environ)
        env.pop("SPACEFL_TRAINING_ACCESS_KEY", None)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[key] = str(threads)
        command = [sys.executable, "-u", "-m", "fl_space.cli", "run", "train",
                   "--session-file", str(output / "session.json"),
                   "--output", str(output / "training_summary.json")]
        process = subprocess.Popen(command, cwd=PROJECT_DIR, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        # Enforce the local deadline even while an HTTPS request is timing out.
        timer = threading.Timer(max_seconds, lambda: process.kill() if process.poll() is None else None)
        timer.daemon = True
        timer.start()

        def read_output():
            with process.stdout:
                for line in process.stdout:
                    with progress_lock:
                        lines.append(line[-4000:])
                        match = PROGRESS.search(line)
                        if match:
                            state.update(progress_round=int(match[1]), accuracy=float(match[3]))
                        (output / "training.log").write_text("".join(lines)[-16000:], encoding="utf-8")

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        state["status"] = "RUNNING"
        while process.poll() is None:
            if time.monotonic() - started > max_seconds:
                process.kill()
                state["status"] = "FAILED"
                print("训练已达到本地设置的时间上限。", flush=True)
                break
            if time.monotonic() - last_report >= 3:
                last_report = time.monotonic()
                try:
                    with progress_lock:
                        report = dict(state)
                    reply = client.post(f"/{client.worker_id}/jobs/{job_id}", report)
                    if reply.get("cancel"):
                        process.kill()
                        state["status"] = "CANCELLED"
                        break
                except HTTPError as exc:
                    if exc.code in {401, 404, 409}:
                        disconnected = True
                        # Pairing loss does not destroy local training work.
                        last_report = float("inf")
                        print("平台连接失效，本次训练继续在本地完成；完成后请重新配对。", flush=True)
                except (URLError, TimeoutError, OSError):
                    pass  # Temporary network loss; local execution and timeout remain active.
            time.sleep(0.2)
        process.wait(timeout=10)
        reader.join(timeout=10)
        if process.returncode == 0 and state["status"] != "CANCELLED":
            result = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
            history = [{"round": row["round"] + 1, "accuracy": row["accuracy"]}
                       for row in result.get("history", [])
                       if isinstance(row.get("round"), int) and row["round"] >= 0
                       and "accuracy" in row]
            if not history:
                raise ValueError("训练结果缺少逐轮准确率")
            state.update(status="DONE", history=history, progress_round=len(history),
                         accuracy=history[-1]["accuracy"])
        elif state["status"] != "CANCELLED":
            state["status"] = "FAILED"
    except KeyboardInterrupt:
        state["status"] = "CANCELLED"
        raise
    except Exception as exc:
        state["status"] = "FAILED"
        (output / "worker-error.txt").write_text(str(exc), encoding="utf-8")
        print(f"任务失败，请查看 {output}", flush=True)
    finally:
        if timer:
            timer.cancel()
        if process and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        state["elapsed_sec"] = round(time.monotonic() - started, 2)
        (output / "job.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        if not disconnected:
            for attempt in range(3):
                try:
                    client.post(f"/{client.worker_id}/jobs/{job_id}", state)
                    break
                except (HTTPError, URLError, TimeoutError, OSError):
                    if attempt == 2:
                        disconnected = True
                        print("结果回传失败，完整结果保留在本机；请重新配对。", flush=True)
                    else:
                        time.sleep(2)
    print(f"任务 {state['status']}，结果目录：{output}", flush=True)
    if disconnected:
        raise ConnectionError("请重新启动本地程序并配对")
    return state


def connect(args):
    root = args.workspace.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    client = RelayClient(args.server)
    registration = client.post("/register", {"name": args.name or socket.gethostname(),
                                            "device": device, "datasets": dataset_metadata(root)})
    client.token, client.worker_id = registration["token"], registration["id"]
    print(f"\n本机设备：{device}；CPU 线程上限：{args.threads}；任务限时：{args.max_seconds} 秒", flush=True)
    print(f"配对码：{registration['pair_code']}（10 分钟内在网页填写，仅交给本人）", flush=True)
    print("等待网页提交训练任务。Ctrl+C 停止本地程序。", flush=True)
    try:
        while True:
            try:
                reply = client.post(f"/{client.worker_id}/poll", {"datasets": dataset_metadata(root)})
                if reply.get("job"):
                    execute_job(client, reply["job"], root, device, args.threads, args.max_seconds)
                time.sleep(3)
            except HTTPError as exc:
                if exc.code in {401, 404}:
                    raise ConnectionError("平台已重启、连接已到期或已解除，请重新启动并配对") from exc
                raise
            except (URLError, TimeoutError):
                print("网络暂不可用，稍后重试。", flush=True)
                time.sleep(5)
    except KeyboardInterrupt:
        print("\n本地训练程序已停止。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="SpaceFL 本地算力连接器：数据和训练留在本人电脑")
    sub = parser.add_subparsers(dest="action", required=True)
    importer = sub.add_parser("import", help="在本机校验并导入 ImageFolder ZIP")
    importer.add_argument("archive", type=Path)
    importer.add_argument("--name", required=True)
    importer.add_argument("--workspace", type=Path, default=PROJECT_DIR / "local-compute")
    worker = sub.add_parser("connect", help="主动连接平台并等待已配对网页的任务")
    worker.add_argument("--server", default="https://spacefl-research-agent.onrender.com")
    worker.add_argument("--workspace", type=Path, default=PROJECT_DIR / "local-compute")
    worker.add_argument("--name")
    worker.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    worker.add_argument("--threads", type=int, default=2)
    worker.add_argument("--max-seconds", type=int, default=3600)
    args = parser.parse_args()
    try:
        if args.action == "import":
            with args.archive.open("rb") as archive:
                result = import_imagefolder(args.workspace.expanduser().resolve() / "datasets", args.name, archive)
            print(f"已导入：{result['name']}，ID={result['id']}，目录={result['data_dir']}")
        else:
            if not 1 <= args.threads <= 64 or not 10 <= args.max_seconds <= 86400:
                parser.error("threads 必须为 1..64，max-seconds 必须为 10..86400")
            connect(args)
    except Exception as exc:
        print(f"连接器停止：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
