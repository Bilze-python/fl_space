"""Small, memory-only relay for opt-in local training workers; imports no torch."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import threading
import time

from fastapi import APIRouter, HTTPException, Request

from web.training_runtime import TrainingRuntime, require_same_origin, validate_cloud_session

PREFIX = "/api/local-compute"
MAX_BODY = 256 * 1024
LEASE_SECONDS = 12 * 3600
ONLINE_SECONDS = 45
LOCAL_LIMITS = TrainingRuntime(
    max_rounds=1000, max_epochs=100, max_satellites=100,
    max_samples=20000, max_batch_size=512, max_seconds=3600,
)
DATA_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def validate_local_session(payload: dict, dataset_id: str | None, datasets: list[dict],
                           root: Path, device: str = "cpu") -> dict:
    """Revalidate on BOTH ends; remote input can never select files or commands."""
    from fl_space.cli import DEFAULT_SESSION

    def reject(message):
        raise HTTPException(422, message)

    if not isinstance(payload, dict) or set(payload) - {"tune", "mount"}:
        reject("Session 格式无效")
    session = deepcopy(DEFAULT_SESSION)
    for section in session:
        values = payload.get(section, {})
        if not isinstance(values, dict) or set(values) - set(session[section]):
            reject(f"不支持的 {section} 参数")
        session[section].update(values)
        for key, default in DEFAULT_SESSION[section].items():
            value = session[section][key]
            if default is None and key == "max_staleness" and value is not None and type(value) is not int:
                reject(f"{key} 必须为整数或 null")
            elif isinstance(default, bool):
                if type(value) is not bool:
                    reject(f"{key} 必须为布尔值")
            elif isinstance(default, str):
                if not isinstance(value, str) or len(value) > 1024:
                    reject(f"{key} 必须为字符串")
            elif isinstance(default, (int, float)) and type(value) not in (int, float):
                reject(f"{key} 必须为数字")
    tune = session["tune"]
    # Ignore all caller-supplied filesystem paths. Only locally imported IDs resolve.
    managed = []
    if tune["dataset"] in {"imagefolder", "custom"}:
        selected = next((item for item in datasets if item["id"] == dataset_id), None)
        if not selected or not DATA_ID.fullmatch(str(dataset_id)):
            reject("请先在本机导入并选择数据集")
        tune["data_dir"] = str(root / "datasets" / dataset_id)
        managed = [{"data_dir": tune["data_dir"]}]
    elif dataset_id is not None:
        reject("内置数据集不需要 dataset_id")
    profiles = tune.get("satellite_data_profiles")
    if not isinstance(profiles, dict) or len(profiles) > 100:
        reject("单星数据画像无效")
    for key, profile in profiles.items():
        if (not re.fullmatch(r"[0-9]{1,2}", str(key))
                or not isinstance(profile, dict)
                or set(profile) != {"preferred_classes", "preference_probability", "max_samples"}):
            reject("单星数据画像无效")
        classes = profile["preferred_classes"]
        if (not isinstance(classes, list) or len(classes) > 100
                or any(type(value) is not int or not 0 <= value < 10000 for value in classes)):
            reject("偏好类别无效")
        probability = profile["preference_probability"]
        if (type(probability) not in (int, float) or not math.isfinite(probability)
                or not 0 <= probability <= 1
                or type(profile["max_samples"]) is not int
                or not 0 <= profile["max_samples"] <= LOCAL_LIMITS.max_samples):
            reject("画像概率或样本上限无效")
    for key, choices in {
        "protocol_mode": {"standard", "paper_approx"},
        "selection_strategy": {"random", "earliest_return"},
    }.items():
        if tune[key] not in choices:
            reject(f"不支持的 {key}")
    if device not in {"cpu", "cuda"}:
        reject("本机设备必须为 cpu 或 cuda")
    # The shared validator enforces all scalar bounds and fixed simulation backend.
    # Device is selected by the local CLI only, never by a web request.
    if tune["device"] != "cpu":
        reject("设备由本地程序选择，请保留网页 device=cpu")
    runtime = TrainingRuntime(**{**LOCAL_LIMITS.__dict__, "root": root})
    try:
        result = validate_cloud_session(session, runtime, managed)
    except HTTPException as exc:
        raise HTTPException(exc.status_code, str(exc.detail).replace("公网训练", "本机连接器")) from exc
    result["tune"]["device"] = device
    return result


def public_datasets(items: list) -> list[dict]:
    if not isinstance(items, list) or len(items) > 20:
        raise HTTPException(422, "最多发布 20 个本机数据集")
    result = []
    for item in items:
        if (not isinstance(item, dict) or not DATA_ID.fullmatch(str(item.get("id", "")))
                or not isinstance(item.get("name"), str) or not 1 <= len(item["name"]) <= 100
                or not isinstance(item.get("classes"), list)
                or not 2 <= len(item["classes"]) <= 1000
                or any(not isinstance(c, str) or len(c) > 100 for c in item["classes"])
                or any(type(item.get(k)) is not int or not 1 <= item[k] <= 50000
                       for k in ("train_images", "test_images"))):
            raise HTTPException(422, "数据集元数据无效")
        result.append({key: item[key] for key in
                       ("id", "name", "classes", "train_images", "test_images")})
    if len({item["id"] for item in result}) != len(result):
        raise HTTPException(422, "重复数据集 ID")
    return result


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def _body(request: Request) -> dict:
    require_same_origin(request)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_BODY:
            raise HTTPException(413, "请求超过 256 KB")
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise HTTPException(422, "JSON 格式无效") from exc
    if not isinstance(payload, dict):
        raise HTTPException(422, "需要 JSON 对象")
    return payload


class LocalComputeRelay:
    """One independent job queue per paired computer, bounded and volatile."""

    def __init__(self, clock=time.time):
        self.clock = clock
        self.workers: dict[str, dict] = {}
        self.hits: dict[str, list[float]] = {}
        self.lock = threading.RLock()

    def clean(self):
        now = self.clock()
        self.workers = {k: v for k, v in self.workers.items() if v["expires"] > now}
        self.hits = {k: [t for t in v if t > now - 600] for k, v in self.hits.items()
                     if any(t > now - 600 for t in v)}

    def rate_limit(self, request):
        self.clean()
        host = request.client.host if request.client else "unknown"
        bucket = self.hits.setdefault(host, [])
        if len(bucket) >= 20 or sum(map(len, self.hits.values())) >= 200:
            raise HTTPException(429, "配对请求过多，请 10 分钟后重试")
        bucket.append(self.clock())

    def authorize(self, request: Request, worker_id: str, role: str):
        self.clean()
        worker = self.workers.get(worker_id)
        header = request.headers.get("authorization", "")
        token = header.removeprefix("Bearer ") if header.startswith("Bearer ") else ""
        if not worker or not token or not secrets.compare_digest(_hash(token), worker[role]):
            raise HTTPException(401, "连接已失效，请重新启动本地程序并配对")
        return worker

    def view(self, worker):
        return {"id": worker["id"], "name": worker["name"], "device": worker["device"],
                "online": self.clock() - worker["seen"] < ONLINE_SECONDS,
                "capabilities": worker["capabilities"],
                "datasets": worker["datasets"], "jobs": list(worker["jobs"].values()),
                "limits": LOCAL_LIMITS.limits()}


def create_local_compute_router(relay: LocalComputeRelay) -> APIRouter:
    router = APIRouter(prefix=PREFIX)

    @router.post("/register")
    async def register(request: Request):
        body = await _body(request)
        datasets = public_datasets(body.get("datasets", []))
        name, device = body.get("name", "本机"), body.get("device", "cpu")
        capabilities = body.get("capabilities", [])
        if (not isinstance(name, str) or not 1 <= len(name) <= 80
                or not isinstance(device, str) or device not in {"cpu", "cuda"}
                or not isinstance(capabilities, list) or any(item != "eurosat_rgb_import" for item in capabilities)):
            raise HTTPException(422, "设备名称或类型无效")
        with relay.lock:
            relay.rate_limit(request)
            if len(relay.workers) >= 32:
                raise HTTPException(429, "本地连接已满，请稍后再试")
            worker_id, token = secrets.token_hex(12), secrets.token_urlsafe(32)
            code = secrets.token_hex(6).upper()
            relay.workers[worker_id] = {
                "id": worker_id, "name": name, "device": device, "datasets": datasets,
                "capabilities": capabilities,
                "worker": _hash(token), "browser": "", "code": _hash(code),
                "code_expires": relay.clock() + 600, "expires": relay.clock() + LEASE_SECONDS,
                "seen": relay.clock(), "jobs": {},
            }
            return {"id": worker_id, "token": token, "pair_code": code, "expires_in": 600}

    @router.post("/pair")
    async def pair(request: Request):
        body = await _body(request)
        code = str(body.get("code", "")).strip().upper()
        with relay.lock:
            relay.rate_limit(request)
            worker = next((w for w in relay.workers.values()
                           if w["code_expires"] > relay.clock() and w["code"] == _hash(code)), None)
            if worker is None:
                raise HTTPException(401, "配对码无效、已使用或已过期")
            token = secrets.token_urlsafe(32)
            worker.update(browser=_hash(token), code="", code_expires=0)
            return {"id": worker["id"], "token": token}

    @router.get("/{worker_id}")
    async def status(worker_id: str, request: Request):
        with relay.lock:
            return deepcopy(relay.view(relay.authorize(request, worker_id, "browser")))

    @router.post("/{worker_id}/jobs")
    async def submit(worker_id: str, request: Request):
        body = await _body(request)
        with relay.lock:
            worker = relay.authorize(request, worker_id, "browser")
            if relay.clock() - worker["seen"] >= ONLINE_SECONDS:
                raise HTTPException(409, "本地程序已离线，请检查电脑连接")
            if any(j["status"] in {"PENDING", "RUNNING"} for j in worker["jobs"].values()):
                raise HTTPException(409, "本机已有训练任务")
            session = validate_local_session(body.get("session"), body.get("dataset_id"),
                                             worker["datasets"], Path("relay"))
            session["tune"]["data_dir"] = ""  # No filesystem paths on the wire.
            job_id = secrets.token_hex(12)
            job = {"id": job_id, "kind": "local_training", "status": "PENDING",
                   "session": session, "dataset_id": body.get("dataset_id"),
                   "rounds": session["tune"]["rounds"], "progress_round": 0,
                   "algorithm": session["mount"]["algo"], "dataset": session["tune"]["dataset"],
                   "message": "等待本机领取", "cancel": False, "created_at": relay.clock()}
            while len(worker["jobs"]) >= 10:
                del worker["jobs"][next(iter(worker["jobs"]))]
            worker["jobs"][job_id] = job
            return deepcopy(job)

    @router.post("/{worker_id}/datasets/import")
    async def import_public_dataset(worker_id: str, request: Request):
        body = await _body(request)
        if body != {"source": "eurosat_rgb"}:
            raise HTTPException(422, "目前仅支持直接添加 EuroSAT RGB")
        with relay.lock:
            worker = relay.authorize(request, worker_id, "browser")
            if relay.clock() - worker["seen"] >= ONLINE_SECONDS:
                raise HTTPException(409, "本地程序已离线，请检查电脑连接")
            if "eurosat_rgb_import" not in worker["capabilities"]:
                raise HTTPException(409, "本地连接器版本过旧，请更新 SpaceFL 并重新启动连接器")
            if any(job["status"] in {"PENDING", "RUNNING"} for job in worker["jobs"].values()):
                raise HTTPException(409, "本机已有训练或导入任务")
            existing = next((item for item in worker["datasets"]
                             if item["name"] == "EuroSAT RGB"), None)
            if existing:
                return {"id": "", "kind": "dataset_import", "status": "DONE",
                        "imported_id": existing["id"], "dataset": "EuroSAT RGB",
                        "message": "EuroSAT RGB 已在本机数据集列表中"}
            job_id = secrets.token_hex(12)
            job = {"id": job_id, "kind": "dataset_import", "source": "eurosat_rgb",
                   "dataset": "EuroSAT RGB", "status": "PENDING", "message": "等待本机下载",
                   "created_at": relay.clock()}
            while len(worker["jobs"]) >= 10:
                del worker["jobs"][next(iter(worker["jobs"]))]
            worker["jobs"][job_id] = job
            return deepcopy(job)

    @router.post("/{worker_id}/cancel")
    async def cancel(worker_id: str, request: Request):
        await _body(request)
        with relay.lock:
            worker = relay.authorize(request, worker_id, "browser")
            for job in worker["jobs"].values():
                if job["status"] == "PENDING":
                    job.update(status="CANCELLED", message="已取消")
                elif job["status"] == "RUNNING":
                    job.update(cancel=True, message="等待本机停止训练")
            return {"ok": True}

    @router.post("/{worker_id}/disconnect")
    async def disconnect(worker_id: str, request: Request):
        await _body(request)
        with relay.lock:
            relay.authorize(request, worker_id, "browser")
            del relay.workers[worker_id]
            return {"ok": True}

    @router.post("/{worker_id}/poll")
    async def poll(worker_id: str, request: Request):
        body = await _body(request)
        datasets = public_datasets(body.get("datasets", []))
        with relay.lock:
            worker = relay.authorize(request, worker_id, "worker")
            worker.update(seen=relay.clock(), datasets=datasets)
            for job in worker["jobs"].values():
                if job["status"] == "PENDING":
                    job.update(status="RUNNING", confirmed=False, message="本机正在领取任务")
                    return {"job": deepcopy(job)}
                if job["status"] == "RUNNING" and not job.get("confirmed", False):
                    # Retrying a lost poll response must deliver the same frozen task.
                    return {"job": deepcopy(job)}
            return {"job": None}

    @router.post("/{worker_id}/jobs/{job_id}")
    async def update(worker_id: str, job_id: str, request: Request):
        body = await _body(request)
        with relay.lock:
            worker = relay.authorize(request, worker_id, "worker")
            job = worker["jobs"].get(job_id)
            if job and job["kind"] == "dataset_import":
                if job["status"] in {"DONE", "FAILED"} and body.get("status") == job["status"]:
                    return {"ok": True}
                if job["status"] != "RUNNING" or body.get("status") not in {"RUNNING", "DONE", "FAILED"}:
                    raise HTTPException(409, "导入任务不存在或已结束")
                message = body.get("message", "")
                if not isinstance(message, str) or not 1 <= len(message) <= 160:
                    raise HTTPException(422, "导入状态无效")
                if body["status"] == "DONE":
                    datasets = public_datasets(body.get("datasets", []))
                    imported_id = body.get("imported_id")
                    if not any(item["id"] == imported_id and item["name"] == "EuroSAT RGB"
                               for item in datasets):
                        raise HTTPException(422, "导入结果不在本机数据集列表中")
                    worker["datasets"] = datasets
                    job["imported_id"] = imported_id
                worker["seen"] = relay.clock()
                job.update(status=body["status"], message=message, confirmed=True)
                if body["status"] != "RUNNING":
                    job["finished_at"] = relay.clock()
                return {"ok": True}
            if (job and job["status"] in {"DONE", "FAILED", "CANCELLED"}
                    and all(body.get(key) == job.get(key) for key in
                            ("status", "progress_round", "accuracy"))
                    and body.get("history", []) == job.get("history", [])):
                worker["seen"] = relay.clock()
                return {"cancel": job["cancel"]}
            if not job or job["status"] != "RUNNING":
                raise HTTPException(409, "任务不存在或已结束")
            status = body.get("status", "RUNNING")
            progress = body.get("progress_round", 0)
            if (not isinstance(status, str) or status not in {"RUNNING", "DONE", "FAILED", "CANCELLED"}
                    or type(progress) is not int or not 0 <= progress <= job["rounds"]):
                raise HTTPException(422, "任务状态无效")
            accuracy = body.get("accuracy")
            if accuracy is not None and (type(accuracy) not in (float, int) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1):
                raise HTTPException(422, "准确率无效")
            history = body.get("history", [])
            if (not isinstance(history, list) or len(history) > job["rounds"]
                    or any(not isinstance(row, dict) or set(row) - {"round", "accuracy", "loss"}
                           or any(type(v) not in (float, int) or not math.isfinite(v) for v in row.values())
                           for row in history)):
                raise HTTPException(422, "结果曲线无效")
            if status == "DONE" and (not history or accuracy is None):
                raise HTTPException(422, "完成任务必须包含结果")
            worker["seen"] = relay.clock()
            job.update(status=status, confirmed=True, progress_round=progress, accuracy=accuracy,
                       message={"RUNNING": "本机正在训练", "DONE": "本机训练完成，完整结果已保存在电脑上",
                                "FAILED": "本机训练失败，请查看本地 training.log",
                                "CANCELLED": "本机已停止训练"}[status])
            if status != "RUNNING":
                job.update(history=history, finished_at=relay.clock())
            return {"cancel": job["cancel"]}

    return router
