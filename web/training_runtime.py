"""Deployment settings and signed access cookies for a shared research workspace."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import os
from pathlib import Path
import secrets
import time

from fastapi import HTTPException, Request

COOKIE_NAME = "spacefl_training"
COOKIE_SECONDS = 12 * 3600


@dataclass(frozen=True)
class TrainingRuntime:
    enabled: bool = False
    root: Path = Path("/var/lib/spacefl")
    access_key: str = ""
    max_rounds: int = 30
    max_epochs: int = 5
    max_satellites: int = 10
    max_samples: int = 2000
    max_batch_size: int = 128
    max_seconds: int = 1800
    cpu_threads: int = 2
    max_datasets: int = 20
    min_free_bytes: int = 2 * 1024**3

    @classmethod
    def from_env(cls) -> TrainingRuntime:
        enabled = os.environ.get("SPACEFL_PUBLIC_TRAINING", "").lower() in {"1", "true", "yes"}
        return cls(
            enabled=enabled,
            root=Path(os.environ.get("SPACEFL_DATA_DIR", "/var/lib/spacefl")),
            access_key=os.environ.get("SPACEFL_TRAINING_ACCESS_KEY", ""),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if len(self.access_key) < 32:
            raise RuntimeError("公网训练需要至少 32 字符的 SPACEFL_TRAINING_ACCESS_KEY")
        if not self.root.is_absolute():
            raise RuntimeError("SPACEFL_DATA_DIR 必须为持久化磁盘上的绝对路径")
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".write-check").open("w", encoding="utf-8") as stream:
            stream.write("ok")
        (self.root / ".write-check").unlink()

    def limits(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in (
            "max_rounds", "max_epochs", "max_satellites", "max_samples",
            "max_batch_size", "max_seconds", "max_datasets",
        )}

    def issue_cookie(self) -> str:
        payload = f"{int(time.time()) + COOKIE_SECONDS}.{secrets.token_hex(16)}"
        signature = hmac.new(self.access_key.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return f"{payload}.{signature}"

    def authenticated(self, request: Request) -> bool:
        raw = request.cookies.get(COOKIE_NAME, "")
        try:
            expiry, nonce, signature = raw.split(".")
            if not self.enabled or not self.access_key or int(expiry) <= time.time():
                return False
            expected = hmac.new(
                self.access_key.encode(), f"{expiry}.{nonce}".encode(), hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(signature, expected)
        except (ValueError, TypeError):
            return False


def require_same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
    if (origin and origin != expected) or request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(status_code=403, detail="跨站点写入请求已拒绝")


def validate_cloud_session(session: dict, runtime: TrainingRuntime, datasets: list[dict]) -> dict:
    """Validate bounded CPU training and restrict input files to managed datasets."""
    from fl_space.cli import DEFAULT_SESSION

    def reject(message: str) -> None:
        raise HTTPException(status_code=422, detail=message)

    for section in ("tune", "mount"):
        if set(session[section]) - set(DEFAULT_SESSION[section]):
            reject(f"不支持的 {section} 参数")
    tune, mount = session["tune"], session["mount"]
    fixed = {
        "tune": {"device": "cpu", "scale": "small", "workers": 1, "data_workers": 0},
        "mount": {"backend": "kepler", "config": None, "body": "earth",
                  "time_model": "slot", "time_model_args": None, "isl": "disabled"},
    }
    for section, values in fixed.items():
        for key, value in values.items():
            if session[section].get(key) != value:
                reject(f"公网训练要求 {section}.{key}={value}")

    def number(section: dict, key: str, low: float, high: float, integer: bool = False) -> None:
        value = section.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not low <= value <= high
                or (integer and int(value) != value)):
            reject(f"{key} 必须在 {low}..{high} 范围内" + ("，且为整数" if integer else ""))
        section[key] = int(value) if integer else float(value)

    for key, low, high in (
        ("rounds", 1, runtime.max_rounds), ("epochs", 1, runtime.max_epochs),
        ("max_contact_epochs", 1, runtime.max_epochs), ("batch_size", 1, runtime.max_batch_size),
        ("max_samples", 1, runtime.max_samples), ("seed", 0, 2**32 - 1),
        ("buffer_size", 1, runtime.max_satellites), ("classes_per_client", 1, 100),
        ("preferred_clients_per_class", 1, runtime.max_satellites),
    ):
        number(tune, key, low, high, True)
    for key, low, high in (
        ("lr", 1e-6, 1), ("mu", 0, 10), ("fedbuff_mu", 0, 10),
        ("alpha", 0.01, 100), ("class_probability", 0, 1), ("early_stop", 0, 1),
    ):
        number(tune, key, low, high)
    if tune["max_staleness"] is not None:
        number(tune, "max_staleness", 0, 1000, True)
    for key, low, high, integer in (
        ("sats", 2, runtime.max_satellites, True), ("stations", 1, 10, True),
        ("sim_hours", 0.1, 24, False), ("timeslot_min", 0.5, 60, False),
        ("altitude", 100, 2000, False), ("inclination", 0, 180, False),
        ("isl_buffer", 0, 100, False), ("isl_step", 1, 3600, False),
    ):
        number(mount, key, low, high, integer)
    if mount["sim_hours"] * 60 / mount["timeslot_min"] > 1800:
        reject("仿真时间槽不能超过 1800 个")
    enums = {
        "algo": (mount, {"fedavg", "fedprox", "fedbuff"}),
        "distribution": (mount, {"uniform", "cluster", "walker"}),
        "dataset": (tune, {"mnist", "fashion_mnist", "cifar10", "imagefolder", "custom"}),
        "partition_strategy": (tune, {"iid", "dirichlet", "shard", "probability"}),
        "sample_cap_strategy": (tune, {"preserve", "balanced"}),
        "preference_mode": (tune, {"class_balanced", "client_window"}),
    }
    for key, (section, choices) in enums.items():
        if section[key] not in choices:
            reject(f"公网训练不支持 {key}={section[key]}")
    for section, keys in ((tune, ("non_iid", "contact_adaptive_epochs")), (mount, ("staleness",))):
        if any(not isinstance(section[key], bool) for key in keys):
            reject("开关参数必须为布尔值")
    for profile in tune["satellite_data_profiles"].values():
        if profile["max_samples"] > runtime.max_samples:
            reject("单星画像样本上限超出公网限制")
    if tune["dataset"] in {"imagefolder", "custom"}:
        allowed = {str(Path(item["data_dir"]).resolve()) for item in datasets}
        if str(Path(str(tune.get("data_dir", ""))).resolve()) not in allowed:
            reject("公网训练只能使用已导入的数据集")
    else:
        tune["data_dir"] = str(runtime.root / "cache")
    return session
