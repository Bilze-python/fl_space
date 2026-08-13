"""SpaceFL visual experiment platform server."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

PROJECT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
CESIUM_DIR = PROJECT_DIR / "node_modules" / "cesium" / "Build" / "Cesium"
SESSION_FILE = PROJECT_DIR / ".fls_session.json"
VALIDATION_SCRIPT = PROJECT_DIR / "scripts" / "validate_fedleo_offloading.py"

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _safe_project_path(relative_path: str) -> Path:
    candidate = (PROJECT_DIR / relative_path).resolve()
    try:
        candidate.relative_to(PROJECT_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Path is outside the project") from exc
    return candidate


def _default_session() -> dict[str, dict[str, Any]]:
    from fl_space.cli import DEFAULT_SESSION

    return {
        "tune": dict(DEFAULT_SESSION["tune"]),
        "mount": dict(DEFAULT_SESSION["mount"]),
    }


def _load_session() -> dict[str, dict[str, Any]]:
    session = _read_json(SESSION_FILE, _default_session())
    defaults = _default_session()
    for section in ("tune", "mount"):
        session.setdefault(section, {})
        for key, value in defaults[section].items():
            session[section].setdefault(key, value)
    return session


def _validate_satellite_profiles(
    raw_profiles: Any,
    satellite_count: int,
) -> dict[str, dict[str, Any]]:
    if raw_profiles in (None, ""):
        return {}
    if not isinstance(raw_profiles, dict):
        raise HTTPException(status_code=422, detail="卫星数据画像必须是 JSON 对象")

    profiles: dict[str, dict[str, Any]] = {}
    for raw_id, raw_profile in raw_profiles.items():
        try:
            satellite_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"无效卫星编号: {raw_id}") from exc
        if not 0 <= satellite_id < satellite_count:
            raise HTTPException(
                status_code=422,
                detail=f"卫星 {satellite_id} 超出当前星座范围 0..{satellite_count - 1}",
            )
        if not isinstance(raw_profile, dict):
            raise HTTPException(status_code=422, detail=f"卫星 {satellite_id} 画像格式无效")
        try:
            preferred_classes = sorted(
                {int(value) for value in raw_profile.get("preferred_classes", [])}
            )
            probability = float(raw_profile.get("preference_probability", 0.8))
            max_samples = int(raw_profile.get("max_samples", 0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"卫星 {satellite_id} 画像含无效数值") from exc
        if any(value < 0 for value in preferred_classes):
            raise HTTPException(status_code=422, detail=f"卫星 {satellite_id} 类别不得为负数")
        if not 0 <= probability <= 1:
            raise HTTPException(status_code=422, detail=f"卫星 {satellite_id} 偏好概率应在 0..1")
        if max_samples < 0:
            raise HTTPException(status_code=422, detail=f"卫星 {satellite_id} 样本上限不得为负数")
        if preferred_classes or max_samples:
            profiles[str(satellite_id)] = {
                "preferred_classes": preferred_classes,
                "preference_probability": probability,
                "max_samples": max_samples,
            }
    return profiles


def _validated_session_update(
    current: dict[str, dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result = {"tune": dict(current["tune"]), "mount": dict(current["mount"])}
    for section in ("tune", "mount"):
        update = payload.get(section)
        if isinstance(update, dict):
            result[section].update(update)

    tune = result["tune"]
    mount = result["mount"]
    protocol = str(tune.get("protocol_mode", "standard")).lower()
    if protocol not in {"standard", "paper_approx"}:
        raise HTTPException(status_code=422, detail="协议模式仅支持 standard 或 paper_approx")
    selection = str(tune.get("selection_strategy", "random")).lower()
    if selection not in {"random", "earliest_return"}:
        raise HTTPException(status_code=422, detail="卫星选择仅支持 random 或 earliest_return")
    satellite_count = max(1, min(int(mount.get("sats", 5)), 100))
    mount["sats"] = satellite_count
    tune["protocol_mode"] = protocol
    tune["selection_strategy"] = selection
    tune["max_contact_epochs"] = max(1, min(int(tune.get("max_contact_epochs", 50)), 500))
    tune["fedbuff_mu"] = max(0.0, float(tune.get("fedbuff_mu", 0.0)))
    raw_max_staleness = tune.get("max_staleness")
    tune["max_staleness"] = (
        None if raw_max_staleness in (None, "") else max(0, int(raw_max_staleness))
    )
    tune["satellite_data_profiles"] = _validate_satellite_profiles(
        tune.get("satellite_data_profiles", {}),
        satellite_count,
    )
    return result


def _scan_experiments() -> list[dict[str, Any]]:
    candidates: list[Path] = []
    for pattern in (
        "experiments/validation_results/*/validation_summary.json",
        "experiments/test_outputs/*/comparison_summary.json",
        "fedleo_local_validation*/validation_summary.json",
        "fedleo_result_1/comparison_summary.json",
        "fedleo_test_output/comparison_summary.json",
    ):
        candidates.extend(PROJECT_DIR.glob(pattern))

    experiments: list[dict[str, Any]] = []
    for path in sorted(set(candidates), key=lambda item: item.stat().st_mtime, reverse=True):
        data = _read_json(path, {})
        relative = path.relative_to(PROJECT_DIR).as_posix()
        if "offload_on" in data:
            config = data.get("config", {})
            experiments.append(
                {
                    "id": path.parent.name,
                    "name": "FedLEO 卸载开关验证",
                    "kind": "fedleo_validation",
                    "status": "DONE" if data.get("passed") else "FAILED",
                    "created_at": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "path": relative,
                    "seed": config.get("seed"),
                    "rounds": config.get("num_rounds"),
                    "dataset": config.get("dataset", "mnist"),
                    "accuracy": data.get("offload_on", {}).get("final_accuracy"),
                    "accuracy_delta": data.get("deltas_on_minus_off", {}).get(
                        "final_accuracy"
                    ),
                    "balance_delta": data.get("deltas_on_minus_off", {}).get(
                        "final_balance"
                    ),
                    "offloaded": data.get("offload_on", {}).get("total_offloaded", 0),
                    "elapsed_sec": data.get("elapsed_sec"),
                    "passed": bool(data.get("passed")),
                }
            )
        elif "fedleo" in data:
            config = data.get("config", {})
            experiments.append(
                {
                    "id": path.parent.name,
                    "name": "FedLEO 与集中式基线",
                    "kind": "fedleo_comparison",
                    "status": "DONE",
                    "created_at": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "path": relative,
                    "seed": config.get("seed", 42),
                    "rounds": config.get("num_rounds"),
                    "dataset": config.get("dataset", "mnist"),
                    "accuracy": data.get("fedleo", {}).get("final_accuracy"),
                    "accuracy_delta": (
                        data.get("fedleo", {}).get("final_accuracy", 0)
                        - data.get("baseline", {}).get("final_accuracy", 0)
                    ),
                    "balance_delta": None,
                    "offloaded": data.get("fedleo", {}).get("total_offloaded", 0),
                    "elapsed_sec": data.get("fedleo", {}).get("elapsed_sec"),
                    "passed": None,
                }
            )
    return experiments


def _experiment_detail(experiment_id: str) -> dict[str, Any]:
    experiment = next(
        (item for item in _scan_experiments() if item["id"] == experiment_id),
        None,
    )
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    data = _read_json(_safe_project_path(experiment["path"]), {})
    detail = dict(experiment)
    detail["raw"] = data
    if experiment["kind"] == "fedleo_validation":
        base = _safe_project_path(experiment["path"]).parent
        detail["history_on"] = _read_json(
            base / "offload_on" / "fedleo_offload.json", {}
        ).get("history", [])
        detail["history_off"] = _read_json(
            base / "offload_off" / "fedleo_no_offload.json", {}
        ).get("history", [])
    else:
        detail["history_on"] = data.get("fedleo", {}).get("history", [])
        detail["history_off"] = data.get("baseline", {}).get("history", [])
    return detail


def _scan_library() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for root_name in ("docs", "文献"):
        root = PROJECT_DIR / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".pdf", ".txt"}:
                continue
            relative = path.relative_to(PROJECT_DIR).as_posix()
            lower = path.name.lower()
            tags = [
                tag
                for tag in (
                    "fedleo",
                    "fedavg",
                    "fedprox",
                    "fedbuff",
                    "satellite",
                    "design",
                )
                if tag in lower
            ]
            items.append(
                {
                    "id": relative,
                    "title": path.stem,
                    "type": path.suffix.lower().lstrip("."),
                    "path": relative,
                    "size": path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "tags": tags,
                }
            )
    return items


def _deepseek_analysis(
    question: str,
    experiment: dict[str, Any] | None,
) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DeepSeek API 尚未配置")

    context: dict[str, Any] = {
        "platform": "SpaceFL visual experiment platform",
        "implementation_level": "lightweight discrete simulation",
        "known_limitations": [
            "No KKT communication/computation power optimization",
            "No dynamic multi-hop contention model",
            "No full continuous P1-P4 paper reproduction",
        ],
    }
    if experiment:
        raw = experiment.get("raw", {})
        context["experiment"] = {
            "name": experiment.get("name"),
            "kind": experiment.get("kind"),
            "status": experiment.get("status"),
            "config": raw.get("config", {}),
            "offload_on": raw.get("offload_on", raw.get("fedleo", {})),
            "offload_off": raw.get("offload_off", raw.get("baseline", {})),
            "deltas": raw.get("deltas_on_minus_off", {}),
            "gates": raw.get("gates", {}),
            "history_on_tail": experiment.get("history_on", [])[-5:],
            "history_off_tail": experiment.get("history_off", [])[-5:],
        }

    payload = {
        "model": "deepseek-chat",
        "temperature": 0.2,
        "max_tokens": 1200,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是 SpaceFL 卫星联邦学习实验分析助手。"
                    "只根据提供的数据作答，区分功能验证、趋势证据和论文级复现。"
                    "优先指出对照是否公平、卸载是否真实发生、样本是否守恒、"
                    "准确率与时延结论是否被当前模型支持。回答使用简洁中文。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"实验上下文：\n{json.dumps(context, ensure_ascii=False)}\n\n"
                    f"用户问题：{question}"
                ),
            },
        ],
    }
    request = Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=75) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"DeepSeek 请求失败：HTTP {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"DeepSeek 网络连接失败：{exc.reason}") from exc

    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("DeepSeek 返回了无法识别的响应") from exc


def _run_validation_job(
    job_id: str,
    output_name: str,
    rounds: int,
    seed: int,
) -> None:
    output_dir = PROJECT_DIR / output_name
    command = [
        sys.executable,
        str(VALIDATION_SCRIPT),
        "--rounds",
        str(rounds),
        "--seed",
        str(seed),
        "--output",
        output_name,
    ]
    started = time.time()
    with JOBS_LOCK:
        JOBS[job_id].update(
            {
                "status": "RUNNING",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "message": "正在运行 FedLEO 卸载开关对照实验",
            }
        )
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
            check=False,
        )
        summary = _read_json(output_dir / "validation_summary.json", None)
        status = "DONE" if completed.returncode == 0 and summary else "FAILED"
        with JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": status,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.time() - started, 2),
                    "result": summary,
                    "message": (
                        "实验完成，全部验收门槛通过"
                        if status == "DONE" and summary.get("passed")
                        else "实验结束，请检查日志和结果"
                    ),
                    "log": (completed.stdout + "\n" + completed.stderr)[-12000:],
                }
            )
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id].update(
                {
                    "status": "FAILED",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.time() - started, 2),
                    "message": str(exc),
                }
            )


def build_orbit_data(
    sim_hours: float = 2.0,
    sats: int = 12,
    gs: int = 5,
    altitude_km: float = 500.0,
    inclination_deg: float = 53.0,
    timeslot_min: float = 2.0,
    isl_enabled: bool = True,
    isl_buffer: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Build orbit positions and link data for the platform."""
    if sim_hours <= 0 or timeslot_min <= 0:
        raise ValueError("Simulation duration and timeslot must be positive")
    if sats < 1 or gs < 1:
        raise ValueError("Satellite and ground station counts must be positive")

    from fl_space.isl.base import ISLConfig
    from fl_space.simulator import OrbitSimulator

    n_slots = max(2, int(sim_hours * 60 / timeslot_min))
    isl_cfg = ISLConfig(
        enabled=isl_enabled,
        calculator="wgs84",
        atmosphere_buffer_km=isl_buffer,
        step_seconds=timeslot_min * 60,
        cluster_mode="plane",
    )
    sim = OrbitSimulator(
        num_satellites=sats,
        num_ground_stations=gs,
        orbit_altitude_km=altitude_km,
        orbit_inclination_deg=inclination_deg,
        timeslot_duration_min=timeslot_min,
        num_timeslots=n_slots,
        isl_config=isl_cfg,
        random_seed=seed,
        verbose=False,
    )
    if isl_enabled:
        sim.compute_isl()

    trajectory_samples = 73
    trajectories = []
    for satellite_id in range(sats):
        points = []
        for sample_index in range(trajectory_samples):
            time_min = sim.orbit_period_min * sample_index / (trajectory_samples - 1)
            lat, lon = sim.orbits[satellite_id].position_at_time_deg(time_min)
            earth_rotation_deg = time_min * 360 / (
                sim.orbits[satellite_id].body.rotation_period_hours * 60
            )
            lon = (lon + earth_rotation_deg + 180) % 360 - 180
            points.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "alt_km": sim.orbit_altitude_km,
                }
            )
        trajectories.append({"sat_id": satellite_id, "positions": points})

    stations = [
        {
            "id": index,
            "name": station.name,
            "lat": station.lat_deg,
            "lon": station.lon_deg,
            "alt_km": station.altitude_km,
        }
        for index, station in enumerate(sim.ground_network)
    ]
    base_dt = datetime(*sim.sim_start_date, tzinfo=timezone.utc)
    timeslots = []
    for slot_index in range(n_slots):
        positions = []
        for satellite_id in range(sats):
            lat, lon = sim.get_sat_position(satellite_id, slot_index)
            positions.append(
                {
                    "sat_id": satellite_id,
                    "lat": lat,
                    "lon": lon,
                    "alt_km": sim.orbit_altitude_km,
                    "plane": satellite_id % max(1, round(sats**0.5)),
                }
            )

        contacts = []
        for satellite_id in range(sats):
            contacts.extend(
                {"sat_id": satellite_id, "gs_id": station_id}
                for station_id in sim.get_all_contacts(satellite_id, slot_index)
            )

        isl_links = []
        if isl_enabled:
            for window in sim.isl_active_at(slot_index):
                a_id = int(window.satellite_a.split("-")[-1])
                b_id = int(window.satellite_b.split("-")[-1])
                isl_links.append({"a_id": a_id, "b_id": b_id})

        timeslots.append(
            {
                "ts": slot_index,
                "time": (
                    base_dt + timedelta(minutes=slot_index * timeslot_min)
                ).isoformat(),
                "positions": positions,
                "contacts": contacts,
                "isl_links": isl_links,
            }
        )

    return {
        "satellites": sats,
        "ground_stations": stations,
        "isl_enabled": isl_enabled,
        "timeslot_duration_min": timeslot_min,
        "sim_hours": sim_hours,
        "projection_days": sim_hours / 24,
        "orbit_period_min": sim.orbit_period_min,
        "trajectories": trajectories,
        "timeslots": timeslots,
    }


def build_experiment_orbit_data(
    experiment_id: str,
    projection_days: float = 1.0,
    projection_step_min: float = 10.0,
    satellites: int | None = None,
    ground_stations: int | None = None,
    isl_enabled: bool | None = None,
) -> dict[str, Any]:
    """Build a replayable orbit timeline from an archived experiment."""
    detail = _experiment_detail(experiment_id)
    raw = detail.get("raw", {})
    config = raw.get("config", {})
    session = _load_session()
    mount = session.get("mount", {})

    history = detail.get("history_on") or detail.get("history_off") or []
    total_sats = config.get("total_sats")
    if not total_sats:
        total_sats = int(config.get("num_planes", 0)) * int(
            config.get("sats_per_plane", 0)
        )
    sats = max(1, int(satellites or total_sats or mount.get("sats", 12)))
    stations = max(
        1,
        int(
            ground_stations
            or config.get("num_ground_stations")
            or mount.get("stations", 5)
        ),
    )
    projection_days = min(max(float(projection_days), 1 / 24), 30.0)
    timeslot_min = min(max(float(projection_step_min), 1.0), 180.0)
    sim_hours = projection_days * 24
    resolved_isl = (
        bool(isl_enabled)
        if isl_enabled is not None
        else str(config.get("isl", mount.get("isl", "disabled"))).lower()
        != "disabled"
    )

    orbit = build_orbit_data(
        sim_hours=sim_hours,
        sats=sats,
        gs=stations,
        altitude_km=float(config.get("altitude_km") or mount.get("altitude", 500)),
        inclination_deg=float(
            config.get("inclination_deg") or mount.get("inclination", 53)
        ),
        timeslot_min=timeslot_min,
        isl_enabled=resolved_isl,
        isl_buffer=float(mount.get("isl_buffer", 0)),
        seed=int(config.get("seed") or detail.get("seed") or 42),
    )

    last_slot_index = max(1, len(orbit["timeslots"]) - 1)
    last_history_index = max(0, len(history) - 1)
    for index, slot in enumerate(orbit["timeslots"]):
        history_index = round(index * last_history_index / last_slot_index)
        row = history[history_index] if history else {}
        slot["experiment"] = {
            key: row.get(key)
            for key in (
                "round",
                "accuracy",
                "train_loss",
                "weight_divergence",
                "total_delay",
                "total_offloaded_samples",
                "num_offload_actions",
                "data_balance_entropy",
                "offload_actions",
            )
            if row.get(key) is not None
        }

    orbit["source"] = "experiment_archive"
    orbit["projection"] = {
        "days": projection_days,
        "step_min": timeslot_min,
        "frames": len(orbit["timeslots"]),
        "profile": "stress" if projection_days >= 15 else "standard",
    }
    orbit["experiment"] = {
        "id": detail["id"],
        "name": detail["name"],
        "kind": detail["kind"],
        "dataset": detail.get("dataset"),
        "rounds": detail.get("rounds"),
        "seed": detail.get("seed"),
        "created_at": detail.get("created_at"),
        "status": detail.get("status"),
    }
    orbit["parameter_sources"] = {
        "satellites": "projection.override" if satellites else (
            "experiment.config" if total_sats else "session.mount"
        ),
        "timeslot": "projection",
        "ground_stations": (
            "projection.override" if ground_stations else (
                "experiment.config"
                if config.get("num_ground_stations")
                else "session.mount"
            )
        ),
        "altitude": "experiment.config" if config.get("altitude_km") else "session.mount",
        "inclination": (
            "experiment.config" if config.get("inclination_deg") else "session.mount"
        ),
    }
    return orbit


def create_app(**sim_kwargs: Any) -> FastAPI:
    app = FastAPI(title="SpaceFL Visual Experiment Platform", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")
    if not CESIUM_DIR.exists():
        raise RuntimeError(
            "Cesium local assets are missing. Run `npm install` in the project directory."
        )
    app.mount("/cesium", StaticFiles(directory=CESIUM_DIR), name="cesium")

    @app.get("/cesium_orbit_viewer.html")
    async def cesium_viewer():
        """Cesium轨道可视化页面"""
        return FileResponse(WEB_DIR / "cesium_orbit_viewer.html")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "SpaceFL Visual Experiment Platform"}

    @app.get("/api/overview")
    async def overview() -> dict[str, Any]:
        experiments = _scan_experiments()
        validations = [item for item in experiments if item["kind"] == "fedleo_validation"]
        latest = validations[0] if validations else None
        deltas = [
            item["accuracy_delta"]
            for item in validations
            if isinstance(item.get("accuracy_delta"), (int, float))
        ]
        return {
            "experiments": len(experiments),
            "validations_passed": sum(item.get("passed") is True for item in validations),
            "mean_accuracy_delta": sum(deltas) / len(deltas) if deltas else 0,
            "latest": latest,
            "session": _load_session(),
            "implementation": {
                "level": "轻量级离散仿真",
                "supported": ["真实样本卸载", "同轨/跨轨分层聚合", "多种子开关消融"],
                "limitations": ["KKT 功率优化", "动态链路竞争", "论文级连续时延模型"],
            },
        }

    @app.get("/api/settings")
    async def settings() -> dict[str, Any]:
        return {
            "ai_provider": "deepseek",
            "ai_model": "deepseek-chat",
            "ai_configured": bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()),
        }

    @app.get("/api/session")
    async def get_session() -> dict[str, dict[str, Any]]:
        return _load_session()

    @app.put("/api/session")
    async def put_session(
        payload: dict[str, dict[str, Any]] = Body(...),
    ) -> dict[str, Any]:
        session = _validated_session_update(_load_session(), payload)
        _write_json(SESSION_FILE, session)
        return {"saved": True, "session": session}

    @app.post("/api/session/reset")
    async def reset_session() -> dict[str, Any]:
        session = _default_session()
        _write_json(SESSION_FILE, session)
        return {"saved": True, "session": session}

    @app.get("/api/experiments")
    async def experiments() -> list[dict[str, Any]]:
        return _scan_experiments()

    @app.get("/api/experiments/{experiment_id}")
    async def experiment_detail(experiment_id: str) -> dict[str, Any]:
        return _experiment_detail(experiment_id)

    @app.get("/api/experiments/{experiment_id}/orbit_data")
    async def experiment_orbit_data(
        experiment_id: str,
        projection_days: float = Query(default=1.0, ge=1 / 24, le=30),
        projection_step_min: float = Query(default=10.0, ge=1, le=180),
        satellites: int | None = Query(default=None, ge=1, le=72),
        ground_stations: int | None = Query(default=None, ge=1, le=20),
        isl_enabled: bool | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            return build_experiment_orbit_data(
                experiment_id,
                projection_days=projection_days,
                projection_step_min=projection_step_min,
                satellites=satellites,
                ground_stations=ground_stations,
                isl_enabled=isl_enabled,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/experiments/fedleo-validation")
    async def create_validation(
        payload: dict[str, Any] = Body(default={}),
    ) -> dict[str, Any]:
        rounds = max(2, min(int(payload.get("rounds", 10)), 30))
        seed = int(payload.get("seed", 20260810))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"fedleo_web_validation_{stamp}_{seed}"
        job_id = f"job_{uuid4().hex[:10]}"
        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id,
                "status": "PENDING",
                "rounds": rounds,
                "seed": seed,
                "output": output_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": "任务已创建",
            }
        threading.Thread(
            target=_run_validation_job,
            args=(job_id, output_name, rounds, seed),
            daemon=True,
        ).start()
        return JOBS[job_id]

    @app.get("/api/jobs")
    async def jobs() -> list[dict[str, Any]]:
        with JOBS_LOCK:
            return sorted(
                (dict(item) for item in JOBS.values()),
                key=lambda item: item["created_at"],
                reverse=True,
            )

    @app.get("/api/jobs/{job_id}")
    async def job(job_id: str) -> dict[str, Any]:
        with JOBS_LOCK:
            if job_id not in JOBS:
                raise HTTPException(status_code=404, detail="Job not found")
            return dict(JOBS[job_id])

    @app.post("/api/ai/analyze")
    async def ai_analyze(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise HTTPException(status_code=422, detail="请输入需要分析的问题")
        if len(question) > 4000:
            raise HTTPException(status_code=422, detail="问题过长，请控制在 4000 字以内")

        experiment_id = str(payload.get("experiment_id", "")).strip()
        experiment = _experiment_detail(experiment_id) if experiment_id else None
        try:
            content = await asyncio.to_thread(
                _deepseek_analysis,
                question,
                experiment,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "content": content,
        }

    @app.get("/api/library")
    async def library(q: str = Query(default="")) -> list[dict[str, Any]]:
        items = _scan_library()
        if not q:
            return items
        needle = q.casefold()
        return [
            item
            for item in items
            if needle in item["title"].casefold()
            or needle in " ".join(item["tags"]).casefold()
        ]

    @app.get("/api/library/content")
    async def library_content(path: str = Query(...)) -> dict[str, Any]:
        target = _safe_project_path(path)
        if not target.exists() or target.suffix.lower() not in {".md", ".txt", ".py", ".json"}:
            raise HTTPException(status_code=404, detail="Readable document not found")
        text = target.read_text(encoding="utf-8", errors="replace")
        return {"path": path, "content": text[:500_000]}

    @app.get("/api/library/file")
    async def library_file(path: str = Query(...)) -> FileResponse:
        target = _safe_project_path(path)
        if not target.exists() or target.suffix.lower() != ".pdf":
            raise HTTPException(status_code=404, detail="PDF not found")
        return FileResponse(target)

    @app.get("/api/orbit_data")
    async def orbit_data(
        sim_hours: float = Query(default=2.0),
        sats: int = Query(default=12),
        gs: int = Query(default=5),
        altitude_km: float = Query(default=500.0),
        inclination_deg: float = Query(default=53.0),
        timeslot_min: float = Query(default=2.0),
        isl_enabled: bool = Query(default=True),
        isl_buffer: float = Query(default=0.0),
        seed: int = Query(default=42),
    ) -> dict[str, Any]:
        kwargs = {
            "sim_hours": sim_hours,
            "sats": sats,
            "gs": gs,
            "altitude_km": altitude_km,
            "inclination_deg": inclination_deg,
            "timeslot_min": timeslot_min,
            "isl_enabled": isl_enabled,
            "isl_buffer": isl_buffer,
            "seed": seed,
        }
        kwargs.update(sim_kwargs)
        return build_orbit_data(**kwargs)

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SpaceFL visual experiment platform")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8700)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
