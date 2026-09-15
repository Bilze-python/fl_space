"""SpaceFL visual experiment platform server."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Query, Request as HttpRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

PROJECT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
CESIUM_DIR = PROJECT_DIR / "node_modules" / "cesium" / "Build" / "Cesium"
NODE_MODULES_DIR = PROJECT_DIR / "node_modules"
SESSION_FILE = PROJECT_DIR / ".fls_session.json"
LITERATURE_CONFIG_FILE = PROJECT_DIR / ".literature_config.json"
AI_CONFIG_FILE = PROJECT_DIR / ".spacefl_ai_config.json"
EXPERIMENT_METADATA_FILE = PROJECT_DIR / ".spacefl_experiment_metadata.json"
EXPERIMENT_REGISTRY_FILE = PROJECT_DIR / ".spacefl_experiment_registry.json"
CC_SWITCH_DB = Path.home() / ".cc-switch" / "cc-switch.db"
VALIDATION_SCRIPT = PROJECT_DIR / "scripts" / "validate_fedleo_offloading.py"
LIBRARY_SUFFIXES = {".md", ".markdown", ".pdf", ".txt"}
EXPERIMENT_SUMMARY_FILENAMES = ("validation_summary.json", "comparison_summary.json")
EXPERIMENT_SCAN_IGNORES = {
    "venv",
    ".venv",
    "env",
    "node_modules",
    ".git",
    "__pycache__",
    "site-packages",
}
MAX_LIBRARY_FILE_SIZE = 30 * 1024 * 1024

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


def _experiment_metadata() -> dict[str, dict[str, Any]]:
    metadata_file = PROJECT_DIR / EXPERIMENT_METADATA_FILE.name
    raw = _read_json(metadata_file, {})
    return raw if isinstance(raw, dict) else {}


def _experiment_display_name(experiment_id: str, default: str) -> str:
    metadata = _experiment_metadata().get(experiment_id, {})
    name = metadata.get("name") if isinstance(metadata, dict) else None
    return str(name).strip()[:160] if str(name or "").strip() else default


def _drop_experiment_metadata(experiment_id: str) -> None:
    metadata = _experiment_metadata()
    if experiment_id in metadata:
        metadata.pop(experiment_id, None)
        _write_json(PROJECT_DIR / EXPERIMENT_METADATA_FILE.name, metadata)


def _set_experiment_name(experiment_id: str, name: str) -> None:
    metadata = _experiment_metadata()
    metadata[experiment_id] = {"name": name}
    _write_json(PROJECT_DIR / EXPERIMENT_METADATA_FILE.name, metadata)


def _experiment_registry() -> list[dict[str, Any]]:
    """读取“＋新建档案”登记表（不依赖结果目录是否存在）。"""
    raw = _read_json(EXPERIMENT_REGISTRY_FILE, [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict) and item.get("id")]


def _write_experiment_registry(entries: list[dict[str, Any]]) -> None:
    _write_json(EXPERIMENT_REGISTRY_FILE, entries)


def _register_archive(entry: dict[str, Any]) -> None:
    """写入/覆盖一条“＋新建档案”登记（按 id 去重）。"""
    registry = [
        item for item in _experiment_registry() if str(item.get("id")) != str(entry["id"])
    ]
    registry.append(entry)
    _write_experiment_registry(registry)


def _registry_entry(experiment_id: str) -> dict[str, Any] | None:
    return next(
        (item for item in _experiment_registry() if str(item.get("id")) == experiment_id),
        None,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _experiment_summary_files() -> list[Path]:
    """列出项目内所有实验汇总文件。

    这里刻意不再维护「输出目录名白名单」：网页验证实验会写到项目根目录的
    ``fedleo_web_validation_<时间戳>_<seed>/``，旧白名单一旦漏写目录名，实验做完了
    档案也永远扫不到。改为扫描两个有界根目录，任何新的输出目录名都能自动入档：

    * 项目根目录下钻一层（CLI 与网页平台的 ``--output`` 结果都落在这里）
    * ``experiments/`` 全量递归（历史归档目录）
    """
    found: set[Path] = set()
    for filename in EXPERIMENT_SUMMARY_FILENAMES:
        found.update(PROJECT_DIR.glob(f"*/{filename}"))
    experiments_dir = PROJECT_DIR / "experiments"
    if experiments_dir.is_dir():
        for filename in EXPERIMENT_SUMMARY_FILENAMES:
            found.update(experiments_dir.rglob(filename))
    return [path for path in found if path.is_file()]


def _summary_file_in(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    for filename in EXPERIMENT_SUMMARY_FILENAMES:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _candidate_summary_files() -> list[Path]:
    """列出可能包含实验结果、但不一定落在自动扫描范围内的汇总文件。

    自动扫描只看“项目根目录下钻一层 + experiments/ 全量递归”；这里额外下钻到三层，
    好让 ``results/``、``output/``、``tmp/`` 之类的自定义输出目录也能被“＋新建档案”登记。
    """
    found: set[Path] = set(_experiment_summary_files())
    for depth in (1, 2, 3):
        for filename in EXPERIMENT_SUMMARY_FILENAMES:
            pattern = "/".join(["*"] * depth + [filename])
            found.update(PROJECT_DIR.glob(pattern))

    candidates: list[Path] = []
    for path in found:
        try:
            parts = path.relative_to(PROJECT_DIR).parts
        except ValueError:
            continue
        if EXPERIMENT_SCAN_IGNORES.intersection(parts) or not path.is_file():
            continue
        candidates.append(path)
    return sorted(candidates, key=_safe_mtime, reverse=True)


def _experiment_from_summary(path: Path) -> dict[str, Any] | None:
    """把一份实验汇总 JSON 解析成档案条目；无法识别时返回 None。"""
    data = _read_json(path, {})
    if not isinstance(data, dict) or not ("offload_on" in data or "fedleo" in data):
        return None
    try:
        relative = path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        relative = path.as_posix()
    experiment_id = path.parent.name
    config = _as_dict(data.get("config"))
    created_at = datetime.fromtimestamp(_safe_mtime(path), tz=timezone.utc).isoformat()

    if "offload_on" in data:
        deltas = _as_dict(data.get("deltas_on_minus_off"))
        on = _as_dict(data.get("offload_on"))
        return {
            "id": experiment_id,
            "name": _experiment_display_name(experiment_id, "FedLEO 卸载开关验证"),
            "kind": "fedleo_validation",
            "status": "DONE" if data.get("passed") else "FAILED",
            "created_at": created_at,
            "path": relative,
            "seed": config.get("seed"),
            "rounds": config.get("num_rounds"),
            "dataset": config.get("dataset", "mnist"),
            "accuracy": on.get("final_accuracy"),
            "accuracy_delta": deltas.get("final_accuracy"),
            "balance_delta": deltas.get("final_balance"),
            "offloaded": on.get("total_offloaded", 0),
            "elapsed_sec": data.get("elapsed_sec"),
            "passed": bool(data.get("passed")),
        }

    fedleo = _as_dict(data.get("fedleo"))
    baseline = _as_dict(data.get("baseline"))
    return {
        "id": experiment_id,
        "name": _experiment_display_name(experiment_id, "FedLEO 与集中式基线"),
        "kind": "fedleo_comparison",
        "status": "DONE",
        "created_at": created_at,
        "path": relative,
        "seed": config.get("seed", 42),
        "rounds": config.get("num_rounds"),
        "dataset": config.get("dataset", "mnist"),
        "accuracy": fedleo.get("final_accuracy"),
        "accuracy_delta": (
            fedleo.get("final_accuracy", 0) - baseline.get("final_accuracy", 0)
        ),
        "balance_delta": None,
        "offloaded": fedleo.get("total_offloaded", 0),
        "elapsed_sec": fedleo.get("elapsed_sec"),
        "passed": None,
    }


def _manual_experiment(entry: dict[str, Any]) -> dict[str, Any]:
    """“＋新建档案”登记了目录但目录里还没有结果文件时的占位档案。"""
    entry_id = str(entry["id"])
    return {
        "id": entry_id,
        "name": _experiment_display_name(entry_id, str(entry.get("name") or "新建档案")),
        "kind": "manual",
        "status": "PENDING",
        "created_at": str(entry.get("created_at") or ""),
        "path": str(entry.get("path") or ""),
        "seed": entry.get("seed"),
        "rounds": entry.get("rounds"),
        "dataset": entry.get("dataset") or "mnist",
        "accuracy": None,
        "accuracy_delta": None,
        "balance_delta": None,
        "offloaded": 0,
        "elapsed_sec": None,
        "passed": None,
        "manual": True,
        "note": str(entry.get("note") or ""),
    }


def _scan_experiments() -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for path in sorted(_experiment_summary_files(), key=_safe_mtime, reverse=True):
        item = _experiment_from_summary(path)
        if item is not None:
            experiments.append(item)

    # “＋新建档案”登记的条目：结果目录可能不在自动扫描范围内，也可能只有配置快照。
    for entry in _experiment_registry():
        relative = str(entry.get("path") or "").strip()
        item: dict[str, Any] | None = None
        if relative:
            try:
                summary = _summary_file_in(_safe_project_path(relative))
            except HTTPException:
                summary = None
            if summary is not None:
                item = _experiment_from_summary(summary)
        if item is None:
            item = _manual_experiment(entry)
        else:
            item["manual"] = True
            item["note"] = str(entry.get("note") or "")
        experiments.append(item)

    # 同名目录可能同时存在于根目录与 experiments/ 下；保留一份，避免 id 冲突。
    # 登记条目优先：它带着用户填写的备注和名称，结果生成后也已经合并了真实数据。
    unique: dict[str, dict[str, Any]] = {}
    for item in experiments:
        existing = unique.get(item["id"])
        if existing is None:
            unique[item["id"]] = item
            continue
        if item.get("manual") and not existing.get("manual"):
            merged = dict(item)
            merged["note"] = item.get("note") or existing.get("note") or ""
            unique[item["id"]] = merged
    return sorted(
        unique.values(),
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    )


def _experiment_detail(experiment_id: str) -> dict[str, Any]:
    experiment = next(
        (item for item in _scan_experiments() if item["id"] == experiment_id),
        None,
    )
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    detail = dict(experiment)
    entry = _registry_entry(experiment_id)
    relative = str(experiment.get("path") or "").strip()

    data: dict[str, Any] = {}
    if relative:
        summary = _safe_project_path(relative)
        if summary.is_file():
            data = _as_dict(_read_json(summary, {}))
    if not data and entry is not None:
        # 只有登记信息、还没有结果文件：把配置快照当成只读上下文返回。
        data = {
            "manual": True,
            "config": _as_dict(entry.get("config")),
            "note": str(entry.get("note") or ""),
        }
    detail["raw"] = data
    detail["note"] = (
        str(entry.get("note") or "") if entry is not None else str(experiment.get("note") or "")
    )

    if experiment["kind"] == "fedleo_validation" and relative:
        base = _safe_project_path(relative).parent
        detail["history_on"] = _as_list(
            _as_dict(_read_json(base / "offload_on" / "fedleo_offload.json", {})).get(
                "history"
            )
        )
        detail["history_off"] = _as_list(
            _as_dict(_read_json(base / "offload_off" / "fedleo_no_offload.json", {})).get(
                "history"
            )
        )
    else:
        detail["history_on"] = _as_list(_as_dict(data.get("fedleo")).get("history"))
        detail["history_off"] = _as_list(_as_dict(data.get("baseline")).get("history"))
    return detail


def _load_literature_config() -> dict[str, Any]:
    raw = _read_json(LITERATURE_CONFIG_FILE, {})
    if not isinstance(raw, dict):
        raw = {}
    default_path = raw.get("default_path")
    if not default_path:
        legacy_paths = raw.get("default_paths", [])
        default_path = legacy_paths[0] if legacy_paths else str(PROJECT_DIR / "文献")
    return {"default_path": str(default_path)}


def _resolve_literature_root(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    return candidate.resolve()


def _library_roots() -> list[tuple[str, Path]]:
    configured = _resolve_literature_root(_load_literature_config()["default_path"])
    candidates = [("项目文档", (PROJECT_DIR / "docs").resolve()), ("默认文献", configured)]
    roots: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, path in candidates:
        if path in seen or not path.exists() or not path.is_dir():
            continue
        seen.add(path)
        roots.append((label, path))
    return roots


def _resolve_library_item(item_path: str) -> Path:
    prefix, separator, relative = item_path.partition(":")
    if separator and prefix.isdigit():
        roots = _library_roots()
        index = int(prefix)
        if index >= len(roots):
            raise HTTPException(status_code=404, detail="文献目录不存在")
        root = roots[index][1]
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="文献路径越界") from exc
        return target
    return _safe_project_path(item_path)


def _scan_library() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for root_index, (source, root) in enumerate(_library_roots()):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in LIBRARY_SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            lower = path.name.lower()
            tags = [
                tag
                for tag in ("fedleo", "fedavg", "fedprox", "fedbuff", "satellite", "design")
                if tag in lower
            ]
            items.append(
                {
                    "id": f"{root_index}:{relative}",
                    "title": path.stem,
                    "type": "md" if path.suffix.lower() == ".markdown" else path.suffix.lower().lstrip("."),
                    "path": f"{root_index}:{relative}",
                    "source": source,
                    "size": path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "tags": tags,
                }
            )
    return items


def _config_value(config: str, key: str) -> str:
    """Read a simple quoted value from CC Switch's TOML-like config."""
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*[\"']([^\"']+)[\"']", config)
    return match.group(1).strip() if match else ""


def _read_cc_switch_provider() -> dict[str, str]:
    if not CC_SWITCH_DB.exists():
        raise RuntimeError("未找到 CC Switch 数据库，请先启动 CC Switch 并配置供应商")
    uri = f"file:{CC_SWITCH_DB.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=3) as connection:
            row = connection.execute(
                "SELECT id, name, app_type, settings_config FROM providers "
                "WHERE app_type IN ('codex', 'claude') AND is_current = 1 "
                "ORDER BY CASE app_type WHEN 'codex' THEN 0 ELSE 1 END LIMIT 1"
            ).fetchone()
    except sqlite3.Error as exc:
        raise RuntimeError(f"无法读取 CC Switch 数据库: {exc}") from exc
    if row is None:
        raise RuntimeError("CC Switch 尚未选择当前供应商")
    try:
        settings = json.loads(row[3])
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("CC Switch 当前供应商配置格式无效") from exc
    env = settings.get("env", {}) if isinstance(settings, dict) else {}
    auth = settings.get("auth", {}) if isinstance(settings, dict) else {}
    config = str(settings.get("config", "")) if isinstance(settings, dict) else ""
    app_type = str(row[2])
    if app_type == "codex":
        token = str(
            auth.get("OPENAI_API_KEY")
            or auth.get("api_key")
            or env.get("OPENAI_API_KEY")
            or ""
        ).strip()
        base_url = _config_value(config, "base_url") or "https://api.openai.com/v1"
        model = str(settings.get("model") or _config_value(config, "model") or "gpt-5.5").strip()
        protocol = _config_value(config, "wire_api") or "responses"
    else:
        token = str(env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_API_KEY") or "").strip()
        base_url = str(env.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").strip()
        model = str(settings.get("model") or env.get("ANTHROPIC_MODEL") or "claude-sonnet-4-5").strip()
        protocol = "anthropic"
    if not token:
        raise RuntimeError("CC Switch 当前供应商没有可用的认证令牌")
    return {
        "id": str(row[0]),
        "name": str(row[1]),
        "app_type": app_type,
        "token": token,
        "base_url": base_url,
        "model": model,
        "protocol": protocol,
    }


def _read_local_ai_provider() -> dict[str, Any]:
    """Read an explicitly configured local provider without exposing its key."""
    config = _read_json(AI_CONFIG_FILE, {})
    if not isinstance(config, dict) or config.get("source") != "local":
        raise RuntimeError("本地 AI 配置不存在")
    token = str(config.get("api_key") or config.get("OPENAI_API_KEY") or "").strip()
    base_url = str(config.get("base_url") or "https://api.openai.com/v1").strip()
    model = str(config.get("model") or "gpt-5.5").strip()
    protocol = str(config.get("wire_api") or "responses").strip()
    if not token:
        raise RuntimeError("本地 AI 配置缺少 API Key")
    return {
        "id": "local",
        "name": str(config.get("provider") or config.get("name") or "iCode").strip(),
        "app_type": "codex",
        "token": token,
        "base_url": base_url,
        "model": model,
        "protocol": protocol,
        "reasoning_effort": str(config.get("model_reasoning_effort") or "high").strip(),
        "disable_response_storage": bool(config.get("disable_response_storage", True)),
    }


def _ai_settings() -> dict[str, Any]:
    config = _read_json(AI_CONFIG_FILE, {})
    if isinstance(config, dict) and config.get("source") == "local":
        try:
            provider = _read_local_ai_provider()
            return {
                "ai_provider": provider["name"],
                "ai_model": provider["model"],
                "ai_configured": True,
                "ai_source": "local",
            }
        except RuntimeError as exc:
            return {
                "ai_provider": str(config.get("provider") or config.get("name") or "iCode"),
                "ai_model": str(config.get("model") or "gpt-5.5"),
                "ai_configured": False,
                "ai_source": "local",
                "ai_error": str(exc),
            }
    if isinstance(config, dict) and config.get("source") == "cc_switch":
        try:
            provider = _read_cc_switch_provider()
            return {
                "ai_provider": provider["name"],
                "ai_model": provider["model"],
                "ai_configured": True,
                "ai_source": "cc_switch",
            }
        except RuntimeError as exc:
            return {
                "ai_provider": "CC Switch",
                "ai_model": "--",
                "ai_configured": False,
                "ai_source": "cc_switch",
                "ai_error": str(exc),
            }
    return {
        "ai_provider": "DeepSeek",
        "ai_model": "deepseek-chat",
        "ai_configured": bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()),
        "ai_source": "environment",
    }


def _analysis_context(experiment: dict[str, Any] | None) -> dict[str, Any]:
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
    return context


def _analysis_system_prompt() -> str:
    return (
        "你是 SpaceFL 卫星联邦学习实验分析助手。"
        "只根据提供的数据作答，区分功能验证、趋势证据和论文级复现。"
        "优先指出对照是否公平、卸载是否真实发生、样本是否守恒、"
        "准确率与时延结论是否被当前模型支持。回答使用简洁中文。"
    )


def _deepseek_analysis(
    question: str,
    experiment: dict[str, Any] | None,
) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DeepSeek API 尚未配置")

    context = _analysis_context(experiment)

    payload = {
        "model": "deepseek-chat",
        "temperature": 0.2,
        "max_tokens": 1200,
        "messages": [
            {
                "role": "system",
                "content": _analysis_system_prompt(),
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


def _openai_responses_analysis(
    question: str,
    experiment: dict[str, Any] | None,
    provider: dict[str, Any],
) -> str:
    base_url = provider["base_url"].rstrip("/")
    endpoint = f"{base_url}/responses"
    user_content = (
        f"实验上下文：\n{json.dumps(_analysis_context(experiment), ensure_ascii=False)}\n\n"
        f"用户问题：{question}"
    )
    payload = {
        "model": provider["model"],
        "instructions": _analysis_system_prompt(),
        "input": user_content,
        "max_output_tokens": 1200,
    }
    if provider.get("reasoning_effort"):
        payload["reasoning"] = {"effort": provider["reasoning_effort"]}
    if provider.get("disable_response_storage"):
        payload["store"] = False
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {provider['token']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"CC Switch 供应商请求失败：HTTP {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"CC Switch 供应商连接失败：{exc.reason}") from exc

    content = data.get("output_text") if isinstance(data, dict) else None
    if not content and isinstance(data, dict):
        chunks: list[str] = []
        for item in data.get("output", []):
            for block in item.get("content", []) if isinstance(item, dict) else []:
                if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                    chunks.append(str(block.get("text", "")))
        content = "".join(chunks)
    if not str(content or "").strip():
        raise RuntimeError("CC Switch 供应商返回了无法识别的响应")
    return str(content).strip()


def _cc_switch_analysis(
    question: str,
    experiment: dict[str, Any] | None,
) -> tuple[str, dict[str, str]]:
    provider = _read_cc_switch_provider()
    if provider["protocol"].lower() == "responses" or provider["app_type"] == "codex":
        content = _openai_responses_analysis(question, experiment, provider)
        return content, {"provider": provider["name"], "model": provider["model"]}
    base_url = provider["base_url"].rstrip("/")
    endpoint = f"{base_url}/messages" if base_url.endswith("/v1") else f"{base_url}/v1/messages"
    user_content = (
        f"实验上下文：\n{json.dumps(_analysis_context(experiment), ensure_ascii=False)}\n\n"
        f"用户问题：{question}"
    )
    payload = {
        "model": provider["model"],
        "temperature": 0.2,
        "max_tokens": 1200,
        "system": _analysis_system_prompt(),
        "messages": [{"role": "user", "content": user_content}],
    }
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {provider['token']}",
            "x-api-key": provider["token"],
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"CC Switch 供应商请求失败：HTTP {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"CC Switch 供应商连接失败：{exc.reason}") from exc
    try:
        content = "".join(
            str(block.get("text", ""))
            for block in data["content"]
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    except (KeyError, TypeError) as exc:
        raise RuntimeError("CC Switch 供应商返回了无法识别的响应") from exc
    if not content:
        raise RuntimeError("CC Switch 供应商没有返回文本内容")
    return content, {"provider": provider["name"], "model": provider["model"]}


def _local_ai_analysis(
    question: str,
    experiment: dict[str, Any] | None,
) -> tuple[str, dict[str, str]]:
    provider = _read_local_ai_provider()
    if provider["protocol"].lower() != "responses":
        raise RuntimeError("本地 AI 配置目前仅支持 wire_api=responses")
    content = _openai_responses_analysis(question, experiment, provider)
    return content, {"provider": provider["name"], "model": provider["model"]}


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
    app.mount("/vendor/d3", StaticFiles(directory=NODE_MODULES_DIR / "d3" / "dist"), name="vendor-d3")
    app.mount(
        "/vendor/topojson",
        StaticFiles(directory=NODE_MODULES_DIR / "topojson-client" / "dist"),
        name="vendor-topojson",
    )
    app.mount(
        "/vendor/world-atlas",
        StaticFiles(directory=NODE_MODULES_DIR / "world-atlas"),
        name="vendor-world-atlas",
    )
    app.mount(
        "/vendor/marked",
        StaticFiles(directory=NODE_MODULES_DIR / "marked" / "lib"),
        name="vendor-marked",
    )
    app.mount(
        "/vendor/dompurify",
        StaticFiles(directory=NODE_MODULES_DIR / "dompurify" / "dist"),
        name="vendor-dompurify",
    )
    if not CESIUM_DIR.exists():
        raise RuntimeError(
            "Cesium local assets are missing. Run `npm install` in the project directory."
        )
    app.mount("/cesium", StaticFiles(directory=CESIUM_DIR), name="cesium")

    @app.get("/cesium_orbit_viewer.html")
    async def cesium_viewer():
        """Cesium轨道可视化页面"""
        return FileResponse(WEB_DIR / "cesium_orbit_viewer.html")

    @app.get("/orbit_2d_viewer.html")
    async def orbit_2d_viewer():
        """2D orbit viewer page"""
        return FileResponse(WEB_DIR / "orbit_2d_viewer.html")

    @app.get("/launcher.html")
    async def launcher():
        return FileResponse(WEB_DIR / "launcher.html")

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
        return _ai_settings()

    @app.post("/api/settings/cc-switch/import")
    async def import_cc_switch() -> dict[str, Any]:
        try:
            provider = _read_cc_switch_provider()
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _write_json(
            AI_CONFIG_FILE,
            {
                "source": "cc_switch",
                "app_type": provider["app_type"],
                "provider_id_at_import": provider["id"],
            },
        )
        return _ai_settings()

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

    # 必须注册在 /api/experiments/{experiment_id} 之前，否则 "candidates" 会被当成实验 id。
    @app.get("/api/experiments/candidates")
    async def experiment_candidates() -> list[dict[str, Any]]:
        archived = {item["id"] for item in _scan_experiments()}
        candidates: list[dict[str, Any]] = []
        for path in _candidate_summary_files():
            item = _experiment_from_summary(path)
            if item is None or item["id"] in archived:
                continue
            candidates.append(
                {
                    "id": item["id"],
                    "path": item["path"],
                    "name": item["name"],
                    "dataset": item["dataset"],
                    "rounds": item["rounds"],
                    "seed": item["seed"],
                    "accuracy": item["accuracy"],
                    "created_at": item["created_at"],
                }
            )
        return candidates

    @app.post("/api/experiments")
    async def create_experiment(
        payload: dict[str, Any] = Body(default={}),
    ) -> dict[str, Any]:
        name = str(payload.get("name", "")).strip()
        if not name:
            raise HTTPException(status_code=422, detail="档案名称不能为空")
        name = name[:160]
        note = str(payload.get("note", "")).strip()[:500]
        source = str(payload.get("source", "session")).strip().lower()
        created_at = datetime.now(timezone.utc).isoformat()

        registry = _experiment_registry()
        archived = {item["id"] for item in _scan_experiments()}
        archived.update(str(item.get("id")) for item in registry)

        if source == "directory":
            raw_path = str(payload.get("path", "")).strip().replace("\\", "/")
            if not raw_path:
                raise HTTPException(status_code=422, detail="请选择要登记的结果目录")
            directory = _safe_project_path(raw_path)
            summary = _summary_file_in(directory)
            if summary is None:
                raise HTTPException(
                    status_code=422,
                    detail="该目录下没有 validation_summary.json 或 comparison_summary.json",
                )
            parsed = _experiment_from_summary(summary)
            if parsed is None:
                raise HTTPException(
                    status_code=422, detail="该汇总文件不是可识别的 SpaceFL 实验结果"
                )
            if parsed["id"] in archived:
                raise HTTPException(status_code=409, detail="该结果目录已经在实验档案中")
            entry: dict[str, Any] = {
                "id": parsed["id"],
                "name": name,
                "note": note,
                "source": "directory",
                "path": parsed["path"],
                "dataset": parsed["dataset"],
                "rounds": parsed["rounds"],
                "seed": parsed["seed"],
                "created_at": created_at,
            }
        elif source == "session":
            session = _load_session()
            entry_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            suffix = 0
            while entry_id in archived:
                suffix += 1
                entry_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
            config = {
                "mount": _as_dict(session.get("mount")),
                "tune": _as_dict(session.get("tune")),
            }
            entry = {
                "id": entry_id,
                "name": name,
                "note": note,
                "source": "session",
                "path": "",
                "dataset": str(
                    payload.get("dataset")
                    or config["tune"].get("dataset")
                    or config["mount"].get("dataset")
                    or "mnist"
                ),
                "rounds": payload.get("rounds") or config["tune"].get("rounds"),
                "seed": config["tune"].get("seed"),
                "config": config,
                "created_at": created_at,
            }
        else:
            raise HTTPException(status_code=422, detail="来源仅支持 directory 或 session")

        registry.append(entry)
        _write_experiment_registry(registry)
        return {
            "created": True,
            "id": entry["id"],
            "name": name,
            "source": entry["source"],
        }

    @app.get("/api/experiments/{experiment_id}")
    async def experiment_detail(experiment_id: str) -> dict[str, Any]:
        return _experiment_detail(experiment_id)

    @app.put("/api/experiments/{experiment_id}/name")
    async def rename_experiment(experiment_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        # Resolve first so arbitrary project files cannot receive metadata.
        experiment = next((item for item in _scan_experiments() if item["id"] == experiment_id), None)
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        name = str(payload.get("name", "")).strip()
        if not name:
            raise HTTPException(status_code=422, detail="实验名称不能为空")
        if len(name) > 160:
            raise HTTPException(status_code=422, detail="实验名称不能超过 160 个字符")
        _set_experiment_name(experiment_id, name)
        return {"saved": True, "id": experiment_id, "name": name}

    @app.delete("/api/experiments/{experiment_id}")
    async def delete_experiment(experiment_id: str) -> dict[str, Any]:
        experiment = next((item for item in _scan_experiments() if item["id"] == experiment_id), None)
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        registry = _experiment_registry()
        registered = any(str(item.get("id")) == experiment_id for item in registry)

        def unregister() -> None:
            _write_experiment_registry(
                [item for item in registry if str(item.get("id")) != experiment_id]
            )
            _drop_experiment_metadata(experiment_id)

        relative = str(experiment.get("path") or "").strip()
        if not relative:
            # “＋新建档案”的占位档案：只注销登记，不碰磁盘。
            unregister()
            return {"deleted": True, "id": experiment_id, "unregistered": True}

        archive_file = _safe_project_path(relative)
        archive_dir = archive_file.parent.resolve()
        # Only remove generated archive directories, never an arbitrary path.
        allowed = archive_dir.parent.name in {"validation_results", "test_outputs"} or archive_dir.name.startswith(("fedleo_", "fedleo-local", "fedleo_local", "fedleo_web_validation_", "manual_"))
        if not allowed or archive_dir == PROJECT_DIR:
            if registered:
                # 手工登记的用户数据目录：只撤销登记，绝不删除。
                unregister()
                return {"deleted": True, "id": experiment_id, "unregistered": True}
            raise HTTPException(status_code=403, detail="该实验档案不允许删除")
        shutil.rmtree(archive_dir)
        if registered:
            unregister()
        else:
            _drop_experiment_metadata(experiment_id)
        return {"deleted": True, "id": experiment_id}

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

        # 从已有档案（例如“＋新建档案”的占位档案）发起运行时会带上 archive_id：
        # 结果直接写进该档案目录，跑完同一条档案就从“待运行”变成有数据的档案。
        requested_id = str(payload.get("archive_id", "")).strip()
        if requested_id:
            if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", requested_id):
                raise HTTPException(
                    status_code=422, detail="档案 id 仅允许字母、数字、点、下划线与中划线"
                )
            output_name = requested_id
        else:
            output_name = f"fedleo_web_validation_{stamp}_{seed}"
        job_id = f"job_{uuid4().hex[:10]}"

        # 网页“＋新建档案 → 立即运行”会带上名称：先把档案登记好，
        # 任务一启动档案里就能看到这条（PENDING），跑完自动变成带数据的档案。
        display_name = str(payload.get("name", "")).strip()[:160]
        if display_name or requested_id:
            existing = _registry_entry(output_name) or {}
            resolved_name = display_name or str(existing.get("name") or "").strip()
            if not resolved_name:
                resolved_name = _experiment_display_name(output_name, "FedLEO 卸载开关验证")
            _register_archive(
                {
                    "id": output_name,
                    "name": resolved_name,
                    "note": (
                        str(payload.get("note", "")).strip()[:500]
                        or str(existing.get("note") or "")
                    ),
                    "source": "run",
                    "path": output_name,
                    "dataset": existing.get("dataset") or "mnist",
                    "rounds": rounds,
                    "seed": seed,
                    "created_at": existing.get("created_at")
                    or datetime.now(timezone.utc).isoformat(),
                }
            )
            _set_experiment_name(output_name, resolved_name)

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
        settings = _ai_settings()
        try:
            if settings.get("ai_source") == "local":
                content, provider_meta = await asyncio.to_thread(
                    _local_ai_analysis,
                    question,
                    experiment,
                )
            elif settings.get("ai_source") == "cc_switch":
                content, provider_meta = await asyncio.to_thread(
                    _cc_switch_analysis,
                    question,
                    experiment,
                )
            else:
                content = await asyncio.to_thread(_deepseek_analysis, question, experiment)
                provider_meta = {"provider": "DeepSeek", "model": "deepseek-chat"}
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            **provider_meta,
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

    @app.get("/api/library/settings")
    async def library_settings() -> dict[str, Any]:
        config = _load_literature_config()
        root = _resolve_literature_root(config["default_path"])
        return {
            "default_path": str(root),
            "exists": root.exists() and root.is_dir(),
            "supported_types": sorted(suffix.lstrip(".") for suffix in LIBRARY_SUFFIXES),
        }

    @app.put("/api/library/settings")
    async def update_library_settings(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        raw_path = str(payload.get("default_path", "")).strip()
        if not raw_path:
            raise HTTPException(status_code=422, detail="请输入默认文献目录")
        root = _resolve_literature_root(raw_path)
        if not root.exists() or not root.is_dir():
            raise HTTPException(status_code=422, detail="默认文献目录不存在或不是文件夹")
        _write_json(LITERATURE_CONFIG_FILE, {"default_path": str(root)})
        return {"saved": True, "default_path": str(root)}

    @app.post("/api/library/import")
    async def import_library_file(
        request: HttpRequest,
        filename: str = Query(..., min_length=1, max_length=240),
    ) -> dict[str, Any]:
        safe_name = Path(filename).name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in LIBRARY_SUFFIXES:
            raise HTTPException(status_code=422, detail="仅支持 PDF、Markdown 和 TXT 文献")
        content_length = int(request.headers.get("content-length", "0") or 0)
        if content_length > MAX_LIBRARY_FILE_SIZE:
            raise HTTPException(status_code=413, detail="单个文献不能超过 30 MB")
        content = await request.body()
        if not content or len(content) > MAX_LIBRARY_FILE_SIZE:
            raise HTTPException(status_code=413, detail="文献为空或超过 30 MB")
        root = _resolve_literature_root(_load_literature_config()["default_path"])
        if not root.exists() or not root.is_dir():
            raise HTTPException(status_code=422, detail="请先设置有效的默认文献目录")
        target = root / safe_name
        if target.exists():
            stem = target.stem
            for index in range(1, 1000):
                candidate = root / f"{stem} ({index}){suffix}"
                if not candidate.exists():
                    target = candidate
                    break
        target.write_bytes(content)
        return {"imported": True, "name": target.name, "size": len(content)}

    @app.get("/api/library/content")
    async def library_content(path: str = Query(...)) -> dict[str, Any]:
        target = _resolve_library_item(path)
        if not target.exists() or target.suffix.lower() not in {".md", ".markdown", ".txt"}:
            raise HTTPException(status_code=404, detail="Readable document not found")
        text = target.read_text(encoding="utf-8", errors="replace")
        return {"path": path, "content": text[:500_000]}

    @app.get("/api/library/file")
    async def library_file(path: str = Query(...)) -> FileResponse:
        target = _resolve_library_item(path)
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
