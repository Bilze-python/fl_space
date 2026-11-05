"""科研智能平台 v1：后台批量实验执行器。

用法::

    python scripts/run_research_plan.py --plan research_results/plan.json \\
        --output research_results/rjob_ab12cd

执行流程：

1. 读取并校验研究计划 JSON（``/api/research/agent/plan`` 的产物）；
2. 逐个 condition 调用 ``fl_space.fedleo.experiment.run_fedleo_vs_baseline``；
3. 每完成一个 condition 原子写 ``<output>/progress.json``；
4. 全部结束后写 ``<output>/research_result.json``（计划 + 逐条件结果 + 汇总指标）。

设计说明：

- 只依赖标准库 + 项目内 FedLEO 实验函数（延迟导入，导入本模块不会加载 PyTorch）；
- 单条件失败只标记 ``FAILED`` 并继续下一个；全部失败退出码 1，否则 0；
- ``KeyboardInterrupt`` 写入 ``status="CANCELLED"`` 并以退出码 2 退出；
- ``--condition-timeout-sec`` 仅为兼容调用方而接受：Windows 下没有可靠的
  单条件超时机制（训练跑在进程内线程里，无 SIGALRM 可用），因此**不强制生效**，
  只记录到 ``progress.json`` 中，由调用方自行放弃该子进程；
- ``--dry-run`` 只打印将执行的参数并写 progress / research_result，不真正训练。
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

PROGRESS_FILENAME = "progress.json"
RESULT_FILENAME = "research_result.json"
SUMMARY_FILENAME = "comparison_summary.json"
DEFAULT_DATA_DIR = "./data"
LOG_TAIL_CHARS = 4000
LOG_TAIL_LINES = 40
ERROR_TEXT_LIMIT = 500

STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"
STATUS_CANCELLED = "CANCELLED"

#: plan 中 condition.params 的缺省值。
PARAM_DEFAULTS: dict[str, Any] = {
    "planes": 3,
    "sats_per_plane": 2,
    "gs": 3,
    "rounds": 3,
    "local_epochs": 1,
    "batch_size": 64,
    "lr": 0.02,
    "seed": 42,
    "offload_every": 2,
    "max_offload_iter": 2,
    "max_samples_per_client": 120,
    "dataset": "mnist",
    "device": "cpu",
}

#: plan 参数名 → 可接受的别名（兼容直接写 run_fedleo_vs_baseline 形参名的计划）。
PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "planes": ("planes", "num_planes"),
    "sats_per_plane": ("sats_per_plane",),
    "gs": ("gs", "num_ground_stations"),
    "rounds": ("rounds", "num_rounds"),
    "local_epochs": ("local_epochs",),
    "batch_size": ("batch_size",),
    "lr": ("lr", "learning_rate"),
    "seed": ("seed",),
    "offload_every": ("offload_every", "offload_every_n_rounds"),
    "max_offload_iter": ("max_offload_iter",),
    "max_samples_per_client": ("max_samples_per_client",),
    "dataset": ("dataset",),
    "device": ("device",),
}


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """返回 UTC ISO 时间字符串。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """原子写 JSON：先写同目录临时文件再 ``replace``，避免 Web 端读到半个文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)


def _log_lines(text: str, limit: int = LOG_TAIL_LINES) -> list[str]:
    """把日志文本压成最近 ``limit`` 行（去掉空行）。"""
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    return lines[-limit:]


def _append_log(current: str, chunk: str) -> str:
    """把新日志追加到缓冲并只保留最后 ``LOG_TAIL_CHARS`` 字。"""
    combined = f"{current}{chunk}" if current else str(chunk or "")
    return combined[-LOG_TAIL_CHARS:]


def _safe_name(value: str) -> str:
    """把 condition id 转成安全的目录名。"""
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    return cleaned.strip("._") or "condition"


def _display_path(path: Path) -> str:
    """把路径格式化成展示用字符串（统一正斜杠）。"""
    return Path(path).as_posix()


def _short_error(error: BaseException) -> str:
    """把异常压成一行简短文本（带异常类型）。"""
    text = f"{type(error).__name__}: {error}".replace("\n", " ").strip()
    return text[:ERROR_TEXT_LIMIT]


def _is_number(value: Any) -> bool:
    """判断是否为有效数值（排除 bool）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any, default: float = 0.0) -> float:
    """宽松转 float（失败返回默认值）。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    """宽松转 int（失败返回默认值）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _strict_int(value: Any, name: str) -> int:
    """严格转 int，非法值抛 ValueError。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"条件参数 {name} 需要整数，实际为 {value!r}") from None
    if not number.is_integer():
        raise ValueError(f"条件参数 {name} 需要整数，实际为 {value!r}")
    return int(number)


def _strict_float(value: Any, name: str) -> float:
    """严格转 float，非法值抛 ValueError。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"条件参数 {name} 需要数值，实际为 {value!r}") from None


# ---------------------------------------------------------------------------
# 纯函数：计划读取 / 参数映射 / 结果提取 / 汇总
# ---------------------------------------------------------------------------


def load_plan(path: Path) -> dict[str, Any]:
    """读取并校验研究计划 JSON。

    校验规则：顶层必须是对象、``conditions`` 必须是非空列表、每个 condition 的
    ``id`` 必须存在且唯一。任何不合法都抛 :class:`ValueError`。

    Parameters
    ----------
    path : Path
        计划 JSON 路径。

    Returns
    -------
    dict
        规范化后的计划（保留未知字段，condition 补齐 ``name`` / ``params``）。
    """
    plan_path = Path(path)
    try:
        raw = plan_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"无法读取计划文件 {plan_path}：{exc}") from None
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"计划文件不是合法 JSON：{exc}") from None
    if not isinstance(data, dict):
        raise ValueError("计划文件必须是 JSON 对象")

    conditions = data.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("计划必须包含至少一个 condition")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, condition in enumerate(conditions, 1):
        if not isinstance(condition, dict):
            raise ValueError(f"第 {index} 个 condition 不是对象")
        condition_id = str(condition.get("id") or "").strip()
        if not condition_id:
            raise ValueError(f"第 {index} 个 condition 缺少 id")
        if condition_id in seen:
            raise ValueError(f"condition id 重复：{condition_id}")
        seen.add(condition_id)
        params = condition.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(f"condition {condition_id} 的 params 不是对象")
        entry = dict(condition)
        entry["id"] = condition_id
        entry["name"] = str(condition.get("name") or f"条件 {condition_id}").strip()
        entry["params"] = dict(params)
        normalized.append(entry)

    plan = dict(data)
    plan["id"] = str(data.get("id") or "").strip()
    plan["title"] = str(data.get("title") or plan["id"] or "未命名研究计划").strip()
    plan["backend"] = str(data.get("backend") or "fedleo").strip() or "fedleo"
    plan["conditions"] = normalized
    return plan


def _pick(params: dict[str, Any], name: str) -> Any:
    """按参数名（含别名）取值，缺省回落 :data:`PARAM_DEFAULTS`。"""
    for key in PARAM_ALIASES.get(name, (name,)):
        value = params.get(key)
        if value is not None and value != "":
            return value
    return PARAM_DEFAULTS[name]


def build_run_kwargs(condition: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """把计划中的 condition 映射成 ``run_fedleo_vs_baseline`` 的关键字参数。

    Parameters
    ----------
    condition : dict
        plan 中的一个条件（含 ``params``）。
    output_dir : Path
        该条件的输出目录。

    Returns
    -------
    dict
        可直接 ``run_fedleo_vs_baseline(**kwargs)`` 的参数。
    """
    if not isinstance(condition, dict):
        raise ValueError("condition 必须是对象")
    params = condition.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("condition.params 必须是对象")

    dataset = str(_pick(params, "dataset") or PARAM_DEFAULTS["dataset"]).strip().lower()
    device = str(_pick(params, "device") or PARAM_DEFAULTS["device"]).strip().lower()
    return {
        "num_planes": _strict_int(_pick(params, "planes"), "planes"),
        "sats_per_plane": _strict_int(_pick(params, "sats_per_plane"), "sats_per_plane"),
        "num_ground_stations": _strict_int(_pick(params, "gs"), "gs"),
        "num_rounds": _strict_int(_pick(params, "rounds"), "rounds"),
        "local_epochs": _strict_int(_pick(params, "local_epochs"), "local_epochs"),
        "batch_size": _strict_int(_pick(params, "batch_size"), "batch_size"),
        "learning_rate": _strict_float(_pick(params, "lr"), "lr"),
        "dataset": dataset,
        "data_dir": DEFAULT_DATA_DIR,
        "device": device,
        "offload_every_n_rounds": _strict_int(
            _pick(params, "offload_every"),
            "offload_every",
        ),
        "max_offload_iter": _strict_int(_pick(params, "max_offload_iter"), "max_offload_iter"),
        "max_samples_per_client": _strict_int(
            _pick(params, "max_samples_per_client"),
            "max_samples_per_client",
        ),
        "seed": _strict_int(_pick(params, "seed"), "seed"),
        "verbose": False,
        "output_dir": str(Path(output_dir)),
        "run_baseline": True,
    }


def _side(summary: dict[str, Any], *names: str) -> dict[str, Any]:
    """取出 summary 中第一个命中的子字典（兼容 offload_on/fedleo 两种命名）。"""
    for name in names:
        value = summary.get(name)
        if isinstance(value, dict):
            return value
    return {}


def _history_sum(side: dict[str, Any], key: str) -> float:
    """把 ``history`` 里某字段累加（comparison_summary.json 未单列该指标时的兜底）。"""
    history = side.get("history")
    if not isinstance(history, list):
        return 0.0
    total = 0.0
    for row in history:
        if isinstance(row, dict):
            total += _as_float(row.get(key), 0.0)
    return total


def _slim_side(side: dict[str, Any]) -> dict[str, Any]:
    """精简单侧指标（不保留完整 history）。"""
    history = side.get("history")
    return {
        "final_accuracy": _as_float(side.get("final_accuracy"), 0.0),
        "peak_accuracy": _as_float(side.get("peak_accuracy"), 0.0),
        "total_offloaded": _as_int(side.get("total_offloaded"), 0),
        "total_delay_slots": _as_float(
            side.get("total_delay_slots"),
            _history_sum(side, "total_delay"),
        ),
        "elapsed_sec": round(_as_float(side.get("elapsed_sec"), 0.0), 3),
        "rounds": len(history) if isinstance(history, list) else 0,
    }


def _slim_summary(summary: dict[str, Any], deltas: dict[str, Any]) -> dict[str, Any]:
    """精简原始 comparison_summary，供前端直接展示。"""
    config = summary.get("config")
    slim: dict[str, Any] = {
        "config": dict(config) if isinstance(config, dict) else {},
        "offload_on": _slim_side(_side(summary, "offload_on", "fedleo")),
        "offload_off": _slim_side(_side(summary, "offload_off", "baseline")),
        "deltas_on_minus_off": dict(deltas),
    }
    gates = summary.get("gates")
    if isinstance(gates, dict):
        slim["gates"] = {str(key): bool(value) for key, value in gates.items()}
    if isinstance(summary.get("passed"), bool):
        slim["passed"] = summary["passed"]
    return slim


def extract_condition_result(
    condition: dict[str, Any],
    summary: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    """从 ``comparison_summary.json`` 提取单个 condition 的指标。

    兼容两种 summary 命名：``offload_on`` / ``offload_off``（校验脚本产物）与
    ``fedleo`` / ``baseline``（``run_fedleo_vs_baseline`` 产物）；缺失的
    ``total_offloaded`` / ``total_delay_slots`` 由 history 累加兜底。

    Parameters
    ----------
    condition : dict
        plan 中的一个条件。
    summary : dict
        解析后的 comparison_summary.json。
    elapsed : float
        该条件实测耗时（秒）。

    Returns
    -------
    dict
        逐条件结果（status 固定为 DONE）。
    """
    data = summary if isinstance(summary, dict) else {}
    on = _side(data, "offload_on", "fedleo")
    off = _side(data, "offload_off", "baseline")
    raw_deltas = data.get("deltas_on_minus_off")
    deltas = dict(raw_deltas) if isinstance(raw_deltas, dict) else {}

    final_accuracy = _as_float(on.get("final_accuracy"), 0.0)
    peak_accuracy = _as_float(on.get("peak_accuracy"), 0.0)
    baseline_accuracy = _as_float(off.get("final_accuracy"), 0.0)
    accuracy_delta = _as_float(deltas.get("final_accuracy"), final_accuracy - baseline_accuracy)
    total_offloaded = _as_int(
        on.get("total_offloaded"),
        int(_history_sum(on, "total_offloaded_samples")),
    )
    total_delay_slots = _as_float(
        on.get("total_delay_slots"),
        _history_sum(on, "total_delay"),
    )

    if isinstance(data.get("passed"), bool):
        passed = bool(data["passed"])
    elif isinstance(data.get("gates"), dict) and data["gates"]:
        passed = all(bool(value) for value in data["gates"].values())
    else:
        passed = bool(final_accuracy > 0.0 and accuracy_delta >= 0.0)

    return {
        "id": str(condition.get("id") or ""),
        "name": str(condition.get("name") or condition.get("id") or ""),
        "status": STATUS_DONE,
        "elapsed_sec": round(float(elapsed), 3),
        "final_accuracy": round(final_accuracy, 6),
        "peak_accuracy": round(peak_accuracy, 6),
        "baseline_accuracy": round(baseline_accuracy, 6),
        "accuracy_delta": round(accuracy_delta, 6),
        "total_offloaded": int(total_offloaded),
        "total_delay_slots": round(total_delay_slots, 3),
        "passed": passed,
        "summary": _slim_summary(data, deltas),
    }


def summarize_conditions(results: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    """汇总所有 condition 的关键指标（只统计 DONE 且有数值结果的条件）。

    Parameters
    ----------
    results : list
        逐条件结果列表。
    elapsed : float
        整个计划的总耗时（秒）。

    Returns
    -------
    dict
        ``mean_final_accuracy`` / ``max_final_accuracy`` / ``mean_delta_accuracy`` /
        ``best_condition``（最佳条件的 id）等汇总字段。
    """
    done = [
        item
        for item in results
        if isinstance(item, dict)
        and item.get("status") == STATUS_DONE
        and _is_number(item.get("final_accuracy"))
    ]
    accuracies = [_as_float(item.get("final_accuracy"), 0.0) for item in done]
    deltas = [_as_float(item.get("accuracy_delta"), 0.0) for item in done]
    best = max(done, key=lambda item: _as_float(item.get("final_accuracy"), 0.0)) if done else {}
    return {
        "mean_final_accuracy": round(sum(accuracies) / len(accuracies), 6) if accuracies else 0.0,
        "max_final_accuracy": round(max(accuracies), 6) if accuracies else 0.0,
        "mean_delta_accuracy": round(sum(deltas) / len(deltas), 6) if deltas else 0.0,
        "best_condition": str(best.get("id") or ""),
        "best_condition_name": str(best.get("name") or ""),
        "total_elapsed_sec": round(float(elapsed), 3),
    }


# ---------------------------------------------------------------------------
# 逐条件执行
# ---------------------------------------------------------------------------


def _base_result(condition: dict[str, Any], status: str, elapsed: float) -> dict[str, Any]:
    """构造逐条件结果骨架。"""
    return {
        "id": str(condition.get("id") or ""),
        "name": str(condition.get("name") or condition.get("id") or ""),
        "status": status,
        "elapsed_sec": round(float(elapsed), 3),
        "final_accuracy": None,
        "peak_accuracy": None,
        "baseline_accuracy": None,
        "accuracy_delta": None,
        "total_offloaded": 0,
        "total_delay_slots": 0.0,
        "passed": False,
        "summary": {},
    }


def _failed_result(
    condition: dict[str, Any],
    error: BaseException,
    elapsed: float,
) -> dict[str, Any]:
    """构造失败结果。"""
    result = _base_result(condition, STATUS_FAILED, elapsed)
    result["error"] = _short_error(error)
    return result


def _dry_run_result(
    condition: dict[str, Any],
    kwargs: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    """构造 dry-run 占位结果（不产生指标，只记录将执行的参数）。"""
    result = _base_result(condition, STATUS_DONE, elapsed)
    result["dry_run"] = True
    result["summary"] = {"dry_run": True, "kwargs": kwargs}
    return result


def _progress_entry(result: dict[str, Any]) -> dict[str, Any]:
    """把逐条件结果压成 progress.json 中的条目。"""
    return {
        "id": result.get("id", ""),
        "name": result.get("name", ""),
        "status": result.get("status", STATUS_PENDING),
        "elapsed_sec": result.get("elapsed_sec", 0.0),
        "final_accuracy": result.get("final_accuracy"),
        "accuracy_delta": result.get("accuracy_delta"),
        "output": result.get("output", ""),
        "error": result.get("error", ""),
    }


def _execute_condition(
    condition: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    """真正运行一个 condition，返回（逐条件结果, 捕获的日志片段）。

    训练过程的 stdout/stderr 会被重定向进缓冲区，因此失败时也能带回日志尾部。
    """
    # 延迟导入：保证 ``--dry-run`` 与单元测试不需要 PyTorch。
    from fl_space.fedleo.experiment import run_fedleo_vs_baseline

    target = Path(output_dir)
    kwargs = build_run_kwargs(condition, target)
    buffer = io.StringIO()
    error: BaseException | None = None
    started = time.monotonic()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            run_fedleo_vs_baseline(**kwargs)
    except Exception as exc:
        error = exc
    elapsed = time.monotonic() - started
    captured = buffer.getvalue()
    if error is not None:
        return _failed_result(condition, error, elapsed), captured

    summary_path = target / SUMMARY_FILENAME
    if not summary_path.is_file():
        missing = RuntimeError(f"未生成对比结果文件：{summary_path}")
        return _failed_result(condition, missing, elapsed), captured
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        invalid = RuntimeError(f"对比结果文件不是合法 JSON：{exc}")
        return _failed_result(condition, invalid, elapsed), captured
    if not isinstance(summary, dict):
        return _failed_result(condition, RuntimeError("对比结果不是 JSON 对象"), elapsed), captured
    return extract_condition_result(condition, summary, elapsed), captured


def execute_plan(
    plan: dict[str, Any],
    output_dir: Path,
    *,
    dry_run: bool = False,
    condition_timeout_sec: int = 1800,
) -> int:
    """执行计划中的全部 condition，持续写 progress.json，最后写 research_result.json。

    Returns
    -------
    int
        退出码：0 至少一个条件成功；1 全部失败；2 被用户中断。
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    conditions = list(plan.get("conditions") or [])
    total = len(conditions)
    results: list[dict[str, Any]] = []
    log_text = ""
    started_at = time.monotonic()
    current_id = ""

    def _snapshot(status: str, done: int, message: str) -> dict[str, Any]:
        """生成一份 progress.json 内容。"""
        return {
            "status": status,
            "progress": round(done / total, 4) if total else 0.0,
            "current": current_id,
            "total": total,
            "message": message,
            "dry_run": bool(dry_run),
            "condition_timeout_sec": int(condition_timeout_sec),
            "conditions": [_progress_entry(item) for item in results],
            "logs": _log_lines(log_text),
            "updated_at": _now_iso(),
        }

    def _flush(status: str, done: int, message: str) -> None:
        """写一次 progress.json。"""
        _write_json_atomic(target_dir / PROGRESS_FILENAME, _snapshot(status, done, message))

    try:
        _flush(STATUS_RUNNING, 0, "开始执行研究计划")
        for index, condition in enumerate(conditions, 1):
            current_id = str(condition.get("id") or "")
            name = str(condition.get("name") or current_id)
            condition_dir = target_dir / _safe_name(current_id)
            _flush(STATUS_RUNNING, index - 1, f"正在运行 条件 {index}/{total}：{name}")
            condition_started = time.monotonic()
            try:
                if dry_run:
                    kwargs = build_run_kwargs(condition, condition_dir)
                    text = (
                        f"[dry-run] 条件 {index}/{total} {current_id} {name}\n"
                        f"{json.dumps(kwargs, ensure_ascii=False, indent=2)}\n"
                    )
                    print(text, end="")
                    log_text = _append_log(log_text, text)
                    result = _dry_run_result(
                        condition,
                        kwargs,
                        time.monotonic() - condition_started,
                    )
                else:
                    result, captured = _execute_condition(condition, condition_dir)
                    log_text = _append_log(log_text, captured)
            except Exception as exc:
                result = _failed_result(condition, exc, time.monotonic() - condition_started)
            result["output"] = _display_path(condition_dir)
            results.append(result)
            _flush(STATUS_RUNNING, index, f"条件 {index}/{total} 结束：{name}")
    except KeyboardInterrupt:
        _flush(STATUS_CANCELLED, len(results), "收到中断信号，任务已取消")
        return 2

    elapsed_total = time.monotonic() - started_at
    succeeded = [item for item in results if item.get("status") == STATUS_DONE]
    summary = summarize_conditions(results, elapsed_total)
    _write_json_atomic(
        target_dir / RESULT_FILENAME,
        {
            "plan_id": str(plan.get("id") or ""),
            "title": str(plan.get("title") or ""),
            "backend": str(plan.get("backend") or "fedleo"),
            "generated_at": _now_iso(),
            "dry_run": bool(dry_run),
            "plan": plan,
            "conditions": results,
            "summary": summary,
        },
    )
    status = STATUS_DONE if succeeded else STATUS_FAILED
    _flush(
        status,
        total,
        f"全部条件执行结束：成功 {len(succeeded)}/{total}",
    )
    return 0 if succeeded else 1


def _write_failure_progress(output_dir: Path, error: str) -> None:
    """计划文件无效时也写一份 progress.json，方便 Web 端感知失败。"""
    with contextlib.suppress(OSError):
        _write_json_atomic(
            Path(output_dir) / PROGRESS_FILENAME,
            {
                "status": STATUS_FAILED,
                "progress": 0.0,
                "current": "",
                "total": 0,
                "message": "计划文件无效",
                "conditions": [],
                "logs": [error],
                "error": error,
                "updated_at": _now_iso(),
            },
        )


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """命令行入口，返回进程退出码。"""
    parser = argparse.ArgumentParser(
        description="SpaceFL 科研智能平台：后台批量实验执行器",
    )
    parser.add_argument("--plan", required=True, help="研究计划 JSON 路径")
    parser.add_argument(
        "--output",
        required=True,
        help="输出目录（写入 progress.json / research_result.json）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将执行的参数并写 progress，不真正训练",
    )
    parser.add_argument(
        "--condition-timeout-sec",
        type=int,
        default=1800,
        help="单条件超时（秒）；当前仅记录，不强制中断",
    )
    args = parser.parse_args(argv)

    if args.condition_timeout_sec != 1800:
        print(
            f"提示：--condition-timeout-sec={args.condition_timeout_sec} 当前不强制生效（仅记录）",
            file=sys.stderr,
        )

    try:
        plan = load_plan(Path(args.plan))
    except ValueError as exc:
        message = _short_error(exc)
        _write_failure_progress(Path(args.output), message)
        print(f"计划文件无效：{message}", file=sys.stderr)
        return 1

    try:
        return execute_plan(
            plan,
            Path(args.output),
            dry_run=bool(args.dry_run),
            condition_timeout_sec=int(args.condition_timeout_sec),
        )
    except KeyboardInterrupt:
        print("任务已取消（KeyboardInterrupt）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
