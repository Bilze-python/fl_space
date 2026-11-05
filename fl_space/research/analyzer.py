"""实验计划评审、实验结果分析与文献精读。

所有函数都遵循同一原则：优先调用大模型，失败或无 Key 时使用确定性规则生成
结构化结论，返回体用 ``mode`` 标注 ``llm`` / ``offline``，并在报告里显式写出
「不能支持的结论」，避免把轻量仿真结果说成论文级复现。
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import re
import statistics
from typing import Any

from fl_space.research import bilingual
from fl_space.research.config import DEFAULT_GOAL

# ── 基础工具 ──────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 2, percent: bool = False) -> str:
    number = _as_float(value)
    if number is None:
        return "--"
    if percent:
        return f"{number * 100:.{digits}f}%"
    return f"{number:.{digits}f}"


def _series(history: Any, key: str) -> list[float]:
    values: list[float] = []
    for row in history if isinstance(history, list) else []:
        if not isinstance(row, dict):
            continue
        number = _as_float(row.get(key))
        if number is not None:
            values.append(number)
    return values


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) >= 2 else 0.0


def _tail(values: list[float], ratio: float = 0.25) -> list[float]:
    if not values:
        return []
    size = max(1, min(len(values), round(len(values) * ratio)))
    return values[-size:]


def series_metrics(history: Any, *, label: str = "") -> dict[str, Any]:
    """把逐轮 history 汇总为指标（兼容 FedLEO 与简易基线两种行结构）。"""
    rounds = [row for row in history if isinstance(row, dict)] if isinstance(history, list) else []
    accuracy = _series(rounds, "accuracy")
    loss = _series(rounds, "train_loss") or _series(rounds, "loss")
    delay = _series(rounds, "total_delay") or _series(rounds, "total_delay_slots")
    offloaded = _series(rounds, "total_offloaded_samples") or _series(
        rounds, "total_offloaded"
    )
    entropy = _series(rounds, "data_balance_entropy")
    sizes = [row.get("data_sizes") for row in rounds if isinstance(row.get("data_sizes"), list)]
    samples_conserved = bool(sizes) and all(
        sum(int(item) for item in row) == sum(int(item) for item in sizes[0]) for row in sizes
    )
    metrics: dict[str, Any] = {
        "label": label or "run",
        "rounds": len(rounds),
        "initial_accuracy": accuracy[0] if accuracy else None,
        "final_accuracy": accuracy[-1] if accuracy else None,
        "peak_accuracy": max(accuracy) if accuracy else None,
        "accuracy_improvement": (accuracy[-1] - accuracy[0]) if len(accuracy) >= 2 else None,
        "accuracy_gain": (max(accuracy) - accuracy[0]) if accuracy else None,
        "tail_accuracy_mean": statistics.fmean(_tail(accuracy)) if accuracy else None,
        "tail_accuracy_std": _std(_tail(accuracy)),
        "convergence_round": None,
        "final_train_loss": loss[-1] if loss else None,
        "train_loss_drop": (loss[0] - loss[-1]) if len(loss) >= 2 else None,
        "total_delay_slots": max(delay) if delay else None,
        "total_offloaded": max(offloaded) if offloaded else None,
        "offload_rounds": sum(1 for value in offloaded if value > 0),
        "initial_balance": entropy[0] if entropy else None,
        "final_balance": entropy[-1] if entropy else None,
        "samples_conserved": samples_conserved,
    }
    if accuracy and len(accuracy) >= 2:
        threshold = accuracy[-1] - 0.02 * max(abs(accuracy[-1]), 1e-6)
        for index, value in enumerate(accuracy):
            if value >= threshold:
                metrics["convergence_round"] = index
                break
    return metrics


def _metric_delta(on: dict[str, Any], off: dict[str, Any], key: str) -> float | None:
    on_value = _as_float(on.get(key))
    off_value = _as_float(off.get(key))
    if on_value is None or off_value is None:
        return None
    return on_value - off_value


def _samples_conserved(raw: dict[str, Any], on: dict[str, Any]) -> Any:
    """优先使用汇总文件里的样本守恒结论，缺失时回退到 history 推导。"""
    block = raw.get("offload_on") if isinstance(raw.get("offload_on"), dict) else {}
    value = block.get("samples_conserved")
    if isinstance(value, bool):
        return value
    return on.get("samples_conserved")


def experiment_metrics(detail: dict[str, Any]) -> dict[str, Any]:
    """从 web 实验详情中提取对照指标。"""
    raw = detail.get("raw") if isinstance(detail.get("raw"), dict) else {}
    history_on = detail.get("history_on") or []
    history_off = detail.get("history_off") or []
    on = series_metrics(history_on, label="卸载开启")
    off = series_metrics(history_off, label="卸载关闭")
    config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
    gates = raw.get("gates") if isinstance(raw.get("gates"), dict) else {}
    deltas = (
        raw.get("deltas_on_minus_off") if isinstance(raw.get("deltas_on_minus_off"), dict) else {}
    )
    # 汇总文件里已有更权威的统计时优先使用
    for prefix, target in (("offload_on", on), ("offload_off", off)):
        block = raw.get(prefix) if isinstance(raw.get(prefix), dict) else {}
        for key in (
            "final_accuracy",
            "peak_accuracy",
            "total_offloaded",
            "total_delay_slots",
            "initial_balance",
            "final_balance",
            "samples_conserved",
        ):
            if key in block and block[key] is not None:
                target[key] = block[key]
    return {
        "name": detail.get("name") or raw.get("experiment") or "实验档案",
        "dataset": detail.get("dataset") or config.get("dataset") or "mnist",
        "rounds": detail.get("rounds") or config.get("num_rounds"),
        "seed": detail.get("seed") or config.get("seed"),
        "config": config,
        "on": on,
        "off": off,
        "deltas": {
            "final_accuracy": _metric_delta(on, off, "final_accuracy"),
            "final_balance": _metric_delta(on, off, "final_balance"),
            "total_delay_slots": _metric_delta(on, off, "total_delay_slots"),
            "total_offloaded": _metric_delta(on, off, "total_offloaded"),
            **{
                str(key): value
                for key, value in deltas.items()
                if isinstance(value, (int, float))
            },
        },
        "gates": gates,
        "passed": raw.get("passed"),
        "samples_conserved": _samples_conserved(raw, on),
    }


# ── 引用格式化 ────────────────────────────────────────────────


def format_citations(citations: list[dict[str, Any]] | None) -> str:
    """把引用列表格式化为 markdown 列表。"""
    if not citations:
        return ""
    lines = []
    for item in citations:
        index = item.get("index")
        title = item.get("title") or "未知文献"
        page = item.get("page")
        page_text = f" p.{page}" if page else ""
        lines.append(f"[{index}] 《{title}》{page_text}")
    return "\n".join(lines)


def _citation_block(citations: list[dict[str, Any]] | None) -> str:
    text = format_citations(citations)
    if not text:
        return "> 本地文献库暂无可用引用；可在「文献库」中导入 PDF 后重试。"
    return text


# ── 离线实验分析 ──────────────────────────────────────────────


def _honesty_flags(metrics: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    rounds = metrics["on"].get("rounds") or metrics.get("rounds") or 0
    if isinstance(rounds, (int, float)) and rounds < 5:
        flags.append(f"当前仅 {int(rounds)} 轮评估，只能说明功能可用与趋势，不足以支撑收敛性结论。")
    if not metrics.get("samples_conserved"):
        flags.append("未检测到样本守恒证据，不能排除卸载过程改变数据分布的可能。")
    gates = metrics.get("gates") or {}
    if not gates:
        flags.append("该档案缺少验收门槛（gates）记录，无法自动核验对照公平性。")
    if metrics["deltas"].get("final_accuracy") is not None:
        delta = float(metrics["deltas"]["final_accuracy"])
        if abs(delta) < 0.01:
            flags.append("两组最终准确率差异小于 1 个百分点，在单 seed 下不具统计意义。")
        elif delta < 0:
            flags.append("本轮卸载组准确率低于对照组，不能据此宣称卸载带来准确率收益。")
    return flags


def _offline_experiment_report(
    metrics: dict[str, Any],
    *,
    question: str,
    goal: str,
    citations: list[dict[str, Any]] | None,
    note: str = "",
) -> str:
    on = metrics["on"]
    off = metrics["off"]
    delta = metrics["deltas"]
    lines: list[str] = []
    lines.append(f"## 实验解读 · {metrics['name']}")
    lines.append("")
    lines.append("### 一、结论摘要")
    summary: list[str] = []
    summary.append(
        f"卸载组最终准确率 {_fmt(on.get('final_accuracy'), 2, True)}，"
        f"对照组 {_fmt(off.get('final_accuracy'), 2, True)}，"
        f"差值 {_fmt(delta.get('final_accuracy'), 2, True)}。"
    )
    offloaded = _as_float(on.get("total_offloaded"))
    if offloaded is not None:
        summary.append(
            f"卸载组累计卸载 {offloaded:.0f} 个样本，发生在 {int(on.get('offload_rounds') or 0)} 轮；"
            f"数据均衡熵由 {_fmt(on.get('initial_balance'), 4)} 变为 {_fmt(on.get('final_balance'), 4)}。"
        )
    delay = _as_float(delta.get("total_delay_slots"))
    if delay is not None:
        summary.append(
            f"卸载组累计时延相对对照组变化 {delay:+.1f} slots，"
            "该数值来自 timeslot 级离散估算，只能同环境比较。"
        )
    lines.extend(f"- {item}" for item in summary)
    lines.append("")

    lines.append("### 二、关键指标对照")
    lines.append("")
    lines.append("| 指标 | 卸载开启 | 卸载关闭 | 差值 |")
    lines.append("| --- | --- | --- | --- |")
    rows = [
        ("最终准确率", "final_accuracy", True, 2),
        ("峰值准确率", "peak_accuracy", True, 2),
        ("尾部准确率均值", "tail_accuracy_mean", True, 2),
        ("尾部准确率标准差", "tail_accuracy_std", False, 3),
        ("数据均衡熵", "final_balance", False, 3),
        ("累计卸载样本", "total_offloaded", False, 0),
        ("累计时延 slots", "total_delay_slots", False, 1),
    ]
    for label, key, percent, digits in rows:
        on_value = on.get(key)
        off_value = off.get(key)
        if percent:
            cells = (
                _fmt(on_value, digits, True),
                _fmt(off_value, digits, True),
                _fmt(delta.get(key), digits, True),
            )
        else:
            cells = (
                _fmt(on_value, digits),
                _fmt(off_value, digits),
                _fmt(_metric_delta(on, off, key), digits),
            )
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    lines.append("### 三、证据链")
    evidence: list[str] = []
    if metrics.get("samples_conserved"):
        evidence.append("样本守恒：两份 history 的 data_sizes 总量保持不变，卸载未凭空增减样本。")
    if (on.get("offload_rounds") or 0) > 0:
        evidence.append("卸载真实发生：history 中存在 total_offloaded_samples > 0 的轮次。")
    if metrics.get("gates"):
        passed = [key for key, value in metrics["gates"].items() if value is True]
        evidence.append(f"验收门槛通过 {len(passed)} 项：" + "、".join(passed[:6]) + "。")
    if not evidence:
        evidence.append("缺少足够的结构化证据，仅能基于曲线本身做趋势判断。")
    lines.extend(f"- {item}" for item in evidence)
    lines.append("")

    if question:
        lines.append("### 四、针对问题的回答")
        lines.extend(f"- {item}" for item in _answer_by_keywords(question, metrics))
        lines.append("")

    lines.append("### 五、与文献结论的对照")
    lines.append(_citation_block(citations))
    if citations:
        lines.append("")
        lines.append(
            "上述文献提供的是概念与指标层面的对照依据；本轮结果只在本平台的轻量离散仿真实现内成立，"
            "不能直接视为对论文数值结论的复现。"
        )
    lines.append("")

    lines.append("### 六、与研究目标的对齐度")
    lines.append(f"- 当前研究目标：{goal or DEFAULT_GOAL}")
    lines.extend(f"- {item}" for item in _goal_alignment(metrics, goal))
    lines.append("")

    lines.append("### 七、尚不能支持的结论")
    flags = _honesty_flags(metrics)
    lines.extend(f"- {item}" for item in flags or ["当前证据不足以支持更强的结论。"])
    lines.append("")

    lines.append("### 八、下一步建议")
    lines.extend(f"- {item}" for item in _next_steps(metrics, question))
    if note:
        lines.append("")
        lines.append(f"> 档案备注：{note}")
    return "\n".join(lines)


def _answer_by_keywords(question: str, metrics: dict[str, Any]) -> list[str]:
    text = str(question or "")
    on = metrics["on"]
    off = metrics["off"]
    delta = metrics["deltas"]
    answers: list[str] = []
    if re.search(r"卸载|offload|卸载是否生效|生效", text, re.I):
        actions = on.get("offload_rounds") or 0
        samples = _as_float(on.get("total_offloaded")) or 0.0
        answers.append(
            f"卸载是否生效：累计卸载 {samples:.0f} 个样本、涉及 {int(actions)} 轮，"
            "说明卸载流程被真实触发；但要证明「有益」还需要准确率/时延的证据。"
        )
    if re.search(r"准确率|精度|准确|accuracy", text, re.I):
        answers.append(
            f"准确率：最终 {_fmt(on.get('final_accuracy'), 2, True)} vs "
            f"{_fmt(off.get('final_accuracy'), 2, True)}（差值 "
            f"{_fmt(delta.get('final_accuracy'), 2, True)}），"
            f"尾部波动标准差 {_fmt(on.get('tail_accuracy_std'), 4)}。"
        )
    if re.search(r"时延|延迟|delay|时间", text, re.I):
        answers.append(
            f"时延：累计时延 slots 差值为 {_fmt(delta.get('total_delay_slots'))}，"
            "属于 timeslot 级离散估算，不能等同于真实星上时延。"
        )
    if re.search(r"均衡|公平|fair|entropy|散度", text, re.I):
        answers.append(
            f"数据均衡：熵 {_fmt(on.get('initial_balance'), 4)} → "
            f"{_fmt(on.get('final_balance'), 4)}（卸载组），"
            f"对照 {_fmt(off.get('initial_balance'), 4)} → {_fmt(off.get('final_balance'), 4)}。"
        )
    if re.search(r"对照|公平|公平性|control|不能证明|局限|证明", text, re.I):
        answers.extend(f"对照公平性：{item}" for item in _honesty_flags(metrics))
    if not answers:
        answers.append(
            "问题未命中预置指标，可参考上方关键指标表；建议明确询问准确率、卸载量、时延或对照公平性。"
        )
    return answers


def _goal_alignment(metrics: dict[str, Any], goal: str) -> list[str]:
    text = str(goal or "")
    items: list[str] = []
    on = metrics["on"]
    delta = metrics["deltas"]
    if re.search(r"准确率|精度|收敛|accuracy", text, re.I):
        items.append(
            f"准确率/收敛：已有 {_fmt(on.get('final_accuracy'), 2, True)} 最终准确率与"
            f"第 {on.get('convergence_round')} 轮收敛点证据。"
        )
    if re.search(r"卸载|offload", text, re.I):
        items.append(
            f"卸载：累计卸载 {_fmt(on.get('total_offloaded'))} 样本，"
            f"准确率差值 {_fmt(delta.get('final_accuracy'), 2, True)}。"
        )
    if re.search(r"时延|延迟|通信开销|delay", text, re.I):
        items.append(
            f"时延/开销：累计时延 slots 差值 {_fmt(delta.get('total_delay_slots'))}，"
            "可作为通信开销的粗粒度代理指标。"
        )
    if re.search(r"均衡|非\s*IID|non.?iid|公平", text, re.I):
        items.append(
            f"数据均衡：均衡熵 {_fmt(on.get('initial_balance'), 4)} → "
            f"{_fmt(on.get('final_balance'), 4)}。"
        )
    if not items:
        items.append("研究目标中的关键词尚未映射到本轮采集的指标，建议在目标中写明关注指标。")
    return items


def _next_steps(metrics: dict[str, Any], question: str) -> list[str]:
    steps: list[str] = []
    on = metrics["on"]
    rounds = int(on.get("rounds") or 0)
    if rounds < 10:
        steps.append("把轮次提升到 10 轮以上，确认趋势是否稳定，再判断收敛差异。")
    steps.append("至少补 3 个随机种子并报告均值±标准差，避免把单次波动当成算法结论。")
    steps.append("固定其它条件，仅改变一个变量（如地面站数量），避免多因素混杂。")
    if re.search(r"时延|delay", str(question or ""), re.I):
        steps.append("补充通信时延分解（卸载/训练/聚合），并说明 timeslot 到真实秒的换算假设。")
    steps.append("在「文献库」中对相关文献做 AI 精读，把本轮指标与文献中的评价口径对齐。")
    return steps


# ── LLM 实验分析 ─────────────────────────────────────────────

_ANALYSIS_SYSTEM_PROMPT = (
    "你是 SpaceFL 卫星联邦学习科研平台的研究智能体。"
    "只根据给定的实验指标、验收门槛与文献片段作答，区分功能验证、趋势证据与论文级复现。"
    "必须指出对照是否公平、卸载是否真实发生、样本是否守恒，以及哪些结论当前不能支持。"
    "使用中文 Markdown，包含小标题；引用文献时使用 [编号] 的形式。"
    "不要编造数据中不存在的数字。"
)


def _call_llm(provider: Any, system: str, user: str) -> str | None:
    if provider is None or not getattr(provider, "configured", False):
        return None
    try:
        from fl_space.research.providers import chat

        content = chat(provider, system, user)
        return str(content or "").strip() or None
    except Exception:
        return None


def _provider_name(provider: Any) -> str:
    if provider is None or not getattr(provider, "configured", False):
        return "离线规则引擎"
    return str(getattr(provider, "provider", "") or getattr(provider, "model", "") or "大模型")


def analyze_experiment(
    detail: dict[str, Any],
    *,
    question: str = "",
    goal: str = "",
    context_text: str = "",
    citations: list[dict[str, Any]] | None = None,
    provider: Any = None,
) -> dict[str, Any]:
    """分析单个实验档案，返回可直接渲染的 Markdown 报告。"""
    metrics = experiment_metrics(detail or {})
    note = str((detail or {}).get("note") or "")
    citations = citations or []
    user_prompt = (
        f"研究目标：{goal or DEFAULT_GOAL}\n"
        f"用户问题：{question or '请给出结果解读'}\n"
        f"实验元信息：{json.dumps({k: metrics[k] for k in ('name', 'dataset', 'rounds', 'seed', 'passed')}, ensure_ascii=False)}\n"
        f"卸载组指标：{json.dumps(metrics['on'], ensure_ascii=False)}\n"
        f"对照组指标：{json.dumps(metrics['off'], ensure_ascii=False)}\n"
        f"差值：{json.dumps(metrics['deltas'], ensure_ascii=False)}\n"
        f"验收门槛：{json.dumps(metrics['gates'], ensure_ascii=False)}\n"
        f"文献片段：\n{context_text or '（暂无）'}\n"
        f"文献列表：\n{format_citations(citations) or '（暂无）'}"
    )
    content = _call_llm(provider, _ANALYSIS_SYSTEM_PROMPT, user_prompt)
    if content:
        return {
            "content": content,
            "metrics": metrics,
            "citations": citations,
            "mode": "llm",
            "provider": _provider_name(provider),
            "generated_at": _now_iso(),
        }
    return {
        "content": _offline_experiment_report(
            metrics,
            question=question,
            goal=goal,
            citations=citations,
            note=note,
        ),
        "metrics": metrics,
        "citations": citations,
        "mode": "offline",
        "provider": "离线规则引擎",
        "generated_at": _now_iso(),
    }


# ── 批量实验结果分析 ─────────────────────────────────────────


def analyze_batch_result(
    result: dict[str, Any],
    *,
    question: str = "",
    goal: str = "",
    context_text: str = "",
    citations: list[dict[str, Any]] | None = None,
    provider: Any = None,
) -> dict[str, Any]:
    """分析后台批量实验任务的结果文件（research_result.json）。"""
    plan = result.get("plan") if isinstance(result.get("plan"), dict) else {}
    conditions = result.get("conditions") if isinstance(result.get("conditions"), list) else []
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}

    def _condition_rows() -> list[dict[str, Any]]:
        rows = []
        for item in conditions:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "name": item.get("name") or item.get("id"),
                    "status": item.get("status"),
                    "final_accuracy": item.get("final_accuracy"),
                    "accuracy_delta": item.get("accuracy_delta"),
                    "total_offloaded": item.get("total_offloaded"),
                    "total_delay_slots": item.get("total_delay_slots"),
                    "elapsed_sec": item.get("elapsed_sec"),
                }
            )
        return rows

    rows = _condition_rows()
    metrics = {"conditions": rows, "summary": summary, "plan": plan}
    offload_deltas = [
        _as_float(row.get("accuracy_delta"))
        for row in rows
        if _as_float(row.get("accuracy_delta")) is not None
    ]
    positive = sum(1 for value in offload_deltas if value > 0)
    user_prompt = (
        f"研究目标：{goal or DEFAULT_GOAL}\n"
        f"用户问题：{question or '请解读这批实验结果'}\n"
        f"计划：{json.dumps(plan, ensure_ascii=False)}\n"
        f"逐条件结果：{json.dumps(rows, ensure_ascii=False)}\n"
        f"汇总：{json.dumps(summary, ensure_ascii=False)}\n"
        f"文献片段：\n{context_text or '（暂无）'}\n"
        f"文献列表：\n{format_citations(citations) or '（暂无）'}"
    )
    content = _call_llm(provider, _ANALYSIS_SYSTEM_PROMPT, user_prompt)
    if content:
        return {
            "content": content,
            "metrics": metrics,
            "citations": citations or [],
            "mode": "llm",
            "provider": _provider_name(provider),
            "generated_at": _now_iso(),
        }

    lines: list[str] = []
    lines.append(f"## 批量实验解读 · {plan.get('title') or result.get('title') or '自定义计划'}")
    lines.append("")
    lines.append("### 一、结论摘要")
    if rows:
        lines.append(
            f"- 本批共 {len(rows)} 个条件，成功 {sum(1 for r in rows if r['status'] == 'DONE')} 个；"
            f"卸载组相对对照组的准确率差值为正的有 {positive}/{len(offload_deltas)} 个条件。"
        )
        best = max(
            (row for row in rows if _as_float(row.get("final_accuracy")) is not None),
            key=lambda row: _as_float(row.get("final_accuracy")) or -1,
            default=None,
        )
        if best:
            lines.append(
                f"- 最高最终准确率出现在「{best['name']}」："
                f"{_fmt(best.get('final_accuracy'), 2, True)}。"
            )
        if summary.get("mean_delta_accuracy") is not None:
            lines.append(
                f"- 平均准确率差值 {_fmt(summary.get('mean_delta_accuracy'), 2, True)}，"
                f"平均最终准确率 {_fmt(summary.get('mean_final_accuracy'), 2, True)}。"
            )
    else:
        lines.append("- 结果文件中没有可用的条件记录。")
    lines.append("")
    lines.append("### 二、逐条件结果")
    lines.append("")
    lines.append("| 条件 | 状态 | 最终准确率 | 准确率差值 | 累计卸载 | 时延 slots | 耗时 s |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    lines.extend(
        f"| {row['name']} | {row['status']} | {_fmt(row.get('final_accuracy'), 2, True)} | "
        f"{_fmt(row.get('accuracy_delta'), 2, True)} | {_fmt(row.get('total_offloaded'))} | "
        f"{_fmt(row.get('total_delay_slots'), 1)} | {_fmt(row.get('elapsed_sec'), 1)} |"
        for row in rows
    )
    lines.append("")
    lines.append("### 三、与研究目标的对齐度")
    lines.append(f"- 研究目标：{goal or DEFAULT_GOAL}")
    if plan.get("hypothesis"):
        lines.append(f"- 计划假设：{plan['hypothesis']}")
    lines.append(f"- 计划预计耗时 {plan.get('expected_duration_sec', '--')} 秒，实际批次耗时见上表。")
    lines.append("")
    lines.append("### 四、尚不能支持的结论")
    honesty: list[str] = []
    if len(rows) < 3:
        honesty.append("条件数量偏少，无法做规模-性能趋势拟合。")
    if rows and all(not (row.get("accuracy_delta") or 0) > 0 for row in rows):
        honesty.append("所有条件下卸载组均未取得更高准确率，当前不能宣称卸载带来准确率收益。")
    honesty.append("每条件默认单 seed，且轮次有限，结论属于趋势证据而非论文级复现。")
    lines.extend(f"- {item}" for item in honesty)
    lines.append("")
    lines.append("### 五、文献对照")
    lines.append(_citation_block(citations))
    lines.append("")
    lines.append("### 六、下一步建议")
    lines.extend(
        f"- {item}"
        for item in (
            "对表现最好的条件补充多个随机种子，报告均值与标准差。",
            "保持其它变量不变，单独扫描地面站数量或卫星规模，定位主要影响因子。",
            "把本批结果登记为实验档案，便于后续与文献指标统一口径对照。",
        )
    )
    return {
        "content": "\n".join(lines),
        "metrics": metrics,
        "citations": citations or [],
        "mode": "offline",
        "provider": "离线规则引擎",
        "generated_at": _now_iso(),
    }


# ── 计划评审 ─────────────────────────────────────────────────


def review_plan(
    plan: dict[str, Any],
    *,
    goal: str = "",
    context_text: str = "",
    citations: list[dict[str, Any]] | None = None,
    provider: Any = None,
) -> dict[str, Any]:
    """评审实验计划设计（变量、对照、统计效力、可执行性）。"""
    conditions = plan.get("conditions") if isinstance(plan.get("conditions"), list) else []
    variables = plan.get("independent_variables") or []
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    add(
        "条件数量",
        len(conditions) >= 2,
        f"共 {len(conditions)} 个条件；单条件只能做功能验证，多条件才可比较规模/参数效应。",
    )
    varying = [
        item
        for item in variables
        if isinstance(item, dict) and len(item.get("values") or []) >= 2
    ]
    add(
        "变量是否真实变化",
        bool(varying),
        "至少需要一个自变量在条件间变化：" + ("、".join(str(v.get("label")) for v in varying) if varying else "未检测到"),
    )
    rounds = [_as_float((c.get("params") or {}).get("rounds")) for c in conditions]
    min_rounds = min((value for value in rounds if value is not None), default=0)
    add(
        "训练轮次",
        min_rounds >= 10,
        f"最小轮次 {min_rounds:.0f}；少于 10 轮时曲线趋势不稳定，建议 10 轮以上。",
    )
    seeds = {int((c.get("params") or {}).get("seed", 0)) for c in conditions if isinstance(c, dict)}
    add(
        "随机种子",
        len(seeds) >= 2,
        f"当前涉及 {len(seeds)} 个种子；单 seed 无法估计方差，建议每配置 ≥3 次重复。",
    )
    controls = plan.get("controls") or []
    add("对照设置", len(controls) >= 2, f"已列出 {len(controls)} 条控制条件。")
    add(
        "指标覆盖",
        len(plan.get("metrics") or []) >= 3,
        "需要同时报告准确率、卸载量与时延，避免单一指标误导。",
    )
    total_seconds = plan.get("expected_duration_sec")
    add(
        "可执行性",
        isinstance(total_seconds, (int, float)) and 0 < total_seconds <= 3600,
        f"预计耗时 {total_seconds} 秒；超过 1 小时建议拆分批次。",
    )

    score = round(100 * sum(1 for item in checks if item["ok"]) / max(len(checks), 1))

    prompt = (
        f"研究目标：{goal or DEFAULT_GOAL}\n"
        f"实验计划：{json.dumps(plan, ensure_ascii=False)}\n"
        f"结构化评审：{json.dumps(checks, ensure_ascii=False)}\n"
        f"文献片段：\n{context_text or '（暂无）'}\n"
        f"文献列表：\n{format_citations(citations) or '（暂无）'}"
    )
    content = _call_llm(
        provider,
        _ANALYSIS_SYSTEM_PROMPT
        + " 本次任务是评审实验计划设计：从变量、对照、统计效力、可执行性四方面给出意见。",
        prompt,
    )
    if content:
        return {
            "content": content,
            "metrics": {"checks": checks, "score": score},
            "citations": citations or [],
            "mode": "llm",
            "provider": _provider_name(provider),
            "generated_at": _now_iso(),
        }

    lines = ["## 实验计划评审", ""]
    lines.append(f"综合评分：**{score}/100**（{sum(1 for c in checks if c['ok'])}/{len(checks)} 项通过）")
    lines.append("")
    lines.append("### 检查项")
    for item in checks:
        mark = "通过" if item["ok"] else "待改进"
        lines.append(f"- **{item['name']}**（{mark}）：{item['detail']}")
    lines.append("")
    if varying:
        lines.append("### 变量设计")
        lines.extend(
            f"- {item.get('label')}：{', '.join(str(v) for v in item.get('values') or [])}"
            for item in varying
        )
        lines.append("")
    lines.append("### 改进建议")
    lines.extend(f"- {item['name']}：{item['detail']}" for item in checks if not item["ok"])
    if len(seeds) < 2:
        lines.append("- 为每个条件补 3 个随机种子，报告中给出均值±标准差。")
    if min_rounds < 10:
        lines.append("- 轮次提升到 10 轮以上，或明确标注结论仅为功能验证。")
    lines.append("")
    lines.append("### 文献对照")
    lines.append(_citation_block(citations))
    lines.append("")
    lines.append("> 该评分为本平台的结构化启发式评分，用于发现设计缺口，不替代同行评议。")
    return {
        "content": "\n".join(lines),
        "metrics": {"checks": checks, "score": score},
        "citations": citations or [],
        "mode": "offline",
        "provider": "离线规则引擎",
        "generated_at": _now_iso(),
    }


# ── 文献精读 ─────────────────────────────────────────────────


def _flatten_text(text: str) -> str:
    """把 PDF 硬换行合并成句子级文本，避免贡献句被行尾截断。"""
    flattened = re.sub(r"[ \t]*\n[ \t]*", " ", str(text or ""))
    return re.sub(r"\s{2,}", " ", flattened)


_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.])\s+(?=[A-Z(])")


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for block in str(text or "").splitlines():
        for sentence in _SENTENCE_SPLIT.split(block):
            cleaned = sentence.strip()
            if len(cleaned) >= 12:
                sentences.append(cleaned)
    return sentences


def _pick_sentences(sentences: list[str], patterns: str, limit: int = 3) -> list[str]:
    regex = re.compile(patterns, re.I)
    picked: list[str] = []
    for sentence in sentences:
        if regex.search(sentence) and sentence not in picked:
            picked.append(sentence)
        if len(picked) >= limit:
            break
    return picked


def _goal_relevance(doc: dict[str, Any], goal: str) -> dict[str, Any]:
    try:
        from fl_space.research.literature import tokenize
    except Exception:
        tokenize = None  # type: ignore[assignment]

    def terms(text: str) -> set[str]:
        if tokenize is not None:
            return set(tokenize(text))
        return set(re.findall(r"[a-zA-Z]{3,}|[\u4e00-\u9fff]{2,4}", str(text).lower()))

    goal_terms = terms(bilingual.expand_text(goal or DEFAULT_GOAL))
    doc_terms = terms(
        " ".join(
            [
                str(doc.get("title") or ""),
                str(doc.get("abstract") or ""),
                " ".join(str(item) for item in (doc.get("keywords") or [])),
            ]
        )
    )
    if not goal_terms:
        return {"score": 0, "reason": "研究目标为空，无法评估相关度。", "goal": goal}
    overlap = goal_terms & doc_terms
    score = min(100, round(100 * len(overlap) / max(len(goal_terms), 1) * 2))
    if overlap:
        reason = "与目标共享关键词：" + "、".join(sorted(overlap)[:12])
    else:
        reason = "标题与摘要未命中研究目标关键词，建议人工判断是否相关。"
    return {"score": score, "reason": reason, "goal": goal}


def digest_document(
    doc: dict[str, Any],
    pages: list[dict[str, Any]],
    *,
    goal: str = "",
    provider: Any = None,
) -> dict[str, Any]:
    """生成文献 AI 精读（离线时抽取摘要/贡献/局限并评估与目标相关度）。"""
    full_text = "\n".join(str(page.get("text") or "") for page in pages)[:20000]
    abstract = str(doc.get("abstract") or "").strip()
    sentences = _split_sentences(_flatten_text(full_text))
    contributions = _pick_sentences(
        sentences,
        r"we propose|we present|we introduce|this paper|本文|提出|贡献|contribution|novel",
        limit=4,
    )
    findings = _pick_sentences(
        sentences,
        r"results? show|experiments?|evaluation|表明|结果显示|实验验证|achiev|improv",
        limit=4,
    )
    limitations = _pick_sentences(
        sentences,
        r"limitation|future work|局限|不足|未来工作|further research|remain",
        limit=3,
    )
    relevance = _goal_relevance(doc, goal)
    summary = abstract or (sentences[0] if sentences else "该文献尚未抽取到摘要。")
    quotes = [{"page": 1, "text": sentence} for sentence in (abstract.split("。")[:2] if abstract else []) if sentence.strip()]
    offline = {
        "summary": summary[:1200],
        "contributions": contributions or ["未从全文中自动识别出显式贡献句，请阅读原文。"],
        "method": (contributions[0] if contributions else "")[:600],
        "findings": findings,
        "limitations": limitations or ["原文未显式声明局限，需人工判断。"],
        "quotes": quotes[:3],
        "relevance": relevance,
    }

    prompt = (
        f"研究目标：{goal or DEFAULT_GOAL}\n"
        f"文献标题：{doc.get('title')}\n"
        f"作者：{doc.get('authors')}\n"
        f"出处：{doc.get('venue')} {doc.get('year') or ''}\n"
        f"摘要：{abstract or '（无）'}\n"
        f"全文节选：\n{full_text[:12000]}"
    )
    content = _call_llm(
        provider,
        _ANALYSIS_SYSTEM_PROMPT + " 本次任务是对单篇文献做结构化精读，输出 JSON。",
        prompt + "\n输出 JSON 字段：summary, contributions[], method, findings[], limitations[], quotes[], relevance_score, relevance_reason。",
    )
    if content:
        return {
            "doc_id": doc.get("id"),
            "mode": "llm",
            "provider": _provider_name(provider),
            "generated_at": _now_iso(),
            **{key: offline[key] for key in ("summary", "contributions", "method", "findings", "limitations", "quotes", "relevance")},
            "llm_notes": content,
        }
    return {
        "doc_id": doc.get("id"),
        "mode": "offline",
        "provider": "离线规则引擎",
        "generated_at": _now_iso(),
        **offline,
    }


def answer_literature_question(
    question: str,
    *,
    context_text: str,
    citations: list[dict[str, Any]] | None,
    goal: str = "",
    provider: Any = None,
) -> dict[str, Any]:
    """基于文献库检索片段回答研究问题。"""
    citations = citations or []
    content = _call_llm(
        provider,
        "你是文献知识库问答助手。只使用给定片段回答，逐条给出 [编号] 引用；"
        "若片段不足以回答，请明确说明并给出检索建议。使用中文。",
        f"研究目标：{goal or DEFAULT_GOAL}\n问题：{question}\n文献片段：\n{context_text}",
    )
    if content:
        return {
            "answer": content,
            "citations": citations,
            "mode": "llm",
            "provider": _provider_name(provider),
        }
    lines = [f"根据本地文献库检索到 {len(citations)} 条相关片段：", ""]
    if citations:
        for item in citations:
            quote = str(item.get("quote") or "").strip()
            lines.append(
                f"**[{item.get('index')}] 《{item.get('title')}》"
                f"{' p.' + str(item.get('page')) if item.get('page') else ''}**"
            )
            if quote:
                lines.append(f"> {quote[:320]}")
            lines.append("")
        lines.append(
            "以上为检索式回答：未配置大模型时无法做跨文献归纳，"
            "请结合原文阅读；在「研究助手」中配置 API 后可获得综合结论。"
        )
    else:
        lines.append("文献库中没有命中片段。请尝试更短的关键词，或先在「文献库」导入 PDF。")
    return {
        "answer": "\n".join(lines),
        "citations": citations,
        "mode": "offline",
        "provider": "离线规则引擎",
    }


def intel_brief(
    items: list[dict[str, Any]],
    *,
    goal: str = "",
    topics: list[str] | None = None,
    provider: Any = None,
) -> dict[str, Any]:
    """把前沿资讯汇总成与研究方向对齐的简报。"""
    items = [item for item in items if isinstance(item, dict)][:12]
    compact = [
        {
            "title": item.get("title"),
            "source": item.get("source"),
            "published": item.get("published"),
            "summary": str(item.get("summary") or "")[:400],
        }
        for item in items
    ]
    content = _call_llm(
        provider,
        "你是航天与联邦学习领域的科研情报分析助手。"
        "将给定资讯按与研究目标的相关度排序，输出 3~5 条要点、每条附来源标题，"
        "并给出 2 条可跟进的研究机会。使用中文 Markdown。",
        f"研究目标：{goal or DEFAULT_GOAL}\n订阅主题：{topics or []}\n"
        f"资讯：{json.dumps(compact, ensure_ascii=False)}",
    )
    if content:
        return {"content": content, "mode": "llm", "provider": _provider_name(provider)}
    lines = ["## 情报简报", ""]
    lines.append(f"**研究目标**：{goal or DEFAULT_GOAL}")
    if topics:
        lines.append(f"**订阅主题**：{'、'.join(str(item) for item in topics)}")
    lines.append("")
    if not items:
        lines.append("暂无情报数据，请先在「前沿情报」页刷新。")
    else:
        lines.append("### 与研究方向最相关的情报")
        for item in items[:5]:
            lines.append(
                f"- **{item.get('title')}**（{item.get('source')}，"
                f"{str(item.get('published') or '')[:10]}）"
            )
            summary = str(item.get("summary") or "").strip()
            if summary:
                lines.append(f"  - {summary[:180]}")
        lines.append("")
        lines.append("### 可跟进的研究机会")
        lines.append("- 把新发表的方法作为对照组，评审现有卸载/聚合策略的改进空间。")
        lines.append("- 关注新数据集与硬件假设，检查本平台配置是否仍具代表性。")
    return {"content": "\n".join(lines), "mode": "offline", "provider": "离线规则引擎"}
