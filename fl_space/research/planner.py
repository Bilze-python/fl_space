"""自然语言 → 结构化实验计划。

规划器有两套实现，界面会如实标注当前使用的模式：

1. **LLM 模式**：配置了大模型后，把研究目标、指令与文献上下文交给模型，
   要求其输出 JSON 计划，再用 :func:`validate_plan` 做结构与范围校验；
2. **离线规则模式**：无 Key 或调用失败时，用中文关键词 + 数字单位解析
   生成确定性计划，保证比赛现场没有外网也能完整演示。

计划是前端与后台实验执行器之间的契约（见 ``docs/RESEARCH_PLATFORM_V1.md``）。
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any, Callable

from fl_space.research.config import DEFAULT_GOAL

# ── 默认实验参数 ───────────────────────────────────────────────

PLAN_DEFAULTS: dict[str, Any] = {
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

METRIC_KEYS = [
    "final_accuracy",
    "peak_accuracy",
    "accuracy_delta",
    "total_offloaded",
    "total_delay_slots",
    "data_balance_entropy",
]

MAX_CONDITIONS = 6
MAX_ROUNDS_V1 = 30
SUPPORTED_DATASETS = {"mnist", "fashion_mnist", "cifar10"}

_CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100}

_FULLWIDTH = str.maketrans(
    "０１２３４５６７８９，、；：．－（）",
    "0123456789,,;:.-()",
)

_VAR_ALIASES: list[tuple[str, str, str]] = [
    (r"地面站(?:数量|数)?|地面接收站|ground\s*stations?|\bgs\b", "gs", "地面站数"),
    (r"轨道面(?:数量|数)?|轨面(?:数量|数)?|\bplanes?\b", "planes", "轨道面数"),
    (
        r"(?:每(?:个)?(?:轨道)?面|每轨)\s*(?:的)?\s*(?:卫星|星)(?:数量|数)?|sats?\s*per\s*plane",
        "sats_per_plane",
        "每轨卫星数",
    ),
    (r"卫星(?:数量|数|总数)?|航天器|客户端|\bsats?\b|\bsatellites?\b", "sats", "卫星总数"),
    (r"训练轮次|轮次|轮数|\brounds?\b", "rounds", "训练轮次"),
    (r"本地\s*epoch|本地训练轮|epochs?", "local_epochs", "本地 Epoch"),
    (r"随机种子|种子|\bseeds?\b", "seed", "随机种子"),
]

_NUM_LIST = r"\d+(?:\s*(?:[/、,，~\-到至和]\s*\d+))+"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plan_id() -> str:
    return f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def default_research_goal() -> str:
    """返回默认研究目标。"""
    return DEFAULT_GOAL


# ── 文本规范化 ────────────────────────────────────────────────


def _cn_number(text: str) -> int:
    """把简单中文数字（含十/百）转换为整数，无法解析返回 -1。"""
    total = 0
    current = 0
    for char in text:
        if char in _CN_DIGITS:
            current = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            unit = _CN_UNITS[char]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
        else:
            return -1
    return total + current


def normalize_instruction(text: str) -> str:
    """统一全角字符、显式写法的中文数字（仅数字+量词场景）。"""
    normalized = str(text or "").translate(_FULLWIDTH)

    def replace_cn(match: re.Match[str]) -> str:
        value = _cn_number(match.group(0))
        return str(value) if value >= 0 else match.group(0)

    return re.sub(
        r"[零一二两三四五六七八九十百]+(?=\s*(?:个|颗|座|台|轮|次|种|星|轨|面|地面站|卫星|轮次|epoch))",
        replace_cn,
        normalized,
    )


def total_sats_to_topology(total: int) -> tuple[int, int]:
    """把「卫星总数」换算为（轨道面数, 每轨卫星数）。"""
    total = max(2, min(int(total), 24))
    if total <= 6:
        return 1, total
    if total % 3 == 0:
        return 3, total // 3
    return 2, max(2, total // 2)


_NUM_TOKEN = r"\d{1,2}"
_NUM_LIST = r"\d+(?:\s*(?:[/、,，~\-到至和]\s*\d+))+"
_NUM_LIST_STRICT = r"\d{1,2}(?:\s*(?:[/、,，~\-到至和])\s*\d{1,2})+"
_MEASURE_WORDS = "个|颗|座|台|种|轮|次"


def _extract_numbers(text: str) -> list[int]:
    return [int(item) for item in re.findall(r"\d+", text)]


def _find_var_values(text: str, alias: str) -> list[int]:
    """抽取变量取值：只接受紧贴单位词的数字，避免把别的参数误当成变量。

    支持两种语序：
    - 数字在前：「3/5/10 个地面站」「5 颗卫星」「10 轮」
    - 单位在前：「地面站 2/3/5」「地面站数量 3」「轮次 10」
    """
    values: list[int] = []
    before = re.compile(
        rf"({_NUM_LIST_STRICT}|{_NUM_TOKEN})\s*(?:{_MEASURE_WORDS})?\s*(?:{alias})",
        flags=re.IGNORECASE,
    )
    for match in before.finditer(text):
        values.extend(_extract_numbers(match.group(1)))
    if not values:
        after = re.compile(
            rf"(?:{alias})\s*(?:数量|数|总数|规模)?\s*(?:为|是|[:：=])?\s*"
            rf"({_NUM_LIST_STRICT}|{_NUM_TOKEN})",
            flags=re.IGNORECASE,
        )
        for match in after.finditer(text):
            values.extend(_extract_numbers(match.group(1)))
    seen: set[int] = set()
    return [value for value in values if not (value in seen or seen.add(value))]


def _extract_scalar(text: str, pattern: str, cast: Callable[[str], Any]) -> Any:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return cast(match.group(1))
    except (TypeError, ValueError):
        return None


# ── 需求解析 ──────────────────────────────────────────────────


def parse_requirements(instruction: str) -> dict[str, Any]:
    """把中文自然语言指令解析为实验需求（确定性规则）。"""
    text = normalize_instruction(instruction)
    scalars: dict[str, Any] = {}
    notes: list[str] = []
    scale: list[dict[str, Any]] = []

    # 1) 紧凑拓扑写法：「3 轨面 2 星」「3×2 卫星」「2 星/轨面」
    pairs = [
        (int(planes), int(per_plane))
        for planes, per_plane in re.findall(
            r"(\d{1,2})\s*(?:个)?\s*轨(?:道)?面\s*(?:[×xX*]|每轨|\s)\s*(\d{1,2})\s*(?:颗|个)?\s*(?:星|卫星)",
            text,
        )
    ]
    if len(pairs) >= 2:
        explicit = []
        for planes, per_plane in pairs:
            if planes * per_plane <= 24:
                explicit.append((planes, per_plane))
        if explicit:
            scale.append(
                {
                    "var": "topology",
                    "label": "星座拓扑",
                    "values": explicit,
                    "explicit": True,
                }
            )
    if pairs and len(pairs) == 1:
        planes, per_plane = pairs[0]
        if 1 <= planes <= 6 and 1 <= per_plane <= 6:
            scalars["planes"] = planes
            scalars["sats_per_plane"] = per_plane
    per_plane_only = re.search(r"(\d{1,2})\s*(?:颗|个)?\s*星\s*/\s*轨(?:道)?面", text)
    if per_plane_only:
        scalars["sats_per_plane"] = int(per_plane_only.group(1))
    per_plane_cn = re.search(r"每(?:个)?(?:轨道)?面\s*(?:的)?\s*(\d{1,2})\s*(?:颗|个)?\s*星", text)
    if per_plane_cn:
        scalars["sats_per_plane"] = int(per_plane_cn.group(1))

    # 2) 其它数值参数
    scalar_rules = [
        ("rounds", r"(?:训练)?(?:轮次|轮数)\s*[:=]?\s*(\d{1,3})|(\d{1,3})\s*轮(?:训练|次)?"),
        (
            "local_epochs",
            r"(?:本地\s*epoch|本地训练轮|local\s*epochs?)\s*[:=]?\s*(\d{1,2})",
        ),
        ("batch_size", r"(?:batch(?:\s*size)?|批大小)\s*[:=]?\s*(\d{1,4})"),
        ("lr", r"(?:lr|学习率)\s*[:=]?\s*(0?\.\d+|\d+\.\d+)"),
        ("seed", r"(?:seed|随机种子)\s*[:=]?\s*(\d+)"),
        (
            "max_samples_per_client",
            r"(?:每(?:颗)?(?:卫星|客户端)|每星)\s*(\d{1,5})\s*(?:个)?\s*样本",
        ),
        ("offload_every", r"每\s*(\d)\s*轮(?:执行|做)?(?:一次)?卸载"),
    ]
    for key, pattern in scalar_rules:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        raw = next((group for group in match.groups() if group), None)
        if raw is None:
            continue
        try:
            scalars[key] = float(raw) if key == "lr" else int(raw)
        except ValueError:
            continue

    # 3) 变量扫描（地面站 / 卫星总数 / 轨道面 / 轮次 / 种子）
    for alias, var, label in _VAR_ALIASES:
        values = _find_var_values(text, alias)
        if not values:
            continue
        values = [value for value in values if value > 0]
        if var == "rounds":
            # 轮次只接受单值，避免把「3 轮、seed 42」误判成网格
            if "rounds" not in scalars and values:
                scalars["rounds"] = values[0]
            continue
        if var == "local_epochs" and "local_epochs" not in scalars and values:
            scalars["local_epochs"] = values[0]
            continue
        if var == "seed":
            unique_seeds = [value for value in values[:4] if value < 10_000_000]
            if len(unique_seeds) >= 2:
                scale.append({"var": "seed", "label": "随机种子", "values": unique_seeds})
            elif unique_seeds:
                scalars["seed"] = unique_seeds[0]
            continue
        unique = [value for value in values if value <= 24]
        if var == "gs":
            unique = [value for value in values if value <= 20]
        if len(unique) >= 2:
            scale.append({"var": var, "label": label, "values": unique[:12]})
        elif len(unique) == 1:
            value = unique[0]
            if var == "sats":
                planes, per_plane = total_sats_to_topology(value)
                scalars["planes"] = planes
                scalars["sats_per_plane"] = per_plane
            elif var == "planes" and 1 <= value <= 6:
                scalars["planes"] = value
            elif var == "sats_per_plane" and 1 <= value <= 6:
                scalars["sats_per_plane"] = value
            elif var == "gs" and 1 <= value <= 20:
                scalars["gs"] = value

    # 4) 意图与开关
    flags = {
        "batch": bool(re.search(r"批量|网格|grid|多组|扫描|扫一遍|组合|多个配置|矩阵", text, re.I)),
        "quick": bool(re.search(r"快速|快一点|小规模|演示|demo|简单跑", text, re.I)),
        "run": bool(re.search(r"运行|跑|执行|开始|启动|训练|批量", text, re.I)),
        "analyze": bool(
            re.search(r"分析|解读|说明|证明|为什么|能否|评估|评审|判断|结论", text, re.I)
        ),
        "intel": bool(re.search(r"资讯|新闻|前沿|最新动态|arxiv|情报", text, re.I)),
        "literature": bool(re.search(r"文献|论文|知识库|综述|引用", text, re.I)),
        "review": bool(re.search(r"评审|评估|设计|变量|对照|方案", text, re.I)),
        "seeds": bool(re.search(r"多种子|多\s*seed|重复\s*\d*\s*次|多次重复", text, re.I)),
    }

    algorithms = [name for name in ("fedleo", "fedavg", "fedprox", "fedbuff") if name in text.lower()]
    if "卸载" in text and "fedleo" not in algorithms:
        algorithms.append("fedleo")

    if flags["quick"]:
        notes.append("已按「快速模式」压缩轮次与样本量，适合演示与冒烟验证。")
    if algorithms and any(item != "fedleo" for item in algorithms):
        notes.append(
            "v1 批量执行器只支持 FedLEO 卸载开/关对照（内置 FedAvg 基线）；"
            "FedProx 等算法请在「实验配置」页单跑。"
        )

    return {
        "text": text,
        "scalars": scalars,
        "scale": scale,
        "flags": flags,
        "algorithms": algorithms,
        "notes": notes,
    }


# ── 计划构建 ──────────────────────────────────────────────────


def _merge_base(requirements: dict[str, Any]) -> dict[str, Any]:
    base = dict(PLAN_DEFAULTS)
    base.update(requirements.get("scalars") or {})
    if requirements["flags"].get("quick"):
        base["rounds"] = min(int(base["rounds"]), 2)
        base["max_samples_per_client"] = min(int(base["max_samples_per_client"]), 80)
    return base


def _topology_label(planes: int, per_plane: int) -> str:
    return f"{planes} 轨面 × {per_plane} 星"


def _dim_overrides(var: str, value: Any) -> dict[str, Any]:
    if var == "topology":
        planes, per_plane = value
        return {"planes": int(planes), "sats_per_plane": int(per_plane)}
    if var == "sats":
        planes, per_plane = total_sats_to_topology(int(value))
        return {"planes": planes, "sats_per_plane": per_plane}
    return {var: value}


def _dim_value_label(var: str, value: Any, override: dict[str, Any]) -> str:
    """把单个变量取值格式化为条件名片段。"""
    if var == "topology":
        return _topology_label(int(override["planes"]), int(override["sats_per_plane"]))
    if var == "sats":
        return f"{int(override['planes']) * int(override['sats_per_plane'])} 星"
    if var == "planes":
        return f"{override['planes']} 轨面"
    if var == "sats_per_plane":
        return f"每轨 {override['sats_per_plane']} 星"
    if var == "seed":
        return f"seed {override['seed']}"
    if var == "gs":
        return f"{override['gs']} 地面站"
    if var == "rounds":
        return f"{override['rounds']} 轮"
    return str(value)


def _build_conditions(
    base: dict[str, Any],
    requirements: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    notes: list[str] = []
    dimensions = [item for item in requirements.get("scale") or [] if item.get("values")]
    if requirements["flags"].get("seeds"):
        seeds = [int(base.get("seed", 42)) + offset for offset in range(3)]
        dimensions.append({"var": "seed", "label": "随机种子", "values": seeds})

    if not dimensions:
        condition = {
            "id": "c1",
            "name": _topology_label(int(base["planes"]), int(base["sats_per_plane"])),
            "params": dict(base),
        }
        return [condition], [], notes

    combos: list[tuple[dict[str, Any], list[str]]] = [({}, [])]
    for dim in dimensions:
        next_combos: list[tuple[dict[str, Any], list[str]]] = []
        for value in dim["values"]:
            override = _dim_overrides(str(dim["var"]), value)
            label = _dim_value_label(str(dim["var"]), value, override)
            for param_acc, label_acc in combos:
                merged = dict(param_acc)
                merged.update(override)
                next_combos.append((merged, [*label_acc, label]))
        combos = next_combos

    if len(combos) > MAX_CONDITIONS:
        notes.append(
            f"条件组合共 {len(combos)} 组，已按 v1 上限截断为前 {MAX_CONDITIONS} 组。"
        )
        combos = combos[:MAX_CONDITIONS]

    conditions: list[dict[str, Any]] = []
    for index, (override, labels) in enumerate(combos, start=1):
        params = dict(base)
        params.update(override)
        conditions.append(
            {
                "id": f"c{index}",
                "name": " / ".join(labels) or _topology_label(
                    int(params["planes"]), int(params["sats_per_plane"])
                ),
                "params": params,
            }
        )
    return conditions, dimensions, notes


def _hypothesis(dimensions: list[dict[str, Any]], requirements: dict[str, Any]) -> str:
    variables = {str(item["var"]) for item in dimensions}
    if "topology" in variables or "sats" in variables or "planes" in variables:
        return (
            "在数据划分、训练预算与评估协议一致的前提下，改变星座规模会改变星地接触窗口与"
            "卸载机会，从而影响最终准确率、峰值准确率与累计卸载量。"
        )
    if "gs" in variables:
        return (
            "地面站数量增加会提升星地接触与聚合频率，从而提升收敛准确率；但超过临界规模后"
            "收益可能递减，甚至因调度冲突带来负向影响。"
        )
    if "seed" in variables:
        return "同一配置在多个随机种子下的结果波动，反映了当前实验设置的统计稳定性。"
    return (
        "在相同数据划分与训练预算下，启用数据卸载能够改善数据均衡度与收敛速度；"
        "但卸载带来的通信时延开销需要与准确率收益一起权衡。"
    )


def _title(dimensions: list[dict[str, Any]], base: dict[str, Any]) -> str:
    if not dimensions:
        return f"FedLEO 卸载开关对照 · {_topology_label(base['planes'], base['sats_per_plane'])}"
    labels = []
    for dim in dimensions[:2]:
        var = str(dim["var"])
        if var == "topology":
            values = " / ".join(
                _topology_label(int(item[0]), int(item[1])) for item in dim["values"]
            )
        elif var == "sats":
            values = " / ".join(f"{value} 星" for value in dim["values"])
        else:
            values = "/".join(str(value) for value in dim["values"])
        labels.append(f"{dim['label']} {values}")
    if len(dimensions) > 2:
        labels.append(f"等 {len(dimensions)} 个变量")
    return f"FedLEO 卸载对照 · {' × '.join(labels)}"


def _estimate_duration(conditions: list[dict[str, Any]]) -> int:
    total = 0.0
    for condition in conditions:
        rounds = int(condition["params"].get("rounds", 3))
        total += 55 + rounds * 18
    return round(total + 20)


def build_default_plan(
    instruction: str,
    *,
    goal: str = "",
    literature: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """离线规则模式：把自然语言指令转成可执行实验计划。"""
    requirements = parse_requirements(instruction)
    base = _merge_base(requirements)
    conditions, dimensions, notes = _build_conditions(base, requirements)
    notes = list(requirements["notes"]) + notes + [
        "每个条件内部同时运行「卸载开启」与「卸载关闭」两组，便于直接对照。",
    ]
    plan: dict[str, Any] = {
        "id": _plan_id(),
        "title": _title(dimensions, base),
        "goal": goal or DEFAULT_GOAL,
        "instruction": str(instruction or "").strip(),
        "backend": "fedleo",
        "hypothesis": _hypothesis(dimensions, requirements),
        "independent_variables": [
            {"name": str(item["var"]), "label": str(item["label"]), "values": item["values"]}
            for item in dimensions
        ],
        "conditions": conditions,
        "controls": [
            "开启/关闭两组共享同一数据集划分、模型初始化与本地 epoch",
            "评估间隔、早停策略与训练轮次保持一致",
            "单机 CPU 训练，准确率看相对趋势，时延只作同环境比较",
        ],
        "metrics": list(METRIC_KEYS),
        "expected_duration_sec": _estimate_duration(conditions),
        "risks": [
            "每个条件默认只跑单个随机种子，统计效力有限",
            "轮次较少，结论用于功能验证与趋势判断，不代表论文级复现",
            "CPU 推理/训练时延不是星上真实算力，不能外推绝对时延",
        ],
        "notes": notes,
        "literature": list(literature or []),
        "created_at": _now_iso(),
        "mode": "offline",
    }
    return plan


# ── 计划校验（LLM 输出与前端提交共用） ────────────────────────

_PARAM_RULES: dict[str, tuple[float, float]] = {
    "planes": (1, 6),
    "sats_per_plane": (1, 6),
    "gs": (1, 20),
    "rounds": (1, MAX_ROUNDS_V1),
    "local_epochs": (1, 5),
    "batch_size": (8, 512),
    "lr": (1e-5, 1.0),
    "seed": (0, 10_000_000),
    "offload_every": (1, 10),
    "max_offload_iter": (1, 5),
    "max_samples_per_client": (20, 2000),
}


def _coerce_param(key: str, value: Any) -> Any:
    if key == "dataset":
        text = str(value or "mnist").lower()
        return text if text in SUPPORTED_DATASETS else "mnist"
    if key == "device":
        return "cuda" if str(value or "").lower() == "cuda" else "cpu"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return PLAN_DEFAULTS[key]
    low, high = _PARAM_RULES[key]
    number = max(low, min(number, high))
    if key == "lr":
        return round(number, 6)
    return round(number)


def validate_plan(plan: Any) -> dict[str, Any]:
    """校验并规范化计划，非法输入抛 ``ValueError``。"""
    if not isinstance(plan, dict):
        raise ValueError("实验计划必须是 JSON 对象")
    conditions_raw = plan.get("conditions")
    if not isinstance(conditions_raw, list) or not conditions_raw:
        raise ValueError("实验计划至少需要一个条件")
    if len(conditions_raw) > MAX_CONDITIONS:
        raise ValueError(f"单个计划最多 {MAX_CONDITIONS} 个条件")

    conditions: list[dict[str, Any]] = []
    clamp_notes: list[str] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(conditions_raw, start=1):
        if not isinstance(raw, dict):
            raise ValueError("条件格式无效")
        condition_id = str(raw.get("id") or f"c{index}").strip()[:40]
        if condition_id in seen_ids:
            condition_id = f"c{index}"
        seen_ids.add(condition_id)
        raw_params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        params = dict(PLAN_DEFAULTS)
        for key in PLAN_DEFAULTS:
            if key in raw_params:
                params[key] = _coerce_param(key, raw_params[key])
        if params["planes"] * params["sats_per_plane"] > 24:
            params["sats_per_plane"] = max(1, 24 // max(1, params["planes"]))
            clamp_notes.append(
                f"条件 {condition_id} 卫星总数超过 24，已自动收缩为 "
                f"{params['planes']} 轨面 × {params['sats_per_plane']} 星。"
            )
        name = str(raw.get("name") or _topology_label(params["planes"], params["sats_per_plane"]))
        conditions.append({"id": condition_id, "name": name[:80], "params": params})

    title = str(plan.get("title") or "").strip()[:120] or "自定义批量实验"
    validated = {
        "id": str(plan.get("id") or _plan_id())[:60],
        "title": title,
        "goal": str(plan.get("goal") or DEFAULT_GOAL)[:2000],
        "instruction": str(plan.get("instruction") or "")[:2000],
        "backend": "fedleo",
        "hypothesis": str(plan.get("hypothesis") or "")[:600],
        "independent_variables": plan.get("independent_variables")
        if isinstance(plan.get("independent_variables"), list)
        else [],
        "conditions": conditions,
        "controls": plan.get("controls") if isinstance(plan.get("controls"), list) else [],
        "metrics": plan.get("metrics") if isinstance(plan.get("metrics"), list) else METRIC_KEYS,
        "expected_duration_sec": _estimate_duration(conditions),
        "risks": plan.get("risks") if isinstance(plan.get("risks"), list) else [],
        "notes": [
            *(plan.get("notes") if isinstance(plan.get("notes"), list) else []),
            *clamp_notes,
        ],
        "literature": plan.get("literature") if isinstance(plan.get("literature"), list) else [],
        "created_at": str(plan.get("created_at") or _now_iso()),
        "mode": str(plan.get("mode") or "offline"),
    }
    return validated


# ── LLM 规划 ─────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = (
    "你是 SpaceFL 卫星联邦学习科研平台的实验规划智能体。"
    "根据用户研究目标与实验指令，输出一个可被后台执行器直接运行的 JSON 实验计划。"
    "只输出 JSON，不要输出解释性文字。"
)

_LLM_SCHEMA_HINT = """{
  "title": "实验标题（中文，<= 40 字）",
  "hypothesis": "可检验假设（中文）",
  "independent_variables": [{"name": "gs|planes|sats_per_plane|sats|seed", "label": "中文标签", "values": [数值]}],
  "conditions": [{"id": "c1", "name": "条件名", "params": {
      "planes": 3, "sats_per_plane": 2, "gs": 3, "rounds": 3, "local_epochs": 1,
      "batch_size": 64, "lr": 0.02, "seed": 42, "offload_every": 2,
      "max_offload_iter": 2, "max_samples_per_client": 120,
      "dataset": "mnist", "device": "cpu"}}],
  "controls": ["对照条件"],
  "metrics": ["final_accuracy", "accuracy_delta", "total_offloaded"],
  "risks": ["风险"],
  "notes": ["说明"]
}"""


def generate_plan(
    instruction: str,
    *,
    goal: str = "",
    provider: Any = None,
    literature: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """生成实验计划：优先 LLM，失败或未配置时退回离线规则。"""
    fallback = build_default_plan(instruction, goal=goal, literature=literature)
    if provider is None or not getattr(provider, "configured", False):
        return fallback
    try:
        from fl_space.research.providers import chat_json

        user_prompt = (
            f"研究目标：{goal or DEFAULT_GOAL}\n"
            f"实验指令：{instruction}\n"
            f"文献上下文（节选）：{(literature or [])[:4]}\n"
            f"输出 JSON 结构要求：{_LLM_SCHEMA_HINT}\n"
            "约束：conditions 1~6 个；每个条件 params 使用给定字段；rounds<=30；"
            "卫星总数<=24；地面站<=20；批量执行器只支持 FedLEO 卸载开/关对照。"
        )
        payload = chat_json(provider, _LLM_SYSTEM_PROMPT, user_prompt, schema_hint=_LLM_SCHEMA_HINT)
        payload.setdefault("goal", goal or DEFAULT_GOAL)
        payload.setdefault("instruction", instruction)
        payload.setdefault("literature", list(literature or []))
        payload.setdefault("notes", [])
        payload["notes"] = [*payload["notes"], "计划由大模型生成，已按执行器约束校验。"]
        payload["mode"] = "llm"
        validated = validate_plan(payload)
        validated["mode"] = "llm"
        return validated
    except Exception:
        fallback["notes"] = [
            *fallback["notes"],
            "大模型规划不可用，已自动切换为离线规则规划。",
        ]
        return fallback


def plan_runtime_hint(plan: dict[str, Any]) -> str:
    """把预计耗时格式化为人类可读文本。"""
    seconds = int(plan.get("expected_duration_sec") or 0)
    if seconds < 90:
        return f"约 {seconds} 秒"
    minutes = math.ceil(seconds / 60)
    return f"约 {minutes} 分钟"
