"""研究目标与订阅主题的持久化配置。

配置文件放在项目根目录 ``.spacefl_research_config.json``，与既有
``.spacefl_ai_config.json``（大模型供应商）分开，互不影响：

- 研究目标（goal）用于计划生成、文献召回、结果分析与情报筛选；
- 订阅主题（topics）用于前沿情报的关键词加权与简报生成。
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any

CONFIG_FILENAME = ".spacefl_research_config.json"

DEFAULT_GOAL = (
    "研究低轨卫星星座中星地链路受限条件下的联邦学习数据卸载与聚合策略，"
    "权衡模型准确率、收敛轮次与卸载时延，并与 FedLEO 等已有工作对照。"
)

DEFAULT_TOPICS = [
    "太空联邦学习",
    "LEO 星座",
    "数据卸载",
    "通信调度",
    "星上计算",
]


def default_research_goal() -> str:
    """返回默认研究目标文本。"""
    return DEFAULT_GOAL


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


class ResearchConfigStore:
    """线程安全的研究配置读写。"""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self.path = self.project_dir / CONFIG_FILENAME
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        raw = _read_json(self.path)
        goal = str(raw.get("goal") or "").strip() or DEFAULT_GOAL
        raw_topics = raw.get("topics")
        topics = [
            str(item).strip()
            for item in (raw_topics if isinstance(raw_topics, list) else DEFAULT_TOPICS)
            if str(item).strip()
        ]
        if not topics:
            topics = list(DEFAULT_TOPICS)
        return {
            "goal": goal,
            "topics": topics,
            "updated_at": str(raw.get("updated_at") or _now_iso()),
        }

    def as_dict(self) -> dict[str, Any]:
        """返回配置副本。"""
        with self._lock:
            return {
                "goal": self._data["goal"],
                "topics": list(self._data["topics"]),
                "updated_at": self._data["updated_at"],
            }

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        """更新研究目标与订阅主题，忽略未知字段。"""
        with self._lock:
            goal = payload.get("goal")
            if goal is not None:
                text = str(goal).strip()
                if not text:
                    raise ValueError("研究目标不能为空")
                if len(text) > 2000:
                    raise ValueError("研究目标请控制在 2000 字以内")
                self._data["goal"] = text
            topics = payload.get("topics")
            if topics is not None:
                if not isinstance(topics, list):
                    raise ValueError("topics 必须是字符串数组")
                cleaned = [str(item).strip() for item in topics if str(item).strip()]
                if len(cleaned) > 12:
                    raise ValueError("订阅主题最多 12 个")
                self._data["topics"] = cleaned or list(DEFAULT_TOPICS)
            self._data["updated_at"] = _now_iso()
            self._save_locked()
            return {
                "goal": self._data["goal"],
                "topics": list(self._data["topics"]),
                "updated_at": self._data["updated_at"],
            }

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)
