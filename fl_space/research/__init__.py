"""SpaceFL 科研智能平台 — 研究层公共入口。

模块划分::

    config     研究目标 / 订阅主题配置
    providers  大模型供应商抽象（离线时自动降级）
    literature 期刊文献索引、全文检索与 RAG 上下文
    intel      前沿资讯抓取（arXiv / RSS）
    planner    自然语言 → 结构化实验计划
    analyzer   实验计划评审、结果分析与文献精读
    service    编排层，供 web/server.py 调用

设计原则：所有 AI 能力都有离线兜底，返回体用 ``mode`` 字段标注
``offline`` 或 ``llm``，便于界面如实展示。
"""

from __future__ import annotations

__all__ = [
    "ResearchService",
    "build_default_plan",
    "default_research_goal",
]

from fl_space.research.planner import build_default_plan, default_research_goal
from fl_space.research.service import ResearchService
