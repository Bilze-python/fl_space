"""
FedLEO — 面向 LEO 星座的去中心化联邦学习框架

论文: "FedLEO: Decentralized Federated Learning over LEO Satellite Constellations"
     (Zhai et al., 2024)

核心思想:
    - 不依赖中心服务器，利用星间链路 (ISL) 做分层聚合
    - 两个阶段: 数据卸载(调整数据分布) → 去中心化训练(本地→同轨→跨轨)
    - 联合优化: 系统时延 + 训练准确率

模块结构:
    - planner.py     : FedLEOPlanner — 离散卸载决策引擎
    - aggregator.py  : FedLEOAggregator — 同轨/跨轨分层聚合
    - scheduler.py   : FedLEOScheduler — 完整调度编排
    - metrics.py     : 权重散度 + 系统时延指标
    - experiment.py  : FedLEO vs 基线对比实验
"""

from fl_space.fedleo.aggregator import FedLEOAggregator
from fl_space.fedleo.metrics import (
    FedLEOMetrics,
    compute_system_delay,
    compute_weight_divergence,
)
from fl_space.fedleo.planner import FedLEOPlanner, OffloadAction, OffloadPlan
from fl_space.fedleo.scheduler import FedLEOConfig, FedLEOScheduler

__all__ = [
    "FedLEOAggregator",
    "FedLEOConfig",
    "FedLEOMetrics",
    "FedLEOPlanner",
    "FedLEOScheduler",
    "OffloadAction",
    "OffloadPlan",
    "compute_system_delay",
    "compute_weight_divergence",
]

