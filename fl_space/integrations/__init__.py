"""
外部框架集成模块 — 可插拔的外部生态适配器。

提供：
    - flower/ — Flower 联邦学习框架适配器
    - spacex_api.py — SpaceX API 只读目录适配器
    - (future) ray/ — Ray 分布式训练适配器
    - (future) tensorflow/ — TensorFlow Federated 适配器

设计原则：
    - 每个集成是独立的可选模块
    - 不修改 SpaceFL 核心代码
    - 通过适配器模式转换数据结构
"""

from .spacex_api import SpaceXApiClient as SpaceXApiClient
from .spacex_api import satellite_design_from_payload as satellite_design_from_payload

__all__ = [
    "SpaceXApiClient",
    "satellite_design_from_payload",
]
