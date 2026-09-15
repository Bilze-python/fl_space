"""
ISL 抽象基类与数据模型。

提供可插拔的 ISL 计算器接口，支持：
    - 内置 WGS84 椭球遮挡计算器
    - 空计算器（禁用 ISL）
    - 用户自定义计算器（继承 ISLCalculator）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ISLWindow:
    """单次 ISL 连续可见窗口。

    Attributes
    ----------
    satellite_a : str
        卫星 A 名称。
    satellite_b : str
        卫星 B 名称。
    cluster_id : str or None
        所属星簇 ID。
    start_utc : datetime
        窗口起始时间 (UTC)。
    end_utc : datetime
        窗口结束时间 (UTC)。
    min_range_km : float
        窗口内最小距离 (km)。
    max_range_km : float
        窗口内最大距离 (km)。
    """

    satellite_a: str
    satellite_b: str
    cluster_id: Optional[str]
    start_utc: datetime
    end_utc: datetime
    min_range_km: float
    max_range_km: float

    @property
    def duration_s(self) -> float:
        """窗口持续时长 (秒)。"""
        return (self.end_utc - self.start_utc).total_seconds()


@dataclass
class ISLConfig:
    """ISL 计算器配置。

    Attributes
    ----------
    enabled : bool
        是否启用 ISL 计算。默认 False（向后兼容）。
    calculator : str
        计算器类型: "wgs84" | "disabled" | "path/to/custom.py:ClassName"。
        默认 "wgs84"。
    atmosphere_buffer_km : float
        WGS84 椭球大气余量 (km)。0 = 纯 WGS84，常用值 80.0。
    step_seconds : float
        ISL 采样步长 (秒)。默认 60.0。
    cluster_mode : str
        星簇分组模式: "plane" (按轨道面) | "none" (不分组)。
    """

    enabled: bool = False
    calculator: str = "wgs84"
    atmosphere_buffer_km: float = 0.0
    step_seconds: float = 60.0
    cluster_mode: str = "plane"

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "calculator": self.calculator,
            "atmosphere_buffer_km": self.atmosphere_buffer_km,
            "step_seconds": self.step_seconds,
            "cluster_mode": self.cluster_mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ISLConfig:
        valid_keys = {
            "enabled",
            "calculator",
            "atmosphere_buffer_km",
            "step_seconds",
            "cluster_mode",
        }
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


class ISLCalculator(ABC):
    """ISL 计算器抽象基类。

    子类必须实现 ``compute()`` 方法。
    用户可通过 ``fl_space.isl.WGS84ISLCalculator`` 使用默认实现，
    或继承本类实现自定义 ISL 判定逻辑。

    输入输出约定：
        - ECEF 坐标使用 km 为单位
        - 时间使用 UTC datetime
        - 返回 ISLWindow 列表按 start_utc 升序排列
    """

    @abstractmethod
    def compute(
        self,
        ecef_positions: dict[str, np.ndarray],
        cluster_assignments: dict[str, Optional[str]],
        sample_times: Sequence[datetime],
        **kwargs,
    ) -> list[ISLWindow]:
        """计算 ISL 窗口。

        Parameters
        ----------
        ecef_positions : dict[str, np.ndarray]
            卫星名称 → ECEF 位置数组，shape (3, N_samples)，单位 km。
        cluster_assignments : dict[str, str or None]
            卫星名称 → 星簇 ID。None 表示不分组。
        sample_times : Sequence[datetime]
            采样时间点列表，长度 = N_samples。

        Returns
        -------
        list[ISLWindow]
            ISL 可见窗口列表，按 start_utc 升序排列。
        """
        ...

    @property
    def name(self) -> str:
        """计算器名称（用于日志和报告）。"""
        return self.__class__.__name__


class NoISLCalculator(ISLCalculator):
    """空 ISL 计算器 — 不计算任何星间链路。

    用于禁用 ISL 的场景，保持接口一致。
    """

    def compute(
        self,
        ecef_positions: dict[str, np.ndarray],
        cluster_assignments: dict[str, Optional[str]],
        sample_times: Sequence[datetime],
        **kwargs,
    ) -> list[ISLWindow]:
        return []

    @property
    def name(self) -> str:
        return "NoISL"
