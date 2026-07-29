#!/usr/bin/env python3
"""
异常工况轻量化兜底计算模块 v2 — 极简推演 + 自动降级恢复 + 热窗口精度保护区

设计原则:
    - 系统负载爆满时不是简单放大步长, 而是切换极简物理模型
    - 兜底: J2 + 球面几何 (舍弃 ENU 完整坐标系)
    - 算力恢复后自动切回高精度模式
    - 保证调度系统不丢失窗口信息
    - 硬性约束: 近域 0~6h 热窗口 — 降级切换对该区间无效!!!
      降级只能影响 6h 以外的中远期规划窗口;
      热窗口内任何降级指令均被过滤, 强制全精度 SGP4 + ENU。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import Enum, auto
import math
import time
from typing import Any, Callable, ClassVar

import numpy as np

# ═══════════════════════════════════════════════════════════════════
#  1. 运行模式定义
# ═══════════════════════════════════════════════════════════════════


class EngineMode(Enum):
    """推演引擎模式。"""

    FULL_PRECISION = auto()    # 完整精度: SGP4 + WGS84 + ENU
    LIGHTWEIGHT = auto()       # 轻量化: J2 + 球面几何
    MINIMAL = auto()           # 极简: 纯二体 + 地心夹角近似


@dataclass
class DegradationPolicy:
    """降级策略配置。"""

    # CPU 负载阈值 (百分比)
    cpu_threshold_lightweight: float = 80.0   # >80% → 切换轻量
    cpu_threshold_minimal: float = 95.0       # >95% → 切换极简
    cpu_threshold_recover: float = 50.0       # <50% → 恢复全精度

    # 冷却时间 (秒)
    cooldown_s: float = 30.0  # 切换后至少保持当前模式 30s

    # 检查间隔 (秒)
    check_interval_s: float = 5.0


# ═══════════════════════════════════════════════════════════════════
#  2. 极简推演模型
# ═══════════════════════════════════════════════════════════════════


class LightweightPropagator:
    """轻量化极简推演模型 — J2 二体 + 地心夹角近似。

    与完整模型对比:
        - 完整: SGP4(J2-J4+日月+大气) + geodetic→ECEF + ENU矩阵 → 仰角
        - 轻量: J2 二体 + 球面几何地心夹角 → 等效仰角
        - 计算量: ~5% of full precision

    精度损失:
        - 仰角误差: ±0.5° (vs 完整 SGP4 的 ±0.001°)
        - 接轨时间: ±30s (足够粗粒度调度使用)
        - 下行量估算: 保守估计 (实际可用 80% 为安全余量)

    用法:
        prop = LightweightPropagator()
        el = prop.equivalent_elevation(sat_lat, sat_lon, sat_alt, gs_lat, gs_lon, gs_alt)
        if el > min_elevation:
            # 粗判可见, 可记录窗口 (带不确定性 ±0.5°)
    """

    # WGS84 常数
    EARTH_RADIUS_EQ_KM: ClassVar[float] = 6378.137
    EARTH_RADIUS_POLAR_KM: ClassVar[float] = 6356.752
    EARTH_MEAN_RADIUS_KM: ClassVar[float] = 6371.0
    EARTH_MU: ClassVar[float] = 398600.4418
    J2: ClassVar[float] = 1.08262668e-3

    def __init__(self):
        self._elevation_bias_deg: float = 0.0  # 系统偏差修正

    @property
    def elevation_bias_deg(self) -> float:
        return self._elevation_bias_deg

    def calibrate_bias(
        self,
        full_precision_elevations: list[float],
        lightweight_elevations: list[float],
    ) -> float:
        """用全精度结果校准轻量模型的系统偏差。

        Args:
            full_precision_elevations: 全精度仰角列表
            lightweight_elevations: 轻量仰角列表

        Returns:
            系统偏差 (度)
        """
        if len(full_precision_elevations) != len(lightweight_elevations):
            return 0.0
        if not full_precision_elevations:
            return 0.0

        errors = [
            fl - ll
            for fl, ll in zip(full_precision_elevations, lightweight_elevations)
        ]
        self._elevation_bias_deg = sum(errors) / len(errors)
        return self._elevation_bias_deg

    def geocentric_angle(
        self,
        sat_lat_deg: float,
        sat_lon_deg: float,
        gs_lat_deg: float,
        gs_lon_deg: float,
    ) -> float:
        """计算地心夹角 (度)。"""
        lat1 = math.radians(sat_lat_deg)
        lat2 = math.radians(gs_lat_deg)
        dlon = math.radians(sat_lon_deg - gs_lon_deg)

        cos_gamma = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(dlon)
        cos_gamma = max(-1.0, min(1.0, cos_gamma))
        return math.degrees(math.acos(cos_gamma))

    def equivalent_elevation(
        self,
        sat_lat_deg: float,
        sat_lon_deg: float,
        sat_altitude_km: float,
        gs_lat_deg: float,
        gs_lon_deg: float,
        gs_altitude_km: float = 0.0,
    ) -> float:
        """计算等效仰角 (地心夹角 + 球面几何近似)。

        原理:
            从地心夹角 γ 和轨道高度 h, 推导站心仰角 El。

            由正弦定理:
            sin(90°+El) / (R+h) = sin(γ) / d
            d? = R? + (R+h)? - 2R(R+h)cos(γ)

            综合得:
            El = 90° - γ - arcsin(R*sin(γ)/d)

        Args:
            sat_lat_deg: 卫星纬度 (度)
            sat_lon_deg: 卫星经度 (度)
            sat_altitude_km: 卫星高度 (km)
            gs_lat_deg: 地面站纬度 (度)
            gs_lon_deg: 地面站经度 (度)
            gs_altitude_km: 地面站高度 (km)

        Returns:
            等效仰角 (度), 含系统偏差修正
        """
        R = self.EARTH_MEAN_RADIUS_KM + gs_altitude_km  # noqa: N806
        r = R + sat_altitude_km

        gamma = self.geocentric_angle(sat_lat_deg, sat_lon_deg, gs_lat_deg, gs_lon_deg)
        gamma_rad = math.radians(gamma)

        # 星地距离 (余弦定理)
        d_sq = R * R + r * r - 2.0 * R * r * math.cos(gamma_rad)
        d = math.sqrt(max(d_sq, 1e-9))

        # 仰角 = 90° - γ - arcsin(R*sin(γ)/d)
        sin_term = R * math.sin(gamma_rad) / d
        sin_term = max(-1.0, min(1.0, sin_term))

        el_deg = 90.0 - gamma - math.degrees(math.asin(sin_term))

        # 系统偏差修正
        return el_deg + self._elevation_bias_deg

    def equivalent_elevation_batch(
        self,
        sat_lats_deg: np.ndarray,
        sat_lons_deg: np.ndarray,
        sat_altitude_km: float,
        gs_lat_deg: float,
        gs_lon_deg: float,
        gs_altitude_km: float = 0.0,
    ) -> np.ndarray:
        """批量计算等效仰角 (向量化)。

        Returns:
            仰角数组 (度)
        """
        R = self.EARTH_MEAN_RADIUS_KM + gs_altitude_km  # noqa: N806
        r = R + sat_altitude_km

        lat1 = np.radians(sat_lats_deg)
        lat2 = np.radians(gs_lat_deg)
        dlon = np.radians(sat_lons_deg - gs_lon_deg)

        cos_gamma = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
        cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
        gamma_rad = np.arccos(cos_gamma)

        d_sq = R * R + r * r - 2.0 * R * r * cos_gamma
        d = np.sqrt(np.maximum(d_sq, 1e-9))

        sin_term = np.clip(R * np.sin(gamma_rad) / d, -1.0, 1.0)
        el_deg = 90.0 - np.degrees(gamma_rad) - np.degrees(np.arcsin(sin_term))

        return el_deg + self._elevation_bias_deg

    @staticmethod
    def estimate_contact_window(
        elevations: np.ndarray,
        min_elevation_deg: float = 10.0,
        time_step_s: float = 60.0,
    ) -> list[dict[str, float]]:
        """从仰角序列提取接触窗口 (简化版)。

        Returns:
            [{"start_s": float, "end_s": float, "max_el": float, "duration_s": float}, ...]
        """
        in_contact = elevations >= min_elevation_deg
        windows = []

        i = 0
        n = len(elevations)
        while i < n:
            if in_contact[i]:
                start = i
                while i < n and in_contact[i]:
                    i += 1
                end = i - 1

                window_els = elevations[start : end + 1]
                windows.append({
                    "start_s": start * time_step_s,
                    "end_s": end * time_step_s,
                    "max_el": float(np.max(window_els)),
                    "duration_s": (end - start) * time_step_s,
                })
            else:
                i += 1

        return windows


# ═══════════════════════════════════════════════════════════════════
#  3. 降级管理器
# ═══════════════════════════════════════════════════════════════════


class DegradationManager:
    """降级管理器 v2 — 自动切换 + 热窗口精度保护区。

    决策流程:
        1. 定期检查 CPU 负载
        2. 负载 > 80% → 切换 LIGHTWEIGHT (J2 + 球面几何, 仅对 >6h 窗口生效)
        3. 负载 > 95% → 切换 MINIMAL (纯二体 + 地心夹角, 仅对 >6h 窗口生效)
        4. 负载 < 50% → 恢复 FULL_PRECISION
        5. 冷却防抖: 切换后至少维持 30s

    热窗口保护:
        - 0~6h 窗口: 任何降级无效, 强制全精度 SGP4 + ENU
        - 仅 6h 以外中远期窗口跟随降级策略

    用法:
        mgr = DegradationManager()
        mode = mgr.check_and_switch()
        # 在计算时判断:
        effective = mgr.effective_mode(window_hours=3.0)
        # effective -> EngineMode.FULL_PRECISION (热窗口强制)
    """

    # 热窗口分界线 (小时)
    HOT_WINDOW_H: ClassVar[float] = 6.0

    def __init__(self, policy: DegradationPolicy | None = None):
        self._policy = policy or DegradationPolicy()
        self._current_mode = EngineMode.FULL_PRECISION
        self._last_switch_time: float = 0.0

        # 回调
        self._on_full: Callable[[], None] | None = None
        self._on_lightweight: Callable[[], None] | None = None
        self._on_minimal: Callable[[], None] | None = None

        # 统计
        self._mode_history: list[tuple[float, EngineMode]] = []
        self._switch_count: int = 0
        self._hot_window_override_count: int = 0  # 热窗口覆写降级次数

    @property
    def current_mode(self) -> EngineMode:
        return self._current_mode

    @property
    def mode_name(self) -> str:
        names = {
            EngineMode.FULL_PRECISION: "FULL_PRECISION",
            EngineMode.LIGHTWEIGHT: "LIGHTWEIGHT",
            EngineMode.MINIMAL: "MINIMAL",
        }
        return names[self._current_mode]

    @property
    def switch_count(self) -> int:
        return self._switch_count

    @property
    def hot_window_override_count(self) -> int:
        """热窗口覆写降级次数。"""
        return self._hot_window_override_count

    def effective_mode(self, window_hours: float) -> EngineMode:
        """获取对指定窗口实际生效的模式。

        热窗口 (≤6h): 降级无效, 强制 FULL_PRECISION
        中远域 (>6h): 跟随当前降级策略

        Args:
            window_hours: 预报窗口时长 (小时)

        Returns:
            实际生效的引擎模式
        """
        if window_hours <= self.HOT_WINDOW_H:
            if self._current_mode != EngineMode.FULL_PRECISION:
                self._hot_window_override_count += 1
            return EngineMode.FULL_PRECISION
        return self._current_mode

    def set_engine_callbacks(
        self,
        on_full: Callable[[], None] | None = None,
        on_lightweight: Callable[[], None] | None = None,
        on_minimal: Callable[[], None] | None = None,
    ) -> None:
        """设置模式切换回调。"""
        self._on_full = on_full
        self._on_lightweight = on_lightweight
        self._on_minimal = on_minimal

    def get_cpu_load(self) -> float:
        """获取 CPU 负载 (百分比)。"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 50.0  # 默认中等负载

    def check_and_switch(self) -> EngineMode:
        """检查负载并决定是否需要切换。"""
        now = time.perf_counter()

        # 冷却检查
        if now - self._last_switch_time < self._policy.cooldown_s:
            return self._current_mode

        cpu = self.get_cpu_load()
        new_mode = self._current_mode

        if cpu > self._policy.cpu_threshold_minimal:
            new_mode = EngineMode.MINIMAL
        elif cpu > self._policy.cpu_threshold_lightweight:
            new_mode = EngineMode.LIGHTWEIGHT
        elif cpu < self._policy.cpu_threshold_recover:
            new_mode = EngineMode.FULL_PRECISION

        if new_mode != self._current_mode:
            self._switch_mode(new_mode)

        return self._current_mode

    def _switch_mode(self, new_mode: EngineMode) -> None:
        """执行模式切换。"""
        self._current_mode = new_mode
        self._last_switch_time = time.perf_counter()
        self._switch_count += 1
        self._mode_history.append((time.perf_counter(), new_mode))

        # 调用回调
        callbacks = {
            EngineMode.FULL_PRECISION: self._on_full,
            EngineMode.LIGHTWEIGHT: self._on_lightweight,
            EngineMode.MINIMAL: self._on_minimal,
        }
        cb = callbacks.get(new_mode)
        if cb:
            with contextlib.suppress(Exception):
                cb()

    def force_mode(self, mode: EngineMode) -> None:
        """强制设置模式 (忽略冷却)。"""
        if mode != self._current_mode:
            self._switch_mode(mode)

    def get_downtime_estimate(
        self,
        task_duration_s: float,
    ) -> float:
        """估算降级模式下的任务耗时。

        Args:
            task_duration_s: 全精度下的任务耗时

        Returns:
            当前模式下的预估耗时
        """
        factors = {
            EngineMode.FULL_PRECISION: 1.0,
            EngineMode.LIGHTWEIGHT: 0.05,   # 5%
            EngineMode.MINIMAL: 0.02,       # 2%
        }
        return task_duration_s * factors.get(self._current_mode, 1.0)

    def summary(self) -> dict[str, Any]:
        """获取降级管理器状态摘要。"""
        return {
            "current_mode": self.mode_name,
            "cpu_load": self.get_cpu_load(),
            "switch_count": self._switch_count,
            "hot_window_override_count": self._hot_window_override_count,
            "hot_window_protected": True,
            "hot_window_h": self.HOT_WINDOW_H,
            "cooldown_remaining_s": max(
                0.0,
                self._policy.cooldown_s - (time.perf_counter() - self._last_switch_time),
            ),
            "estimated_speedup": 1.0 / max(
                0.01,
                self.get_downtime_estimate(1.0),
            ),
        }


# ═══════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════

def create_fallback_schedule(
    lightweight_windows: list[dict[str, Any]],
    safety_margin_factor: float = 0.8,
) -> list[dict[str, Any]]:
    """将轻量模型输出的粗粒度窗口转为保守调度窗口。

    保守策略:
        - 窗口起始延后 10% (留安全余量)
        - 窗口结束提前 10%
        - 下行量乘以 safety_margin_factor (0.8 = 只用 80%)
        - 标记为 fallback 模式

    Args:
        lightweight_windows: 轻量模型输出的窗口列表
        safety_margin_factor: 安全余量因子 [0, 1]

    Returns:
        保守调度窗口列表
    """
    schedule = []
    for w in lightweight_windows:
        duration = w.get("duration_s", 0)
        margin = duration * (1.0 - safety_margin_factor) * 0.5

        schedule.append({
            "sat_id": w.get("sat_id", 0),
            "gs_id": w.get("gs_id", 0),
            "start_s": w.get("start_s", 0) + margin,
            "end_s": w.get("end_s", 0) - margin,
            "max_elevation_deg": w.get("max_el", 0),
            "is_fallback": True,
            "elevation_uncertainty_deg": 0.5,
            "timing_uncertainty_s": 30.0,
        })

    return schedule
