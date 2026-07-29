#!/usr/bin/env python3
"""
可视窗口前置预筛选拓扑优化模块 — 几何包围盒预判 + 仰角变化率剪枝
D6: Pre-filtering topology optimization for visibility window detection.

设计目标:
    - 包围盒预判: 卫星/地面站空间包围盒无交集 → 直接跳过全部仰角计算
    - 仰角变化率剪枝: 卫星远离时批量跳过连续采样点
    - 预期减少 60-80% 无效仰角计算
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import numpy as np

# ═══════════════════════════════════════════════════════════════════
#  1. 轨道地面站几何包围盒预判
# ═══════════════════════════════════════════════════════════════════


@dataclass
class BoundingBox3D:
    """三维空间包围盒 (ECEF 坐标系)。"""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def intersects(self, other: BoundingBox3D, margin_km: float = 100.0) -> bool:
        """判断两个包围盒是否相交 (含边距)。

        Args:
            other: 另一个包围盒
            margin_km: 扩展边距 (km), 允许近似相交

        Returns:
            True 如果两个包围盒 (扩展后) 有交集
        """
        return (
            self.x_min - margin_km <= other.x_max + margin_km
            and self.x_max + margin_km >= other.x_min - margin_km
            and self.y_min - margin_km <= other.y_max + margin_km
            and self.y_max + margin_km >= other.y_min - margin_km
            and self.z_min - margin_km <= other.z_max + margin_km
            and self.z_max + margin_km >= other.z_min - margin_km
        )

    def expand(self, margin_km: float) -> BoundingBox3D:
        """扩展包围盒。"""
        return BoundingBox3D(
            x_min=self.x_min - margin_km,
            x_max=self.x_max + margin_km,
            y_min=self.y_min - margin_km,
            y_max=self.y_max + margin_km,
            z_min=self.z_min - margin_km,
            z_max=self.z_max + margin_km,
        )

    @property
    def volume_km3(self) -> float:
        """包围盒体积 (km?)。"""
        dx = self.x_max - self.x_min
        dy = self.y_max - self.y_min
        dz = self.z_max - self.z_min
        return max(0.0, dx * dy * dz)


class OrbitalBoundingBox:
    """轨道包围盒计算器 — 为卫星运行轨迹构建时间段内三维包围盒。

    原理:
        1. 给定轨道参数, 计算时间段内卫星 ECEF 坐标的范围
        2. 对地面站集合构建地理包围盒
        3. 若卫星包围盒与地面站包围盒无交集, 直接判定无可视窗口

    用法:
        obb = OrbitalBoundingBox()
        sat_bbox = obb.compute_satellite_bbox(
            semimajor_axis_km=6878.0,
            inclination_deg=53.0,
            raan_deg=0.0,
            t_start_s=0.0,
            t_end_s=3600.0,
        )
        gs_bbox = obb.compute_gs_bbox(gs_ecef_list)
        if not sat_bbox.intersects(gs_bbox):
            print("No possible visibility, skip!")
    """

    # 轨道周期常数
    EARTH_MU: ClassVar[float] = 398600.4418  # km?/s?

    @staticmethod
    def compute_satellite_bbox(
        semimajor_axis_km: float,
        inclination_deg: float,
        raan_deg: float = 0.0,
        eccentricity: float = 0.0,
        t_start_s: float = 0.0,
        t_end_s: float = 3600.0,
        n_samples: int = 12,
    ) -> BoundingBox3D:
        """计算卫星在时间段内的 ECEF 空间包围盒。

        Args:
            semimajor_axis_km: 半长轴 (km)
            inclination_deg: 轨道倾角 (度)
            raan_deg: 升交点赤经 (度)
            eccentricity: 偏心率
            t_start_s: 起始时刻 (秒)
            t_end_s: 结束时刻 (秒)
            n_samples: 采样点数

        Returns:
            ECEF 包围盒
        """
        # 轨道周期
        period_s = 2.0 * math.pi * math.sqrt(semimajor_axis_km**3 / OrbitalBoundingBox.EARTH_MU)
        mean_motion = 2.0 * math.pi / period_s
        incl_rad = math.radians(inclination_deg)
        raan_rad = math.radians(raan_deg)

        ts = np.linspace(t_start_s, t_end_s, n_samples)

        # 简化: 圆轨道近似 + 地球自转
        # 实际应调用完整 SGP4, 此处提供近似包围盒
        omega_earth = 7.2921159e-5  # rad/s

        xs = []
        ys = []
        zs = []

        r = semimajor_axis_km  # 圆轨道近似

        for t in ts:
            # 轨道面内角度 (近点角)
            theta = (mean_motion * t) % (2.0 * math.pi)

            # 轨道面内坐标
            x_orb = r * math.cos(theta)
            y_orb = r * math.sin(theta)

            # 旋转到 ECI
            x_eci = x_orb * math.cos(raan_rad) - y_orb * math.cos(incl_rad) * math.sin(raan_rad)
            y_eci = x_orb * math.sin(raan_rad) + y_orb * math.cos(incl_rad) * math.cos(raan_rad)
            z_eci = y_orb * math.sin(incl_rad)

            # ECI → ECEF (地球自转)
            gmst = omega_earth * t
            x_ecef = x_eci * math.cos(gmst) + y_eci * math.sin(gmst)
            y_ecef = -x_eci * math.sin(gmst) + y_eci * math.cos(gmst)
            z_ecef = z_eci

            xs.append(x_ecef)
            ys.append(y_ecef)
            zs.append(z_ecef)

        return BoundingBox3D(
            x_min=min(xs), x_max=max(xs),
            y_min=min(ys), y_max=max(ys),
            z_min=min(zs), z_max=max(zs),
        )

    @staticmethod
    def compute_gs_bbox(
        gs_ecef: list[tuple[float, float, float]],
        communication_range_km: float = 3000.0,
    ) -> BoundingBox3D:
        """计算地面站集合的地理包围盒 (含通信距离扩展)。

        Args:
            gs_ecef: 地面站 ECEF 坐标列表 [(x, y, z), ...]
            communication_range_km: 通信范围扩展 (km)

        Returns:
            地面站包围盒
        """
        if not gs_ecef:
            return BoundingBox3D(0, 0, 0, 0, 0, 0)

        xs = [p[0] for p in gs_ecef]
        ys = [p[1] for p in gs_ecef]
        zs = [p[2] for p in gs_ecef]

        bbox = BoundingBox3D(
            x_min=min(xs), x_max=max(xs),
            y_min=min(ys), y_max=max(ys),
            z_min=min(zs), z_max=max(zs),
        )
        return bbox.expand(communication_range_km)

    @staticmethod
    def can_have_visibility(
        sat_bbox: BoundingBox3D,
        gs_bbox: BoundingBox3D,
        min_elevation_deg: float = 10.0,
    ) -> bool:
        """快速判断卫星和地面站集合是否可能有可视窗口。

        Args:
            sat_bbox: 卫星包围盒
            gs_bbox: 地面站包围盒
            min_elevation_deg: 最小通信仰角 (度)

        Returns:
            True = 可能有可视窗口, 需进一步精细计算
            False = 确定无可视窗口, 可跳过
        """
        # 扩展边距: 基于最小仰角的近似
        min_el_rad = math.radians(min_elevation_deg)
        # 若仰角极低, 包围盒可能需要很大边距
        margin = 2000.0 * (1.0 - math.sin(min_el_rad)) + 500.0

        return sat_bbox.intersects(gs_bbox, margin)


# ═══════════════════════════════════════════════════════════════════
#  2. 仰角变化率预判剪枝
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ElevationState:
    """仰角状态快照 — 用于变化率剪枝。"""

    elevation_deg: float
    rate_deg_per_s: float  # 仰角变化率 (°/s)
    timeslot: int
    sat_ecef: tuple[float, float, float]


class ElevationRatePruner:
    """仰角变化率预判剪枝器 — 批量跳过仰角持续下降的采样点。

    原理:
        使用相邻时刻卫星位置求解仰角变化速率:
        - 若当前仰角 << min_elevation, 且变化率为负 (卫星继续下沉)
          → 可跳过一批连续采样点
        - 仅当变化率由负转正或靠近阈值区间时恢复精细计算

    预期跳过: 轨道周期的 70-85% 的采样点 (卫星在地面站地平线以下时段)

    用法:
        pruner = ElevationRatePruner(min_elevation_deg=10.0)
        skip_count = pruner.should_skip(current_el=3.0, rate=-0.02, distance=8000)
        # 若 skip_count > 0, 直接跳过 skip_count 个采样点
    """

    def __init__(self, min_elevation_deg: float = 10.0):
        """
        Args:
            min_elevation_deg: 最低通信仰角 (度)
        """
        self._min_el = min_elevation_deg
        self._min_el_rad = math.radians(min_elevation_deg)

        # 剪枝参数
        self._far_threshold = self._min_el * 0.5  # "远低于"阈值 = 最低仰角的 50%
        self._near_threshold = self._min_el * 0.8  # "接近"阈值 = 最低仰角的 80%

        # 统计
        self._total_checks: int = 0
        self._skipped: int = 0

    @property
    def skip_ratio(self) -> float:
        if self._total_checks == 0:
            return 0.0
        return self._skipped / self._total_checks

    def should_skip(
        self,
        current_elevation_deg: float,
        rate_deg_per_s: float,
        distance_km: float = 5000.0,
        time_step_s: float = 60.0,
    ) -> int:
        """判断是否需要跳过后续采样点。

        Args:
            current_elevation_deg: 当前仰角 (度)
            rate_deg_per_s: 仰角变化率 (°/s), 正值=上升, 负值=下降
            distance_km: 星地距离 (km), 用于估算恢复时间
            time_step_s: 采样步长 (秒)

        Returns:
            > 0: 可以跳过的采样点数
            0: 需要继续精细计算
        """
        self._total_checks += 1

        # 情况1: 仰角高于阈值 → 不跳过
        if current_elevation_deg >= self._near_threshold:
            return 0

        # 情况2: 仰角在上升 (变化率为正) → 不跳过, 即将进入可视区间
        if rate_deg_per_s > 0.0:
            return 0

        # 情况3: 仰角远低于阈值且持续下降 → 批量跳过
        if current_elevation_deg < self._far_threshold and rate_deg_per_s < 0.0:
            # 估算恢复到阈值的时间
            self._min_el - current_elevation_deg
            # 仰角将先降到最低点再回升, 绕地球半周
            # 简化: 跳过 500-2000 秒 (取决于距离)
            skip_seconds = max(300, min(2000, distance_km * 0.1))
            skip_slots = max(1, int(skip_seconds / time_step_s))
            self._skipped += skip_slots
            return skip_slots

        # 情况4: 仰角在阈值附近但变化率接近零 → 谨慎跳过少量
        if abs(rate_deg_per_s) < 0.005:
            # 变化极慢, 可能是远距离过顶, 跳过少量
            self._skipped += 1
            return 1

        return 0

    def compute_rate(
        self,
        el_prev_deg: float,
        el_curr_deg: float,
        dt_s: float,
    ) -> float:
        """计算仰角变化率 (°/s)。

        Args:
            el_prev_deg: 前一时刻仰角
            el_curr_deg: 当前时刻仰角
            dt_s: 时间间隔 (秒)

        Returns:
            仰角变化率 (°/s)
        """
        if dt_s < 1e-9:
            return 0.0
        return (el_curr_deg - el_prev_deg) / dt_s

    @staticmethod
    def estimate_recovery_time(
        current_elevation_deg: float,
        rate_deg_per_s: float,
        min_elevation_deg: float,
    ) -> float:
        """估算仰角恢复到最低阈值所需的时间 (秒)。

        简化模型: 假设仰角近似正弦波:
            el(t) = A * sin(2π * t / T)
            rate(t) = A * 2π / T * cos(2π * t / T)

        仰角恢复时间 ≈ 轨道周期的 1/4 (从最低点到最高点)
        对 LEO ~ 90min 周期 ≈ 1350s
        """
        if rate_deg_per_s >= 0.0:
            return 0.0  # 已在恢复中

        # 简化: 近似半个可见弧段的时间
        # 对 500km 轨道: 可见弧段约 10min (600s)
        # 从低于阈值到恢复约一半: 300s
        deficit = min_elevation_deg - current_elevation_deg
        if deficit <= 0:
            return 0.0

        # 假设变化率不变 (保守估计)
        if abs(rate_deg_per_s) > 1e-9:
            return deficit / abs(rate_deg_per_s)
        return 600.0  # 默认 10min

    def reset_stats(self) -> None:
        self._total_checks = 0
        self._skipped = 0


# ═══════════════════════════════════════════════════════════════════
#  3. 组合预筛选器
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PreFilterResult:
    """预筛选结果。"""

    possible: bool  # 是否可能有可视窗口
    skip_slots: int  # 可跳过的时隙数 (0=需要精细计算)
    reason: str  # 筛选原因


class CombinedPreFilter:
    """组合预筛选器 — 先包围盒粗判, 后仰角变化率剪枝。

    流程:
        1. 包围盒预判: 无交集 → 直接返回 impossible
        2. 仰角变化率剪枝: 判断是否可批量跳过
        3. 返回预筛选结果

    用法:
        filt = CombinedPreFilter(min_elevation_deg=10.0)
        result = filt.filter(
            sat_bbox=bbox1, gs_bbox=bbox2,
            current_el=3.0, rate=-0.02, distance=8000,
        )
        if not result.possible:
            continue  # 跳过该星站对
        if result.skip_slots > 0:
            ts += result.skip_slots  # 批量跳过
    """

    def __init__(self, min_elevation_deg: float = 10.0):
        self._obb = OrbitalBoundingBox()
        self._pruner = ElevationRatePruner(min_elevation_deg)
        self._min_el = min_elevation_deg

    def filter_bbox(
        self,
        sat_bbox: BoundingBox3D,
        gs_bbox: BoundingBox3D,
    ) -> PreFilterResult:
        """包围盒预判。"""
        if self._obb.can_have_visibility(sat_bbox, gs_bbox, self._min_el):
            return PreFilterResult(True, 0, "bbox_possible")
        return PreFilterResult(False, 0, "bbox_no_intersection")

    def filter_elevation(
        self,
        current_elevation_deg: float,
        rate_deg_per_s: float,
        distance_km: float = 5000.0,
        time_step_s: float = 60.0,
    ) -> PreFilterResult:
        """仰角变化率剪枝。"""
        skip = self._pruner.should_skip(
            current_elevation_deg, rate_deg_per_s, distance_km, time_step_s
        )
        if skip > 0:
            return PreFilterResult(True, skip, "elevation_rate_skip")
        return PreFilterResult(True, 0, "elevation_fine")

    @property
    def stats(self) -> dict[str, float]:
        return {
            "pruner_skip_ratio": self._pruner.skip_ratio,
        }
