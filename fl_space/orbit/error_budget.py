#!/usr/bin/env python3
"""
分层误差分配 + 闭环校验模块 — 残差监控·超阈值告警·自动回退全精度

功能:
    1. 分层误差分配: 总误差 → 轨道求解/几何计算/窗口插值
    2. 残差阈值监控: 每层独立监控, 超标触发告警
    3. 闭环校验: 比对实测 vs 预报接轨时间, 统计误差分布
    4. 自动回退: 超阈值自动切换完整 SGP4 全阶模型

设计原则:
    - 所有拟合/简化模型强制配置残差阈值监控
    - 误差超标立刻强制全量 SGP4 重算
    - 误差补偿仅作为两次定时刷新间的小幅修正, 不能替代定时轨道更新
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
import math
import time
from typing import Any

import numpy as np

# ═══════════════════════════════════════════════════════════════════
#  告警级别
# ═══════════════════════════════════════════════════════════════════

class AlertLevel(Enum):
    """告警级别。"""
    OK = auto()        # 正常
    WARNING = auto()   # 逼近上限 (80%)
    CRITICAL = auto()  # 超标


# ═══════════════════════════════════════════════════════════════════
#  分层误差追踪
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LayerError:
    """单层误差追踪。"""
    name: str                     # 层级名称
    max_budget_m: float           # 本层最大允许位置偏差 (m)
    max_budget_s: float = 0.0     # 本层最大允许时间偏差 (s)

    # 滑动窗口统计 (保留最近 N 个样本)
    _errors: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _max_history: int = 100

    @property
    def current_rms_m(self) -> float:
        if not self._errors:
            return 0.0
        return math.sqrt(sum(e * e for e in self._errors) / len(self._errors))

    @property
    def current_max_m(self) -> float:
        if not self._errors:
            return 0.0
        return max(self._errors)

    @property
    def current_mean_m(self) -> float:
        if not self._errors:
            return 0.0
        return sum(self._errors) / len(self._errors)

    @property
    def sample_count(self) -> int:
        return len(self._errors)

    @property
    def ratio(self) -> float:
        """当前 RMS 占预算的比例。"""
        if self.max_budget_m <= 0:
            return 1.0
        return self.current_rms_m / self.max_budget_m

    @property
    def alert_level(self) -> AlertLevel:
        r = self.ratio
        if r > 1.0:
            return AlertLevel.CRITICAL
        if r > 0.8:
            return AlertLevel.WARNING
        return AlertLevel.OK

    def record(self, error_m: float) -> None:
        """记录一次误差。"""
        self._errors.append(error_m)

    def reset(self) -> None:
        self._errors.clear()

    def summary(self) -> dict:
        return {
            "name": self.name,
            "budget_m": self.max_budget_m,
            "rms_m": round(self.current_rms_m, 3),
            "max_m": round(self.current_max_m, 3),
            "mean_m": round(self.current_mean_m, 3),
            "ratio": round(self.ratio, 3),
            "alert": self.alert_level.name,
            "samples": self.sample_count,
        }


# ═══════════════════════════════════════════════════════════════════
#  分层误差管理器
# ═══════════════════════════════════════════════════════════════════

class LayeredErrorManager:
    """分层误差管理器 — 将总误差预算拆分至各计算环节独立追踪。

    三层误差:
        orbit:    轨道求解误差 (SGP4 截断/简化导致的坐标偏差)
        geometry: 几何计算误差 (ECEF→ENU 坐标转换精度)
        interp:   窗口插值误差 (多项式拟合残差)

    用法:
        mgr = LayeredErrorManager(total_budget=error_budget)
        mgr.record_orbit(orbit_error_km * 1000)  # 转为米
        mgr.record_geometry(0.5)   # 0.5m
        mgr.record_interp(0.1)     # 0.1m
        if mgr.any_critical:
            raise FallbackToFullModel("轨道误差超标, 回退全精度")
    """

    def __init__(self, from_budget: Any = None, **kwargs):
        """从 ErrorBudget 或手动参数初始化。

        Args:
            from_budget: ErrorBudget 对象 (from fl_space.orbit.semi_analytic)
            **kwargs: 手动指定各层预算 (orbit_m, geometry_m, interp_m)
        """
        if from_budget is not None:
            self._orbit = LayerError("orbit", from_budget.orbit_budget_m,
                                     from_budget.orbit_budget_s)
            self._geometry = LayerError("geometry", from_budget.geometry_budget_m)
            self._interp = LayerError("interp", from_budget.interp_budget_m)
            self._total_budget_m = from_budget.max_position_m
            self._total_budget_s = from_budget.max_timing_s
        else:
            self._orbit = LayerError("orbit", kwargs.get("orbit_m", 50.0),
                                     kwargs.get("orbit_s", 2.5))
            self._geometry = LayerError("geometry", kwargs.get("geometry_m", 30.0))
            self._interp = LayerError("interp", kwargs.get("interp_m", 20.0))
            self._total_budget_m = kwargs.get("total_m", 100.0)
            self._total_budget_s = kwargs.get("total_s", 5.0)

    @property
    def orbit(self) -> LayerError:
        return self._orbit

    @property
    def geometry(self) -> LayerError:
        return self._geometry

    @property
    def interp(self) -> LayerError:
        return self._interp

    @property
    def total_rms_m(self) -> float:
        return math.sqrt(
            self._orbit.current_rms_m ** 2
            + self._geometry.current_rms_m ** 2
            + self._interp.current_rms_m ** 2
        )

    @property
    def total_ratio(self) -> float:
        if self._total_budget_m <= 0:
            return 1.0
        return self.total_rms_m / self._total_budget_m

    @property
    def any_critical(self) -> bool:
        return any(
            layer.alert_level == AlertLevel.CRITICAL
            for layer in [self._orbit, self._geometry, self._interp]
        )

    @property
    def any_warning(self) -> bool:
        return any(
            layer.alert_level in (AlertLevel.WARNING, AlertLevel.CRITICAL)
            for layer in [self._orbit, self._geometry, self._interp]
        )

    def record_orbit(self, error_m: float) -> None:
        self._orbit.record(error_m)

    def record_geometry(self, error_m: float) -> None:
        self._geometry.record(error_m)

    def record_interp(self, error_m: float) -> None:
        self._interp.record(error_m)

    def reset_all(self) -> None:
        for layer in [self._orbit, self._geometry, self._interp]:
            layer.reset()

    def summary(self) -> dict:
        return {
            "total_rms_m": round(self.total_rms_m, 3),
            "total_budget_m": self._total_budget_m,
            "total_ratio": round(self.total_ratio, 3),
            "total_budget_s": self._total_budget_s,
            "any_critical": self.any_critical,
            "any_warning": self.any_warning,
            "layers": {
                "orbit": self._orbit.summary(),
                "geometry": self._geometry.summary(),
                "interp": self._interp.summary(),
            },
        }


# ═══════════════════════════════════════════════════════════════════
#  闭环校验: ClosedLoopValidator
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PassObservation:
    """一次过境实测 vs 预报比对记录。"""
    sat_id: int
    gs_id: int
    predicted_start_s: float   # 预报接轨开始时刻 (相对 epoch 秒)
    observed_start_s: float    # 实测接轨开始时刻
    predicted_end_s: float
    observed_end_s: float
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass
class ValidationStats:
    """闭环校验统计。"""
    total_observations: int = 0
    mean_error_s: float = 0.0    # 平均时间偏差 (预报-实测)
    rms_error_s: float = 0.0     # 均方根偏差
    max_error_s: float = 0.0     # 最大偏差
    p95_error_s: float = 0.0     # 95 百分位
    alert_count: int = 0         # 超阈值告警次数
    fallback_count: int = 0      # 回退全精度次数


class ClosedLoopValidator:
    """闭环校验器 — 比对实测星地接轨时间与预报值, 统计误差分布。

    核心机制:
        1. 记录每次实测过境 vs 预报值的偏差
        2. 滑动窗口统计误差分布 (均值, RMS, P95, 最大值)
        3. 偏差超过当前时域允许上限 → 触发告警并自动回退全精度
        4. 周期性输出校验报告, 供偏差补偿表迭代优化

    回退策略:
        - 单次超阈值: WARNING 告警, 继续观察
        - 连续 3 次超 80% 阈值: 回退全精度
        - 单次超 150% 阈值: 立即回退全精度
        - 回退后至少保持全精度 10 分钟 (冷却)

    用法:
        validator = ClosedLoopValidator(max_timing_error_s=5.0)
        validator.record(sat_id=0, gs_id=0,
                         predicted_start_s=100.0, observed_start_s=102.5,
                         predicted_end_s=300.0, observed_end_s=302.0)
        if validator.should_fallback():
            switch_to_full_sgp4()
    """

    def __init__(
        self,
        max_timing_error_s: float = 5.0,  # 接轨时刻允许偏差
        window_size: int = 50,            # 滑动窗口大小
        consecutive_threshold: int = 3,   # 连续超 80% 阈值触发回退
        immediate_fallback_factor: float = 1.5,  # 超 N 倍阈值立刻回退
        cooldown_s: float = 600.0,        # 回退冷却时间 (10min)
    ):
        self._max_error_s = max_timing_error_s
        self._window_size = window_size
        self._consecutive_threshold = consecutive_threshold
        self._immediate_factor = immediate_fallback_factor
        self._cooldown_s = cooldown_s

        self._observations: deque[PassObservation] = deque(maxlen=window_size)
        self._errors: deque[float] = deque(maxlen=window_size)  # 时间偏差 (预报-实测)
        self._position_errors: deque[float] = deque(maxlen=window_size)

        self._alert_count: int = 0
        self._fallback_count: int = 0
        self._consecutive_warnings: int = 0
        self._last_fallback_time: float = 0.0
        self._is_fallback_mode: bool = False

    @property
    def is_fallback_mode(self) -> bool:
        return self._is_fallback_mode

    @property
    def stats(self) -> ValidationStats:
        errors = list(self._errors)
        if not errors:
            return ValidationStats()
        arr = np.array(errors)
        return ValidationStats(
            total_observations=len(self._observations),
            mean_error_s=float(np.mean(arr)),
            rms_error_s=float(np.sqrt(np.mean(arr ** 2))),
            max_error_s=float(np.max(np.abs(arr))),
            p95_error_s=float(np.percentile(np.abs(arr), 95)),
            alert_count=self._alert_count,
            fallback_count=self._fallback_count,
        )

    def record(
        self,
        sat_id: int,
        gs_id: int,
        predicted_start_s: float,
        observed_start_s: float,
        predicted_end_s: float = 0.0,
        observed_end_s: float = 0.0,
        position_error_m: float = 0.0,
    ) -> None:
        """记录一次实测 vs 预报比对。

        Args:
            predicted_start_s: 预报接轨开始时刻
            observed_start_s: 实测接轨开始时刻
            position_error_m: 预报位置偏差 (用于分层误差分配)
        """
        obs = PassObservation(
            sat_id=sat_id, gs_id=gs_id,
            predicted_start_s=predicted_start_s,
            observed_start_s=observed_start_s,
            predicted_end_s=predicted_end_s,
            observed_end_s=observed_end_s,
        )
        timing_error = predicted_start_s - observed_start_s

        self._observations.append(obs)
        self._errors.append(timing_error)
        self._position_errors.append(position_error_m)

        # 检查是否需要回退
        abs_err = abs(timing_error)
        if abs_err > self._max_error_s * self._immediate_factor:
            # 单次超 150% → 立即回退
            self._trigger_fallback(f"单次偏差 {abs_err:.1f}s 超 {self._immediate_factor}x 阈值")
        elif abs_err > self._max_error_s * 0.8:
            self._consecutive_warnings += 1
            self._alert_count += 1
            if self._consecutive_warnings >= self._consecutive_threshold:
                self._trigger_fallback(
                    f"连续 {self._consecutive_warnings} 次超 80% 阈值"
                )
        else:
            self._consecutive_warnings = 0

    def _trigger_fallback(self, reason: str) -> None:
        """触发回退全精度。"""
        now = time.perf_counter()
        if now - self._last_fallback_time < self._cooldown_s:
            return  # 冷却中
        self._is_fallback_mode = True
        self._fallback_count += 1
        self._last_fallback_time = now
        self._consecutive_warnings = 0

    def should_fallback(self) -> bool:
        """外部查询: 是否应该切换到全精度模式。"""
        if not self._is_fallback_mode:
            return False
        # 回退后检查是否可以恢复
        now = time.perf_counter()
        if now - self._last_fallback_time > self._cooldown_s:
            self._is_fallback_mode = False
            return False
        return True

    def force_recover(self) -> None:
        """强制恢复 (外部手动触发)。"""
        self._is_fallback_mode = False
        self._consecutive_warnings = 0

    def error_distribution(self) -> dict[str, float]:
        """误差分布统计。"""
        if not self._errors:
            return {}
        arr = np.array(list(self._errors))
        abs_arr = np.abs(arr)
        return {
            "mean_s": round(float(np.mean(arr)), 3),
            "rms_s": round(float(np.sqrt(np.mean(arr ** 2))), 3),
            "max_s": round(float(np.max(abs_arr)), 3),
            "p50_s": round(float(np.percentile(abs_arr, 50)), 3),
            "p95_s": round(float(np.percentile(abs_arr, 95)), 3),
            "p99_s": round(float(np.percentile(abs_arr, 99)), 3),
            "std_s": round(float(np.std(arr)), 3),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "stats": self.stats.__dict__,
            "distribution": self.error_distribution(),
            "is_fallback": self._is_fallback_mode,
            "consecutive_warnings": self._consecutive_warnings,
            "threshold_s": self._max_error_s,
        }


# ═══════════════════════════════════════════════════════════════════
#  全模型回退信号
# ═══════════════════════════════════════════════════════════════════

class FallbackToFullModelError(Exception):
    """回退全精度异常 — 被上层捕获后切换完整 SGP4。"""
    def __init__(self, reason: str = ""):
        super().__init__(f"回退完整 SGP4 全阶模型: {reason}" if reason
                         else "回退完整 SGP4 全阶模型")


# ═══════════════════════════════════════════════════════════════════
#  全局残差监控组合器
# ═══════════════════════════════════════════════════════════════════

class GlobalResidualMonitor:
    """全局残差监控组合器 — 统一管理分层误差 + 闭环校验。

    使用方式:
        monitor = GlobalResidualMonitor(error_budget, max_timing_s=5.0)
        # 每一轮定时计算后:
        monitor.record_orbit_error(orbit_err_m)
        monitor.record_pass_observation(...)
        # 检查是否需要回退:
        if monitor.should_use_full_model():
            switch_to_full_sgp4()
    """

    def __init__(self, error_budget, max_timing_error_s: float = 5.0):
        self._layered = LayeredErrorManager(from_budget=error_budget)
        self._validator = ClosedLoopValidator(max_timing_error_s=max_timing_error_s)
        self._force_full_model: bool = False
        self._force_reason: str = ""

    @property
    def layered(self) -> LayeredErrorManager:
        return self._layered

    @property
    def validator(self) -> ClosedLoopValidator:
        return self._validator

    def record_orbit_error(self, error_m: float) -> None:
        self._layered.record_orbit(error_m)

    def record_geometry_error(self, error_m: float) -> None:
        self._layered.record_geometry(error_m)

    def record_interp_error(self, error_m: float) -> None:
        self._layered.record_interp(error_m)

    def record_pass_observation(
        self, sat_id: int, gs_id: int,
        predicted_start_s: float, observed_start_s: float,
        predicted_end_s: float = 0.0, observed_end_s: float = 0.0,
        position_error_m: float = 0.0,
    ) -> None:
        self._validator.record(sat_id, gs_id, predicted_start_s, observed_start_s,
                               predicted_end_s, observed_end_s, position_error_m)

    def should_use_full_model(self) -> tuple[bool, str]:
        """判定是否应使用完整 SGP4 全阶模型。

        Returns:
            (should_use_full, reason)
        """
        # 检查分层误差
        if self._layered.any_critical:
            return True, "分层误差超标"

        # 检查闭环校验
        if self._validator.should_fallback():
            return True, "闭环校验超阈值触发回退"

        # 检查强制全模型标记
        if self._force_full_model:
            return True, self._force_reason

        return False, ""

    def force_full_model(self, reason: str = "") -> None:
        """外部强制启用全模型。"""
        self._force_full_model = True
        self._force_reason = reason

    def release_full_model(self) -> None:
        self._force_full_model = False
        self._force_reason = ""
        self._validator.force_recover()

    def summary(self) -> dict:
        return {
            "layered_errors": self._layered.summary(),
            "closed_loop": self._validator.summary(),
            "force_full_model": self._force_full_model,
        }
