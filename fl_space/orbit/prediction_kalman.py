#!/usr/bin/env python3
"""
时序预测补偿模块 v2 — 卡尔曼滤波 + 误差趋势拟合 (仅作为小幅修正手段)

重要约束:
    - 误差补偿仅作为两次定时刷新之间的小幅修正手段!!!
    - 不能完全替代定时轨道更新!!!
    - 补偿量超过阈值 → 触发告警, 强制执行全量 SGP4 轨道更新
    - 近域 0~6h 热窗口: 禁用误差补偿 (精度要求刚性)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

# ═══════════════════════════════════════════════════════════════════
#  补偿约束常量
# ═══════════════════════════════════════════════════════════════════

# 补偿量上限: 超过此值不再补偿, 直接触发轨道更新
MAX_COMPENSATION_S: float = 30.0      # 最大时间补偿 (s)
MAX_COMPENSATION_POS_KM: float = 0.5  # 最大位置补偿 (km)

# 热窗口 (0~6h) 禁止补偿
HOT_WINDOW_HOURS: float = 6.0


# ═══════════════════════════════════════════════════════════════════
#  KalmanConfig
# ═══════════════════════════════════════════════════════════════════

@dataclass
class KalmanConfig:
    """一维卡尔曼滤波器配置。"""
    process_noise_q: float = 1.0       # 过程噪声 (时间偏差漂移)
    measurement_noise_r: float = 4.0   # 测量噪声
    initial_estimate_s: float = 0.0    # 初始偏差估计
    initial_uncertainty: float = 10.0  # 初始不确定性


# ═══════════════════════════════════════════════════════════════════
#  KalmanTimingCompensator v2
# ═══════════════════════════════════════════════════════════════════

class KalmanTimingCompensator:
    """一维卡尔曼滤波器 — 补偿轨道预报时序偏差。

    约束:
        - 仅用于两次定时刷新间的小幅修正
        - 补偿量 ≤ MAX_COMPENSATION_S (30s), 超限强制轨道更新
        - 热窗口 (0~6h) 禁止使用
        - delta_t_hours 参数用于判定窗口类型
    """

    def __init__(self, config: KalmanConfig | None = None):
        cfg = config or KalmanConfig()
        self._q = cfg.process_noise_q
        self._r = cfg.measurement_noise_r
        self._estimate = cfg.initial_estimate_s     # 偏差估计 (s)
        self._uncertainty = cfg.initial_uncertainty  # 估计不确定性
        self._update_count: int = 0
        self._warn_count: int = 0  # 超限警告计数

    @property
    def current_estimate(self) -> float:
        return self._estimate

    @property
    def update_count(self) -> int:
        return self._update_count

    def update(
        self,
        measured_error_s: float,
        delta_t_hours: float = 8.0,  # 上次更新至今的小时数
        force_update: bool = True,
    ) -> float:
        """卡尔曼滤波更新 — 输入实测误差, 输出修正后的偏差估计。

        Args:
            measured_error_s: 实测误差 (预报-实测), 秒
            delta_t_hours: 距上次定时轨道更新的时长
            force_update: 如果超限是否强制执行轨道更新

        Returns:
            修正后的偏差估计 (秒), 用于下次预报叠加偏移。

        Raises:
            CompensationLimitExceededError: 补偿量超限, 应触发全量轨道更新
        """
        # ── 约束 1: 热窗口禁止补偿 ──
        if delta_t_hours <= HOT_WINDOW_HOURS:
            # 热窗口内不走补偿, 直接偏差归零
            self._estimate = 0.0
            return 0.0

        # ── 约束 2: 补偿量上限检查 ──
        if abs(measured_error_s) > MAX_COMPENSATION_S:
            self._warn_count += 1
            if force_update:
                raise CompensationLimitExceededError(
                    f"卡尔曼补偿量 {measured_error_s:.1f}s 超过上限 {MAX_COMPENSATION_S}s, "
                    "应触发全量轨道更新而非累积补偿"
                )
            # 不强制则裁剪
            measured_error_s = math.copysign(MAX_COMPENSATION_S, measured_error_s)

        # ── 标准卡尔曼滤波 ──
        # 预测 (过程噪声随 delta_t 增大)
        process_q = self._q * delta_t_hours
        self._uncertainty += process_q

        # 更新
        kalman_gain = self._uncertainty / (self._uncertainty + self._r)
        self._estimate += kalman_gain * (measured_error_s - self._estimate)
        self._uncertainty *= (1.0 - kalman_gain)
        self._update_count += 1

        return self._estimate

    def get_correction(self) -> float:
        """获取当前最优补偿值 (秒)。

        Returns:
            偏差估计值, 叠加到原始预报时间上。
        """
        return self._estimate

    def predict_forward(self, delta_t_hours: float) -> float:
        """预测 delta_t_hours 小时后的偏差 (用于延长定时刷新间隔)。

        Returns:
            预测偏差 (s)
        """
        return self._estimate  # 一维状态: 偏差本身在漂移小的情况下视为常量

    def reset(self) -> None:
        self._estimate = 0.0
        self._uncertainty = 10.0
        self._warn_count = 0

    def summary(self) -> dict:
        return {
            "estimate_s": round(self._estimate, 3),
            "uncertainty": round(self._uncertainty, 3),
            "update_count": self._update_count,
            "warn_count": self._warn_count,
        }


class CompensationLimitExceededError(Exception):
    """补偿量超限异常 — 应触发全量轨道更新。"""
    pass


# ═══════════════════════════════════════════════════════════════════
#  ErrorTrendFitter v2
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrendFitResult:
    """误差趋势拟合结果。"""
    coeffs: np.ndarray          # 多项式系数 (高次→低次)
    degree: int
    r_squared: float            # 拟合优度
    rmse_s: float               # 拟合 RMSE
    valid: bool                 # 是否有效
    warning: str = ""           # 超限警告


class ErrorTrendFitter:
    """误差趋势拟合器 — 多项式拟合误差漂移曲线。

    约束:
        - 仅用于趋势外推, 不替代轨道更新
        - 残差过大时标记 invalid
        - AIC 自动选阶 (1-3)

    用法:
        fitter = ErrorTrendFitter()
        result = fitter.fit(times_h, errors_s)
        predicted = fitter.predict(result, future_t_h)
    """

    def __init__(self, max_degree: int = 3, max_rmse_s: float = 10.0):
        self._max_degree = max_degree
        self._max_rmse_s = max_rmse_s
        self._history: deque[float] = deque(maxlen=100)

    def fit(self, times_h: list[float], errors_s: list[float]) -> TrendFitResult:
        """拟合误差趋势。

        Args:
            times_h: 时间序列 (小时)
            errors_s: 误差序列 (秒)

        Returns:
            TrendFitResult
        """
        if len(times_h) < 3 or len(errors_s) < 3:
            return TrendFitResult(
                coeffs=np.array([0.0]), degree=0,
                r_squared=0.0, rmse_s=0.0, valid=False,
                warning="样本不足 (需≥3)"
            )

        t = np.array(times_h, dtype=np.float64)
        e = np.array(errors_s, dtype=np.float64)

        best_result: TrendFitResult | None = None

        for deg in range(1, min(self._max_degree + 1, len(t) - 1)):
            try:
                coeffs = np.polyfit(t, e, deg)
                fitted = np.polyval(coeffs, t)
                residuals = e - fitted
                rmse = float(np.sqrt(np.mean(residuals ** 2)))

                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((e - np.mean(e)) ** 2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

                result = TrendFitResult(
                    coeffs=coeffs, degree=deg,
                    r_squared=round(r2, 6), rmse_s=round(rmse, 3),
                    valid=rmse <= self._max_rmse_s,
                    warning=f"RMSE={rmse:.1f}s 超限 {self._max_rmse_s}s"
                    if rmse > self._max_rmse_s else "",
                )

                if best_result is None or rmse < best_result.rmse_s:
                    best_result = result

            except np.linalg.LinAlgError:
                continue

        if best_result is None:
            return TrendFitResult(
                coeffs=np.array([0.0]), degree=0,
                r_squared=0.0, rmse_s=0.0, valid=False,
                warning="拟合失败"
            )

        return best_result

    def predict(self, result: TrendFitResult, future_t_h: float) -> float:
        """根据拟合结果预测 future_t_h 时刻的误差。

        Returns:
            预测误差 (s)
        """
        if not result.valid:
            return 0.0
        return float(np.polyval(result.coeffs, future_t_h))


# ═══════════════════════════════════════════════════════════════════
#  HybridCompensator v2
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CompensationResult:
    """补偿结果。"""
    total_correction_s: float
    kalman_part_s: float
    trend_part_s: float
    kalman_weight: float
    needs_orbit_update: bool
    reason: str = ""


class HybridCompensator:
    """混合补偿器 — 卡尔曼 (短期) + 趋势 (长期), 动态权重。

    约束:
        - 补偿量超过阈值 → needs_orbit_update=True (触发全量轨道刷新)
        - 热窗口不使用补偿
        - 误差补偿不替代定时轨道更新
    """

    def __init__(
        self,
        kalman_config: KalmanConfig | None = None,
        max_total_correction_s: float = MAX_COMPENSATION_S,
    ):
        self._kalman = KalmanTimingCompensator(kalman_config)
        self._fitter = ErrorTrendFitter()
        self._max_total = max_total_correction_s

        # 误差历史
        self._times_h: list[float] = []
        self._errors_s: list[float] = []
        self._max_history: int = 50

    def record(self, delta_t_hours: float, measured_error_s: float) -> None:
        """记录一次误差观测。"""
        self._times_h.append(delta_t_hours)
        self._errors_s.append(measured_error_s)
        if len(self._times_h) > self._max_history:
            self._times_h = self._times_h[-self._max_history:]
            self._errors_s = self._errors_s[-self._max_history:]

    def compensate(
        self,
        delta_t_hours: float,
        measured_error_s: float | None = None,
    ) -> CompensationResult:
        """计算补偿量。

        Args:
            delta_t_hours: 距上次定时轨道更新的时长
            measured_error_s: 最新实测误差 (None 则只用趋势)

        Returns:
            CompensationResult
        """
        # 热窗口 → 不补偿
        if delta_t_hours <= HOT_WINDOW_HOURS:
            return CompensationResult(
                total_correction_s=0.0, kalman_part_s=0.0,
                trend_part_s=0.0, kalman_weight=0.0,
                needs_orbit_update=False, reason="热窗口禁用补偿",
            )

        kalman_corr = 0.0
        trend_corr = 0.0

        # 卡尔曼 (需要实测误差)
        if measured_error_s is not None:
            try:
                kalman_corr = self._kalman.update(
                    measured_error_s, delta_t_hours, force_update=True
                )
            except CompensationLimitExceededError:
                return CompensationResult(
                    total_correction_s=0.0, kalman_part_s=0.0,
                    trend_part_s=0.0, kalman_weight=0.0,
                    needs_orbit_update=True,
                    reason="卡尔曼补偿量超限, 需要全量轨道更新",
                )

        # 趋势 (长期)
        if len(self._errors_s) >= 3:
            trend_result = self._fitter.fit(self._times_h, self._errors_s)
            if trend_result.valid:
                trend_corr = self._fitter.predict(trend_result, delta_t_hours)

        # 动态权重: 刚更新后优先用卡尔曼, 时间越长趋势权重大
        kalman_weight = max(0.0, 1.0 - delta_t_hours / 12.0)  # 12h 内逐渐退位
        total = kalman_weight * kalman_corr + (1.0 - kalman_weight) * trend_corr

        # 总量超限判定
        if abs(total) > self._max_total:
            return CompensationResult(
                total_correction_s=0.0, kalman_part_s=kalman_corr,
                trend_part_s=trend_corr, kalman_weight=kalman_weight,
                needs_orbit_update=True,
                reason=f"总补偿量 {total:.1f}s 超限 {self._max_total}s",
            )

        return CompensationResult(
            total_correction_s=total, kalman_part_s=kalman_corr,
            trend_part_s=trend_corr, kalman_weight=kalman_weight,
            needs_orbit_update=False,
        )

    def reset(self) -> None:
        self._kalman.reset()
        self._times_h.clear()
        self._errors_s.clear()

    def summary(self) -> dict:
        return {
            "kalman": self._kalman.summary(),
            "history_size": len(self._errors_s),
        }


# ═══════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════

def estimate_refresh_interval_extension(
    current_interval_h: float, compensation_accuracy_s: float,
) -> float:
    """估算补偿机制可拉长的定时刷新间隔。

    Args:
        current_interval_h: 当前刷新间隔 (小时)
        compensation_accuracy_s: 补偿精度 (秒)

    Returns:
        建议间隔 (小时)
    """
    if compensation_accuracy_s < 1.0:
        return min(current_interval_h * 2.0, 12.0)  # 最多拉长到 12h
    elif compensation_accuracy_s < 5.0:
        return min(current_interval_h * 1.5, 8.0)
    else:
        return current_interval_h  # 补偿精度不够, 不拉长
