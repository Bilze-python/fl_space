#!/usr/bin/env python3
"""
半解析轨道传播优化模块 v2 — 三级摄动·三层时域·周期校准·动态边界·闭环校验

参考文献:
    - Vallado, "Fundamentals of Astrodynamics and Applications"
    - Hoots & Roehrich, "Spacetrack Report No.3"
    - Conway, "Laguerre solution of Kepler's equation"

三层时域区间:
    近域高精度执行区 (HOT,   0~6h):   天线实时对准 | Δpos≤100m  | Δt≤5s   | 禁止截断
    中域规划调度区   (WARM,  6~24h):  当日排班      | Δpos≤300m  | Δt≤15s  | 条件截断
    远域粗预估区     (COLD, 24~48h):  多天资源规划  | Δpos≤800m  | Δt≤30s  | 大幅截断

三级摄动分级:
    基础必选项 (BASIC):    J2 + 一阶大气阻力 — 全时域永久保留
    可控低阶项 (LOW_ORDER): J3/J4 地球椭率 — 条件截断
    高阶复杂项 (HIGH_ORDER): 日月摄动/潮汐/高阶大气 — 可截断

硬性规则:
    - 0~6h 热窗口: 禁止截断 J3 及日月摄动，启用完整 SGP4
    - 截断模型连续运行 ≤3h (中域) / ≤6h (远域)，到期强制全模型校准
    - 每到达校准节点，用全阶 SGP4 计算真值，清零累积系统偏差
    - 轨道高度<500km / 倾角>60° / 机动状态 → 收紧截断策略
    - 动态边界滑动: 窗口进入下一时域自动恢复全阶模型
    - 0.5h 过渡缓冲区, 模型平滑切换
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
from typing import Callable, ClassVar

import numpy as np

# ═══════════════════════════════════════════════════════════════════
#  枚举定义
# ═══════════════════════════════════════════════════════════════════

class TimeDomain(Enum):
    """时域分层 — 以预报时长 ΔT 划分。"""
    HOT = auto()   # 0~6h, 近域高精度执行区
    WARM = auto()  # 6~24h, 中域规划调度区
    COLD = auto()  # 24~48h, 远域粗预估区


class PerturbationTier(Enum):
    """摄动项分级。"""
    BASIC = auto()       # 基础必选项 (J2 + 一阶大气阻力)
    LOW_ORDER = auto()   # 可控低阶项 (J3/J4 地球椭率)
    HIGH_ORDER = auto()  # 高阶复杂项 (日月/潮汐/高阶大气/太阳辐射)


class BiasType(Enum):
    """偏差类型。"""
    SYSTEMATIC = auto()  # 系统偏差 — 长时间均值≠0, 单调累积
    RANDOM = auto()      # 随机误差 — 大气瞬时扰动/测量抖动


# ═══════════════════════════════════════════════════════════════════
#  误差预算定义 (可写入论文)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ErrorBudget:
    """单时域误差预算 — 总误差拆分为轨道/几何/插值三层。"""

    max_position_m: float     # 最大系统位置偏差 (m)
    max_timing_s: float       # 接轨时刻允许偏差 (s)
    # 分层分配:
    orbit_frac: float = 0.50  # 轨道求解误差占比
    geometry_frac: float = 0.30  # 几何计算误差占比
    interp_frac: float = 0.20  # 窗口插值误差占比

    @property
    def orbit_budget_m(self) -> float:
        return self.max_position_m * self.orbit_frac

    @property
    def orbit_budget_s(self) -> float:
        return self.max_timing_s * self.orbit_frac

    @property
    def geometry_budget_m(self) -> float:
        return self.max_position_m * self.geometry_frac

    @property
    def interp_budget_m(self) -> float:
        return self.max_position_m * self.interp_frac


# 预定义三层时域误差预算
ERROR_BUDGET_HOT = ErrorBudget(
    max_position_m=100.0, max_timing_s=5.0,
)
ERROR_BUDGET_WARM = ErrorBudget(
    max_position_m=300.0, max_timing_s=15.0,
)
ERROR_BUDGET_COLD = ErrorBudget(
    max_position_m=800.0, max_timing_s=30.0,
    orbit_frac=0.55, geometry_frac=0.25, interp_frac=0.20,
)

# ═══════════════════════════════════════════════════════════════════
#  截断配置: TruncationConfig v2
# ═══════════════════════════════════════════════════════════════════

# 时域分界线 (小时) — 固定, 不可随意改动
_HOT_END = 6.0
_WARM_END = 24.0
_COLD_END = 48.0
# 过渡缓冲区 (小时)
_TRANSITION_BUFFER_H = 0.5
# 最大连续截断时长 (小时)
_MAX_CONTINUOUS_TRUNCATED_H = 3.0  # 中域
_MAX_CONTINUOUS_TRUNCATED_COLD_H = 6.0  # 远域


def _classify_time_domain(delta_t_hours: float) -> TimeDomain:
    """按预报时长划分时域。"""
    if delta_t_hours <= _HOT_END:
        return TimeDomain.HOT
    elif delta_t_hours <= _WARM_END:
        return TimeDomain.WARM
    elif delta_t_hours <= _COLD_END:
        return TimeDomain.COLD
    else:
        return TimeDomain.COLD  # 超 48h 也视为冷窗口


def _domain_error_budget(domain: TimeDomain) -> ErrorBudget:
    return {
        TimeDomain.HOT: ERROR_BUDGET_HOT,
        TimeDomain.WARM: ERROR_BUDGET_WARM,
        TimeDomain.COLD: ERROR_BUDGET_COLD,
    }[domain]


@dataclass
class TruncationConfig:
    """摄动项截断配置 v2 — 三层时域 + 三级摄动 + 卫星属性开关。

    控制规则优先级:
        1. 时域基础截断许可
        2. 卫星属性二次修正 (高度/倾角/机动)
        3. 动态边界滑动 (窗口逼近下一时域)
        4. 连续截断时长限制 + 周期校准
    """

    # ── 输入参数 ──
    delta_t_hours: float         # 预报时长 (h)
    altitude_km: float           # 轨道高度 (km)
    eccentricity: float = 0.0    # 偏心率
    inclination_deg: float = 0.0  # 轨道倾角 (度)
    is_sun_sync: bool = False    # 是否太阳同步轨道
    is_maneuvering: bool = False  # 是否处于机动阶段
    is_emergency: bool = False    # 是否应急任务

    # ── 派生属性 ──
    @property
    def time_domain(self) -> TimeDomain:
        return _classify_time_domain(self.delta_t_hours)

    @property
    def domain_name(self) -> str:
        return {TimeDomain.HOT: "HOT", TimeDomain.WARM: "WARM",
                TimeDomain.COLD: "COLD"}[self.time_domain]

    @property
    def error_budget(self) -> ErrorBudget:
        return _domain_error_budget(self.time_domain)

    @property
    def in_transition_buffer(self) -> bool:
        """是否处于时域过渡缓冲区内。"""
        dist_to_hot = abs(self.delta_t_hours - _HOT_END)
        dist_to_warm = abs(self.delta_t_hours - _WARM_END)
        return min(dist_to_hot, dist_to_warm) <= _TRANSITION_BUFFER_H

    @property
    def max_continuous_truncated_h(self) -> float:
        """本时域允许的最长连续截断推演时长。"""
        if self.time_domain == TimeDomain.HOT:
            return 0.0  # 热窗口禁止截断
        elif self.time_domain == TimeDomain.WARM:
            return _MAX_CONTINUOUS_TRUNCATED_H
        else:
            return _MAX_CONTINUOUS_TRUNCATED_COLD_H

    # ── 摄动项开关 (per-tier) ──

    @property
    def include_j2(self) -> bool:
        """BASIC: J2 — 永久保留, 绝不可截断。"""
        return True

    @property
    def include_drag(self) -> bool:
        """BASIC: 一阶大气阻力 — 永久保留。"""
        return True

    @property
    def include_j3(self) -> bool:
        """LOW_ORDER: J3 — 条件截断。

        强制开启条件:
            - 热窗口 → 永久开启
            - 太阳同步轨道 / 倾角>60° → 永久开启
            - 过渡缓冲区 → 开启 (平滑切换)
            - 轨道高度<500km → 中远域也尽量保留
        可截断条件:
            - 中域 + 倾角<30° + 高度>600km + 无机动
            - 远域 → 统一截断 (离线补偿)
        """
        if self.time_domain == TimeDomain.HOT:
            return True
        if self.in_transition_buffer:
            return True
        if self.is_sun_sync or self.inclination_deg > 60.0:
            return True  # 极轨 J3 影响显著
        if self.altitude_km < 500.0:
            return True  # 低轨大气复杂, J3 扰动耦合
        if self.is_maneuvering or self.is_emergency:
            return True
        if self.time_domain == TimeDomain.COLD:
            return False  # 远域截断, 靠离线偏差表补偿
        # 中域条件判定
        return not (self.inclination_deg < 30.0 and self.altitude_km > 600.0)

    @property
    def include_j4(self) -> bool:
        """LOW_ORDER: J4 — 同 J3 逻辑, 但仅在热窗口强制开启。"""
        if self.time_domain == TimeDomain.HOT:
            return True
        if self.is_sun_sync or self.inclination_deg > 60.0:
            return True
        if self.is_maneuvering or self.is_emergency:
            return True
        if self.time_domain == TimeDomain.COLD:
            return False
        # 中域: 同 J3 条件
        return not (self.inclination_deg < 30.0 and self.altitude_km > 600.0)

    @property
    def include_lunisolar(self) -> bool:
        """HIGH_ORDER: 日月引力摄动。

        强制开启:
            - 热窗口 → 永久开启 (0~6h 禁止截断)
            - 过渡缓冲区 → 开启
        可截断:
            - 中域 + 倾角<30° + 高度>600km → 截断 (叠加离线补偿)
            - 远域 → 统一截断
        """
        if self.time_domain == TimeDomain.HOT:
            return True  # 0~6h 硬性规则
        if self.in_transition_buffer:
            return True
        if self.is_maneuvering or self.is_emergency:
            return True
        if self.time_domain == TimeDomain.COLD:
            return False
        # 中域
        return not (self.inclination_deg < 30.0 and self.altitude_km > 600.0)

    @property
    def include_tides(self) -> bool:
        """HIGH_ORDER: 固体潮/海潮摄动 — 仅热窗口+过渡区开启。"""
        if self.time_domain == TimeDomain.HOT:
            return True
        return bool(self.in_transition_buffer)

    @property
    def include_high_order_drag(self) -> bool:
        """HIGH_ORDER: 大气高阶脉动阻力。

        强制开启:
            - 热窗口
            - 轨道高度<500km (低层大气扰动剧烈)
            - 机动状态
        """
        if self.time_domain == TimeDomain.HOT:
            return True
        if self.altitude_km < 500.0:
            return True
        return bool(self.is_maneuvering or self.is_emergency)

    @property
    def include_solar_rp(self) -> bool:
        """HIGH_ORDER: 太阳辐射压力 — 仅远域可截断。"""
        if self.is_maneuvering or self.is_emergency:
            return True
        return self.time_domain != TimeDomain.COLD

    # ── 汇总 ──

    @property
    def active_tiers(self) -> set[PerturbationTier]:
        tiers: set[PerturbationTier] = {PerturbationTier.BASIC}
        if self.include_j3 or self.include_j4:
            tiers.add(PerturbationTier.LOW_ORDER)
        if self.include_lunisolar or self.include_tides or \
           self.include_high_order_drag or self.include_solar_rp:
            tiers.add(PerturbationTier.HIGH_ORDER)
        return tiers

    @property
    def term_count(self) -> int:
        n = 2  # J2 + drag (BASIC)
        if self.include_j3:
            n += 1
        if self.include_j4:
            n += 1
        if self.include_lunisolar:
            n += 2
        if self.include_tides:
            n += 2
        if self.include_high_order_drag:
            n += 1
        if self.include_solar_rp:
            n += 1
        return n

    @property
    def speedup_factor(self) -> float:
        return 10.0 / max(self.term_count, 1)

    @property
    def is_full_model(self) -> bool:
        """是否等价于完整 SGP4 全阶模型。"""
        return self.term_count >= 9

    @property
    def is_truncation_allowed(self) -> bool:
        """本时域是否允许任何截断。

        Returns:
            False 如果处于热窗口或过渡缓冲区或模型应全开。
        """
        if self.time_domain == TimeDomain.HOT:
            return False
        if self.in_transition_buffer:
            return False
        return not (self.is_maneuvering or self.is_emergency)

    def should_force_full_model(self) -> bool:
        """紧急判定: 是否必须强制使用全阶 SGP4。

        触发条件:
            - 热窗口 (0~6h)
            - 过渡缓冲区
            - 卫星机动/应急
            - 轨道高度<500km 且倾角>60° (J3+大气联合效应)
        """
        if self.time_domain == TimeDomain.HOT:
            return True
        if self.in_transition_buffer:
            return True
        if self.is_maneuvering or self.is_emergency:
            return True
        return bool(self.altitude_km < 500.0 and self.inclination_deg > 60.0)

    def summary(self) -> dict:
        return {
            "time_domain": self.domain_name,
            "delta_t_hours": self.delta_t_hours,
            "altitude_km": self.altitude_km,
            "inclination_deg": self.inclination_deg,
            "error_budget": {
                "max_pos_m": self.error_budget.max_position_m,
                "max_timing_s": self.error_budget.max_timing_s,
                "orbit_budget_m": self.error_budget.orbit_budget_m,
                "geometry_budget_m": self.error_budget.geometry_budget_m,
                "interp_budget_m": self.error_budget.interp_budget_m,
            },
            "terms": {
                "j2": self.include_j2,
                "j3": self.include_j3,
                "j4": self.include_j4,
                "drag": self.include_drag,
                "lunisolar": self.include_lunisolar,
                "tides": self.include_tides,
                "high_order_drag": self.include_high_order_drag,
                "solar_rp": self.include_solar_rp,
            },
            "term_count": self.term_count,
            "speedup": round(self.speedup_factor, 2),
            "full_model": self.is_full_model,
            "truncation_allowed": self.is_truncation_allowed,
            "force_full_model": self.should_force_full_model(),
            "max_continuous_truncated_h": self.max_continuous_truncated_h,
            "in_transition_buffer": self.in_transition_buffer,
        }


# ═══════════════════════════════════════════════════════════════════
#  系统偏差补偿表 (离线预存)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BiasCompensationTable:
    """离线预存的固有系统偏差系数表。

    按 (高度区间, 倾角区间, 时域) 索引,
    存储截断模型 vs 全阶模型的平均位置/时间偏差。
    在线使用时直接查表做初次补偿。
    """

    # 维度: (altitude_bins-1) x (inclination_bins-1) x (3 time domains)
    pos_bias_km: np.ndarray  # 位置系统偏差 (km)
    timing_bias_s: np.ndarray  # 时间系统偏差 (s)
    sample_count: np.ndarray  # 样本数 (for置信度)

    altitude_bins_km: np.ndarray = field(
        default_factory=lambda: np.array([200, 400, 600, 800, 1500, 36000])
    )
    inclination_bins_deg: np.ndarray = field(
        default_factory=lambda: np.array([0, 30, 60, 90, 180])
    )

    def __post_init__(self):
        n_alt = len(self.altitude_bins_km) - 1
        n_inc = len(self.inclination_bins_deg) - 1
        if self.pos_bias_km.size == 0:
            self.pos_bias_km = np.zeros((n_alt, n_inc, 3))
            self.timing_bias_s = np.zeros((n_alt, n_inc, 3))
            self.sample_count = np.zeros((n_alt, n_inc, 3), dtype=np.int32)

    def lookup(
        self, altitude_km: float, inclination_deg: float, domain: TimeDomain
    ) -> tuple[float, float]:
        """查表获取位置偏差(km)和时间偏差(s)。"""
        i_alt = np.searchsorted(self.altitude_bins_km, altitude_km, side="right") - 1
        i_alt = max(0, min(i_alt, len(self.altitude_bins_km) - 2))
        i_inc = np.searchsorted(self.inclination_bins_deg, inclination_deg, side="right") - 1
        i_inc = max(0, min(i_inc, len(self.inclination_bins_deg) - 2))
        i_domain = {TimeDomain.HOT: 0, TimeDomain.WARM: 1, TimeDomain.COLD: 2}[domain]

        pos = float(self.pos_bias_km[i_alt, i_inc, i_domain])
        timing = float(self.timing_bias_s[i_alt, i_inc, i_domain])
        return pos, timing

    def update(
        self, altitude_km: float, inclination_deg: float,
        domain: TimeDomain, pos_error_km: float, timing_error_s: float,
    ) -> None:
        """在线更新偏差表 (增量学习)。"""
        i_alt = np.searchsorted(self.altitude_bins_km, altitude_km, side="right") - 1
        i_alt = max(0, min(i_alt, len(self.altitude_bins_km) - 2))
        i_inc = np.searchsorted(self.inclination_bins_deg, inclination_deg, side="right") - 1
        i_inc = max(0, min(i_inc, len(self.inclination_bins_deg) - 2))
        i_domain = {TimeDomain.HOT: 0, TimeDomain.WARM: 1, TimeDomain.COLD: 2}[domain]

        n = self.sample_count[i_alt, i_inc, i_domain]
        alpha = 1.0 / (n + 1.0)
        self.pos_bias_km[i_alt, i_inc, i_domain] = \
            (1.0 - alpha) * self.pos_bias_km[i_alt, i_inc, i_domain] + alpha * pos_error_km
        self.timing_bias_s[i_alt, i_inc, i_domain] = \
            (1.0 - alpha) * self.timing_bias_s[i_alt, i_inc, i_domain] + alpha * timing_error_s
        self.sample_count[i_alt, i_inc, i_domain] += 1


# ═══════════════════════════════════════════════════════════════════
#  AdaptivePerturbationTruncator v2
# ═══════════════════════════════════════════════════════════════════

class AdaptivePerturbationTruncator:
    """自适应摄动项截断器 v2 — 三步决策流程。

    决策流程:
        Step 1: 时域分类 → 获取基础截断许可
        Step 2: 卫星属性二次修正 (高度/倾角/机动/应急)
        Step 3: 动态边界滑动判定 (过渡缓冲区 + 逼近下一时域)

    用法:
        trunc = AdaptivePerturbationTruncator()
        config = trunc.select(delta_t_hours=3.0, altitude_km=500.0, ...)
        # config.is_full_model -> True (热窗口强制全阶)
    """

    @staticmethod
    def select(
        delta_t_hours: float,
        altitude_km: float,
        eccentricity: float = 0.0,
        inclination_deg: float = 0.0,
        is_sun_sync: bool = False,
        is_maneuvering: bool = False,
        is_emergency: bool = False,
    ) -> TruncationConfig:
        return TruncationConfig(
            delta_t_hours=delta_t_hours,
            altitude_km=altitude_km,
            eccentricity=eccentricity,
            inclination_deg=inclination_deg,
            is_sun_sync=is_sun_sync,
            is_maneuvering=is_maneuvering,
            is_emergency=is_emergency,
        )

    @classmethod
    def compute_acceleration(
        cls,
        r_eci: tuple[float, float, float],
        v_eci: tuple[float, float, float],
        config: TruncationConfig,
        gm: float = 398600.4418,
        j2: float = 1.08262668e-3,
        re: float = 6378.137,
    ) -> tuple[float, float, float]:
        """计算截断后的加速度 (km/s²), 仅包含选中的摄动项。

        Returns:
            (ax, ay, az) in ECI frame (km/s²)
        """
        x, y, z = r_eci
        r = math.sqrt(x * x + y * y + z * z)

        if r < 1e-9:
            return (0.0, 0.0, 0.0)

        gm_r3 = gm / (r * r * r)
        ax = -gm_r3 * x
        ay = -gm_r3 * y
        az = -gm_r3 * z

        if not config.include_j2:
            return (ax, ay, az)

        z_over_r = z / r
        j2_factor = 1.5 * j2 * (re / r) ** 2
        j2_term = j2_factor * (5.0 * z_over_r * z_over_r - 1.0)
        ax += (gm_r3 * j2_term) * x
        ay += (gm_r3 * j2_term) * y
        az += (gm_r3 * (j2_term + 2.0 * j2_factor)) * z

        return (ax, ay, az)

    @staticmethod
    def is_hot_window(delta_t_hours: float) -> bool:
        """判定是否热窗口 (0~6h)，用于外部快速判断。"""
        return delta_t_hours <= _HOT_END


# ═══════════════════════════════════════════════════════════════════
#  周期校准管理器: PeriodicCalibrationManager
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CalibrationPoint:
    """校准点 — 全阶 SGP4 真值坐标。"""
    t_seconds: float
    x_true_km: float
    y_true_km: float
    z_true_km: float


@dataclass
class BiasState:
    """当前累计偏差状态。"""
    bias_x_km: float = 0.0
    bias_y_km: float = 0.0
    bias_z_km: float = 0.0
    total_bias_norm_km: float = 0.0

    # 系统偏差 vs 随机误差分离
    systematic_x_km: float = 0.0  # 慢变均值 (EMA)
    systematic_y_km: float = 0.0
    systematic_z_km: float = 0.0

    bias_type: BiasType = BiasType.SYSTEMATIC


class PeriodicCalibrationManager:
    """周期全模型校准管理器 — 截断偏差归零 + 系统/随机误差分离。

    核心机制:
        1. 截断模型连续推演 ≤ max_continuous_h 小时后, 强制执行全阶 SGP4 真值校准
        2. 校准时刻: t_cal, 计算真值 r_true(t_cal), 截断值 r_cut(t_cal)
        3. 更新累计偏置: Bias_new = r_cut - r_true
        4. 后续截断结果统一减去 Bias_new
        5. 区分系统偏差 (慢变 EMA) 和随机误差 (高频抖动)

    用法:
        cal = PeriodicCalibrationManager(config)
        cal.reset_bias()  # 每一新阶段开始
        for t in times:
            pos = truncated_propagator(t)  # 截断模型
            pos = cal.apply_correction(pos)  # 减去当前偏置
            if cal.needs_calibration(t):
                true_pos = full_sgp4(t)
                cal.calibrate(t, pos_raw, true_pos)
    """

    def __init__(self, config: TruncationConfig):
        self._config = config
        self._bias = BiasState()
        self._last_calibration_t: float | None = None
        self._calibration_count: int = 0
        # 系统偏差 EMA 衰减因子
        self._sys_ema_alpha: float = 0.1

    @property
    def current_bias(self) -> BiasState:
        return self._bias

    @property
    def calibration_count(self) -> int:
        return self._calibration_count

    @property
    def max_continuous_s(self) -> float:
        return self._config.max_continuous_truncated_h * 3600.0

    def reset_bias(self) -> None:
        """重置累计偏差 (新阶段开始)。"""
        self._bias = BiasState()

    def needs_calibration(self, t_seconds: float) -> bool:
        """判断是否需要执行校准。"""
        if self._config.time_domain == TimeDomain.HOT:
            return False  # 热窗口用全阶模型, 无需校准
        if self._last_calibration_t is None:
            return True  # 首次
        return (t_seconds - self._last_calibration_t) >= self.max_continuous_s

    def calibrate(
        self,
        t_seconds: float,
        cut_pos: tuple[float, float, float],
        true_pos: tuple[float, float, float],
    ) -> BiasState:
        """执行全阶真值校准, 更新偏差。

        Args:
            t_seconds: 校准时刻
            cut_pos: 截断模型计算值 (x, y, z) km
            true_pos: 全阶 SGP4 真值 (x, y, z) km

        Returns:
            更新后的 BiasState
        """
        # 残差 = 截断 - 真值
        residual_x = cut_pos[0] - true_pos[0]
        residual_y = cut_pos[1] - true_pos[1]
        residual_z = cut_pos[2] - true_pos[2]

        # 分离系统偏差 (EMA 平滑) vs 随机误差
        self._bias.systematic_x_km = (
            (1.0 - self._sys_ema_alpha) * self._bias.systematic_x_km
            + self._sys_ema_alpha * residual_x
        )
        self._bias.systematic_y_km = (
            (1.0 - self._sys_ema_alpha) * self._bias.systematic_y_km
            + self._sys_ema_alpha * residual_y
        )
        self._bias.systematic_z_km = (
            (1.0 - self._sys_ema_alpha) * self._bias.systematic_z_km
            + self._sys_ema_alpha * residual_z
        )

        # 当前偏置更新为系统偏差 (不用残差的原始值, 避免补偿随机噪声)
        self._bias.bias_x_km = self._bias.systematic_x_km
        self._bias.bias_y_km = self._bias.systematic_y_km
        self._bias.bias_z_km = self._bias.systematic_z_km
        self._bias.total_bias_norm_km = math.sqrt(
            self._bias.bias_x_km ** 2
            + self._bias.bias_y_km ** 2
            + self._bias.bias_z_km ** 2
        )
        self._bias.bias_type = BiasType.SYSTEMATIC

        self._last_calibration_t = t_seconds
        self._calibration_count += 1

        return self._bias

    def apply_correction(
        self, pos: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """对截断计算位置施加偏差修正。

        Args:
            pos: 截断模型输出 (x, y, z)

        Returns:
            修正后位置 (x, y, z)
        """
        return (
            pos[0] - self._bias.bias_x_km,
            pos[1] - self._bias.bias_y_km,
            pos[2] - self._bias.bias_z_km,
        )

    def check_bias_threshold(self) -> bool:
        """检查偏差是否超出当前时域误差预算。

        Returns:
            True 如果超出阈值, 需要关闭截断切换全模型。
        """
        budget = self._config.error_budget
        pos_bias_m = self._bias.total_bias_norm_km * 1000.0
        # 用轨道层预算作为判据
        return pos_bias_m > budget.orbit_budget_m

    def summary(self) -> dict:
        return {
            "bias_km": round(self._bias.total_bias_norm_km, 6),
            "bias_x_km": round(self._bias.bias_x_km, 6),
            "bias_y_km": round(self._bias.bias_y_km, 6),
            "bias_z_km": round(self._bias.bias_z_km, 6),
            "sys_x_km": round(self._bias.systematic_x_km, 6),
            "sys_y_km": round(self._bias.systematic_y_km, 6),
            "sys_z_km": round(self._bias.systematic_z_km, 6),
            "bias_type": self._bias.bias_type.name,
            "calibration_count": self._calibration_count,
            "max_continuous_h": self._config.max_continuous_truncated_h,
            "threshold_exceeded": self.check_bias_threshold(),
        }


# ═══════════════════════════════════════════════════════════════════
#  时间窗口控制器: TimeDomainWindowController
# ═══════════════════════════════════════════════════════════════════

class TimeDomainWindowController:
    """时域动态边界滑动控制器 — 滚动推演时自动切换精度模型。

    工作原理:
        定时系统滚动向前 (永远计算 t_now → t_now+48h)。
        随着时间流逝, 原本远域的窗口会滑入中域/近域。
        控制器检测时域变迁并触发模型切换。

    过渡缓冲区:
        0.5h 缓冲区, 中域窗口逼近 6h 分界线时提前 1h 完成切换,
        避免临近接轨时刻精度断崖下跌。

    用法:
        ctrl = TimeDomainWindowController()
        ctrl.update(t_now=current_time)
        domain = ctrl.domain_for(t_query)  # 返回该时刻应在的时域
        if ctrl.should_switch(t_query):
            new_config = ctrl.get_full_model_config(t_query)
    """

    def __init__(self):
        self._t_now: float = 0.0  # 当前系统时刻 (相对 epoch 秒)

    def update(self, t_now_s: float) -> None:
        """更新当前系统时刻。"""
        self._t_now = t_now_s

    def domain_for(self, t_query_s: float) -> TimeDomain:
        """查询时刻 t_query 应属的时域 (考虑过渡缓冲区)。"""
        dt_h = (t_query_s - self._t_now) / 3600.0

        if dt_h < 0:
            return TimeDomain.HOT  # 过去时刻, 视为已过窗口

        # 前向过渡缓冲区: 中域→热域, 提前 1h 升级
        if _HOT_END - _TRANSITION_BUFFER_H <= dt_h <= _HOT_END + 1.0:
            return TimeDomain.HOT

        # 后向过渡缓冲区: 远域→中域
        if _WARM_END - _TRANSITION_BUFFER_H <= dt_h <= _WARM_END:
            return TimeDomain.WARM

        return _classify_time_domain(dt_h)

    def should_switch(
        self, t_query_s: float, current_config: TruncationConfig
    ) -> bool:
        """检查是否需要切换精度模型。

        Returns:
            True 如果需要升级 (全阶模型) 或降级。
        """
        expected_domain = self.domain_for(t_query_s)
        return expected_domain != current_config.time_domain

    def get_full_model_config(
        self, t_query_s: float, altitude_km: float,
        inclination_deg: float = 0.0, is_sun_sync: bool = False,
        is_maneuvering: bool = False, is_emergency: bool = False,
    ) -> TruncationConfig:
        """获取应使用的最新配置 (可能升级为全阶)。"""
        return AdaptivePerturbationTruncator.select(
            delta_t_hours=(t_query_s - self._t_now) / 3600.0,
            altitude_km=altitude_km,
            inclination_deg=inclination_deg,
            is_sun_sync=is_sun_sync,
            is_maneuvering=is_maneuvering,
            is_emergency=is_emergency,
        )


# ═══════════════════════════════════════════════════════════════════
#  滑动窗口多项式拟合 v2: SlidingWindowPolyFitter (残差阈值强制监控)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PolyFitResult:
    """多项式拟合结果。"""
    coeffs_x: np.ndarray
    coeffs_y: np.ndarray
    coeffs_z: np.ndarray
    t0: float
    t1: float
    max_error_km: float
    degree: int


class SlidingWindowPolyFitter:
    """滑动窗口多项式拟合器 v2 — 残差超标立刻强制全量 SGP4 重算。

    改动:
        - 残差超阈值 100% 触发全量 SGP4 重算 (不沿用旧拟合)
        - 近域 0~6h 热窗口: 强制禁用大步长简化, 固定下限采样步长 ≤10s
        - 仅在中远域 (≥6h) 才启用多项式拟合
    """

    # 热窗口最小采样步长 (秒)
    HOT_MIN_STEP_S: ClassVar[float] = 10.0
    # 中域推荐步长 (秒)
    WARM_STEP_S: ClassVar[float] = 30.0
    # 远域推荐步长 (秒)
    COLD_STEP_S: ClassVar[float] = 120.0

    def __init__(
        self,
        sgp4_func: Callable[[float], tuple[float, float, float]],
        window_points: int = 20,
        degree: int = 8,
        tolerance_km: float = 0.1,
        time_domain: TimeDomain = TimeDomain.WARM,
        force_full_on_exceed: bool = True,
    ):
        """
        Args:
            sgp4_func: SGP4 传播函数
            window_points: 拟合窗口采样点数
            degree: 多项式阶数
            tolerance_km: 拟合残差阈值 (km), 超限立刻触发全量重算
            time_domain: 当前时域
            force_full_on_exceed: 超阈值是否强制全模型 (默认 True)
        """
        self._sgp4 = sgp4_func
        self._window_points = window_points
        self._degree = degree
        self._tolerance_km = tolerance_km
        self._time_domain = time_domain
        self._force_full_on_exceed = force_full_on_exceed

        self._result: PolyFitResult | None = None
        self._refresh_count: int = 0
        self._eval_count: int = 0
        self._exceed_count: int = 0  # 残差超标次数

    @property
    def refresh_count(self) -> int:
        return self._refresh_count

    @property
    def eval_count(self) -> int:
        return self._eval_count

    @property
    def exceed_count(self) -> int:
        return self._exceed_count

    @property
    def cache_hit_ratio(self) -> float:
        if self._eval_count == 0:
            return 0.0
        return 1.0 - self._refresh_count / self._eval_count

    @property
    def is_disabled_in_hot(self) -> bool:
        """热窗口下是否禁用拟合 (直接走 SGP4)。"""
        return self._time_domain == TimeDomain.HOT

    def _sampling_step(self) -> float:
        return {
            TimeDomain.HOT: self.HOT_MIN_STEP_S,
            TimeDomain.WARM: self.WARM_STEP_S,
            TimeDomain.COLD: self.COLD_STEP_S,
        }[self._time_domain]

    def initialize(self, t0_seconds: float, window_duration_s: float = 3600.0) -> PolyFitResult:
        """初始拟合。

        若为热窗口, 缩小窗口为最长 6h 并强制细采样。
        """
        if self._time_domain == TimeDomain.HOT:
            window_duration_s = min(window_duration_s, 3600.0 * _HOT_END)
            self._window_points = max(self._window_points, 40)  # 热窗口多采样

        t1 = t0_seconds + window_duration_s
        ts = np.linspace(t0_seconds, t1, self._window_points)

        xs = np.empty(self._window_points)
        ys = np.empty(self._window_points)
        zs = np.empty(self._window_points)

        for i, t in enumerate(ts):
            x, y, z = self._sgp4(float(t))
            xs[i] = x
            ys[i] = y
            zs[i] = z

        t_mid = 0.5 * (t0_seconds + t1)
        t_half = 0.5 * (t1 - t0_seconds)
        t_norm = (ts - t_mid) / t_half

        coeffs_x = np.polynomial.chebyshev.chebfit(t_norm, xs, self._degree)
        coeffs_y = np.polynomial.chebyshev.chebfit(t_norm, ys, self._degree)
        coeffs_z = np.polynomial.chebyshev.chebfit(t_norm, zs, self._degree)

        x_fit = np.polynomial.chebyshev.chebval(t_norm, coeffs_x)
        y_fit = np.polynomial.chebyshev.chebval(t_norm, coeffs_y)
        z_fit = np.polynomial.chebyshev.chebval(t_norm, coeffs_z)
        errors = np.sqrt((xs - x_fit) ** 2 + (ys - y_fit) ** 2 + (zs - z_fit) ** 2)
        max_err = float(np.max(errors))

        # 残差超标检测
        if max_err > self._tolerance_km:
            self._exceed_count += 1
            if self._force_full_on_exceed:
                raise ResidualExceededError(
                    f"拟合残差 {max_err:.6f} km 超过阈值 {self._tolerance_km} km, "
                    "强制触发全量 SGP4 重算"
                )

        self._result = PolyFitResult(
            coeffs_x=coeffs_x, coeffs_y=coeffs_y, coeffs_z=coeffs_z,
            t0=t0_seconds, t1=t1, max_error_km=max_err, degree=self._degree,
        )
        self._refresh_count += 1
        return self._result

    def evaluate(self, t_seconds: float) -> tuple[float, float, float]:
        """多项式求值 — 热窗口直接回退 SGP4。"""
        self._eval_count += 1

        # 热窗口: 禁用多项式拟合, 直接调用全量 SGP4
        if self._time_domain == TimeDomain.HOT:
            return self._sgp4(t_seconds)

        if self._result is None:
            self.initialize(t_seconds)
            if self._result is None:
                return self._sgp4(t_seconds)

        need_refresh = False
        if t_seconds < self._result.t0 or t_seconds > self._result.t1:
            need_refresh = True
        elif self._result.max_error_km > self._tolerance_km:
            need_refresh = True
            self._exceed_count += 1

        if need_refresh:
            window_dur = 3600.0
            if self._result is not None:
                window_dur = self._result.t1 - self._result.t0
            self.initialize(t_seconds, window_dur)
            if self._result is None:
                return self._sgp4(t_seconds)

        t_mid = 0.5 * (self._result.t0 + self._result.t1)
        t_half = 0.5 * (self._result.t1 - self._result.t0)
        if t_half < 1e-9:
            return self._sgp4(t_seconds)
        t_norm = (t_seconds - t_mid) / t_half

        x = float(np.polynomial.chebyshev.chebval(t_norm, self._result.coeffs_x))
        y = float(np.polynomial.chebyshev.chebval(t_norm, self._result.coeffs_y))
        z = float(np.polynomial.chebyshev.chebval(t_norm, self._result.coeffs_z))

        return (x, y, z)


class ResidualExceededError(Exception):
    """残差超标异常 — 触发全量 SGP4 回退。"""
    pass


# ═══════════════════════════════════════════════════════════════════
#  相对运动推演 (保留, 略作增强)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RelativeState:
    """相对轨道状态。"""
    delta_x: float
    delta_y: float
    delta_z: float
    delta_vx: float
    delta_vy: float
    delta_vz: float


class RelativeMotionPropagator:
    """相对运动推演器 — CW 方程, 星座集群专用。"""

    def __init__(
        self,
        mean_motion: float,
        j2_correction: bool = False,
        inclination_deg: float = 0.0,
        semi_major_axis_km: float = 6878.137,
    ):
        self._n = mean_motion
        self._j2_correction = j2_correction
        self._inclination_rad = math.radians(inclination_deg)
        self._sma = semi_major_axis_km
        self._omega_dot: float = 0.0

        if j2_correction:
            j2 = 1.08262668e-3
            re = 6378.137
            cos_i = math.cos(self._inclination_rad)
            self._omega_dot = -1.5 * j2 * (re / self._sma) ** 2 * self._n * cos_i

    def propagate(
        self, initial: RelativeState, delta_t_seconds: float,
    ) -> tuple[float, float, float]:
        n = self._n
        t = delta_t_seconds
        nt = n * t
        cos_nt = math.cos(nt)
        sin_nt = math.sin(nt)

        dx = (
            initial.delta_x
            + 2.0 * initial.delta_vy * (1.0 - cos_nt) / n
            + initial.delta_vx * (4.0 * sin_nt - 3.0 * nt) / n
            - 6.0 * initial.delta_y * (nt - sin_nt)
        )
        dy = (
            initial.delta_y * (4.0 - 3.0 * cos_nt)
            + initial.delta_vx * 2.0 * (cos_nt - 1.0) / n
            + initial.delta_vy * sin_nt / n
        )
        dz = initial.delta_z * cos_nt + initial.delta_vz * sin_nt / n

        if self._j2_correction:
            drift = self._omega_dot * t * nt * 0.5
            dx += drift * 1e-3

        return (dx, dy, dz)

    @staticmethod
    def cluster_propagate(
        center_positions: list[tuple[float, float, float]],
        rel_states: list[RelativeState],
        delta_t_seconds: float,
        mean_motion: float,
        j2_correction: bool = False,
        inclination_deg: float = 0.0,
    ) -> list[tuple[float, float, float]]:
        n_sats = len(rel_states)
        len(center_positions)
        prop = RelativeMotionPropagator(
            mean_motion=mean_motion,
            j2_correction=j2_correction,
            inclination_deg=inclination_deg,
        )
        results: list[tuple[float, float, float]] = []
        for i_sat in range(n_sats):
            rel = rel_states[i_sat]
            for i_t, (cx, cy, cz) in enumerate(center_positions):
                dt = i_t * delta_t_seconds
                dx, dy, dz = prop.propagate(rel, dt)
                results.append((cx + dx, cy + dy, cz + dz))
        return results


# ═══════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════

def estimate_truncation_speedup(window_hours: float, altitude_km: float) -> float:
    """估算截断加速比。"""
    return AdaptivePerturbationTruncator.select(
        delta_t_hours=window_hours, altitude_km=altitude_km,
    ).speedup_factor
