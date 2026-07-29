"""
自适应多层定时调度与动态负载降级模块
=========================================

本模块实现论文"七维分层改进"中的：

- **维度一：分层自适应定时 + 事件增量触发混合调度**
  - 近域高精度层（0~6h）：周期 2-3min，步长 10s
  - 中域规划层（6~24h）：周期 10min，步长 30s
  - 远域粗规划层（24~48h）：周期 30min，步长 60-120s
  - 增量事件触发：单星变轨/单站检修 -> 局部重算；多星批量变轨 -> 全量重推
  - 错峰分时：轨道递推/可视计算/窗口分配分时执行，避免算力峰值

- **维度七：动态负载自适应降载机制**
  - CPU/内存负载监控
  - 三级降级：轻度（放大远域步长）/ 中度（暂停精细插值）/ 重度（仅保近域高精度）
  - 自动恢复完整模式

Notes
-----
本模块不直接修改 OrbitSimulator / PassTimetable / GroundStationAllocator，
而是作为顶层编排调度器，控制定时推演的执行节奏与粒度。
"""

from __future__ import annotationsfrom dataclasses import dataclass, fieldfrom enum import IntEnumimport timefrom typing import TYPE_CHECKINGif TYPE_CHECKING:
    from fl_space.simulator.orbit_simulator import OrbitSimulator    from fl_space.simulator.pass_scheduler import AllocationResult, PassTimetable


# ============================================================
# 多粒度分层周期定义（论文维度一整表1）
# ============================================================

class TimingLayer(IntEnum):
    """三层定时任务分层。"""
    NEAR = 0     # 近域高精度 0~6h
    MID = 1      # 中域规划 6~24h
    FAR = 2      # 远域粗规划 24~48h


# 各层配置：周期(min)、采样步长(s)、仰角精度级别
LAYER_CONFIG = {
    TimingLayer.NEAR: {
        "period_min": 2.5,          # 2-3 min
        "sample_step_s": 10,        # 10 s
        "elevation_precision": "full",      # 完整 ENU + 岁差章动
        "store_elevation_curve": True,
        "enable_interpolation": True,
    },
    TimingLayer.MID: {
        "period_min": 10.0,         # 10 min
        "sample_step_s": 30,        # 30 s
        "elevation_precision": "standard",   # 标准 ENU，无章动
        "store_elevation_curve": False,
        "enable_interpolation": True,
    },
    TimingLayer.FAR: {
        "period_min": 30.0,         # 30 min
        "sample_step_s": 90,        # 60-120 s
        "elevation_precision": "coarse",     # 一阶地球自转修正
        "store_elevation_curve": False,
        "enable_interpolation": False,
    },
}

# 近域/中域/远域时间分界（小时）
NEAR_HORIZON_H = 6.0
MID_HORIZON_H = 24.0
FAR_HORIZON_H = 48.0

# 错峰执行偏移（分钟）：不同任务分时启动，避免算力峰值
STAGGER_OFFSETS = {
    "orbit_propagation": 0.0,       # 第 0 min 执行
    "visibility_compute": 1.0,      # 第 1 min 执行
    "window_assembly_allocate": 2.0, # 第 2 min 执行
}


# ============================================================
# 动态负载自适应降载等级（维度七）
# ============================================================

class LoadLevel(IntEnum):
    """系统负载等级。"""
    NORMAL = 0      # 正常：完整高精度
    LIGHT = 1       # 轻度：放大远域步长 2x
    MODERATE = 2    # 中度：暂停精细插值 + 关闭附加指标
    HEAVY = 3       # 重度：仅保留近域高精度，远域暂停


# 各降级等级的调节系数
DEGRADATION_CONFIG = {
    LoadLevel.NORMAL: {
        "far_step_multiplier": 1.0,
        "mid_step_multiplier": 1.0,
        "near_step_multiplier": 1.0,
        "enable_interpolation": True,
        "store_elevation_curve": True,
        "far_enabled": True,
        "mid_enabled": True,
    },
    LoadLevel.LIGHT: {
        "far_step_multiplier": 2.0,
        "mid_step_multiplier": 1.0,
        "near_step_multiplier": 1.0,
        "enable_interpolation": True,
        "store_elevation_curve": True,
        "far_enabled": True,
        "mid_enabled": True,
    },
    LoadLevel.MODERATE: {
        "far_step_multiplier": 3.0,
        "mid_step_multiplier": 1.5,
        "near_step_multiplier": 1.0,
        "enable_interpolation": False,
        "store_elevation_curve": False,
        "far_enabled": True,
        "mid_enabled": True,
    },
    LoadLevel.HEAVY: {
        "far_step_multiplier": 1.0,
        "mid_step_multiplier": 1.0,
        "near_step_multiplier": 1.0,
        "enable_interpolation": False,
        "store_elevation_curve": False,
        "far_enabled": False,
        "mid_enabled": False,
    },
}

# CPU 负载阈值（0-100%）
CPU_THRESHOLD_LIGHT = 70.0
CPU_THRESHOLD_MODERATE = 85.0
CPU_THRESHOLD_HEAVY = 95.0


# ============================================================
# 增量事件类型
# ============================================================

@dataclass
class IncrementalEvent:
    """增量触发事件。

    当发生关键事件时不再等待定时周期，立即局部/全量重算。
    """
    event_type: str                  # "sat_maneuver" / "gs_maintenance" / "batch_maneuver" / "emergency_mission"
    affected_sat_ids: list[int] = field(default_factory=list)
    affected_gs_ids: list[int] = field(default_factory=list)
    timestamp: float = 0.0           # 事件发生时刻 (Unix 时间)
    force_full: bool = False         # 是否强制全量重推


# ============================================================
# 自适应多层定时调度器
# ============================================================

class AdaptiveTimingScheduler:
    """多层自适应定时调度器 + 动态负载降级。

    替代固定周期定时逻辑，按三层粒度 + 事件触发 + 错峰分时编排定时演算，
    同时监控系统负载并自动降级/恢复。

    Parameters
    ----------
    simulator : OrbitSimulator
        轨道模拟器实例。
    """

    def __init__(self, simulator: OrbitSimulator):
        self._sim = simulator
        self._current_layer = TimingLayer.NEAR
        self._current_load_level = LoadLevel.NORMAL

        # 各层最近一次执行时间
        self._last_run: dict[TimingLayer, float] = dict.fromkeys(TimingLayer, 0.0)

        # 增量事件队列
        self._event_queue: list[IncrementalEvent] = []

        # 负载监控
        self._cpu_samples: list[float] = []
        self._load_history_size = 10

        # 错峰执行状态
        self._stagger_phase = 0  # 0=轨道, 1=可视, 2=分配

        # 当前生效的步长（受降级影响）
        self._effective_steps: dict[TimingLayer, float] = {
            layer: cfg["sample_step_s"] for layer, cfg in LAYER_CONFIG.items()
        }

        # 定期更新检查
        self._last_load_check = 0.0
        self._load_check_interval_s = 30.0  # 每 30s 检查一次负载

        # 降级恢复冷却（避免抖动）
        self._degrade_cooldown = 0.0
        self._recover_cooldown = 0.0
        self._cooldown_s = 60.0  # 冷却 60s

    # ---- 分层周期决策 ----

    def get_layer_for_horizon(self, hours_ahead: float) -> TimingLayer:
        """根据推演提前量返回对应粒度层。"""
        if hours_ahead <= NEAR_HORIZON_H:
            return TimingLayer.NEAR
        if hours_ahead <= MID_HORIZON_H:
            return TimingLayer.MID
        return TimingLayer.FAR

    def get_sample_step_for_layer(self, layer: TimingLayer) -> float:
        """返回某层当前有效步长（含降级因子）。"""
        base = LAYER_CONFIG[layer]["sample_step_s"]
        cfg = DEGRADATION_CONFIG[self._current_load_level]
        if layer == TimingLayer.NEAR:
            mult = cfg["near_step_multiplier"]
        elif layer == TimingLayer.MID:
            mult = cfg["mid_step_multiplier"]
        else:
            mult = cfg["far_step_multiplier"]
        return base * mult

    def should_run_layer(self, layer: TimingLayer, current_time: float) -> bool:
        """判断该层是否到达定时触发时刻。"""
        cfg = DEGRADATION_CONFIG[self._current_load_level]
        # 重度降级时关闭中/远域
        if layer == TimingLayer.FAR and not cfg["far_enabled"]:
            return False
        if layer == TimingLayer.MID and not cfg["mid_enabled"]:
            return False

        period_s = LAYER_CONFIG[layer]["period_min"] * 60.0
        return (current_time - self._last_run[layer]) >= period_s

    # ---- 错峰分时执行 ----

    def get_stagger_phase(self) -> str:
        """返回当前应执行的错峰阶段（0→1→2→0 循环）。"""
        phases = ["orbit_propagation", "visibility_compute", "window_assembly_allocate"]
        phase = phases[self._stagger_phase]
        self._stagger_phase = (self._stagger_phase + 1) % 3
        return phase

    def stagger_offset_min(self, phase: str) -> float:
        """返回某阶段的错峰偏移（分钟）。"""
        return STAGGER_OFFSETS.get(phase, 0.0)

    # ---- 增量事件触发 ----

    def push_event(self, event: IncrementalEvent) -> None:
        """将增量事件加入队列，下次调度循环优先处理。"""
        self._event_queue.append(event)

    def pop_events(self) -> list[IncrementalEvent]:
        """取出所有待处理事件并清空队列。"""
        events = self._event_queue.copy()
        self._event_queue.clear()
        return events

    def classify_event_scope(self, event: IncrementalEvent) -> str:
        """判断事件是局部重算还是全量重推。

        Returns
        -------
        "local" 或 "full"。
        """
        if event.force_full:
            return "full"
        if event.event_type == "batch_maneuver":
            return "full"
        if event.event_type == "emergency_mission":
            return "full"
        if event.event_type in ("sat_maneuver", "gs_maintenance"):
            return "local"
        return "local"

    # ---- 动态负载自适应 ----

    def _estimate_cpu_load(self) -> float:
        """模拟 CPU 负载估计（生产环境用 psutil）。"""
        try:
            import psutil  # optional dependency
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            # 简易加权移动平均模拟
            import os
            try:
                load = os.getloadavg() if hasattr(os, 'getloadavg') else None
            except (AttributeError, OSError):
                load = None
            return 50.0 if load is None else min(100.0, load[0] * 25.0)

    def update_load_level(self) -> LoadLevel:
        """更新负载等级（冷却防抖）。"""
        now = time.time()
        if now - self._last_load_check < self._load_check_interval_s:
            return self._current_load_level
        self._last_load_check = now

        cpu = self._estimate_cpu_load()
        self._cpu_samples.append(cpu)
        if len(self._cpu_samples) > self._load_history_size:
            self._cpu_samples.pop(0)

        avg_cpu = sum(self._cpu_samples) / len(self._cpu_samples)

        old_level = self._current_load_level

        if avg_cpu >= CPU_THRESHOLD_HEAVY and now >= self._degrade_cooldown:
            self._current_load_level = LoadLevel.HEAVY
            self._degrade_cooldown = now + self._cooldown_s
        elif avg_cpu >= CPU_THRESHOLD_MODERATE and now >= self._degrade_cooldown:
            self._current_load_level = LoadLevel.MODERATE
            self._degrade_cooldown = now + self._cooldown_s
        elif avg_cpu >= CPU_THRESHOLD_LIGHT and now >= self._degrade_cooldown:
            self._current_load_level = LoadLevel.LIGHT
            self._degrade_cooldown = now + self._cooldown_s
        elif avg_cpu < CPU_THRESHOLD_LIGHT and self._current_load_level != LoadLevel.NORMAL:
            if now >= self._recover_cooldown:
                self._current_load_level = max(
                    LoadLevel.NORMAL, LoadLevel(self._current_load_level - 1)
                )
                self._recover_cooldown = now + self._cooldown_s

        if old_level != self._current_load_level:
            self._update_effective_steps()

        return self._current_load_level

    def _update_effective_steps(self) -> None:
        """同步有效步长到各层（受降级因子影响）。"""
        for layer in TimingLayer:
            self._effective_steps[layer] = self.get_sample_step_for_layer(layer)

    @property
    def load_level(self) -> LoadLevel:
        return self._current_load_level

    @property
    def effective_steps(self) -> dict[TimingLayer, float]:
        return dict(self._effective_steps)

    def interpolation_enabled(self) -> bool:
        return DEGRADATION_CONFIG[self._current_load_level]["enable_interpolation"]

    def elevation_curve_enabled(self) -> bool:
        return DEGRADATION_CONFIG[self._current_load_level]["store_elevation_curve"]

    # ---- 诊断/日志 ----

    def status(self) -> dict:
        """返回调度器当前状态（可序列化）。"""
        return {
            "current_layer": self._current_layer.name,
            "load_level": self._current_load_level.name,
            "avg_cpu_percent": round(
                sum(self._cpu_samples) / len(self._cpu_samples), 1
            ) if self._cpu_samples else 0.0,
            "effective_steps": {
                layer.name: round(step, 1)
                for layer, step in self._effective_steps.items()
            },
            "interpolation_enabled": self.interpolation_enabled(),
            "elevation_curve_enabled": self.elevation_curve_enabled(),
            "pending_events": len(self._event_queue),
            "stagger_phase": self._stagger_phase,
        }


# ============================================================
# 一步式集成：分层定时编排过境调度
# ============================================================

def run_layered_schedule(
    simulator: OrbitSimulator,
    timing: AdaptiveTimingScheduler | None = None,
    min_duration_slots: int = 1,
    sat_priorities: dict[int, int] | None = None,
    timestep_s: float | None = None,
) -> tuple[PassTimetable, AllocationResult, dict]:
    """以分层定时策略编排一次完整的定时推演+窗口提取+分配。

    论文维度一核心入口。

    Parameters
    ----------
    simulator : OrbitSimulator
    timing : AdaptiveTimingScheduler or None
        若为 None 则自动创建。
    min_duration_slots : int
    sat_priorities : dict
    timestep_s : float or None
        覆盖 Default 步长。用于自适应采样。

    Returns
    -------
    (PassTimetable, AllocationResult, timing_status_dict)
    """
    from fl_space.simulator.pass_scheduler import GroundStationAllocator, PassTimetable

    if timing is None:
        timing = AdaptiveTimingScheduler(simulator)

    # 更新负载等级
    timing.update_load_level()

    # 决定当前有效采样步长
    hours = simulator.num_timeslots * simulator.timeslot_duration_min / 60.0
    layer = timing.get_layer_for_horizon(hours)
    if timestep_s is None:
        timestep_s = timing.get_sample_step_for_layer(layer)

    # 执行过境时间表构建（传递步长信息给 timetable）
    tt = PassTimetable(
        simulator,
        min_duration_slots=min_duration_slots,
        sat_priorities=sat_priorities,
        predict_slots=simulator.num_timeslots,
        enable_interpolation=timing.interpolation_enabled(),
        effective_step_s=timestep_s,
    )

    # 阶段 2: 分配
    allocator = GroundStationAllocator(tt)
    alloc = allocator.allocate()

    # 更新各层运行时间
    now = time.time()
    timing._last_run[layer] = now

    return tt, alloc, timing.status()
