"""
业务侧联合优化模块 (论文维度七)
================================

实现:
1. 任务优先级驱动局部优先推演 (高优卫星独立高频推演)
2. 传输负载预判前置计算 (提前识别未来 2-4h 资源拥堵)
3. 多地面站协同合并窗口 (临近站点接力过境合并)

将调度需求嵌入定时计算, 减少二次运算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fl_space.simulator.orbit_simulator import OrbitSimulator
    from fl_space.simulator.pass_scheduler import PassRecord, PassTimetable


# ============================================================
# 1. 优先级驱动局部推演
# ============================================================

@dataclass
class PriorityTask:
    """业务任务 (驱动局部优先推演)。"""

    sat_id: int
    priority: int   # 1=紧急, 2=常规, 3=低优
    task_type: str = "normal"  # emergency/data_dump/telemetry
    deadline_ts: int = 0       # 截止 timeslot


class PriorityDrivenScheduler:
    """优先级驱动推演调度器。

    高优先级卫星独立高频小范围推演, 低优先级卫星统一低频批量计算。
    """

    def __init__(
        self,
        sim: OrbitSimulator,
        high_pri_freq_min: float = 2.0,    # 高优刷新间隔 (min)
        low_pri_freq_min: float = 15.0,    # 低优刷新间隔 (min)
    ):
        self._sim = sim
        self._high_freq = high_pri_freq_min
        self._low_freq = low_pri_freq_min
        self._tasks: dict[int, PriorityTask] = {}
        self._last_push: dict[int, float] = {}  # {sat_id: last_update_timeslot}

    def register_task(self, task: PriorityTask) -> None:
        self._tasks[task.sat_id] = task

    def needs_refresh(self, sat_id: int, current_ts: int) -> bool:
        """判断是否需要局部刷新 (基于优先级 + 距上次刷新间隔)。"""
        task = self._tasks.get(sat_id)
        if task is None:
            return True
        last = self._last_push.get(sat_id)
        if last is None:
            return True
        slot_min = self._sim.timeslot_duration_min
        elapsed_min = (current_ts - last) * slot_min
        if task.priority == 1:
            return elapsed_min >= self._high_freq
        if task.priority == 2:
            return elapsed_min >= self._low_freq
        return elapsed_min >= self._low_freq * 2

    def mark_refreshed(self, sat_id: int, current_ts: int) -> None:
        self._last_push[sat_id] = current_ts

    def get_high_priority_sats(self) -> list[int]:
        """获取高优先级卫星列表 (需独立高频推演)。"""
        return sorted(
            [sid for sid, t in self._tasks.items() if t.priority == 1]
        )

    def get_batch_candidates(self) -> list[int]:
        """获取可批量处理的低优先级卫星列表。"""
        return sorted(
            [sid for sid, t in self._tasks.items() if t.priority >= 2]
        )


# ============================================================
# 2. 传输负载预判前置计算
# ============================================================

@dataclass
class LoadForecast:
    """地面站未来负载预判。"""

    gs_id: int
    future_load: dict[int, float] = field(default_factory=dict)  # {ts: load_mb}
    peak_ts: int = 0
    peak_load: float = 0.0
    congestion_flag: bool = False   # 拥堵预警


class GsLoadForecaster:
    """地面站未来负载预判器。

    定时推演同步计算每座地面站未来负载曲线,
    提前识别未来 2-4h 资源拥堵时段。
    """

    def __init__(self, sim: OrbitSimulator, forecast_horizon_slots: int = 240):
        """forecast_horizon_slots: 预判窗口 (默认 240 slots = 4h @ 1min/slot)。"""
        self._sim = sim
        self._horizon = forecast_horizon_slots
        self._forecasts: dict[int, LoadForecast] = {}

    def compute_forecast(
        self,
        start_ts: int,
        timetable: PassTimetable,
    ) -> dict[int, LoadForecast]:
        """计算各站点未来负载曲线。

        Parameters
        ----------
        start_ts : int
            当前 timeslot。
        timetable : PassTimetable
            已生成的过境时间表。

        Returns
        -------
        dict[int, LoadForecast]
            {gs_id: 负载预判}。
        """
        end_ts = start_ts + self._horizon
        n_gs = self._sim.num_ground_stations

        forecasts: dict[int, LoadForecast] = {}
        for gs_id in range(n_gs):
            fc = LoadForecast(gs_id=gs_id)
            gs_records = timetable.records_by_gs(gs_id)
            for r in gs_records:
                if r.ts_start > end_ts or r.ts_end < start_ts:
                    continue
                # 分配负载到各 timeslot
                for ts in range(max(start_ts, r.ts_start), min(end_ts, r.ts_end) + 1):
                    fc.future_load[ts] = fc.future_load.get(ts, 0.0) + r.est_downlink_mb / max(1, r.duration_slots)

            if fc.future_load:
                peak_ts = max(fc.future_load, key=lambda t: fc.future_load[t])
                fc.peak_ts = peak_ts
                fc.peak_load = fc.future_load[peak_ts]
                # 拥堵判断: 峰值超过该站单时刻理论下行容量
                gs = self._sim.ground_network[gs_id]
                slot_capacity_mb = gs.downlink_rate_mbps * self._sim.timeslot_duration_min * 60.0 / 8.0 * gs.max_concurrent_sats
                fc.congestion_flag = fc.peak_load > slot_capacity_mb * 0.8

            forecasts[gs_id] = fc
        self._forecasts = forecasts
        return forecasts

    def get_congested_gs(self) -> list[int]:
        """返回存在拥堵风险的站点列表。"""
        return [fc.gs_id for fc in self._forecasts.values() if fc.congestion_flag]

    def get_load_spread_score(self, gs_ids: list[int]) -> float:
        """计算负载分散度得分 (越高越均衡)。"""
        loads = [self._forecasts.get(gid, LoadForecast(gs_id=gid)).peak_load for gid in gs_ids]
        n = len(loads)
        if n <= 1:
            return 1.0
        mean = sum(loads) / n
        if mean < 1e-9:
            return 1.0
        variance = sum((v - mean) ** 2 for v in loads) / n
        cv = math.sqrt(variance) / mean  # 变异系数
        return round(1.0 / (1.0 + cv), 4)


# ============================================================
# 3. 多地面站协同合并窗口
# ============================================================

@dataclass
class RelayChain:
    """星站接力传输序列。"""

    sat_id: int
    segments: list[tuple[int, int, int]]  # [(gs_id, ts_start, ts_end), ...]
    total_duration_slots: int = 0
    total_duration_min: float = 0.0


class StationRelayMerger:
    """多地面站协同接力窗口合并。

    对距离相近、可协同接力接收的地面站,
    在定时演算阶段自动合并连续过境窗口。
    """

    def __init__(
        self,
        max_distance_km: float = 3000.0,   # 最大站点间距 (可接力)
        max_handoff_slots: int = 5,         # 移交间隙 (timeslots)
    ):
        self._max_dist_km = max_distance_km
        self._max_handoff = max_handoff_slots

    def _gs_distance_km(
        self,
        sim: OrbitSimulator,
        gs_a: int,
        gs_b: int,
    ) -> float:
        """计算两地面的大圆距离 (km)。"""
        import math as _m
        gsa = sim.ground_network[gs_a]
        gsb = sim.ground_network[gs_b]
        lat1 = _m.radians(gsa.lat_deg)
        lon1 = _m.radians(gsa.lon_deg)
        lat2 = _m.radians(gsb.lat_deg)
        lon2 = _m.radians(gsb.lon_deg)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a_val = _m.sin(dlat / 2) ** 2 + _m.cos(lat1) * _m.cos(lat2) * _m.sin(dlon / 2) ** 2
        c_val = 2 * _m.atan2(_m.sqrt(a_val), _m.sqrt(1 - a_val))
        return 6371.0 * c_val  # 地球半径

    def build_relay_chains(
        self,
        sim: OrbitSimulator,
        sat_id: int,
        passes: list[PassRecord],
    ) -> list[RelayChain]:
        """为单颗卫星构建接力传输序列。

        Parameters
        ----------
        sim : OrbitSimulator
        sat_id : int
            目标卫星 ID。
        passes : list[PassRecord]
            该卫星的所有过境窗口。

        Returns
        -------
        list[RelayChain]
            (可能) 合并后的接力序列。
        """
        if not passes:
            return []

        # 按时间排序
        sorted_passes = sorted(passes, key=lambda r: r.ts_start)

        chains: list[RelayChain] = []
        current_chain: list[tuple[int, int, int]] = []
        sorted_passes[0]

        for _i, r in enumerate(sorted_passes):
            if not current_chain:
                current_chain.append((r.gs_id, r.ts_start, r.ts_end))
                continue

            # 检查与前一个窗口的关系
            prev_end = current_chain[-1][2]
            prev_gs = current_chain[-1][0]

            # 同一站点: 扩展 (由碎片合并处理, 此处仅接力)
            if r.gs_id == prev_gs and r.ts_start <= prev_end + self._max_handoff:
                current_chain[-1] = (prev_gs, current_chain[-1][1], max(prev_end, r.ts_end))
                continue

            # 不同站点: 检查是否可接力
            dist = self._gs_distance_km(sim, prev_gs, r.gs_id)
            if dist <= self._max_dist_km and r.ts_start <= prev_end + self._max_handoff:
                current_chain.append((r.gs_id, r.ts_start, r.ts_end))
            else:
                # 不能接力 → 完成当前链, 开始新链
                if len(current_chain) >= 2:  # 至少 2 站才算接力
                    total_slots = current_chain[-1][2] - current_chain[0][1] + 1
                    chains.append(RelayChain(
                        sat_id=sat_id,
                        segments=list(current_chain),
                        total_duration_slots=total_slots,
                        total_duration_min=total_slots * sim.timeslot_duration_min,
                    ))
                current_chain = [(r.gs_id, r.ts_start, r.ts_end)]

        # 末尾链
        if len(current_chain) >= 2:
            total_slots = current_chain[-1][2] - current_chain[0][1] + 1
            chains.append(RelayChain(
                sat_id=sat_id,
                segments=list(current_chain),
                total_duration_slots=total_slots,
                total_duration_min=total_slots * sim.timeslot_duration_min,
            ))

        return chains
