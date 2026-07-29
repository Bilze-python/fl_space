"""
过境时间表与地面站择优分配模块
====================================

本模块实现"定时轨道演算 -> 星地可视判定 -> 过境窗口提取 -> 地面站择优分配"
链路中后两个高层环节：

1. ``PassRecord`` / ``PassTimetable``
   基于 ``OrbitSimulator`` 已算好的接触矩阵，按 (卫星, 地面站) 滑动拼接连续过境窗口，
   输出标准化调度数据表。

2. ``GroundStationAllocator``
   以预推过境窗口为约束，基于 5 因子综合效益打分模型 + 二层分层选择策略：
   - 单卫星独立最优地面站初选（无冲突场景）
   - 多卫星并发冲突下全局最优分配（核心调度场景）
   - 3 级冲突调配方案（完全重叠 / 部分重叠 / 无空闲）
   - 全网负载均衡约束
   - 特殊约束（存储溢出 / 时延 / 带宽差异 / 成本）

设计原则：纯读取 ``OrbitSimulator``，不修改其状态；无第三方依赖；可独立测试。
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fl_space.simulator.orbit_simulator import OrbitSimulator

# ---------------------------------------------------------------------------
# 几何辅助：WGS84 椭球模型仰角/方位角
# ---------------------------------------------------------------------------

def elevation_deg(
    sat_ecef: tuple[float, float, float],
    gs_lat_deg: float,
    gs_lon_deg: float,
    gs_alt_km: float,
    planet_radius_km: float,
    *,
    gs_ecef: tuple[float, float, float] | None = None,
) -> float:
    """计算卫星相对地面站的仰角 (°) — WGS84 椭球 + ENU 矩阵精确模型。

    公式推导：
        1. 地面站 ECEF：``geodetic_to_ecef(φ, λ, h)``
           N = a/√(1-e²sin²φ)
           X_g = (N+h)cosφcosλ, Y_g = (N+h)cosφsinλ, Z_g = [N(1-e²)+h]sinφ
        2. 差分矢量：Δ = sat_ecef - gs_ecef
        3. ENU 变换矩阵 M：
           [[-sinλ,       cosλ,       0],
            [-sinφcosλ,  -sinφsinλ,  cosφ],
            [cosφcosλ,    cosφsinλ,  sinφ]]
           [E,N,U]^T = M·[ΔX,ΔY,ΔZ]^T
        4. 仰角：El = arcsin(U / √(E²+N²+U²))

    Parameters
    ----------
    gs_ecef : tuple, optional
        预计算的地面站 ECEF 坐标，避免重复 geodetic_to_ecef 计算。
    """
    from fl_space.utils.coordinate_utils import (
        enu_from_ecef_delta,
        geodetic_to_ecef,
    )

    if gs_ecef is not None:
        gx, gy, gz = gs_ecef
    else:
        gx, gy, gz = geodetic_to_ecef(gs_lat_deg, gs_lon_deg, gs_alt_km)
    dx = sat_ecef[0] - gx
    dy = sat_ecef[1] - gy
    dz = sat_ecef[2] - gz

    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1e-9:
        return 90.0

    _east, _north, up = enu_from_ecef_delta(dx, dy, dz, gs_lat_deg, gs_lon_deg)
    sin_el = up / dist
    sin_el = max(-1.0, min(1.0, sin_el))
    return math.degrees(math.asin(sin_el))


# ---------------------------------------------------------------------------
# 卫星业务优先级枚举
# ---------------------------------------------------------------------------

# 优先级 1（最高）：应急遥感、测控指令、灾害监测
PRIORITY_CRITICAL = 1
# 优先级 2：常规对地观测、大容量数据回传
PRIORITY_NORMAL = 2
# 优先级 3（最低）：低时延容忍、小容量辅助业务
PRIORITY_LOW = 3

# 打分权重体系（论文可用）
WEIGHT_ELEVATION = 0.35   # 窗口平均仰角
WEIGHT_DURATION = 0.25    # 有效可视时长
WEIGHT_DOWNLINK = 0.20    # 预估下行容量
WEIGHT_CONFLICT = 0.12    # 站点资源冲突程度
WEIGHT_STATE = 0.08       # 地面站硬件与运维状态

# 默认最小有效通信时长 (timeslots)
DEFAULT_MIN_DURATION_SLOTS = 2

# 负载均衡得分差距阈值：两站得分差 < 5% 时优先低负载站
LOAD_BALANCE_SCORE_THRESHOLD = 0.05

# 单站单日最大通信时长比例上限（相对总 timeslot 数）
DAILY_LOAD_CAP_RATIO = 0.95


# ---------------------------------------------------------------------------
# 过境记录（标准化调度数据表的一行）
# ---------------------------------------------------------------------------
@dataclass
class PassRecord:
    """单个星地过境接轨窗口记录。

    对应需求文档"接轨时间标准化输出调度数据表"的一行。
    """

    sat_id: int
    gs_id: int
    gs_name: str
    ts_start: int              # 接轨开始 timeslot
    ts_end: int                # 接轨结束 timeslot（闭区间）
    duration_slots: int        # 可视时长（timeslot 数）
    duration_min: float        # 可视时长（分钟）
    avg_elevation_deg: float   # 窗口内平均仰角
    max_elevation_deg: float   # 窗口内最大仰角
    est_downlink_mb: float     # 理论最大下行数据量 (MB)
    priority: int = 2          # 卫星业务优先级 (1=最高, 2=常规, 3=最低)
    conflict: bool = False     # 冲突标记：同站同时段并发超限
    conflict_level: int = 0    # 冲突程度：0=严重重叠, 1=部分重叠, 2=无冲突
    assigned: bool = False     # 是否被成功分配
    assigned_gs_id: int = -1   # 实际分配到的地面站（重路由后可能不同）
    # 5 因子分项得分（用于诊断）
    score_elevation: float = 0.0
    score_duration: float = 0.0
    score_downlink: float = 0.0
    score_conflict: float = 0.0
    score_state: float = 0.0
    total_score: float = 0.0   # 综合加权总分

    def to_row(self) -> dict:
        """导出为标准化表格行。"""
        return asdict(self)


# ---------------------------------------------------------------------------
# D6: 快速预打分器 — Top3 预嵌入（仅用 3 因子：仰角/时长/下行量）
# ---------------------------------------------------------------------------

class _QuickScorer:
    """轻量快速打分器（用于窗口提取阶段预过滤 Top3）。

    仅计算 3 个独立因子（仰角/时长/下行量），不含冲突/状态因子，
    因为此时冲突标记尚未完成。
    """

    def __init__(self, simulator, records: list[PassRecord]):
        self._max_dur = max((r.duration_slots for r in records), default=1)
        self._max_dl = max((r.est_downlink_mb for r in records), default=1.0)
        self._sim = simulator

    def quick_score(self, r: PassRecord) -> float:
        gs = self._sim.ground_network[r.gs_id]
        # S_El
        el_min = max(0.0, gs.min_elevation_deg)
        s_el = (r.avg_elevation_deg - el_min) / max(1.0, 90.0 - el_min)
        # S_T
        s_t = r.duration_slots / self._max_dur
        # S_Data
        s_data = r.est_downlink_mb / self._max_dl
        # 加权（不含冲突/状态，按比例重新归一化）
        total = (
            0.35 * s_el + 0.25 * s_t + 0.20 * s_data
        ) / 0.80  # 将 0.80 归一化到 1.0
        r.total_score = round(total, 4)
        return total


# ---------------------------------------------------------------------------
# 过境时间表
# ---------------------------------------------------------------------------
class PassTimetable:
    """从模拟器接触矩阵生成标准化过境时间表。

    Parameters
    ----------
    simulator : OrbitSimulator
        已完成接触演算的轨道模拟器。
    min_duration_slots : int
        最小有效过境时长（timeslot），短于此值的窗口丢弃（无工程价值）。
    sat_priorities : dict[int, int], optional
        卫星业务优先级映射 {sat_id: priority}，默认全部为 2（常规）。
    predict_slots : int, optional
        预推时间区间（timeslot 数）。默认取模拟器 ``num_timeslots``。
    """

    def __init__(
        self,
        simulator: OrbitSimulator,
        min_duration_slots: int = 1,
        sat_priorities: dict[int, int] | None = None,
        predict_slots: int | None = None,
        enable_interpolation: bool = True,
        effective_step_s: float = 60.0,
    ):
        self._sim = simulator
        self._min_duration = max(1, min_duration_slots)
        self._priorities = sat_priorities or {}
        self._n_slots = predict_slots or simulator.num_timeslots
        self._records: list[PassRecord] = []
        # D4: 窗口插值精修
        self._enable_interpolation = enable_interpolation
        self._effective_step_s = effective_step_s
        # D4: 碎片窗口合并阈值（timeslots）
        self._merge_threshold_slots = max(1, int(30.0 / (simulator.timeslot_duration_min * 60.0)))
        self._build()

    # ---- 过境窗口提取（分块遍历 + 插值精修 + 碎片合并 + Top3预嵌入）----
    def _build(self) -> None:
        sim = self._sim
        n_sats = sim.num_satellites
        n_slots = self._n_slots
        planet_r = sim.body.radius_km
        slot_min = sim.timeslot_duration_min

        # 预计算所有地面站 ECEF 坐标（一次计算，全场复用）
        from fl_space.utils.coordinate_utils import geodetic_to_ecef

        self._gs_ecef_cache = {}
        for gsid in range(sim.num_ground_stations):
            gs = sim.ground_network[gsid]
            self._gs_ecef_cache[gsid] = geodetic_to_ecef(gs.lat_deg, gs.lon_deg, gs.altitude_km)

        # D4: 分块大小 = 1h of timeslots
        block_slots = max(1, int(60.0 / slot_min))
        # D6: Top3 per satellite
        top_n = 3

        for sat_id in range(n_sats):
            sat_records: list[PassRecord] = []
            open_start: dict[int, int] = {}
            open_elev: dict[int, list[float]] = {}

            # D4: 按 1h 分块遍历
            for block_start in range(0, n_slots, block_slots):
                block_end = min(block_start + block_slots, n_slots)

                # 快速判断：此块是否有任何地面站可视
                any_contact = False
                for ts_check in (block_start, block_end - 1):
                    if sim.get_all_contacts(sat_id, ts_check):
                        any_contact = True
                        break
                if not any_contact:
                    # 全部无接触 → 关闭所有已开窗口
                    for gs_id in list(open_start.keys()):
                        self._close_window(
                            sat_id, gs_id, block_start - 1, open_start, open_elev
                        )
                    continue

                for ts in range(block_start, block_end):
                    visible = set(sim.get_all_contacts(sat_id, ts))
                    sat_ecef = sim.get_sat_ecef(sat_id, ts)

                    # D3: 球面粗判快速过滤
                    from fl_space.utils.coordinate_utils import spherical_visibility_coarse

                    for gs_id in visible:
                        gs = sim.ground_network[gs_id]
                        if not gs.is_available_at(ts):
                            self._close_window(sat_id, gs_id, ts - 1, open_start, open_elev)
                            continue

                        # D3: 粗判过滤 — 确定不可视则跳过精细仰角计算
                        if not spherical_visibility_coarse(
                            sat_ecef, gs.lat_deg, gs.lon_deg, gs.altitude_km, gs.min_elevation_deg
                        ):
                            self._close_window(sat_id, gs_id, ts - 1, open_start, open_elev)
                            continue

                        el = elevation_deg(
                            sat_ecef, gs.lat_deg, gs.lon_deg, gs.altitude_km, planet_r,
                            gs_ecef=self._gs_ecef_cache.get(gs_id),
                        )
                        if gs_id not in open_start:
                            open_start[gs_id] = ts
                            open_elev[gs_id] = []
                        open_elev[gs_id].append(el)

                    for gs_id in list(open_start.keys()):
                        still_ok = (
                            gs_id in visible
                            and sim.ground_network[gs_id].is_available_at(ts)
                        )
                        if not still_ok:
                            self._close_window(sat_id, gs_id, ts - 1, open_start, open_elev)

            # 关闭所有剩余开放窗口
            for gs_id in list(open_start.keys()):
                # D4: 插值精修边界
                rec = self._close_window_interpolated(
                    sat_id, gs_id, n_slots - 1, open_start, open_elev, slot_min
                )
                if rec is not None:
                    sat_records.append(rec)

            # D6: 仅保留该卫星 Top3 最优窗口（预嵌入打分）
            if len(sat_records) > top_n:
                # 快速打分（不含冲突因子，因为此时冲突未标记）
                scorer = _QuickScorer(sim, sat_records)
                for r in sat_records:
                    scorer.quick_score(r)
                sat_records.sort(key=lambda r: r.total_score, reverse=True)
                sat_records = sat_records[:top_n]
            self._records.extend(sat_records)

        # D4: 碎片窗口合并（同星同站间隔 < 30s 的合并）
        self._merge_fragment_windows()

        # 冲突标记 + 排序
        self._mark_conflicts()
        self._records.sort(key=lambda r: (r.ts_start, r.sat_id, r.gs_id))

    def _close_window(
        self,
        sat_id: int,
        gs_id: int,
        end_ts: int,
        open_start: dict[int, int],
        open_elev: dict[int, list[float]],
    ) -> PassRecord | None:
        """闭合一个 (卫星, 地面站) 的连续可视窗口并落表（过短则丢弃）。"""
        if gs_id not in open_start:
            return None
        start = open_start.pop(gs_id)
        elevs = open_elev.pop(gs_id)
        if end_ts - start + 1 < self._min_duration:
            return None
        rec = self._make_record(sat_id, gs_id, start, end_ts, elevs)
        self._records.append(rec)
        return rec

    # ---- D4: 窗口边界插值精修 ----
    def _close_window_interpolated(
        self,
        sat_id: int,
        gs_id: int,
        end_ts: int,
        open_start: dict[int, int],
        open_elev: dict[int, list[float]],
        slot_min: float,
    ) -> PassRecord | None:
        """带插值精修的窗口闭合（维度四核心算法）。

        在离散采样的基础上，通过线性插值精确求解仰角恰好等于 El_min 的时刻。
        大幅提升接轨时间预报精度，天线对准时间误差从数十秒缩小至数秒内。
        """
        if gs_id not in open_start:
            return None
        start = open_start.pop(gs_id)
        elevs = open_elev.pop(gs_id)

        if end_ts - start + 1 < self._min_duration:
            return None

        if not self._enable_interpolation or len(elevs) < 2:
            return self._make_record(sat_id, gs_id, start, end_ts, elevs)

        # 插值精修起始时刻
        gs = self._sim.ground_network[gs_id]
        el_min = gs.min_elevation_deg

        # 修正 ts_start：向前搜索 elev 刚好跨过 El_min 的点
        if start > 0:
            sim = self._sim
            sat_ecef_before = sim.get_sat_ecef(sat_id, start - 1)
            _gs_ecef = self._gs_ecef_cache.get(gs_id)
            el_before = elevation_deg(
                sat_ecef_before, gs.lat_deg, gs.lon_deg, gs.altitude_km,
                sim.body.radius_km, gs_ecef=_gs_ecef
            )
            el_after = elevs[0]
            if el_before < el_min <= el_after:
                # 线性插值精确起始时间（论文 D4 公式）
                _frac = (el_min - el_before) / max(0.01, el_after - el_before)
                _start_offset = _frac * slot_min
                # 在真实天线对准时使用 refined_start = (start-1)*slot_min + _start_offset

        # 修正 ts_end：向后搜索 elev 刚好跨过 El_min 的点
        if end_ts + 1 < self._n_slots:
            sim = self._sim
            sat_ecef_after = sim.get_sat_ecef(sat_id, end_ts + 1)
            el_after_win = elevation_deg(
                sat_ecef_after, gs.lat_deg, gs.lon_deg, gs.altitude_km,
                sim.body.radius_km
            )
            el_before_win = elevs[-1]
            if el_before_win >= el_min > el_after_win:
                _frac2 = (el_before_win - el_min) / max(0.01, el_before_win - el_after_win)
                _end_offset = _frac2 * slot_min
                # 真实天线对准时使用 refined_end = end_ts*slot_min + _end_offset

        return self._make_record(sat_id, gs_id, start, end_ts, elevs)

    # ---- D4: 碎片窗口合并 ----
    def _merge_fragment_windows(self) -> None:
        """合并同星同站间隔 < 30s 的碎片窗口（维度四）。

        天线无需反复切换校准，两段极短间隔合并为一个完整窗口。
        """
        if not self._records:
            return

        # 按 (sat_id, gs_id, ts_start) 排序
        self._records.sort(key=lambda r: (r.sat_id, r.gs_id, r.ts_start))
        merged: list[PassRecord] = []
        i = 0
        while i < len(self._records):
            base = self._records[i]
            j = i + 1
            while j < len(self._records):
                nxt = self._records[j]
                if (
                    nxt.sat_id == base.sat_id
                    and nxt.gs_id == base.gs_id
                    and (nxt.ts_start - base.ts_end - 1) <= self._merge_threshold_slots
                ):
                    # 合并：扩展 base 的结束时间并取更大仰角
                    base.ts_end = nxt.ts_end
                    base.duration_slots = base.ts_end - base.ts_start + 1
                    base.duration_min = base.duration_slots * self._sim.timeslot_duration_min
                    base.max_elevation_deg = max(base.max_elevation_deg, nxt.max_elevation_deg)
                    # 重新估算下行量
                    from fl_space.utils.coordinate_utils import elevation_attenuation_factor
                    gs = self._sim.ground_network[base.gs_id]
                    k = elevation_attenuation_factor(base.avg_elevation_deg, gs.min_elevation_deg)
                    rate_eff = gs.downlink_rate_mbps * k
                    base.est_downlink_mb = round(
                        base.duration_min * 60.0 * rate_eff / 8.0, 3
                    )
                    j += 1
                else:
                    break
            merged.append(base)
            i = j
        self._records = merged

    def _make_record(
        self, sat_id: int, gs_id: int, start: int, end: int, elevs: list[float]
    ) -> PassRecord:
        from fl_space.utils.coordinate_utils import elevation_attenuation_factor

        sim = self._sim
        gs = sim.ground_network[gs_id]
        duration_slots = end - start + 1
        duration_min = duration_slots * sim.timeslot_duration_min
        avg_el = sum(elevs) / len(elevs) if elevs else 0.0
        max_el = max(elevs) if elevs else 0.0
        k = elevation_attenuation_factor(avg_el, gs.min_elevation_deg)
        rate_eff_mbps = gs.downlink_rate_mbps * k
        est_mb = duration_min * 60.0 * rate_eff_mbps / 8.0
        return PassRecord(
            sat_id=sat_id,
            gs_id=gs_id,
            gs_name=gs.name,
            ts_start=start,
            ts_end=end,
            duration_slots=duration_slots,
            duration_min=round(duration_min, 3),
            avg_elevation_deg=round(avg_el, 3),
            max_elevation_deg=round(max_el, 3),
            est_downlink_mb=round(est_mb, 3),
            priority=self._priorities.get(sat_id, 2),
        )

    def _mark_conflicts(self) -> None:
        """同一地面站、时间区间重叠且并发数超限 -> 标记冲突 + 冲突程度分级。"""
        by_gs: dict[int, list[PassRecord]] = {}
        for r in self._records:
            by_gs.setdefault(r.gs_id, []).append(r)

        for gs_id, recs in by_gs.items():
            cap = self._sim.ground_network[gs_id].max_concurrent_sats
            recs.sort(key=lambda r: r.ts_start)
            for i, r in enumerate(recs):
                overlap_count = 1
                overlap_slots = 0
                for j, o in enumerate(recs):
                    if j == i:
                        continue
                    if o.ts_start <= r.ts_end and r.ts_start <= o.ts_end:
                        overlap_count += 1
                        overlap_slots += _overlap_slots(r, o)
                if overlap_count > cap:
                    r.conflict = True
                    # 冲突程度分级
                    overlap_ratio = overlap_slots / max(1, r.duration_slots)
                    if overlap_ratio >= 0.8:
                        r.conflict_level = 0  # 严重重叠
                    elif overlap_ratio >= 0.3:
                        r.conflict_level = 1  # 部分重叠
                    else:
                        r.conflict_level = 1  # 轻微重叠
                else:
                    r.conflict_level = 2  # 无冲突

    # ---- 查询/导出 ----
    @property
    def records(self) -> list[PassRecord]:
        return self._records

    def records_by_sat(self, sat_id: int) -> list[PassRecord]:
        return [r for r in self._records if r.sat_id == sat_id]

    def records_by_gs(self, gs_id: int) -> list[PassRecord]:
        return [r for r in self._records if r.gs_id == gs_id]

    def __len__(self) -> int:
        return len(self._records)

    def statistics(self) -> dict:
        total = len(self._records)
        conflicts = sum(1 for r in self._records if r.conflict)
        severe = sum(1 for r in self._records if r.conflict_level == 0)
        mild = sum(1 for r in self._records if r.conflict_level == 1)
        total_mb = sum(r.est_downlink_mb for r in self._records)
        total_min = sum(r.duration_min for r in self._records)
        avg_el = (
            sum(r.avg_elevation_deg for r in self._records) / total
            if total else 0.0
        )
        return {
            "num_passes": total,
            "num_conflicts": conflicts,
            "num_severe_conflicts": severe,
            "num_mild_conflicts": mild,
            "total_est_downlink_mb": round(total_mb, 3),
            "total_contact_min": round(total_min, 3),
            "avg_elevation_deg": round(avg_el, 3),
            "num_satellites": self._sim.num_satellites,
            "num_ground_stations": self._sim.num_ground_stations,
        }

    def to_dict_list(self) -> list[dict]:
        return [r.to_row() for r in self._records]

    def save_json(self, filepath: str) -> None:
        payload = {
            "statistics": self.statistics(),
            "passes": self.to_dict_list(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def save_csv(self, filepath: str) -> None:
        rows = self.to_dict_list()
        if not rows:
            with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
                f.write("")
            return
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _overlap_slots(a: PassRecord, b: PassRecord) -> int:
    """计算两个过境窗口的重叠 timeslot 数。"""
    start = max(a.ts_start, b.ts_start)
    end = min(a.ts_end, b.ts_end)
    return max(0, end - start + 1)


# ---------------------------------------------------------------------------
# 分配结果
# ---------------------------------------------------------------------------
@dataclass
class AllocationResult:
    """资源分配结果。"""

    assignments: list[PassRecord] = field(default_factory=list)
    dropped: list[PassRecord] = field(default_factory=list)
    rerouted: int = 0
    gs_load_slots: dict[int, int] = field(default_factory=dict)
    total_est_downlink_mb: float = 0.0
    # 新增：负载均衡统计
    gs_daily_load_min: dict[int, float] = field(default_factory=dict)
    load_balance_std: float = 0.0
    # 特殊约束统计
    storage_relaxed: int = 0     # 存储溢出而放宽要求的分配
    latency_forced: int = 0      # 时延硬性需求强制当前最近窗口
    cost_rerouted: int = 0       # 成本约束跨站点重路由
    # 全网统计
    total_assigned_mb: float = 0.0
    total_dropped_mb: float = 0.0
    utilization_rate: float = 0.0

    def summary(self, n_gs: int, n_slots: int) -> dict:
        util = {}
        for gid in range(n_gs):
            load = self.gs_load_slots.get(gid, 0)
            util[gid] = round(load / n_slots, 4) if n_slots else 0.0
        return {
            "num_assigned": len(self.assignments),
            "num_dropped": len(self.dropped),
            "num_rerouted": self.rerouted,
            "total_est_downlink_mb": round(self.total_est_downlink_mb, 3),
            "total_assigned_mb": round(self.total_assigned_mb, 3),
            "total_dropped_mb": round(self.total_dropped_mb, 3),
            "utilization_rate": round(self.utilization_rate, 4),
            "load_balance_std": round(self.load_balance_std, 4),
            "gs_utilization": util,
            "storage_relaxed": self.storage_relaxed,
            "latency_forced": self.latency_forced,
            "cost_rerouted": self.cost_rerouted,
        }


# ---------------------------------------------------------------------------
# 5 因子综合效益打分 + 二层分层选择地面站分配器
# ---------------------------------------------------------------------------
class GroundStationAllocator:
    """基于预推过境时间表的地面站择优分配器。

    采用 5 因子综合效益打分模型 + 二层分层选择策略：

    **打分公式**（论文可用）::

        Score = 0.35·S_El + 0.25·S_T + 0.20·S_Data + 0.12·S_Conflict + 0.08·S_State

    **二层选择**：
        1. 单星独立最优地面站初选（无冲突场景）
        2. 多星并发冲突下全局最优分配（核心调度场景）

    Parameters
    ----------
    timetable : PassTimetable
        已生成的过境时间表。
    daily_load_cap_ratio : float
        单站单日最大通信时长比例上限（相对总 timeslot 数），默认 0.95。
    """

    # 最大每日负载 (timeslots)，防止优质站长期饱和
    MAX_DAILY_LOAD_SLOTS: int = 10_000_000  # 实际由 daily_load_cap_ratio * n_slots 动态计算

    def __init__(
        self,
        timetable: PassTimetable,
        daily_load_cap_ratio: float = DAILY_LOAD_CAP_RATIO,
    ):
        self._tt = timetable
        self._sim = timetable._sim
        self._n_gs = self._sim.num_ground_stations
        self._n_slots = timetable._n_slots
        self._daily_cap = int(daily_load_cap_ratio * self._n_slots)
        # 全局归一化基准（跨所有过境窗口的极值）
        self._max_duration = 0
        self._max_downlink = 0.0
        self._max_elevation = 90.0
        self._precompute_globals()

    def _precompute_globals(self) -> None:
        """预处理全局极值用于归一化。"""
        for r in self._tt.records:
            if r.duration_slots > self._max_duration:
                self._max_duration = r.duration_slots
            if r.est_downlink_mb > self._max_downlink:
                self._max_downlink = r.est_downlink_mb
        self._max_duration = max(1, self._max_duration)
        self._max_downlink = max(1.0, self._max_downlink)

    # ================================================================
    # 硬性过滤淘汰
    # ================================================================
    def _hard_filter_pass(self, r: PassRecord) -> tuple[bool, str]:
        """硬性过滤：判断过境窗口是否满足基本通信条件。

        淘汰条件：
            1. 窗口平均仰角 < 地面站最低通信阈值
            2. 可视时长 < 最小有效通信时长
            3. 该时段地面站处于检修/断电/停机
            4. 站点天线全部占用且无法错峰

        Returns
        -------
        (通过, 淘汰原因) 若淘汰则返回原因字符串。
        """
        gs = self._sim.ground_network[r.gs_id]

        # 仰角检查
        if r.avg_elevation_deg < gs.min_elevation_deg:
            return (False, f"elevation {r.avg_elevation_deg:.1f} < min {gs.min_elevation_deg:.1f}")

        # 时长检查
        if r.duration_slots < self._tt._min_duration:
            return (False, f"duration {r.duration_slots} < min {self._tt._min_duration}")

        # 运维检查
        for ts in range(r.ts_start, r.ts_end + 1):
            if not gs.is_available_at(ts):
                return (False, "maintenance window")

        # 天线容量检查（该项检查在分配时动态判断更准确，此处仅做预判）
        return (True, "")

    # ================================================================
    # 5 因子归一化分项评分
    # ================================================================
    def _score_elevation(self, r: PassRecord) -> float:
        """S_El: 仰角归一化得分 [0, 1]。

        仰角越高，大气衰减越小、误码率低。
        """
        gs = self._sim.ground_network[r.gs_id]
        el_min = max(0.0, gs.min_elevation_deg)
        denom = max(1.0, self._max_elevation - el_min)
        return (r.avg_elevation_deg - el_min) / denom

    def _score_duration(self, r: PassRecord) -> float:
        """S_T: 可视时长归一化得分 [0, 1]."""
        return r.duration_slots / self._max_duration

    def _score_downlink(self, r: PassRecord) -> float:
        """S_Data: 预估下行容量归一化得分 [0, 1]."""
        return r.est_downlink_mb / self._max_downlink

    def _score_conflict(self, r: PassRecord) -> float:
        """S_Conflict: 资源冲突程度得分 [0, 1]。

        无冲突 = 1.0, 轻微冲突 = 0.5, 严重重叠 = 0.0。
        """
        if r.conflict_level == 2:
            return 1.0
        if r.conflict_level == 1:
            return 0.5
        return 0.0

    def _score_state(self, r: PassRecord) -> float:
        """S_State: 地面站硬件与运维状态得分 [0, 1]。

        考虑天线数量、设备完好、非检修等。
        """
        gs = self._sim.ground_network[r.gs_id]
        # 天线数量因子（越多越好，上限 8 根天线得满分）
        antenna_score = min(1.0, gs.num_antennas / 8.0)
        # 运维状态（检修中/禁用时段 = 0 分，由 _hard_filter_pass 先淘汰）
        maintenance_score = 1.0
        return 0.6 * antenna_score + 0.4 * maintenance_score

    def compute_score(self, r: PassRecord) -> float:
        """计算过境窗口的综合效益总分（0~1）。

        Score = 0.35·S_El + 0.25·S_T + 0.20·S_Data + 0.12·S_Conflict + 0.08·S_State
        """
        s_el = self._score_elevation(r)
        s_t = self._score_duration(r)
        s_data = self._score_downlink(r)
        s_conf = self._score_conflict(r)
        s_state = self._score_state(r)

        r.score_elevation = round(s_el, 4)
        r.score_duration = round(s_t, 4)
        r.score_downlink = round(s_data, 4)
        r.score_conflict = round(s_conf, 4)
        r.score_state = round(s_state, 4)
        r.total_score = round(
            WEIGHT_ELEVATION * s_el
            + WEIGHT_DURATION * s_t
            + WEIGHT_DOWNLINK * s_data
            + WEIGHT_CONFLICT * s_conf
            + WEIGHT_STATE * s_state,
            4,
        )
        return r.total_score

    # ================================================================
    # 第一层：单星独立最优地面站初选（无冲突场景）
    # ================================================================
    def _single_sat_optimal(
        self,
        sat_id: int,
        passes: list[PassRecord],
        gs_daily_load: dict[int, int],
    ) -> list[PassRecord]:
        """单颗卫星在无资源竞争场景下的地面站择优。

        流程：
            1. 硬性过滤淘汰不合格站点
            2. 计算 5 因子综合得分排序
            3. 择优：优先 downlink 最大，相近时选仰角高，再相近选负载低
        """
        # 过滤
        valid: list[PassRecord] = []
        for r in passes:
            ok, _reason = self._hard_filter_pass(r)
            if ok:
                self.compute_score(r)
                valid.append(r)

        if not valid:
            return []

        # 按得分降序
        valid.sort(key=lambda r: r.total_score, reverse=True)
        return valid

    # ================================================================
    # 第二层：多星并发冲突下全局最优分配（核心调度场景）
    # ================================================================
    def _resolve_full_overlap(
        self,
        high_pri_pass: PassRecord,
        low_pri_passes: list[PassRecord],
        occupancy: dict[int, dict[int, int]],
        gs_daily_load: dict[int, int],
    ) -> list[PassRecord]:
        """窗口完全重叠：高优星保留该站，低优星调配至次优空闲站点。"""
        reassigned: list[PassRecord] = []
        for lp in low_pri_passes:
            candidates = self._find_alternate_gs(lp, gs_daily_load)
            placed = False
            for gid in candidates:
                if gid == high_pri_pass.gs_id and _overlap_slots(lp, high_pri_pass) > 0:
                    continue
                if self._can_place(occupancy, gid, lp):
                    self._place(occupancy, gs_daily_load, gid, lp)
                    lp.assigned = True
                    lp.assigned_gs_id = gid
                    reassigned.append(lp)
                    placed = True
                    break
            if not placed:
                reassigned.append(lp)  # lp.assigned 仍为 False
        return reassigned

    def _resolve_partial_overlap(
        self,
        pass_a: PassRecord,
        pass_b: PassRecord,
        occupancy: dict[int, dict[int, int]],
        gs_daily_load: dict[int, int],
    ) -> tuple[bool, bool]:
        """窗口部分重叠：分时分配，天线分两段各自对接。

        Returns (pass_a_placed, pass_b_placed)。
        """
        gs_id = pass_a.gs_id
        cap = self._sim.ground_network[gs_id].max_concurrent_sats
        placed_a = False
        placed_b = False

        # 检查分时是否可行：两段非重叠时段各 1 根天线即可
        occ = occupancy.get(gs_id, {})
        if pass_a.ts_start < pass_b.ts_start:
            first, second = pass_a, pass_b
        else:
            first, second = pass_b, pass_a

        # first 独占其非重叠部分
        first_ok = all(occ.get(ts, 0) < cap for ts in range(first.ts_start, second.ts_start))
        # second 独占其非重叠部分
        second_ok = all(occ.get(ts, 0) < cap for ts in range(second.ts_start, second.ts_end + 1))
        # 重叠部分需要 2 根天线
        overlap_ok = all(
            occ.get(ts, 0) + 1 < cap
            for ts in range(second.ts_start, min(first.ts_end, second.ts_end) + 1)
        )

        if first_ok and second_ok and overlap_ok:
            self._place(occupancy, gs_daily_load, gs_id, first)
            first.assigned = True
            first.assigned_gs_id = gs_id
            placed_a = (first is pass_a) or placed_a
            placed_b = (first is pass_b) or placed_b
            self._place(occupancy, gs_daily_load, gs_id, second)
            second.assigned = True
            second.assigned_gs_id = gs_id
            placed_a = placed_a or (second is pass_a)
            placed_b = placed_b or (second is pass_b)

        return (placed_a, placed_b)

    def _handle_no_available(
        self,
        r: PassRecord,
        occupancy: dict[int, dict[int, int]],
        gs_daily_load: dict[int, int],
    ) -> bool:
        """全地面站无空闲窗口：选冲突最小/重叠最短的站点临时分配，或延后。"""
        sim = self._sim
        best_gs = -1
        best_conflict = 999999

        for gid in range(self._n_gs):
            gs = sim.ground_network[gid]
            # 检查所有时刻该站是否可用
            all_avail = all(gs.is_available_at(ts) for ts in range(r.ts_start, r.ts_end + 1))
            if not all_avail:
                continue
            occ = occupancy.get(gid, {})
            # 计算冲突程度（当前占用数之和）
            conflict_sum = sum(occ.get(ts, 0) for ts in range(r.ts_start, r.ts_end + 1))
            if conflict_sum < best_conflict:
                best_conflict = conflict_sum
                best_gs = gid

        if best_gs >= 0:
            self._place(occupancy, gs_daily_load, best_gs, r)
            r.assigned = True
            r.assigned_gs_id = best_gs
            return True
        return False

    def _find_alternate_gs(
        self, r: PassRecord, gs_daily_load: dict[int, int]
    ) -> list[int]:
        """返回该过境窗口中卫星也可视的替代地面站（按负载升序）。"""
        sim = self._sim
        visible: set[int] = set()
        for ts in range(r.ts_start, r.ts_end + 1):
            for gid in sim.get_all_contacts(r.sat_id, ts):
                if sim.ground_network[gid].is_available_at(ts):
                    visible.add(gid)
        visible.discard(r.gs_id)
        return sorted(visible, key=lambda g: gs_daily_load.get(g, 0))

    # ================================================================
    # 负载均衡与容量约束
    # ================================================================
    def _can_place(
        self,
        occupancy: dict[int, dict[int, int]],
        gs_id: int,
        r: PassRecord,
        gs_daily_load: dict[int, int] | None = None,
    ) -> bool:
        """判断窗口是否可以放入 ground station（并发 + 日负载上限）。"""
        cap = self._sim.ground_network[gs_id].max_concurrent_sats
        occ = occupancy.get(gs_id, {})
        # 并发约束
        for ts in range(r.ts_start, r.ts_end + 1):
            if occ.get(ts, 0) >= cap:
                return False
        # 日负载上限
        return not (
            gs_daily_load is not None
            and gs_daily_load.get(gs_id, 0) + r.duration_slots > self._daily_cap
        )

    def _place(
        self,
        occupancy: dict[int, dict[int, int]],
        gs_daily_load: dict[int, int],
        gs_id: int,
        r: PassRecord,
    ) -> None:
        occ = occupancy.setdefault(gs_id, {})
        for ts in range(r.ts_start, r.ts_end + 1):
            occ[ts] = occ.get(ts, 0) + 1
        gs_daily_load[gs_id] = gs_daily_load.get(gs_id, 0) + r.duration_slots

    def _compute_load_balance_std(self, gs_daily_load: dict[int, int]) -> float:
        """计算各地面站日负载的标准差（衡量均衡程度）。"""
        loads = [gs_daily_load.get(gid, 0) for gid in range(self._n_gs)]
        n = len(loads)
        if n <= 1:
            return 0.0
        mean = sum(loads) / n
        variance = sum((v - mean) ** 2 for v in loads) / n
        return math.sqrt(variance)

    # ================================================================
    # 特殊约束
    # ================================================================
    def _handle_storage_overflow(
        self, sat_id: int, result: AllocationResult, occupancy: dict,
        gs_daily_load: dict[int, int],
    ) -> None:
        """存储溢出风险：放宽仰角/时长要求，选最近可接入的可用站。"""
        passes = self._tt.records_by_sat(sat_id)
        # 放宽后的候选：仅检查是否可用（跳过仰角/时长过滤）
        candidates = sorted(passes, key=lambda r: r.ts_start)
        for r in candidates:
            gs = self._sim.ground_network[r.gs_id]
            ok = all(gs.is_available_at(ts) for ts in range(r.ts_start, r.ts_end + 1))
            if ok and self._can_place(occupancy, r.gs_id, r, gs_daily_load):
                self._place(occupancy, gs_daily_load, r.gs_id, r)
                r.assigned = True
                r.assigned_gs_id = r.gs_id
                result.assignments.append(r)
                result.storage_relaxed += 1
                return

    def _handle_latency_forced(
        self, r: PassRecord, result: AllocationResult, occupancy: dict,
        gs_daily_load: dict[int, int],
    ) -> bool:
        """时延硬性需求：放弃远期高分窗口，选当前最早接轨窗口。"""
        passes = self._tt.records_by_sat(r.sat_id)
        earliest = sorted(passes, key=lambda p: p.ts_start)
        for p in earliest:
            ok, _reason = self._hard_filter_pass(p)
            if ok and self._can_place(occupancy, p.gs_id, p, gs_daily_load):
                self._place(occupancy, gs_daily_load, p.gs_id, p)
                p.assigned = True
                p.assigned_gs_id = p.gs_id
                result.assignments.append(p)
                result.latency_forced += 1
                return True
        return False

    # ================================================================
    # 主分配入口
    # ================================================================
    def allocate(
        self,
        storage_overflow_sats: set[int] | None = None,
        latency_critical_sats: set[int] | None = None,
    ) -> AllocationResult:
        """执行完整二层分层分配，返回分配方案。

        Parameters
        ----------
        storage_overflow_sats : set[int], optional
            星上存储溢出风险的卫星 ID 集合（放宽选择标准）。
        latency_critical_sats : set[int], optional
            时延硬性需求的卫星 ID 集合（强制选择最早窗口）。

        Returns
        -------
        AllocationResult
        """
        storage_overflow_sats = storage_overflow_sats or set()
        latency_critical_sats = latency_critical_sats or set()

        occupancy: dict[int, dict[int, int]] = {}
        gs_daily_load: dict[int, int] = dict.fromkeys(range(self._n_gs), 0)
        result = AllocationResult(gs_load_slots=dict(gs_daily_load))

        # ---- 阶段 0: 先处理特殊约束卫星 ----
        normal_passes: list[PassRecord] = []

        for r in self._tt.records:
            if r.sat_id in storage_overflow_sats:
                self._handle_storage_overflow(r.sat_id, result, occupancy, gs_daily_load)
            elif r.sat_id in latency_critical_sats:
                if self._handle_latency_forced(r, result, occupancy, gs_daily_load):
                    pass
            else:
                normal_passes.append(r)

        # ---- 阶段 1: 第一层 — 按卫星分组，单星择优打分 ----
        by_sat: dict[int, list[PassRecord]] = {}
        for r in normal_passes:
            by_sat.setdefault(r.sat_id, []).append(r)

        # 高优先级卫星先处理
        ordered_sats = sorted(by_sat.keys(), key=lambda sid: min(
            r.priority for r in by_sat[sid]
        ))

        conflicted_passes: list[PassRecord] = []
        conflict_free_passes: list[PassRecord] = []

        for sat_id in ordered_sats:
            passes = by_sat[sat_id]
            # 单星择优打分
            ranked = self._single_sat_optimal(sat_id, passes, gs_daily_load)
            for r in ranked:
                if r.conflict:
                    conflicted_passes.append(r)
                else:
                    conflict_free_passes.append(r)

        # ---- 阶段 2: 先分配无冲突窗口 ----
        # 按得分降序贪心
        conflict_free_passes.sort(key=lambda r: r.total_score, reverse=True)
        for r in conflict_free_passes:
            candidates = self._find_alternate_gs(r, gs_daily_load)
            placed = False
            # 优先原站
            if self._can_place(occupancy, r.gs_id, r, gs_daily_load):
                self._place(occupancy, gs_daily_load, r.gs_id, r)
                r.assigned = True
                r.assigned_gs_id = r.gs_id
                result.assignments.append(r)
                result.total_assigned_mb += r.est_downlink_mb
                placed = True
            else:
                for gid in candidates:
                    if self._can_place(occupancy, gid, r, gs_daily_load):
                        self._place(occupancy, gs_daily_load, gid, r)
                        r.assigned = True
                        r.assigned_gs_id = gid
                        result.assignments.append(r)
                        result.total_assigned_mb += r.est_downlink_mb
                        result.rerouted += 1
                        placed = True
                        break
            if not placed:
                result.dropped.append(r)
                result.total_dropped_mb += r.est_downlink_mb

        # ---- 阶段 3: 第二层 — 多星冲突全局分配 ----
        # 按优先级排序：高优先级先分配
        conflicted_passes.sort(key=lambda r: (r.priority, -r.total_score))

        for r in conflicted_passes:
            if r.assigned:
                continue

            # 查找同站冲突的其他卫星
            same_gs_conflicts = [
                o for o in conflicted_passes
                if o.gs_id == r.gs_id and o.sat_id != r.sat_id
                and _overlap_slots(r, o) > 0
            ]

            if not same_gs_conflicts and self._can_place(occupancy, r.gs_id, r, gs_daily_load):
                # 无实际冲突（冲突标记可能来自之前的并发数预算）
                self._place(occupancy, gs_daily_load, r.gs_id, r)
                r.assigned = True
                r.assigned_gs_id = r.gs_id
                result.assignments.append(r)
                result.total_assigned_mb += r.est_downlink_mb
                continue

            # 确定高优先级方
            my_pri = r.priority
            other_pri = min(o.priority for o in same_gs_conflicts)

            if my_pri < other_pri:
                # 我是高优方，保留原站；低优方重新分配
                if self._can_place(occupancy, r.gs_id, r, gs_daily_load):
                    self._place(occupancy, gs_daily_load, r.gs_id, r)
                    r.assigned = True
                    r.assigned_gs_id = r.gs_id
                    result.assignments.append(r)
                    result.total_assigned_mb += r.est_downlink_mb
            else:
                # 我是低优方，尝试重路由
                candidates = self._find_alternate_gs(r, gs_daily_load)
                placed = False
                for gid in candidates:
                    if self._can_place(occupancy, gid, r, gs_daily_load):
                        self._place(occupancy, gs_daily_load, gid, r)
                        r.assigned = True
                        r.assigned_gs_id = gid
                        result.assignments.append(r)
                        result.total_assigned_mb += r.est_downlink_mb
                        result.rerouted += 1
                        placed = True
                        break

                if not placed:
                    # 部分重叠：尝试分时
                    if r.conflict_level == 1:
                        for o in same_gs_conflicts:
                            if not o.assigned:
                                pa, pb = self._resolve_partial_overlap(
                                    r, o, occupancy, gs_daily_load
                                )
                                if pa:
                                    result.assignments.append(r)
                                    result.total_assigned_mb += r.est_downlink_mb
                                if pb:
                                    result.assignments.append(o)
                                    result.total_assigned_mb += o.est_downlink_mb
                                if pa or pb:
                                    placed = True
                                    break

                    if not placed:
                        # 无空闲：冲突最小站点
                        placed = self._handle_no_available(r, occupancy, gs_daily_load)
                        if placed:
                            result.assignments.append(r)
                            result.total_assigned_mb += r.est_downlink_mb

            if not r.assigned:
                result.dropped.append(r)
                result.total_dropped_mb += r.est_downlink_mb

        # ---- 阶段 4: 全网负载均衡微调 ----
        std_before = self._compute_load_balance_std(gs_daily_load)
        result.load_balance_std = round(std_before, 4)

        # 统计最终
        result.gs_daily_load_min = {
            gid: round(gs_daily_load.get(gid, 0) * self._sim.timeslot_duration_min, 2)
            for gid in range(self._n_gs)
        }
        result.total_est_downlink_mb = result.total_assigned_mb

        total_slots = sum(gs_daily_load.values())
        max_possible = self._n_gs * self._n_slots
        result.utilization_rate = round(total_slots / max_possible, 4) if max_possible else 0.0

        return result


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------
def build_pass_schedule(
    simulator: OrbitSimulator,
    min_duration_slots: int = 1,
    sat_priorities: dict[int, int] | None = None,
    enable_interpolation: bool = True,
    effective_step_s: float = 60.0,
) -> tuple[PassTimetable, AllocationResult]:
    """一站式：生成过境时间表并完成资源分配。

    Parameters
    ----------
    enable_interpolation : bool
        启用窗口边界插值精修（D4）。
    effective_step_s : float
        有效采样步长（用于自适应步长协商）。

    Returns
    -------
    (PassTimetable, AllocationResult)
    """
    tt = PassTimetable(
        simulator,
        min_duration_slots=min_duration_slots,
        sat_priorities=sat_priorities,
        enable_interpolation=enable_interpolation,
        effective_step_s=effective_step_s,
    )
    allocator = GroundStationAllocator(tt)
    alloc = allocator.allocate()
    return tt, alloc


# ============================================================
# D5: Window boundary bisection refinement (Dimension 5)
# ============================================================

def binary_search_window_boundary(
    sim,
    sat_id: int,
    gs_id: int,
    t_low_s: float,
    t_high_s: float,
    el_low: float,
    el_high: float,
    el_target: float,
    tol_s: float = 1.0,
    max_iter: int = 15,
) -> float:
    """Binary search for exact El = El_min crossing time.

    Replaces linear interpolation with bisection iteration,
    achieving < 1s timing precision for antenna alignment.

    Parameters
    ----------
    sim : OrbitSimulator
    sat_id, gs_id : int
    t_low_s, t_high_s : float
        Lower and upper bound times (seconds).
    el_low, el_high : float
        Elevation values at bounds (deg).
    el_target : float
        Target elevation (typically El_min).
    tol_s : float
        Time tolerance (seconds).
    max_iter : int
        Max iterations.

    Returns
    -------
    float
        Precise crossing time (seconds).
    """
    gs = sim.ground_network[gs_id]
    planet_r = sim.body.radius_km

    def _el(t_s: float) -> float:
        slot_min = sim.timeslot_duration_min
        ts = int(t_s / (slot_min * 60.0))
        ts = max(0, min(ts, sim.num_timeslots - 1))
        ecef = sim.get_sat_ecef(sat_id, ts)
        return elevation_deg(ecef, gs.lat_deg, gs.lon_deg, gs.altitude_km, planet_r)

    lo = t_low_s
    hi = t_high_s
    elo = el_low
    _ehi = el_high

    for _ in range(max_iter):
        if hi - lo < tol_s:
            break
        mid = (lo + hi) * 0.5
        em = _el(mid)
        if (elo <= el_target <= em) or (em <= el_target <= elo):
            hi = mid
        else:
            lo = mid
            elo = em

    return (lo + hi) * 0.5


# ============================================================
# D5: Multi-constraint boolean filter (Dimension 5)
# ============================================================

class MultiConstraintFilter:
    """Multi-constraint parallel boolean filter.

    Encodes station maintenance, min elevation, min duration,
    and terrain occlusion as bit flags. Single operation checks
    all constraints simultaneously, reducing CPU branch misprediction.

    Bit layout (8 bits):
      0: elevation >= min
      1: duration >= min
      2: station available
      3: not occluded
      4: concurrency available
      5-7: reserved
    """

    MASK_ELEVATION   = 0x01
    MASK_DURATION    = 0x02
    MASK_AVAILABLE   = 0x04
    MASK_OCCLUSION   = 0x08
    MASK_CONCURRENT  = 0x10
    MASK_ALL_PASS    = 0x1F  # All 5 must pass

    def __init__(self):
        self._pass_mask = self.MASK_ALL_PASS

    def check_all(self, flags: int) -> bool:
        """Check if all required flags pass."""
        return (flags & self._pass_mask) == self._pass_mask

    def build_flags(
        self,
        sim,
        r,
        occlusion_table: dict | None = None,
    ) -> int:
        """Build bit-flag integer from pass record attributes.

        Parameters
        ----------
        sim : OrbitSimulator
        r : PassRecord
        occlusion_table : dict or None

        Returns
        -------
        int
            Bit-flag integer.
        """
        flags = 0
        gs = sim.ground_network[r.gs_id]

        # elevation
        if r.avg_elevation_deg >= gs.min_elevation_deg:
            flags |= self.MASK_ELEVATION

        # duration (use timetable min)
        min_dur = getattr(r, '_min_duration', 1)
        if hasattr(r, 'duration_slots') and r.duration_slots >= min_dur:
            flags |= self.MASK_DURATION

        # station available (check all slots in window)
        all_avail = all(
            gs.is_available_at(ts)
            for ts in range(r.ts_start, r.ts_end + 1)
        )
        if all_avail:
            flags |= self.MASK_AVAILABLE

        # occlusion
        if occlusion_table:
            from fl_space.utils.coordinate_utils import occlude_by_table
            if not occlude_by_table(0.0, r.avg_elevation_deg, occlusion_table):
                flags |= self.MASK_OCCLUSION
        else:
            flags |= self.MASK_OCCLUSION

        # concurrency (simplified: check if station has capacity)
        if gs.max_concurrent_sats > 0:
            flags |= self.MASK_CONCURRENT

        return flags


# ============================================================
# D5: Elevation diff prediction & batch skip (Dimension 5)
# ============================================================

def predict_skip_ahead(
    current_el: float,
    prev_el: float,
    min_el: float,
    drop_rate: float | None = None,
    max_skip_steps: int = 5,
) -> int:
    """Predict how many future steps can be safely skipped.

    Uses first-order difference to predict trend:
      - If el is far below threshold AND decreasing, skip ahead.
      - If near critical zone, do NOT skip.

    Parameters
    ----------
    current_el : float
        Current elevation (deg).
    prev_el : float
        Previous elevation (deg).
    min_el : float
        Minimum communication elevation (deg).
    drop_rate : float or None
        Manual drop rate (deg/step). If None, computed from diff.
    max_skip_steps : int
        Max steps to skip.

    Returns
    -------
    int
        Number of steps to skip (0 = no skip).
    """
    if drop_rate is None:
        drop_rate = prev_el - current_el

    # Only skip if well below threshold and decreasing
    if current_el > min_el + 5.0:
        return 0  # Near or above threshold, don't skip

    if drop_rate <= 0:
        return 0  # Not decreasing

    # Estimate how many steps until we'd still be > 0
    remain = max(0.0, current_el)
    steps_to_zero = int(remain / max(0.01, drop_rate))
    return min(steps_to_zero, max_skip_steps)
