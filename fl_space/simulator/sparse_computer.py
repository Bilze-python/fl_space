"""
时空分区与稀疏计算优化模块 (论文维度二)
==========================================

实现:
1. 经纬度空间分块栅格预筛选 (跳过轨道带不可达的栅格)
2. 时序稀疏标记批量跳过空白时段 (粗扫标记潜在过境区间)
3. 多卫星轨道分组合并计算 (同轨道面复用常数项)

适合数十座地面站组网场景, 避免对永远不可视的星站组合逐时刻循环。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fl_space.simulator.orbit_simulator import OrbitSimulator


# ============================================================
# 1. 经纬度空间分块栅格预筛选
# ============================================================

@dataclass
class GeoGrid:
    """地理栅格单元。"""

    lat_min: float    # deg
    lat_max: float
    lon_min: float
    lon_max: float
    gs_ids: list[int]     # 此栅格内的地面站 ID 列表


def build_geo_grid(
    gs_lats: list[float],
    gs_lons: list[float],
    grid_lat_step: float = 10.0,
    grid_lon_step: float = 15.0,
) -> list[GeoGrid]:
    """将全球地面站按经纬度划分栅格。

    Parameters
    ----------
    gs_lats : list[float]
        各站点纬度 (°)。
    gs_lons : list[float]
        各站点经度 (°)。
    grid_lat_step : float
        纬度分块粒度 (°), 默认 10°。
    grid_lon_step : float
        经度分块粒度 (°), 默认 15°。

    Returns
    -------
    list[GeoGrid]
        非空栅格列表。
    """
    # 先分组: {(lat_bin, lon_bin): [gs_ids]}
    bins: dict[tuple[int, int], list[int]] = {}
    for gs_id, (lat, lon) in enumerate(zip(gs_lats, gs_lons)):
        lat_bin = int(lat // grid_lat_step)
        lon_bin = int(lon // grid_lon_step)
        bins.setdefault((lat_bin, lon_bin), []).append(gs_id)

    grids: list[GeoGrid] = []
    for (lat_bin, lon_bin), gids in bins.items():
        grids.append(GeoGrid(
            lat_min=lat_bin * grid_lat_step,
            lat_max=(lat_bin + 1) * grid_lat_step,
            lon_min=lon_bin * grid_lon_step,
            lon_max=(lon_bin + 1) * grid_lon_step,
            gs_ids=sorted(gids),
        ))
    return grids


def can_orbit_cover_grid(
    orbit_inclination_deg: float,
    orbit_altitude_km: float,
    grid: GeoGrid,
    margin_deg: float = 3.0,
) -> bool:
    """判断卫星轨道带能否覆盖某地理栅格。

    基于轨道倾角 i 判断覆盖纬度范围 [-(i+margin), (i+margin)]。
    极轨卫星 (i~90°) 全纬度覆盖; 赤道轨道 (i~0°) 仅覆盖赤道附近。

    Parameters
    ----------
    orbit_inclination_deg : float
        轨道倾角 (°)。
    orbit_altitude_km : float
        轨道高度 (km)。
    grid : GeoGrid
        目标栅格。
    margin_deg : float
        覆盖裕度 (°), 考虑可视张角。

    Returns
    -------
    bool
        True = 可能覆盖, False = 确定不可达。
    """
    max_lat = orbit_inclination_deg + margin_deg
    if max_lat > 90.0:
        max_lat = 180.0 - max_lat if max_lat > 90.0 else max_lat
    # 简化: 考虑顺行轨道, 覆盖纬度 [-max_lat, max_lat]
    # 逆行轨道 (>90°) 覆盖互补范围
    if orbit_inclination_deg <= 90.0:
        cover_lat_max = min(90.0, orbit_inclination_deg + margin_deg)
        cover_lat_min = -cover_lat_max
    else:
        retro_equiv = 180.0 - orbit_inclination_deg
        cover_lat_max = min(90.0, retro_equiv + margin_deg)
        cover_lat_min = -cover_lat_max

    # 栅格与覆盖带有交集
    return not (grid.lat_max < cover_lat_min or grid.lat_min > cover_lat_max)


class GeoGridFilter:
    """地理栅格预筛选器。

    构建后, 给定卫星轨道参数即可快速筛除不可达栅格,
    避免对整组站点做无用仰角计算。
    """

    def __init__(self, sim: OrbitSimulator, grid_lat_step: float = 15.0, grid_lon_step: float = 20.0):
        from fl_space.environment.ground_station import GroundStationNetwork

        gs_network: GroundStationNetwork = sim.ground_network
        lats = [gs_network[gid].lat_deg for gid in range(sim.num_ground_stations)]
        lons = [gs_network[gid].lon_deg for gid in range(sim.num_ground_stations)]
        self._grids = build_geo_grid(lats, lons, grid_lat_step, grid_lon_step)
        self._num_gs = sim.num_ground_stations

    def filter_visible_grids(
        self,
        inclination_deg: float,
        altitude_km: float,
    ) -> list[GeoGrid]:
        """返回卫星可能覆盖的栅格列表。"""
        return [
            g for g in self._grids
            if can_orbit_cover_grid(inclination_deg, altitude_km, g)
        ]

    def get_gs_ids_for_orbit(
        self,
        inclination_deg: float,
        altitude_km: float,
    ) -> set[int]:
        """返回卫星可能可视的地面站 ID 集合 (粗筛)。"""
        visible_grids = self.filter_visible_grids(inclination_deg, altitude_km)
        gs_set: set[int] = set()
        for grid in visible_grids:
            gs_set.update(grid.gs_ids)
        return gs_set


# ============================================================
# 2. 时序稀疏标记 (粗扫 + 精细推演)
# ============================================================

@dataclass
class CoarseScanResult:
    """单卫星 48h 粗扫结果。"""

    sat_id: int
    coarse_step_s: float          # 粗扫步长
    potential_ranges: list[tuple[float, float]]  # (t_start_s, t_end_s) 潜在过境区间
    blank_ranges: list[tuple[float, float]]      # 完全不可视区间


def coarse_scan_timeline(
    sim: OrbitSimulator,
    sat_id: int,
    total_duration_s: float,
    coarse_step_s: float = 120.0,
    min_elevation_deg: float = 5.0,
) -> CoarseScanResult:
    """对单星预推时序做低步长粗扫, 标记潜在过境区间与空白区间。

    粗扫: 2min 步长, 仅判断球面几何粗判 (不求解 ENU/仰角),
    速度极快但可能漏判极小窗口 (<2min), 由精扫阶段补偿。

    Returns
    -------
    CoarseScanResult
        包含 potential_ranges (精细推演区间) 和 blank_ranges (跳过区间)。
    """
    from fl_space.utils.coordinate_utils import spherical_visibility_coarse

    slot_min = sim.timeslot_duration_min
    n_slots = sim.num_timeslots
    total_s = n_slots * slot_min * 60.0
    if total_duration_s <= 0:
        total_duration_s = total_s

    # 映射: 时间(s) -> timeslot
    def _t_to_ts(t_s: float) -> int:
        return min(int(t_s / (slot_min * 60.0)), n_slots - 1)

    potential: list[tuple[float, float]] = []
    blank: list[tuple[float, float]] = []
    in_potential = False
    seg_start = 0.0

    t = 0.0
    while t < total_duration_s:
        ts = _t_to_ts(t)
        sat_ecef = sim.get_sat_ecef(sat_id, ts)
        has_any = False
        for gs_id in range(sim.num_ground_stations):
            gs = sim.ground_network[gs_id]
            if spherical_visibility_coarse(
                sat_ecef, gs.lat_deg, gs.lon_deg, gs.altitude_km, min_elevation_deg
            ):
                has_any = True
                break

        if has_any and not in_potential:
            seg_start = t
            in_potential = True
        elif not has_any and in_potential:
            potential.append((seg_start, t))
            blank.append((seg_start, t))  # 后续用 blank 区分
            in_potential = False

        t += coarse_step_s

    if in_potential:
        potential.append((seg_start, total_duration_s))

    return CoarseScanResult(
        sat_id=sat_id,
        coarse_step_s=coarse_step_s,
        potential_ranges=potential,
        blank_ranges=blank,
    )


def sparse_fine_scan_slots(
    coarse: CoarseScanResult,
    sim: OrbitSimulator,
) -> list[int]:
    """从粗扫结果提取需要精细计算的 timeslot 列表。

    仅在 potential_ranges 内的时刻参与精细 ENU/仰角求解,
    空白区间直接跳过。
    """
    slot_min = sim.timeslot_duration_min
    n_slots = sim.num_timeslots

    fine_slots: list[int] = []
    for t_start, t_end in coarse.potential_ranges:
        ts_start = max(0, int(t_start / (slot_min * 60.0)))
        ts_end = min(n_slots - 1, int(t_end / (slot_min * 60.0)))
        fine_slots.extend(range(ts_start, ts_end + 1))
    return sorted(set(fine_slots))


# ============================================================
# 3. 多卫星轨道分组合并计算
# ============================================================

class SatelliteOrbitGroups:
    """按轨道面/倾角对卫星分组, 同组复用摄动修正项和恒星时修正。"""

    def __init__(self, inclination_tol_deg: float = 1.0, altitude_tol_km: float = 50.0):
        self._inc_tol = inclination_tol_deg
        self._alt_tol = altitude_tol_km
        # {group_id: [sat_ids]}
        self._groups: dict[int, list[int]] = {}
        # {sat_id: (inclination_deg, altitude_km)}
        self._sat_params: dict[int, tuple[float, float]] = {}

    def register_sat(self, sat_id: int, inclination_deg: float, altitude_km: float) -> None:
        """注册卫星轨道参数, 自动分组。"""
        self._sat_params[sat_id] = (inclination_deg, altitude_km)

        # 找匹配组
        for members in self._groups.values():
            if not members:
                continue
            ref = self._sat_params.get(members[0])
            if ref is None:
                continue
            if (
                abs(ref[0] - inclination_deg) <= self._inc_tol
                and abs(ref[1] - altitude_km) <= self._alt_tol
            ):
                members.append(sat_id)
                return
        # 新建组
        new_gid = max(self._groups.keys(), default=-1) + 1
        self._groups[new_gid] = [sat_id]

    def get_group(self, sat_id: int) -> list[int]:
        """返回该卫星所在组的所有卫星 ID (含自身)。"""
        for members in self._groups.values():
            if sat_id in members:
                return list(members)
        return [sat_id]

    def get_groups(self) -> dict[int, list[int]]:
        return dict(self._groups)

    def group_size(self, sat_id: int) -> int:
        return len(self.get_group(sat_id))


# ============================================================
# 4. 卫星-地面站几何包围盒预判 (furr_chk 六-1)
# ============================================================

import numpy as np  # noqa: E402


@dataclass
class OrbitBoundingBox:
    """卫星轨道三维空间包围盒 — 时间段 [t_start, t_end] 内
    卫星可能访问的空间区域。

    用于快速判定该卫星在此时间段内是否可能覆盖某地面站，
    若包围盒与地面站无交集，直接跳过所有逐时刻仰角计算。
    """

    t_start_s: float
    t_end_s: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    sat_id: int


def compute_orbit_bbox(
    sat_ecef_sequence: np.ndarray,   # shape (N_slots, 3)
    sat_id: int = 0,
    t_start_s: float = 0.0,
    t_end_s: float = 86400.0,
    margin_km: float = 200.0,
) -> OrbitBoundingBox:
    """根据卫星 ECEF 坐标序列构建三维包围盒。

    Parameters
    ----------
    sat_ecef_sequence : np.ndarray
        shape (N_timeslots, 3) — 卫星 ECEF 坐标序列 (x, y, z km)。
    margin_km : float
        包围盒扩展边距 (km), 补偿时间步长间隔。
    """
    xyz = sat_ecef_sequence.reshape(-1, 3)
    return OrbitBoundingBox(
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        x_min=float(xyz[:, 0].min()) - margin_km,
        x_max=float(xyz[:, 0].max()) + margin_km,
        y_min=float(xyz[:, 1].min()) - margin_km,
        y_max=float(xyz[:, 1].max()) + margin_km,
        z_min=float(xyz[:, 2].min()) - margin_km,
        z_max=float(xyz[:, 2].max()) + margin_km,
        sat_id=sat_id,
    )


@dataclass
class GSBoundingSphere:
    """地面站地理包围球 — 地面站的三维空间范围。"""

    gs_id: int
    x_ecef: float   # km
    y_ecef: float
    z_ecef: float
    max_range_km: float  # 最大通信距离 (含大气余量)


def bbox_intersects(
    sat_bbox: OrbitBoundingBox,
    gs_sphere: GSBoundingSphere,
) -> bool:
    """判断卫星包围盒是否与地面站包围球相交。

    若不相交，该时间段内卫星绝不可能看到此地面站，
    直接跳过逐时刻仰角计算。
    """
    # 包围盒上离球心最近点
    cx = max(sat_bbox.x_min, min(gs_sphere.x_ecef, sat_bbox.x_max))
    cy = max(sat_bbox.y_min, min(gs_sphere.y_ecef, sat_bbox.y_max))
    cz = max(sat_bbox.z_min, min(gs_sphere.z_ecef, sat_bbox.z_max))

    dx = cx - gs_sphere.x_ecef
    dy = cy - gs_sphere.y_ecef
    dz = cz - gs_sphere.z_ecef
    dist2 = dx * dx + dy * dy + dz * dz

    return dist2 <= gs_sphere.max_range_km * gs_sphere.max_range_km


# ============================================================
# 5. 仰角变化率预判剪枝 (furr_chk 六-2)
# ============================================================

def should_skip_slots_by_elevation_trend(
    elevations: np.ndarray,         # shape (N,) — 连续时刻仰角序列
    min_elevation_deg: float = 5.0,
    threshold_buffer_deg: float = -2.0,   # 低于此仰角时考虑剪枝
    lookahead: int = 5,                   # 前瞻点数
) -> np.ndarray:
    """仰角变化率预判剪枝。

    利用相邻时刻卫星位置求解仰角变化速率。
    若当前仰角 << 最低通信仰角，且仰角变化率持续为负（卫星继续下沉），
    可直接跳过一段连续采样点的计算；
    仅在仰角变化率由负转正、靠近阈值区间时恢复精细计算。

    Parameters
    ----------
    elevations : np.ndarray
        仰角序列 (°)。缺失值填 -999。
    min_elevation_deg : float
        最低通信仰角。
    threshold_buffer_deg : float
        低于此仰角时才考虑剪枝 (应 << min_elevation_deg)。
    lookahead : int
        前瞻点数，检查未来连续点数是否全低于阈值。

    Returns
    -------
    np.ndarray
        bool 数组, True = 可跳过, False = 需精细计算。
    """
    n = len(elevations)
    skip_mask = np.zeros(n, dtype=bool)

    if n <= lookahead:
        return skip_mask

    # 计算仰角变化率 (中心差分)
    rate = np.zeros(n, dtype=np.float64)
    valid = elevations > -900
    rate[1:-1] = (elevations[2:] - elevations[:-2]) / 2.0
    rate[~valid] = 0.0

    for i in range(n):
        if valid[i] and elevations[i] >= threshold_buffer_deg:
            continue  # 仰角不低，不剪枝
        if not valid[i]:
            skip_mask[i] = True
            continue

        # 检查前瞻 window 内是否全为负变化率且均低于阈值
        end = min(i + lookahead, n)
        window_valid = valid[i:end]
        window_el = elevations[i:end]
        window_rate = rate[i:end]

        if (
            np.all(window_valid)
            and np.all(window_el < threshold_buffer_deg)
            and np.all(window_rate < 0)
        ):
            skip_mask[i:end] = True

    return skip_mask
