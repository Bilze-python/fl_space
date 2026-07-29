#!/usr/bin/env python3
"""
多地面站组网联合计算减负模块 — 邻近站点批量计算 + 接力过境窗口合并
D8: Multi-ground-station network joint computation optimization.

设计目标:
    - 邻近地面站批量 ENU 修正: 一组 ENU 数据 ± 地理偏移修正
    - 接力过境窗口合并: 相邻站点接续时段合并为连续链路
    - 减少重复坐标转换 60-80%
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

# ═══════════════════════════════════════════════════════════════════
#  1. 地理位置邻近地面站批量统一计算
# ═══════════════════════════════════════════════════════════════════


@dataclass
class GSCluster:
    """邻近地面站聚类 — 地理位置相近的一批地面站。"""

    cluster_id: int
    center_gs_id: int  # 中心参考站
    member_gs_ids: list[int]  # 成员站 ID (含中心站)
    center_lat_deg: float
    center_lon_deg: float
    center_ecef: tuple[float, float, float]  # 中心站 ECEF
    offsets_ecef: dict[int, tuple[float, float, float]] = field(default_factory=dict)  # gs_id -> ΔECEF


class ProximityGSClusterer:
    """邻近地面站聚类器 — 将地理位置相近的地面站分组。

    聚类阈值:
        - 默认 500 km 半径内视为邻近
        - 参考站点 (中心站) 做完整 ENU 计算
        - 成员站点基于中心结果 + 偏移量修正

    用法:
        clusterer = ProximityGSClusterer(cluster_radius_km=500.0)
        clusters = clusterer.cluster(gs_ecef_list)
        for cluster in clusters:
            # 1. 对中心站做完整 ENU 计算
            enu_center = compute_enu(sat_ecef, cluster.center_ecef)
            # 2. 成员站批量修正
            for gs_id in cluster.member_gs_ids:
                offset = cluster.offsets_ecef[gs_id]
                enu_member = enu_center + offset_correction(offset)
    """

    def __init__(self, cluster_radius_km: float = 500.0):
        self._radius = cluster_radius_km

    def cluster(
        self,
        gs_data: list[dict[str, Any]],
    ) -> list[GSCluster]:
        """对地面站进行空间聚类。

        Args:
            gs_data: 地面站数据列表 [{"gs_id": int, "lat_deg": float, "lon_deg": float,
                       "ecef": (x,y,z)}, ...]

        Returns:
            GSCluster 列表
        """
        n = len(gs_data)
        visited = [False] * n
        clusters: list[GSCluster] = []

        for i in range(n):
            if visited[i]:
                continue

            center = gs_data[i]
            cluster_members = [center["gs_id"]]
            offsets: dict[int, tuple[float, float, float]] = {}

            cx, cy, cz = center["ecef"]

            for j in range(i + 1, n):
                if visited[j]:
                    continue
                other = gs_data[j]
                ox, oy, oz = other["ecef"]
                dist = math.sqrt((ox - cx) ** 2 + (oy - cy) ** 2 + (oz - cz) ** 2)

                if dist <= self._radius:
                    visited[j] = True
                    cluster_members.append(other["gs_id"])
                    offsets[other["gs_id"]] = (
                        ox - cx,
                        oy - cy,
                        oz - cz,
                    )

            visited[i] = True
            clusters.append(GSCluster(
                cluster_id=len(clusters),
                center_gs_id=center["gs_id"],
                member_gs_ids=cluster_members,
                center_lat_deg=center["lat_deg"],
                center_lon_deg=center["lon_deg"],
                center_ecef=center["ecef"],
                offsets_ecef=offsets,
            ))

        return clusters


class BatchENUComputer:
    """批量 ENU 计算器 — 基于邻近站点偏移修正。

    节省计算:
        - 传统: N 个地面站 → N 次 geodetic→ECEF + N 次 ENU 旋转矩阵
        - 批量:  1 次中心站 → (N-1) 次简单偏移修正
        - 当 N 大时, 节省 (N-1) * 2 次三角运算

    用法:
        batch = BatchENUComputer(clusters)
        enu_center, az_center, el_center = compute_center(sat_ecef, cluster.center_ecef)
        results = batch.correct_all(cluster, enu_center, sat_ecef)
    """

    def __init__(self, clusters: list[GSCluster]):
        self._clusters = {c.cluster_id: c for c in clusters}
        self._gs_to_cluster: dict[int, int] = {}
        for c in clusters:
            for gs_id in c.member_gs_ids:
                self._gs_to_cluster[gs_id] = c.cluster_id

    def get_cluster(self, gs_id: int) -> GSCluster | None:
        cid = self._gs_to_cluster.get(gs_id)
        if cid is None:
            return None
        return self._clusters.get(cid)

    @staticmethod
    def correct_elevation(
        center_elevation_deg: float,
        center_azimuth_deg: float,
        offset_ecef: tuple[float, float, float],
        sat_distance_km: float,
    ) -> tuple[float, float]:
        """基于中心站仰角/方位角, 修正成员站的仰角和方位角。

        近似: 小偏移 (< 500km) 下, 仰角变化 = -offset_U / distance * 180/π
              方位角变化 = offset_horizontal / (distance * cos(el)) * 180/π

        Args:
            center_elevation_deg: 中心站仰角
            center_azimuth_deg: 中心站方位角
            offset_ecef: 成员站相对中心站的 ECEF 偏移 (km)
            sat_distance_km: 卫星到中心站的距离 (km)

        Returns:
            (corrected_elevation_deg, corrected_azimuth_deg)
        """
        if sat_distance_km < 1.0:
            return (center_elevation_deg, center_azimuth_deg)

        # 将 ECEF 偏移转换到 ENU 框架 (使用中心站的方位角/仰角)
        el_rad = math.radians(center_elevation_deg)
        az_rad = math.radians(center_azimuth_deg)

        # ENU 基向量 (中心站)
        # East(方位角90°顺时针) → cos(az), -sin(az), 0 的水平分量
        # North(方位角0°) → -sin(az)*sin(el), -cos(az)*sin(el), cos(el)
        # Up → sin(az)*cos(el), cos(az)*cos(el), sin(el)... 实际上:
        # Up = (cos(az)*cos(el), sin(az)*cos(el), sin(el)) in terms of direction to satellite

        # 简化: 只计算轴向分量
        dx, dy, dz = offset_ecef

        # 仰角修正: 偏移在 Up 方向的分量
        # Up 方向近似 = 卫星方向
        # Δel ≈ -offset_up / distance
        cos_az = math.cos(az_rad)
        sin_az = math.sin(az_rad)
        cos_el = math.cos(el_rad)
        sin_el = math.sin(el_rad)

        # ENU → ECEF 矩阵的逆 (近似)
        # Up 分量 = -sin(lat)*cos(lon)*dx - sin(lat)*sin(lon)*dy + cos(lat)*dz
        # 此处用简化: offset 在指向卫星方向的投影
        # 卫星方向单位矢量 (ECEF)
        sat_dir_x = cos_el * sin_az  # 简化
        sat_dir_y = cos_el * cos_az
        sat_dir_z = sin_el

        offset_along_sat = dx * sat_dir_x + dy * sat_dir_y + dz * sat_dir_z
        delta_el_rad = -offset_along_sat / sat_distance_km
        delta_el_deg = math.degrees(delta_el_rad)

        # 方位角修正: 偏移在水平面的分量
        offset_horizontal = math.sqrt(
            (dx - offset_along_sat * sat_dir_x) ** 2
            + (dy - offset_along_sat * sat_dir_y) ** 2
            + (dz - offset_along_sat * sat_dir_z) ** 2
        )
        horizontal_range = sat_distance_km * cos_el
        if horizontal_range > 0.01:
            delta_az_rad = offset_horizontal / horizontal_range
            delta_az_deg = math.degrees(delta_az_rad)
        else:
            delta_az_deg = 0.0

        return (
            center_elevation_deg + delta_el_deg,
            (center_azimuth_deg + delta_az_deg) % 360.0,
        )

    def correct_all(
        self,
        cluster: GSCluster,
        center_elevation_deg: float,
        center_azimuth_deg: float,
        sat_distance_km: float,
    ) -> dict[int, tuple[float, float]]:
        """批量修正集群内所有成员站。

        Returns:
            {gs_id: (elevation_deg, azimuth_deg)}
        """
        results: dict[int, tuple[float, float]] = {
            cluster.center_gs_id: (center_elevation_deg, center_azimuth_deg),
        }

        for gs_id, offset in cluster.offsets_ecef.items():
            el, az = self.correct_elevation(
                center_elevation_deg, center_azimuth_deg, offset, sat_distance_km
            )
            results[gs_id] = (el, az)

        return results


# ═══════════════════════════════════════════════════════════════════
#  2. 接力过境窗口合并预计算
# ═══════════════════════════════════════════════════════════════════


@dataclass
class RelayWindow:
    """接力窗口 — 多个地面站接续覆盖的连续通信时段。"""

    relay_id: int
    sat_id: int
    segments: list[RelaySegment]  # 按时间排序的站点接力段
    total_start_s: float
    total_end_s: float
    total_duration_s: float

    @property
    def gs_sequence(self) -> list[int]:
        return [s.gs_id for s in self.segments]


@dataclass
class RelaySegment:
    """接力段 — 单个地面站的可视窗口。"""

    gs_id: int
    start_s: float
    end_s: float
    duration_s: float
    max_elevation_deg: float


class RelayWindowMerger:
    """接力过境窗口合并器 — 识别相邻站点的接续可视时段, 合并为连续链路。

    算法:
        1. 按时间排序所有地面站的过境窗口
        2. 扫描窗口序列:
           - 若相邻窗口有时间重叠或间隔 < gap_threshold → 合并
           - 否则作为新的接力段
        3. 输出接力传输时刻表

    优势:
        - 预处理阶段完成合并, 后处理分配算法无需二次遍历
        - 接力窗口作为整体分配, 减少调度碎片

    用法:
        merger = RelayWindowMerger(gap_threshold_s=30.0)
        windows = [
            {"sat_id": 0, "gs_id": 1, "start_s": 100, "end_s": 500},
            {"sat_id": 0, "gs_id": 2, "start_s": 480, "end_s": 900},
        ]
        relays = merger.merge(sat_id=0, windows=windows)
        # 输出: 1 个从 100s 到 900s 的接力窗口, 含 GS1→GS2 两段
    """

    def __init__(self, gap_threshold_s: float = 30.0, min_segment_duration_s: float = 10.0):
        """
        Args:
            gap_threshold_s: 两段之间的最大间隔 (秒), 超过则断开接力
            min_segment_duration_s: 最小段时长 (秒), 过滤碎片窗口
        """
        self._gap_threshold = gap_threshold_s
        self._min_duration = min_segment_duration_s

    def merge(
        self,
        sat_id: int,
        windows: list[dict[str, Any]],
    ) -> list[RelayWindow]:
        """合并过境窗口为接力链路。

        Args:
            sat_id: 卫星 ID
            windows: 过境窗口列表 [{gs_id, start_s, end_s, duration_s, max_elevation_deg}, ...]

        Returns:
            RelayWindow 列表
        """
        # 过滤太短的窗口
        valid = [w for w in windows if w.get("duration_s", 0) >= self._min_duration]

        if not valid:
            return []

        # 按起始时间排序
        valid.sort(key=lambda w: w["start_s"])

        relays: list[RelayWindow] = []
        current_segments: list[RelaySegment] = []
        current_end = -float("inf")
        used_gs: set[int] = set()

        for w in valid:
            seg = RelaySegment(
                gs_id=w["gs_id"],
                start_s=w["start_s"],
                end_s=w["end_s"],
                duration_s=w.get("duration_s", w["end_s"] - w["start_s"]),
                max_elevation_deg=w.get("max_elevation_deg", 0.0),
            )

            if not current_segments:
                # 开始新接力链
                current_segments.append(seg)
                current_end = seg.end_s
                used_gs.add(seg.gs_id)
                continue

            # 检查是否与上一个段可接力
            gap = seg.start_s - current_end

            if gap <= self._gap_threshold and seg.gs_id not in used_gs:
                # 可接力 (不同站点, 间隔小于阈值)
                current_segments.append(seg)
                current_end = max(current_end, seg.end_s)
                used_gs.add(seg.gs_id)
            elif seg.start_s <= current_end:
                # 时间重叠 (可能是同一卫星的不同地面站同时可见)
                # 选较优的站 (仰角更高)
                last_seg = current_segments[-1]
                if seg.max_elevation_deg > last_seg.max_elevation_deg:
                    # 替换最后一段
                    used_gs.discard(last_seg.gs_id)
                    current_segments[-1] = seg
                    current_end = max(current_end, seg.end_s)
                    used_gs.add(seg.gs_id)
                # 否则忽略重叠的劣质段
            else:
                # 间隔过大, 结束当前接力链
                if len(current_segments) >= 1:
                    relays.append(self._build_relay(sat_id, len(relays), current_segments))
                current_segments = [seg]
                current_end = seg.end_s
                used_gs = {seg.gs_id}

        # 最后一条接力链
        if current_segments:
            relays.append(self._build_relay(sat_id, len(relays), current_segments))

        return relays

    def _build_relay(
        self,
        sat_id: int,
        relay_id: int,
        segments: list[RelaySegment],
    ) -> RelayWindow:
        total_start = segments[0].start_s
        total_end = segments[-1].end_s
        return RelayWindow(
            relay_id=relay_id,
            sat_id=sat_id,
            segments=segments,
            total_start_s=total_start,
            total_end_s=total_end,
            total_duration_s=total_end - total_start,
        )

    @staticmethod
    def format_relay_schedule(
        relays: list[RelayWindow],
    ) -> list[dict[str, Any]]:
        """格式化接力时刻表 (用于 JSON 输出或后续调度)。"""
        schedule = []
        for r in relays:
            schedule.append({  # noqa: PERF401
                "relay_id": r.relay_id,
                "sat_id": r.sat_id,
                "total_start_s": r.total_start_s,
                "total_end_s": r.total_end_s,
                "total_duration_s": r.total_duration_s,
                "gs_sequence": r.gs_sequence,
                "num_hops": len(r.segments),
                "segments": [
                    {
                        "gs_id": s.gs_id,
                        "start_s": s.start_s,
                        "end_s": s.end_s,
                        "max_elevation_deg": s.max_elevation_deg,
                    }
                    for s in r.segments
                ],
            })
        return schedule


# ═══════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════

def estimate_batch_speedup(num_gs: int, cluster_ratio: float = 0.6) -> float:
    """估算批量计算加速比。

    Args:
        num_gs: 总地面站数
        cluster_ratio: 聚类比例 (多少站可归入邻近集群)

    Returns:
        相对逐个计算的加速比
    """
    # 聚类内的站: 中心站 1x + 成员站 0.3x (偏移修正开销)
    # 孤立站: 1x
    clustered = num_gs * cluster_ratio
    isolated = num_gs * (1 - cluster_ratio)

    # 假设每个集群 5 个站: 1 中心 + 4 成员
    cluster_count = max(1, clustered / 5)
    cost_clustered = cluster_count * 1.0 + (clustered - cluster_count) * 0.3
    cost_isolated = isolated * 1.0
    total_cost = cost_clustered + cost_isolated
    baseline = num_gs * 1.0

    return baseline / max(total_cost, 1e-9)
