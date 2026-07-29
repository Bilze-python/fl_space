"""
接力过境窗口合并 + 邻近地面站批量计算 (furr_chk 八-1 / 八-2)
=============================================================

1. **接力过境窗口合并** (furr_chk 八-2):
   在定时演算提取窗口阶段，自动识别相邻地面站的接续可视时段，
   合并为一条连续通信链路。同步输出接力传输时刻表，
   后处理分配算法无需二次遍历匹配。

2. **邻近地面站批量统一计算** (furr_chk 八-1):
   地理位置相距很近的一批地面站，卫星相对矢量差值极小。
   计算出一组 ENU/仰角数据后，基于地理位置偏移量批量修正，
   不用对每一个站点重复整套坐标转换运算。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MergedRelayWindow:
    """合并后的接力过境窗口。"""

    sat_id: int
    relay_chain: list[int]           # 接力地面站链 [gs_a, gs_b, gs_c, ...]
    timeslot_start: int
    timeslot_end: int
    duration_slots: int
    is_merged: bool = True           # 是否为合并窗口 (vs 单站窗口)


def merge_relay_windows(
    windows: list[dict],       # 每个: {"sat_id": int, "gs_id": int, "ts_start": int, "ts_end": int}
    gap_tolerance_slots: int = 2,
    min_elevation_deg: float = 5.0,
) -> list[MergedRelayWindow]:
    """合并同卫星相邻地面站的接力过境窗口。

    识别规则:
        1. 同卫星的窗口按 timeslot_start 排序
        2. 若窗口 A 结束 与 窗口 B 开始 间隙 <= gap_tolerance_slots
           → 合并为接力窗口 (A → B)
        3. 合并后窗口的 relay_chain 按时序连接

    Parameters
    ----------
    windows : list[dict]
        原始过境窗口列表。
    gap_tolerance_slots : int
        合并容忍间隙 (timeslots)。
    min_elevation_deg : float
        最低仰角 (保留字段，供后续扩展)。

    Returns
    -------
    list[MergedRelayWindow]
        合并后的窗口列表 (含原始单站窗口和合并接力窗口)。
    """
    # 按卫星分组、排序
    by_sat: dict[int, list[dict]] = {}
    for w in windows:
        by_sat.setdefault(w["sat_id"], []).append(w)

    for sat_windows in by_sat.values():
        sat_windows.sort(key=lambda w: w["ts_start"])

    merged: list[MergedRelayWindow] = []

    for sat_id, sat_windows in by_sat.items():
        i = 0
        while i < len(sat_windows):
            chain_start = i
            chain_end = i

            # 向前延伸：寻找可接续的后续窗口
            while chain_end + 1 < len(sat_windows):
                cur_end = sat_windows[chain_end]["ts_end"]
                next_start = sat_windows[chain_end + 1]["ts_start"]
                gap = next_start - cur_end
                if gap <= gap_tolerance_slots and gap >= -1:
                    chain_end += 1
                else:
                    break

            if chain_end > chain_start:
                # 合并
                chain_gs = [sat_windows[j]["gs_id"] for j in range(chain_start, chain_end + 1)]
                merged.append(MergedRelayWindow(
                    sat_id=sat_id,
                    relay_chain=chain_gs,
                    timeslot_start=sat_windows[chain_start]["ts_start"],
                    timeslot_end=sat_windows[chain_end]["ts_end"],
                    duration_slots=sat_windows[chain_end]["ts_end"] - sat_windows[chain_start]["ts_start"] + 1,
                    is_merged=True,
                ))
            else:
                # 单站窗口
                w = sat_windows[chain_start]
                merged.append(MergedRelayWindow(
                    sat_id=sat_id,
                    relay_chain=[w["gs_id"]],
                    timeslot_start=w["ts_start"],
                    timeslot_end=w["ts_end"],
                    duration_slots=w["ts_end"] - w["ts_start"] + 1,
                    is_merged=False,
                ))

            i = chain_end + 1

    return merged


# ============================================================
# 邻近地面站批量统一计算 (furr_chk 八-1)
# ============================================================

@dataclass
class GSProximityGroup:
    """邻近地面站组 — 地理位置相距很近的一批站点。"""

    group_id: int
    gs_ids: list[int]
    center_lat_deg: float          # 组中心纬度
    center_lon_deg: float          # 组中心经度
    reference_gs_id: int           # 参考站 (全量 ENU 计算)
    max_offset_km: float           # 组内最大偏移 (km)


def group_nearby_gs(
    gs_lats: np.ndarray,    # (N,)
    gs_lons: np.ndarray,    # (N,)
    proximity_threshold_km: float = 500.0,
) -> list[GSProximityGroup]:
    """将地理位置相近的地面站分组，同组批量修正。

    原理:
        同组内仅对 reference_gs_id 做完整 ENU/仰角计算，
        其余站点基于地理位置偏移量 + 小角度近似校正。

    Parameters
    ----------
    gs_lats : np.ndarray
        各站点纬度 (°)。
    gs_lons : np.ndarray
        各站点经度 (°)。
    proximity_threshold_km : float
        邻近判定阈值 (km), 小于此距离视为同组。

    Returns
    -------
    list[GSProximityGroup]
    """
    n = len(gs_lats)
    assigned = np.zeros(n, dtype=bool)
    groups: list[GSProximityGroup] = []
    gid = 0

    # 简单的贪心分组: 对每个未分配的站点, 找附近站点组成簇
    for i in range(n):
        if assigned[i]:
            continue

        members = [i]
        assigned[i] = True

        for j in range(i + 1, n):
            if assigned[j]:
                continue
            dist_km = _haversine_km(gs_lats[i], gs_lons[i], gs_lats[j], gs_lons[j])
            if dist_km <= proximity_threshold_km:
                members.append(j)
                assigned[j] = True

        if len(members) == 1:
            # 单站组: 不需要批量修正，但保留组结构
            groups.append(GSProximityGroup(
                group_id=gid,
                gs_ids=members,
                center_lat_deg=float(gs_lats[i]),
                center_lon_deg=float(gs_lons[i]),
                reference_gs_id=i,
                max_offset_km=0.0,
            ))
        else:
            # 多站组: 以中心点为参考
            c_lat = float(np.mean(gs_lats[members]))
            c_lon = float(np.mean(gs_lons[members]))
            offsets = [_haversine_km(c_lat, c_lon, float(gs_lats[m]), float(gs_lons[m])) for m in members]
            groups.append(GSProximityGroup(
                group_id=gid,
                gs_ids=members,
                center_lat_deg=c_lat,
                center_lon_deg=c_lon,
                reference_gs_id=members[0],
                max_offset_km=max(offsets) if offsets else 0.0,
            ))

        gid += 1

    return groups


def batch_correct_elevation(
    ref_elevation_deg: float,
    ref_gs_lat: float,
    ref_gs_lon: float,
    target_gs_lat: float,
    target_gs_lon: float,
    sat_range_km: float = 1500.0,
) -> float:
    """基于参考站仰角 + 地理位置偏移批量修正目标站仰角。

    原理:
        对于地理位置相近的两个地面站 A (参考) 和 B (目标):
        - 卫星到两站的视线方向差异很小 (≈ offset / range 弧度级)
        - 仰角差异 ≈ Δθ ≈ (offset · cos(azimuth)) / range

    Parameters
    ----------
    ref_elevation_deg : float
        参考站仰角 (°)。
    ref_gs_lat, ref_gs_lon : float
        参考站坐标 (°)。
    target_gs_lat, target_gs_lon : float
        目标站坐标 (°)。
    sat_range_km : float
        卫星到地面站的距离 (km), 默认 1500km (LEO)。

    Returns
    -------
    float
        修正后的目标站仰角 (°)。
    """
    offset_km = _haversine_km(ref_gs_lat, ref_gs_lon, target_gs_lat, target_gs_lon)

    # 一阶近似: Δel ≈ offset / range (弧度), 保守估计取最大值
    delta_el_deg = np.degrees(offset_km / max(sat_range_km, 100.0))

    # 保守修正: 取偏低修正 (确保不虚报窗口)
    return ref_elevation_deg - delta_el_deg


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine 地表距离 (km)。"""
    R = 6371.0  # noqa: N806
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(min(1.0, np.sqrt(a)))
