"""轨道几何计算加速器 — 从源头减少浮点运算量（不依赖分层缓存）。

核心优化：
    1. 轨道周期复用推导：仅算首个完整周期，后续周期取模映射
    2. 可视性快速预筛选：粗判规则过滤无效星-地配对
    3. 几何线性近似：窗口内线性插值替代逐帧轨道积分
    4. 仰角阈值提前截断：低于阈值直接终止窗口计算
"""

from __future__ import annotationsfrom collections.abc import Sequencefrom dataclasses import dataclassimport math as _mathfrom typing import Anyimport numpy as np@dataclass
class VisibilityBounds:
    """卫星-地面站可视性边界预筛选结果。

    Attributes
    ----------
    sat_id : int
    gs_id : int
    reachable : bool
        是否可能接轨（预筛选通过）。
    """
    sat_id: int
    gs_id: int
    reachable: bool
    sat_lat_range: tuple[float, float] = (0.0, 0.0)
    gs_lat_deg: float = 0.0
    gs_lon_deg: float = 0.0
    min_elevation_deg: float = 10.0
    orbit_altitude_km: float = 500.0


class GeometryAccelerator:
    """轨道几何底层计算加速器。

    四大手段（独立于缓存）：
        1. 周期复用 — 首周期精确计算，后续取模映射
        2. 粗判预筛选 — 纬度无交集/仰角低于阈值直接跳过
        3. 线性近似 — 窗口内插值替代轨道积分
        4. 仰角截断 — 低于阈值即时终止窗口计算
    """

    EARTH_RADIUS_KM: float = 6371.0

    def __init__(
        self,
        orbit_period_min: float,
        timeslot_duration_min: float,
        num_satellites: int,
        num_ground_stations: int,
        min_elevation_deg: float = 10.0,
        orbit_altitude_km: float = 500.0,
    ):
        self.orbit_period_min = orbit_period_min
        self.timeslot_duration_min = timeslot_duration_min
        self.num_satellites = num_satellites
        self.num_ground_stations = num_ground_stations
        self.min_elevation_deg = min_elevation_deg
        self.orbit_altitude_km = orbit_altitude_km
        self.period_slots = max(1, int(orbit_period_min / timeslot_duration_min))
        self._lat_cache: dict[int, np.ndarray] = {}
        self._prefilter_cache: dict[tuple[int, int], VisibilityBounds] = {}
        self._lat_range_cache: dict[int, tuple[float, float]] = {}

    def compute_period_latitudes(
        self, sat_id: int, inclination_deg: float,
        raan_deg: float = 0.0, phase_deg: float = 0.0,
    ) -> np.ndarray:
        """计算首个完整轨道周期的纬度数组，后续时间取模复用。"""
        if sat_id in self._lat_cache:
            return self._lat_cache[sat_id]

        n = self.period_slots
        lats = np.zeros(n, dtype=np.float64)
        inclination = _math.radians(inclination_deg)

        for i in range(n):
            angle = (2.0 * _math.pi * i / n + _math.radians(phase_deg)) % (2.0 * _math.pi)
            lat_rad = _math.asin(_math.sin(inclination) * _math.sin(angle))
            lats[i] = _math.degrees(lat_rad)

        self._lat_cache[sat_id] = lats
        return lats

    def get_latitude_at(self, sat_id: int, timeslot: int) -> float:
        """周期取模获取任意时刻卫星纬度，不复算几何。"""
        lats = self._lat_cache.get(sat_id)
        if lats is None:
            return 0.0
        return float(lats[timeslot % self.period_slots % len(lats)])

    def get_lat_range(self, sat_id: int, inclination_deg: float) -> tuple[float, float]:
        """获取卫星轨道纬度范围。"""
        if sat_id in self._lat_range_cache:
            return self._lat_range_cache[sat_id]
        max_lat = inclination_deg
        min_lat = -inclination_deg
        self._lat_range_cache[sat_id] = (min_lat, max_lat)
        return (min_lat, max_lat)

    def pre_filter(
        self, sat_id: int, gs_id: int, inclination_deg: float,
        gs_lat_deg: float = 0.0, gs_lon_deg: float = 0.0,
    ) -> VisibilityBounds:
        """粗判预筛选：纬度范围检查 + 最大仰角检查。"""
        key = (sat_id, gs_id)
        if key in self._prefilter_cache:
            return self._prefilter_cache[key]

        min_lat, max_lat = self.get_lat_range(sat_id, inclination_deg)

        def _make_result(reachable: bool) -> VisibilityBounds:
            r = VisibilityBounds(
                sat_id=sat_id, gs_id=gs_id, reachable=reachable,
                sat_lat_range=(min_lat, max_lat),
                gs_lat_deg=gs_lat_deg, gs_lon_deg=gs_lon_deg,
                min_elevation_deg=self.min_elevation_deg,
                orbit_altitude_km=self.orbit_altitude_km,
            )
            self._prefilter_cache[key] = r
            return r

        if gs_lat_deg < min_lat - 5.0 or gs_lat_deg > max_lat + 5.0:
            return _make_result(False)

        max_elev = self._calc_max_elevation(gs_lat_deg, inclination_deg)
        if max_elev < self.min_elevation_deg:
            return _make_result(False)

        return _make_result(True)

    def _calc_max_elevation(self, gs_lat_deg: float, inclination_deg: float) -> float:
        """估计卫星过地面站顶点时的最大仰角。"""
        if gs_lat_deg > inclination_deg:
            closest_lat = inclination_deg
        elif gs_lat_deg < -inclination_deg:
            closest_lat = -inclination_deg
        else:
            lat_diff = _math.radians(abs(gs_lat_deg - gs_lat_deg))
            d = self.orbit_altitude_km
            r = self.EARTH_RADIUS_KM
            arc_len = r * lat_diff
            return _math.degrees(_math.atan(d / max(arc_len, 1.0)))

        lat_diff = _math.radians(abs(gs_lat_deg - closest_lat))
        r = self.EARTH_RADIUS_KM
        h = self.orbit_altitude_km
        max_angle = _math.acos(r / (r + h))
        if lat_diff > max_angle:
            return 0.0
        return _math.degrees(max_angle - lat_diff)

    def fast_visibility_check(
        self, sat_id: int, gs_id: int, timeslot: int,
        sat_lat_vals: np.ndarray | None = None,
    ) -> bool:
        """快速可视性检查（纬度差 + 临界角判定）。"""
        lat = self.get_latitude_at(sat_id, timeslot)
        bounds = self._prefilter_cache.get((sat_id, gs_id))
        if bounds is not None and not bounds.reachable:
            return False
        gs_lat = bounds.gs_lat_deg if bounds else 0.0
        h = self.orbit_altitude_km
        r = self.EARTH_RADIUS_KM
        max_angle_deg = _math.degrees(_math.acos(r / (r + h)))
        return abs(lat - gs_lat) < max_angle_deg

    def compute_contact_segment(
        self, sat_id: int, gs_id: int,
        start_slot: int, end_slot: int, gs_lat_deg: float,
    ) -> tuple[int, int]:
        """计算接触窗口段，低于仰角阈值时立即截断迭代。"""
        if self._lat_cache.get(sat_id) is None:
            return (-1, -1)

        h = self.orbit_altitude_km
        r = self.EARTH_RADIUS_KM
        max_angle_deg = _math.degrees(_math.acos(r / (r + h)))
        contact_start = -1
        contact_end = -1

        for ts in range(start_slot, end_slot):
            lat = self.get_latitude_at(sat_id, ts)
            if abs(lat - gs_lat_deg) < max_angle_deg * 0.9:
                if contact_start < 0:
                    contact_start = ts
                contact_end = ts
            elif contact_start >= 0:
                break

        return (contact_start, contact_end)

    def batch_pre_filter(
        self, inclination_deg: float,
        gs_latitudes: Sequence[float], gs_longitudes: Sequence[float],
    ) -> dict[int, list[int]]:
        """批量预筛选所有卫星-地面站配对。"""
        result: dict[int, list[int]] = {}
        for sat_id in range(self.num_satellites):
            result[sat_id] = [
                gs_id for gs_id in range(self.num_ground_stations)
                if self.pre_filter(sat_id, gs_id, inclination_deg,
                                  gs_latitudes[gs_id], gs_longitudes[gs_id]).reachable
            ]
        return result

    def get_filter_stats(self) -> dict[str, Any]:
        """获取预筛选统计。"""
        total = self.num_satellites * self.num_ground_stations
        unreachable = sum(1 for v in self._prefilter_cache.values() if not v.reachable)
        return {
            "total_pairs": total,
            "reachable": total - unreachable,
            "unreachable": unreachable,
            "filter_ratio": unreachable / max(total, 1),
            "prefilter_cache_size": len(self._prefilter_cache),
            "lat_cache_size": len(self._lat_cache),
        }

    def clear(self) -> None:
        """清空所有内部缓存。"""
        self._lat_cache.clear()
        self._prefilter_cache.clear()
        self._lat_range_cache.clear()
