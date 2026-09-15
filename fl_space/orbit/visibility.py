"""
可见性计算模块 — 卫星-地面站可见性判断

支持:
    - 视线判断（直线几何遮挡）
    - 多地面站同时可见检测
    - 最小仰角约束
    - 向量化批量计算

未来扩展: 大气衰减、多普勒频移、链路预算
"""

from typing import Optional

import numpy as np

from fl_space.environment import CelestialBody, GroundStationNetwork
from fl_space.environment.coordinate_utils import (
    batch_angular_distances_rad,
    horizon_angle_rad,
)

from .kepler_orbit import KeplerOrbit


class VisibilityEngine:
    """
    可见性计算引擎。

    根据卫星轨道和地面站位置，判断每颗卫星在给定时刻
    可以看到哪些地面站。

    使用示例::

        from fl_space.environment import CelestialBody, GroundStationNetwork
        from fl_space.orbit import KeplerOrbit, VisibilityEngine

        earth = CelestialBody.earth()
        network = GroundStationNetwork.from_tuples([
            ("Sioux Falls", 43.55, -96.72),
            ("Sanya", 18.25, 109.5),
        ])
        orbit = create_circular_orbit(500, 90, 0, 0, earth)
        engine = VisibilityEngine(earth, orbit, network)

        visible = engine.visible_stations_at_time(60.0)
        print(visible)  # [0, 1] 或 []
    """

    def __init__(
        self,
        body: CelestialBody,
        orbit: KeplerOrbit,
        network: GroundStationNetwork,
    ):
        """
        Parameters
        ----------
        body : CelestialBody
            中心天体。
        orbit : KeplerOrbit
            卫星轨道。
        network : GroundStationNetwork
            地面站网络。
        """
        self.body = body
        self.orbit = orbit
        self.network = network

        # 预计算地表坐标数组 (N, 2) 用于批量运算
        self._gs_coords_rad = np.array([
            [gs.lat_rad, gs.lon_rad] for gs in network
        ]) if network.count > 0 else np.zeros((0, 2))

        # 最小仰角数组
        self._gs_min_elev_rad = np.array([
            gs.min_elevation_rad for gs in network
        ]) if network.count > 0 else np.zeros(0)

        # 地平可见半角
        self._horizon_angle = horizon_angle_rad(
            body.radius_km, orbit.orbit_radius_km, body.atmosphere_height_km,
        )

    @property
    def horizon_angle_rad(self) -> float:
        """地平可见半角 (弧度)。"""
        return self._horizon_angle

    def visible_stations_at_time(
        self, time_min: float
    ) -> list[int]:
        """
        返回 t 时刻卫星可见的所有地面站 ID 列表。

        Parameters
        ----------
        time_min : float
            从 t=0 起的时间 (分钟)。

        Returns
        -------
        List[int]
            可见地面站 ID 列表（按索引顺序）。
        """
        if self.network.count == 0:
            return []

        sat_lat, sat_lon = self.orbit.position_at_time(time_min)
        ang_dists = batch_angular_distances_rad(
            sat_lat, sat_lon, self._gs_coords_rad,
        )

        # 可见条件: 角距 < 地平半角 - 最小仰角
        thresholds = self._horizon_angle - self._gs_min_elev_rad
        visible_mask = ang_dists < thresholds

        return np.where(visible_mask)[0].tolist()

    def first_visible_station_at_time(
        self, time_min: float
    ) -> Optional[int]:
        """
        返回 t 时刻第一个可见的地面站 ID。

        与原 orbit_sim_v2.py 的接触矩阵行为兼容。

        Parameters
        ----------
        time_min : float
            从 t=0 起的时间 (分钟)。

        Returns
        -------
        Optional[int]
            第一个可见地面站 ID，或 None。
        """
        visible = self.visible_stations_at_time(time_min)
        return visible[0] if visible else None

    def all_visible_at_time(
        self, time_min: float
    ) -> list[int]:
        """同 visible_stations_at_time。"""
        return self.visible_stations_at_time(time_min)


class MultiSatVisibility:
    """
    多卫星可见性批量计算器。

    管理多颗卫星对同一地面站网络的可见性判断。
    支持向量化批量计算以提高性能。
    """

    def __init__(
        self,
        body: CelestialBody,
        orbits: list[KeplerOrbit],
        network: GroundStationNetwork,
    ):
        self.body = body
        self.orbits = orbits
        self.network = network

        # 为每颗卫星创建 VisibilityEngine
        self._engines = [
            VisibilityEngine(body, orb, network) for orb in orbits
        ]

    def visible_matrix_at_time(
        self, time_min: float
    ) -> list[list[int]]:
        """
        返回 t 时刻所有卫星的可见地面站矩阵。

        Returns
        -------
        List[List[int]]
            每颗卫星的可见地面站 ID 列表。
        """
        return [eng.visible_stations_at_time(time_min) for eng in self._engines]

    def first_contact_matrix_at_time(
        self, time_min: float
    ) -> list[Optional[int]]:
        """
        返回 t 时刻每颗卫星的第一个可见地面站 ID。

        Returns
        -------
        List[Optional[int]]
            [gs_id or None, ...]，长度 = 卫星数。
        """
        return [eng.first_visible_station_at_time(time_min) for eng in self._engines]
