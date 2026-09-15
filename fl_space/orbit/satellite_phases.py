"""
卫星相位/星座分布模块 — 灵活的星座设计

支持 Walker 星座、星簇分布、自定义分布。
"""

from dataclasses import dataclass
from typing import Literal, Optional

from fl_space.environment import CelestialBody

from .kepler_orbit import KeplerOrbit, create_circular_orbit

DistributionType = Literal["walker", "cluster", "custom", "uniform"]


@dataclass
class ConstellationConfig:
    """星座配置描述（纯数据类）。"""
    num_satellites: int = 3
    num_planes: int = 1
    inclination_deg: float = 90.0
    altitude_km: float = 500.0
    distribution: DistributionType = "walker"
    phasing_factor: int = 0

    def __post_init__(self):
        if self.num_planes > self.num_satellites:
            raise ValueError(f"num_planes({self.num_planes}) > num_satellites({self.num_satellites})")


def generate_walker_phases(
    num_satellites: int,
    num_planes: int = 1,
    phasing_factor: int = 0,
    inclination_deg: float = 90.0,
    altitude_km: float = 500.0,
    body: Optional[CelestialBody] = None,
) -> list[KeplerOrbit]:
    """
    Walker T/P/F 星座分布。

    卫星在轨道面上均匀分布，轨道面间的相位差 = F·360°/T。
    """
    if num_planes > num_satellites:
        raise ValueError(f"num_planes({num_planes}) > num_satellites({num_satellites})")

    if body is None:
        body = CelestialBody.earth()

    sats_per_plane = num_satellites // num_planes
    remainder = num_satellites % num_planes

    orbits = []
    for plane in range(num_planes):
        n_in_plane = sats_per_plane + (1 if plane < remainder else 0)
        raan = plane * 360.0 / num_planes
        for j in range(n_in_plane):
            ta = j * 360.0 / max(n_in_plane, 1) + phasing_factor * plane * 360.0 / num_satellites
            orbits.append(create_circular_orbit(
                altitude_km=altitude_km, inclination_deg=inclination_deg,
                raan_deg=raan, true_anomaly_deg=ta, body=body,
            ))
    return orbits


def generate_cluster_phases(
    num_satellites: int,
    num_clusters: int = 1,
    inclination_deg: float = 90.0,
    altitude_km: float = 500.0,
    body: Optional[CelestialBody] = None,
) -> list[KeplerOrbit]:
    """
    星簇分布：RAAN 按簇分组，真近点角在簇内错开。

    原 orbit_sim_v2.py 的大规模星座分布策略。
    """
    if body is None:
        body = CelestialBody.earth()

    if num_clusters == 1:
        # 退化为均匀分布
        sats_per_cluster = num_satellites
    else:
        sats_per_cluster = num_satellites // num_clusters

    orbits = []
    for i in range(num_satellites):
        cluster = i % num_clusters
        raan = cluster * (360.0 / num_clusters)
        ta = (i // num_clusters) * (360.0 / (sats_per_cluster + 1))
        orbits.append(create_circular_orbit(
            altitude_km=altitude_km, inclination_deg=inclination_deg,
            raan_deg=raan, true_anomaly_deg=ta, body=body,
        ))
    return orbits


def generate_uniform_phases(
    num_satellites: int,
    inclination_deg: float = 90.0,
    altitude_km: float = 500.0,
    body: Optional[CelestialBody] = None,
) -> list[KeplerOrbit]:
    """均匀分布：RAAN 和 TA 均等间隔分布在 [0, 360°] 范围内。"""
    if body is None:
        body = CelestialBody.earth()
    orbits = []
    for i in range(num_satellites):
        raan = i * (360.0 / num_satellites)
        ta = i * (360.0 / num_satellites)
        orbits.append(create_circular_orbit(
            altitude_km=altitude_km, inclination_deg=inclination_deg,
            raan_deg=raan, true_anomaly_deg=ta, body=body,
        ))
    return orbits


def generate_orbits(
    config: ConstellationConfig,
    body: Optional[CelestialBody] = None,
) -> list[KeplerOrbit]:
    """
    根据配置生成所有卫星轨道。

    Parameters
    ----------
    config : ConstellationConfig
        星座配置。
    body : CelestialBody, optional
        中心天体。

    Returns
    -------
    list of KeplerOrbit
    """
    if body is None:
        body = CelestialBody.earth()

    strategy = config.distribution
    if strategy == "walker":
        return generate_walker_phases(
            num_satellites=config.num_satellites,
            num_planes=config.num_planes,
            phasing_factor=config.phasing_factor,
            inclination_deg=config.inclination_deg,
            altitude_km=config.altitude_km,
            body=body,
        )
    elif strategy == "cluster":
        return generate_cluster_phases(
            num_satellites=config.num_satellites,
            num_clusters=config.num_planes,  # 复用 num_planes 作为星簇数
            inclination_deg=config.inclination_deg,
            altitude_km=config.altitude_km,
            body=body,
        )
    elif strategy in ("uniform", "custom"):
        return generate_uniform_phases(
            num_satellites=config.num_satellites,
            inclination_deg=config.inclination_deg,
            altitude_km=config.altitude_km,
            body=body,
        )
    else:
        raise ValueError(f"Unknown distribution strategy: {strategy}")
