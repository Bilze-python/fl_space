"""
卫星配置模块 — 多星簇、自定义卫星参数、插件式注册。

支持:
    - 多星簇配置（不同高度、倾角、分布策略）
    - 单星精细配置（自定义轨道六要素）
    - 用户自定义卫星注册表
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from fl_space.environment import CelestialBody

from .kepler_orbit import KeplerOrbit, OrbitalElements
from .satellite_phases import (
    ConstellationConfig,
    generate_cluster_phases,
    generate_uniform_phases,
    generate_walker_phases,
)

if TYPE_CHECKING:
    pass

DistributionType = Literal["walker", "cluster", "uniform", "custom"]


# ============================================================
# 单星精细配置
# ============================================================


@dataclass
class SatelliteSpec:
    """单个卫星的精细配置。

    支持两种创建方式:
    1. 圆形轨道: 指定 altitude_km + inclination_deg + raan_deg + true_anomaly_deg
    2. 通用轨道: 指定所有六要素（semi_major_axis_km, eccentricity, ...）

    Attributes
    ----------
    name : str
        卫星名称。
    altitude_km : float
        轨道高度 (km)。圆形轨道时使用。
    inclination_deg : float
        轨道倾角 (度)。
    raan_deg : float
        升交点赤经 (度)。
    true_anomaly_deg : float
        真近点角 (度)。
    semi_major_axis_km : float, optional
        半长轴 (km)。指定后 altitude_km 被忽略。
    eccentricity : float
        偏心率。0=圆形。
    arg_perigee_deg : float
        近地点幅角 (度)。
    comm_range_km : float
        通信距离上限 (km), 0 = 无限制。
    tx_power_dbm : float
        发射功率 (dBm)。
    """

    name: str = "SAT"
    altitude_km: float = 500.0
    inclination_deg: float = 90.0
    raan_deg: float = 0.0
    true_anomaly_deg: float = 0.0
    semi_major_axis_km: Optional[float] = None
    eccentricity: float = 0.0
    arg_perigee_deg: float = 0.0
    comm_range_km: float = 0.0
    tx_power_dbm: float = 30.0

    def to_orbit(self, body: CelestialBody) -> KeplerOrbit:
        """转换为 KeplerOrbit 对象。"""
        if self.semi_major_axis_km is not None:
            a = self.semi_major_axis_km
        else:
            a = body.radius_km + self.altitude_km

        oe = OrbitalElements(
            semi_major_axis_km=a,
            eccentricity=self.eccentricity,
            inclination_deg=self.inclination_deg,
            raan_deg=self.raan_deg,
            arg_perigee_deg=self.arg_perigee_deg,
            true_anomaly_deg=self.true_anomaly_deg,
        )
        return KeplerOrbit(oe, body)


# ============================================================
# 星簇规格
# ============================================================


@dataclass
class ClusterSpec:
    """单个星簇的规格。

    一个星簇是一组共享相同轨道高度和倾角的卫星，
    在 RAAN 和真近点角上按分布策略排列。

    Attributes
    ----------
    name : str
        星簇名称 (如 "polar_cluster", "equatorial_cluster")。
    num_satellites : int
        簇内卫星数。
    altitude_km : float
        轨道高度 (km)。
    inclination_deg : float
        轨道倾角 (度)。
    distribution : str
        簇内分布策略: "walker", "cluster", "uniform"。
    num_planes : int
        Walker 分布的轨道面数 (仅 walker 模式)。
    phasing_factor : int
        Walker 相位因子 (仅 walker 模式)。
    raan_offset_deg : float
        整个星簇的 RAAN 偏移量 (度), 用于多簇间错开。
    """

    name: str = "cluster"
    num_satellites: int = 3
    altitude_km: float = 500.0
    inclination_deg: float = 90.0
    distribution: DistributionType = "uniform"
    num_planes: int = 1
    phasing_factor: int = 0
    raan_offset_deg: float = 0.0

    def __post_init__(self):
        if self.num_planes > self.num_satellites:
            self.num_planes = self.num_satellites

    def generate_orbits(self, body: CelestialBody) -> list[KeplerOrbit]:
        """为此星簇生成所有卫星轨道。"""
        config = ConstellationConfig(
            num_satellites=self.num_satellites,
            num_planes=self.num_planes,
            inclination_deg=self.inclination_deg,
            altitude_km=self.altitude_km,
            distribution=self.distribution,
            phasing_factor=self.phasing_factor,
        )
        orbits = _generate_with_distribution(config, body)

        # 应用 RAAN 偏移
        if self.raan_offset_deg != 0.0:
            for orb in orbits:
                orb.elements.raan_deg = (orb.elements.raan_deg + self.raan_offset_deg) % 360.0

        return orbits


def _generate_with_distribution(
    config: ConstellationConfig, body: CelestialBody
) -> list[KeplerOrbit]:
    """根据分布策略生成轨道（内部辅助）。"""
    if config.distribution == "walker":
        return generate_walker_phases(
            num_satellites=config.num_satellites,
            num_planes=config.num_planes,
            phasing_factor=config.phasing_factor,
            inclination_deg=config.inclination_deg,
            altitude_km=config.altitude_km,
            body=body,
        )
    elif config.distribution == "cluster":
        return generate_cluster_phases(
            num_satellites=config.num_satellites,
            num_clusters=max(1, config.num_planes),
            inclination_deg=config.inclination_deg,
            altitude_km=config.altitude_km,
            body=body,
        )
    else:  # uniform
        return generate_uniform_phases(
            num_satellites=config.num_satellites,
            inclination_deg=config.inclination_deg,
            altitude_km=config.altitude_km,
            body=body,
        )


# ============================================================
# 多星簇配置（用户主接口）
# ============================================================


@dataclass
class MultiClusterConfig:
    """多星簇星座配置。

    用户可以定义多个星簇，每个簇有不同的高度、倾角、卫星数。
    系统自动为所有星簇生成卫星轨道并统一编号。

    Examples
    --------
    >>> config = MultiClusterConfig(
    ...     clusters=[
    ...         ClusterSpec("polar", num_satellites=4, altitude_km=500, inclination_deg=90),
    ...         ClusterSpec("equatorial", num_satellites=3, altitude_km=600, inclination_deg=0),
    ...     ]
    ... )
    >>> orbits = config.generate_orbits(CelestialBody.earth())
    >>> len(orbits)
    7
    """

    clusters: list[ClusterSpec] = field(default_factory=list)
    custom_satellites: list[SatelliteSpec] = field(default_factory=list)

    @property
    def total_satellites(self) -> int:
        """总卫星数。"""
        return sum(c.num_satellites for c in self.clusters) + len(self.custom_satellites)

    def generate_orbits(
        self, body: Optional[CelestialBody] = None
    ) -> tuple[list[KeplerOrbit], dict[str, list[int]]]:
        """生成所有卫星轨道，返回 (轨道列表, {簇名: [卫星索引...]})。

        Parameters
        ----------
        body : CelestialBody, optional
            中心天体，默认地球。

        Returns
        -------
        orbits : list of KeplerOrbit
            所有卫星轨道（按簇排序）。
        cluster_map : dict
            {簇名: [该簇卫星的全局索引列表]}。
        """
        if body is None:
            body = CelestialBody.earth()

        all_orbits: list[KeplerOrbit] = []
        cluster_map: dict[str, list[int]] = {}

        for cluster in self.clusters:
            start_idx = len(all_orbits)
            cluster_orbits = cluster.generate_orbits(body)
            all_orbits.extend(cluster_orbits)
            cluster_map[cluster.name] = list(range(start_idx, len(all_orbits)))

        # 自定义卫星
        if self.custom_satellites:
            custom_start = len(all_orbits)
            for spec in self.custom_satellites:
                all_orbits.append(spec.to_orbit(body))
            cluster_map["_custom"] = list(range(custom_start, len(all_orbits)))

        return all_orbits, cluster_map

    def to_dict(self) -> dict:
        """导出为可序列化的字典。"""
        return {
            "clusters": [
                {
                    "name": c.name,
                    "num_satellites": c.num_satellites,
                    "altitude_km": c.altitude_km,
                    "inclination_deg": c.inclination_deg,
                    "distribution": c.distribution,
                    "num_planes": c.num_planes,
                    "phasing_factor": c.phasing_factor,
                    "raan_offset_deg": c.raan_offset_deg,
                }
                for c in self.clusters
            ],
            "custom_satellites": [
                {
                    "name": s.name,
                    "altitude_km": s.altitude_km,
                    "inclination_deg": s.inclination_deg,
                    "raan_deg": s.raan_deg,
                    "true_anomaly_deg": s.true_anomaly_deg,
                }
                for s in self.custom_satellites
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> MultiClusterConfig:
        """从字典创建。"""
        clusters = [ClusterSpec(**c) for c in data.get("clusters", [])]
        customs = [SatelliteSpec(**s) for s in data.get("custom_satellites", [])]
        return cls(clusters=clusters, custom_satellites=customs)

    # ---- 预设工厂 ----

    @classmethod
    def polar_only(cls, num_sats: int = 6, altitude_km: float = 500.0) -> MultiClusterConfig:
        """纯极轨星座。"""
        return cls(clusters=[
            ClusterSpec("polar", num_satellites=num_sats, altitude_km=altitude_km,
                        inclination_deg=90, distribution="uniform"),
        ])

    @classmethod
    def starlink_like(cls, num_sats: int = 20, altitude_km: float = 550.0) -> MultiClusterConfig:
        """Starlink 风格：53度倾角 Walker 分布。"""
        return cls(clusters=[
            ClusterSpec("starlink_shell", num_satellites=num_sats, altitude_km=altitude_km,
                        inclination_deg=53, distribution="walker",
                        num_planes=max(1, num_sats // 4)),
        ])

    @classmethod
    def mixed_orbit(
        cls,
        num_polar: int = 4,
        num_mid: int = 4,
        num_eq: int = 2,
    ) -> MultiClusterConfig:
        """混合轨道星座：极轨 + 中倾角 + 赤道。"""
        return cls(clusters=[
            ClusterSpec("polar", num_satellites=num_polar, altitude_km=500,
                        inclination_deg=90, distribution="uniform", raan_offset_deg=0),
            ClusterSpec("mid_lat", num_satellites=num_mid, altitude_km=550,
                        inclination_deg=53, distribution="walker",
                        num_planes=2, raan_offset_deg=15),
            ClusterSpec("equatorial", num_satellites=num_eq, altitude_km=600,
                        inclination_deg=0, distribution="uniform", raan_offset_deg=30),
        ])

    @classmethod
    def demo_default(cls) -> MultiClusterConfig:
        """演示用默认配置：极轨 + 中倾角双星簇。"""
        return cls(clusters=[
            ClusterSpec("polar", num_satellites=4, altitude_km=500,
                        inclination_deg=90, distribution="uniform"),
            ClusterSpec("leo_shell", num_satellites=6, altitude_km=550,
                        inclination_deg=53, distribution="walker",
                        num_planes=3, raan_offset_deg=20),
        ])


# ============================================================
# 兼容旧接口
# ============================================================


def orbits_from_legacy_config(
    config: ConstellationConfig,
    body: Optional[CelestialBody] = None,
) -> tuple[list[KeplerOrbit], dict[str, list[int]]]:
    """从旧版 ConstellationConfig 创建轨道（兼容过渡）。

    返回 (orbits, {"default": indices})。
    """
    if body is None:
        body = CelestialBody.earth()

    cluster = ClusterSpec(
        name="default",
        num_satellites=config.num_satellites,
        altitude_km=config.altitude_km,
        inclination_deg=config.inclination_deg,
        distribution=config.distribution,
        num_planes=config.num_planes,
        phasing_factor=config.phasing_factor,
    )
    mc = MultiClusterConfig(clusters=[cluster])
    return mc.generate_orbits(body)
