"""
开普勒轨道力学 — 二体问题下的轨道计算

功能:
    - 轨道六要素定义
    - 真近点角随时间演化
    - 星下点位置 (纬度、经度) 计算
    - 支持圆形、椭圆轨道
    - 支持任意轨道倾角（不再硬编码极轨）

轨道模型:
    - 二体近似（仅中心天体引力）
    - 圆形轨道: e=0, 真近点角匀速变化
    - 椭圆轨道: e>0, 通过开普勒方程求解

参考文献:
    - Vallado, "Fundamentals of Astrodynamics and Applications"
    - Bate, Mueller, White, "Fundamentals of Astrodynamics"
"""

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fl_space.environment.celestial_body import CelestialBody


@dataclass
class OrbitalElements:
    """
    经典轨道六要素 (开普勒轨道根数)。

    所有角度单位为度，半长轴单位为 km。

    Attributes
    ----------
    semi_major_axis_km : float
        半长轴 a (km)。对于圆形轨道 a = R_planet + altitude。
    eccentricity : float
        偏心率 e。0 = 圆形, 0 < e < 1 = 椭圆。
    inclination_deg : float
        轨道倾角 i (°)。0 = 赤道轨道, 90 = 极轨道。
    raan_deg : float
        升交点赤经 Ω (°)。
    arg_perigee_deg : float
        近地点幅角 ω (°)。对于圆形轨道通常设为 0。
    true_anomaly_deg : float
        t=0 时的真近点角 ν₀ (°)。
    """

    semi_major_axis_km: float
    eccentricity: float = 0.0
    inclination_deg: float = 90.0
    raan_deg: float = 0.0
    arg_perigee_deg: float = 0.0
    true_anomaly_deg: float = 0.0

    def __post_init__(self):
        # 确保角度在合理范围
        self.inclination_deg = self.inclination_deg % 360.0
        self.raan_deg = self.raan_deg % 360.0
        self.arg_perigee_deg = self.arg_perigee_deg % 360.0
        self.true_anomaly_deg = self.true_anomaly_deg % 360.0

    # ---- 导出属性 (弧度) ----

    @property
    def eccentricity_rad(self) -> float:
        """偏心率（无量纲，非角度。保留 _rad 后缀为接口一致）。"""
        return self.eccentricity

    @property
    def inclination_rad(self) -> float:
        return math.radians(self.inclination_deg)

    @property
    def raan_rad(self) -> float:
        return math.radians(self.raan_deg)

    @property
    def arg_perigee_rad(self) -> float:
        return math.radians(self.arg_perigee_deg)

    @property
    def true_anomaly_rad(self) -> float:
        return math.radians(self.true_anomaly_deg)

    @property
    def is_circular(self) -> bool:
        """是否为圆形轨道。"""
        return self.eccentricity < 1e-9

    def to_dict(self) -> dict:
        return {
            "semi_major_axis_km": self.semi_major_axis_km,
            "eccentricity": self.eccentricity,
            "inclination_deg": self.inclination_deg,
            "raan_deg": self.raan_deg,
            "arg_perigee_deg": self.arg_perigee_deg,
            "true_anomaly_deg": self.true_anomaly_deg,
        }

    def __repr__(self) -> str:
        return (
            f"OrbitalElements(a={self.semi_major_axis_km:.1f}km, "
            f"e={self.eccentricity:.4f}, i={self.inclination_deg:.1f}°, "
            f"Ω={self.raan_deg:.1f}°, ν₀={self.true_anomaly_deg:.1f}°)"
        )


class KeplerOrbit:
    """
    开普勒轨道计算器。

    根据给定的轨道要素和天体参数，计算任意时刻的卫星位置。

    使用示例::

        from fl_space.environment import CelestialBody
        from fl_space.orbit import KeplerOrbit

        earth = CelestialBody.earth()
        oe = OrbitalElements(
            semi_major_axis_km=earth.radius_km + 500,
            eccentricity=0.0,
            inclination_deg=90.0,
            raan_deg=30.0,
        )
        orbit = KeplerOrbit(oe, earth)
        lat, lon = orbit.position_at_time(60.0)  # 1分钟后的位置
    """

    def __init__(self, elements: OrbitalElements, body: "CelestialBody"):
        """
        Parameters
        ----------
        elements : OrbitalElements
            轨道要素。
        body : CelestialBody
            中心天体。
        """
        self.elements = elements
        self.body = body

        # 轨道周期 (分钟)
        self._period_min = (
            2 * math.pi * math.sqrt(elements.semi_major_axis_km**3 / body.GM)
        ) / 60.0

        # 平均运动 (rad/min)
        self._mean_motion = 2 * math.pi / self._period_min

        # 轨道半径 (对圆形轨道 = semi_major_axis)
        self._orbit_radius_km = elements.semi_major_axis_km

    # ---- 属性 ----

    @property
    def period_min(self) -> float:
        """轨道周期 (分钟)。"""
        return self._period_min

    @property
    def mean_motion_rad_per_min(self) -> float:
        """平均运动 (rad/min)。"""
        return self._mean_motion

    @property
    def orbit_radius_km(self) -> float:
        """轨道半径 (km)。"""
        return self._orbit_radius_km

    @property
    def altitude_km(self) -> float:
        """轨道高度 = 轨道半径 - 行星半径。"""
        return self._orbit_radius_km - self.body.radius_km

    # ---- 核心计算 ----

    def true_anomaly_at_time(self, time_min: float) -> float:
        """
        计算 t 时刻的真近点角 (弧度)。

        Parameters
        ----------
        time_min : float
            从 t=0 起的时间 (分钟)。

        Returns
        -------
        float
            真近点角 (弧度)。
        """
        if self.elements.is_circular:
            # 圆形轨道: 真近点角均匀增加
            ta = self.elements.true_anomaly_rad + self._mean_motion * time_min
            return ta % (2 * math.pi)
        else:
            # 椭圆轨道: 开普勒方程
            return self._true_anomaly_elliptical(time_min)

    def _true_anomaly_elliptical(self, time_min: float) -> float:
        """
        椭圆轨道的真近点角求解。

        步骤:
        1. 平近点角 M = M₀ + n·t
        2. 开普勒方程 M = E - e·sin(E) → 牛顿法求解E
        3. 偏近点角E → 真近点角ν
        """
        e = self.elements.eccentricity
        M0 = self._mean_anomaly_from_true(self.elements.true_anomaly_rad, e)
        M = M0 + self._mean_motion * time_min
        M = M % (2 * math.pi)

        # 牛顿法求解开普勒方程
        E = M  # 初始猜测
        for _ in range(20):
            dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
            E += dE
            if abs(dE) < 1e-12:
                break

        # 偏近点角 → 真近点角
        sin_nu = math.sqrt(1 - e**2) * math.sin(E) / (1 - e * math.cos(E))
        cos_nu = (math.cos(E) - e) / (1 - e * math.cos(E))
        nu = math.atan2(sin_nu, cos_nu)
        return nu % (2 * math.pi)

    @staticmethod
    def _mean_anomaly_from_true(true_anomaly_rad: float, e: float) -> float:
        """真近点角 → 平近点角。"""
        cos_nu = math.cos(true_anomaly_rad)
        # 偏近点角
        cos_E = (e + cos_nu) / (1 + e * cos_nu)
        sin_E = math.sqrt(1 - e**2) * math.sin(true_anomaly_rad) / (1 + e * cos_nu)
        E = math.atan2(sin_E, cos_E)
        # 开普勒方程
        M = E - e * math.sin(E)
        return M % (2 * math.pi)

    def position_at_time(self, time_min: float) -> tuple[float, float]:
        """
        计算 t 时刻卫星的星下点位置 (纬度, 经度)。

        算法:
        1. 计算当前真近点角 ν(t)
        2. 由真近点角 → 地心纬度: sin(lat) = sin(i)·sin(ω+ν)
        3. 由 RAAN 和地球自转 → 经度

        Parameters
        ----------
        time_min : float
            从 t=0 起的时间 (分钟)。

        Returns
        -------
        (lat_rad, lon_rad) : Tuple[float, float]
            星下点纬度和经度 (弧度)。
        """
        # 当前真近点角
        nu = self.true_anomaly_at_time(time_min)

        # 纬度角参数 u = ω + ν
        u = self.elements.arg_perigee_rad + nu

        # 纬度: sin(δ) = sin(i)·sin(u)
        sat_lat_rad = math.asin(math.sin(self.elements.inclination_rad) * math.sin(u))

        # 经度 (轨道平面内): λ' = Ω + arctan(cos(i)·tan(u))
        # 处理 tan(u) 的奇点
        cos_i = math.cos(self.elements.inclination_rad)
        if abs(math.cos(u)) < 1e-12:
            lon_in_plane = self.elements.raan_rad + (math.pi / 2) * math.copysign(1, math.sin(u))
        else:
            lon_in_plane = self.elements.raan_rad + math.atan2(cos_i * math.sin(u), math.cos(u))

        # 减去地球自转
        rotation_offset = math.radians(time_min * (360.0 / (self.body.rotation_period_hours * 60)))
        sat_lon_rad = lon_in_plane - rotation_offset

        # 归一化经度
        sat_lon_rad = math.atan2(math.sin(sat_lon_rad), math.cos(sat_lon_rad))

        return sat_lat_rad, sat_lon_rad

    def position_at_time_deg(self, time_min: float) -> tuple[float, float]:
        """
        计算 t 时刻卫星的星下点位置 (纬度°, 经度°)。

        Parameters
        ----------
        time_min : float
            从 t=0 起的时间 (分钟)。

        Returns
        -------
        (lat_deg, lon_deg) : Tuple[float, float]
        """
        lat_rad, lon_rad = self.position_at_time(time_min)
        return math.degrees(lat_rad), math.degrees(lon_rad)

    def ecef_at_time(self, time_min: float) -> tuple[float, float, float]:
        """计算 t 时刻卫星的 ECEF 坐标 (km)。

        基于开普勒轨道的星下点纬度/经度 + 轨道半径转换到 ECEF。
        用于 ISL 星间链路计算。

        Parameters
        ----------
        time_min : float
            从 t=0 起的时间 (分钟)。

        Returns
        -------
        (x_km, y_km, z_km) : Tuple[float, float, float]
            ECEF 坐标 (km)。
        """
        lat_rad, lon_rad = self.position_at_time(time_min)
        r = self._orbit_radius_km
        cos_lat = math.cos(lat_rad)
        x = r * cos_lat * math.cos(lon_rad)
        y = r * cos_lat * math.sin(lon_rad)
        z = r * math.sin(lat_rad)
        return (x, y, z)

    def __repr__(self) -> str:
        return (
            f"KeplerOrbit(a={self._orbit_radius_km:.0f}km, "
            f"e={self.elements.eccentricity:.3f}, "
            f"i={self.elements.inclination_deg:.1f}°, "
            f"T={self._period_min:.1f}min)"
        )


# ---- 工厂函数 ----


def create_circular_orbit(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    true_anomaly_deg: float,
    body: "CelestialBody",
) -> KeplerOrbit:
    """
    创建圆形轨道。

    Parameters
    ----------
    altitude_km : float
        轨道高度 (km)。
    inclination_deg : float
        轨道倾角 (°)。
    raan_deg : float
        升交点赤经 (°)。
    true_anomaly_deg : float
        初始真近点角 (°)。
    body : CelestialBody
        中心天体。

    Returns
    -------
    KeplerOrbit
    """
    oe = OrbitalElements(
        semi_major_axis_km=body.radius_km + altitude_km,
        eccentricity=0.0,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        arg_perigee_deg=0.0,
        true_anomaly_deg=true_anomaly_deg,
    )
    return KeplerOrbit(oe, body)


def create_polar_orbit(
    satellite_id: int,
    num_satellites: int,
    altitude_km: float = 500.0,
    body: Optional["CelestialBody"] = None,
) -> KeplerOrbit:
    """
    创建Walker极轨分布中的一条轨道。

    这是原 orbit_sim_v2.py 中极轨模型的通用版本，
    卫星在升交点赤经和真近点角上均匀分布。

    Parameters
    ----------
    satellite_id : int
        卫星编号 (0-based)。
    num_satellites : int
        总卫星数。
    altitude_km : float
        轨道高度 (km)。
    body : CelestialBody, optional
        中心天体，默认地球。

    Returns
    -------
    KeplerOrbit
    """
    if body is None:
        from fl_space.environment import CelestialBody

        body = CelestialBody.earth()

    raan = satellite_id * (360.0 / num_satellites)
    ta = satellite_id * (360.0 / num_satellites)

    return create_circular_orbit(
        altitude_km=altitude_km,
        inclination_deg=90.0,
        raan_deg=raan,
        true_anomaly_deg=ta,
        body=body,
    )
