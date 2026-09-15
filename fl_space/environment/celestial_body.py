"""
天体定义模块 — 可配置的行星/天体参数

支持:
    - 内置预设 (地球、火星、月球)
    - 自定义天体参数
    - 天体参数查询接口

使用示例::

    from fl_space.environment import CelestialBody

    earth = CelestialBody.earth()
    mars = CelestialBody.mars()
    custom = CelestialBody(
        name="Kepler-22b",
        radius_km=16100,
        GM=4.8e6,
        rotation_period_hours=15.2,
    )
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CelestialBody:
    """
    天体参数定义。

    所有长度单位为 km，时间单位为小时，GM 单位为 km³/s²。

    Attributes
    ----------
    name : str
        天体名称。
    radius_km : float
        天体赤道半径 (km)。
    GM : float
        引力常数 × 天体质量 (km³/s²)。
    rotation_period_hours : float
        自转周期 (小时)。正值为西向东自转（与地球相同）。
    flattening : float
        扁率 f = (a-b)/a，球体为 0。
    atmosphere_height_km : float
        有效大气层高度 (km)，用于计算可见半角。
    surface_gravity_ms2 : float
        表面重力加速度 (m/s²)。
    """
    name: str
    radius_km: float
    GM: float
    rotation_period_hours: float = 24.0
    flattening: float = 0.0
    atmosphere_height_km: float = 0.0
    surface_gravity_ms2: Optional[float] = None

    def __post_init__(self):
        if self.surface_gravity_ms2 is None:
            # g = GM / R² (转换为 m/s²)
            self.surface_gravity_ms2 = (
                self.GM / (self.radius_km ** 2)
            ) * 1000.0

    @property
    def rotation_rate_deg_per_min(self) -> float:
        """自转角速度 (°/min)。"""
        return 360.0 / (self.rotation_period_hours * 60.0)

    @property
    def rotation_rate_rad_per_min(self) -> float:
        """自转角速度 (rad/min)。"""
        import math
        return math.radians(self.rotation_rate_deg_per_min)

    @property
    def orbit_visible_horizon_angle_rad(self) -> float:
        """
        从轨道高度看地平线的有效可见半角 (弧度)。

        考虑大气层高度后，可见范围会略微增大。
        """
        import math
        effective_radius = self.radius_km + self.atmosphere_height_km
        return math.acos(self.radius_km / effective_radius)

    def effective_horizon_angle_for_altitude(self, altitude_km: float) -> float:
        """
        计算给定轨道高度处的地平可见半角。

        Parameters
        ----------
        altitude_km : float
            轨道高度 (km)。

        Returns
        -------
        float
            可见半角 (弧度)。
        """
        import math
        orbit_radius = self.radius_km + altitude_km
        effective_radius = self.radius_km + self.atmosphere_height_km
        return math.acos(effective_radius / orbit_radius)

    def __repr__(self) -> str:
        return (
            f"CelestialBody(name='{self.name}', R={self.radius_km}km, "
            f"GM={self.GM:.1f}, T_rot={self.rotation_period_hours}h)"
        )

    # ---- 预设天体工厂方法 ----

    @classmethod
    def earth(cls, precise: bool = False) -> "CelestialBody":
        """地球（默认天体）。

        Parameters
        ----------
        precise : bool
            如果 True，尝试从 Skyfield/JPL 获取更高精度参数。
        """
        params = {
            "name": "Earth",
            "radius_km": 6371.0,
            "GM": 398600.4418,
            "rotation_period_hours": 24.0,
            "flattening": 1.0 / 298.257,
            "atmosphere_height_km": 100.0,
            "surface_gravity_ms2": 9.80665,
        }
        if precise:
            try:
                from fl_space.orbit.skyfield_backend import get_precise_body_params
                jpl = get_precise_body_params('earth')
                if jpl:
                    params.update({
                        k: v for k, v in jpl.items()
                        if k in cls.__dataclass_fields__
                    })
            except Exception:
                pass
        return cls(**params)

    @classmethod
    def mars(cls, precise: bool = False) -> "CelestialBody":
        """火星。"""
        params = {
            "name": "Mars",
            "radius_km": 3389.5,
            "GM": 42828.3,
            "rotation_period_hours": 24.6597,
            "flattening": 1.0 / 169.8,
            "atmosphere_height_km": 80.0,
            "surface_gravity_ms2": 3.72076,
        }
        if precise:
            try:
                from fl_space.orbit.skyfield_backend import get_precise_body_params
                jpl = get_precise_body_params('mars')
                if jpl:
                    params.update({
                        k: v for k, v in jpl.items()
                        if k in cls.__dataclass_fields__
                    })
            except Exception:
                pass
        return cls(**params)

    @classmethod
    def from_name(
        cls, name: str, precise: bool = False
    ) -> Optional["CelestialBody"]:
        """
        按名称创建天体，自动选择最精确的可用参数。

        支持: earth, mars, moon, jupiter, saturn, venus
        当 precise=True 且 Skyfield 可用时，使用 JPL 星历参数。

        Parameters
        ----------
        name : str
            天体名称（不区分大小写）。
        precise : bool
            是否使用 JPL 高精度参数。

        Returns
        -------
        CelestialBody 或 None
        """
        name_lower = name.lower()
        builtin = {
            'earth': cls.earth,
            'mars': cls.mars,
            'moon': cls.moon,
            'jupiter': cls.jupiter,
            'saturn': cls.saturn,
            'venus': cls.venus,
        }
        factory = builtin.get(name_lower)
        if factory:
            return factory(precise=precise)
        return None

    @classmethod
    def moon(cls, precise: bool = False) -> "CelestialBody":
        """月球。"""
        params = {
            "name": "Moon", "radius_km": 1737.4, "GM": 4902.8,
            "rotation_period_hours": 27.32 * 24, "flattening": 0.0,
            "atmosphere_height_km": 0.0, "surface_gravity_ms2": 1.62,
        }
        if precise:
            try:
                from fl_space.orbit.skyfield_backend import get_precise_body_params
                jpl = get_precise_body_params('moon')
                if jpl:
                    params.update({k: v for k, v in jpl.items()
                                   if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls(**params)

    @classmethod
    def jupiter(cls, precise: bool = False) -> "CelestialBody":
        """木星。"""
        params = {
            "name": "Jupiter", "radius_km": 69911.0, "GM": 126686534.0,
            "rotation_period_hours": 9.925, "flattening": 0.06487,
            "atmosphere_height_km": 5000.0,
        }
        if precise:
            try:
                from fl_space.orbit.skyfield_backend import get_precise_body_params
                jpl = get_precise_body_params('jupiter')
                if jpl:
                    params.update({k: v for k, v in jpl.items()
                                   if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls(**params)

    @classmethod
    def saturn(cls, precise: bool = False) -> "CelestialBody":
        """土星。"""
        params = {
            "name": "Saturn", "radius_km": 58232.0, "GM": 37931187.0,
            "rotation_period_hours": 10.656, "flattening": 0.09796,
            "atmosphere_height_km": 3000.0,
        }
        if precise:
            try:
                from fl_space.orbit.skyfield_backend import get_precise_body_params
                jpl = get_precise_body_params('saturn')
                if jpl:
                    params.update({k: v for k, v in jpl.items()
                                   if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls(**params)

    @classmethod
    def venus(cls, precise: bool = False) -> "CelestialBody":
        """金星。"""
        params = {
            "name": "Venus", "radius_km": 6051.8, "GM": 324859.0,
            "rotation_period_hours": -5832.5, "flattening": 0.0,
            "atmosphere_height_km": 250.0,
        }
        if precise:
            try:
                from fl_space.orbit.skyfield_backend import get_precise_body_params
                jpl = get_precise_body_params('venus')
                if jpl:
                    params.update({k: v for k, v in jpl.items()
                                   if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls(**params)

    def to_dict(self) -> dict:
        """导出为字典（便于序列化和配置存储）。"""
        return {
            "name": self.name,
            "radius_km": self.radius_km,
            "GM": self.GM,
            "rotation_period_hours": self.rotation_period_hours,
            "flattening": self.flattening,
            "atmosphere_height_km": self.atmosphere_height_km,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CelestialBody":
        """从字典创建天体。"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
