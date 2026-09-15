"""
环境模拟层 — 天体定义、大气模型、地面站管理、坐标工具

主要导出:
    CelestialBody      — 可配置的行星/天体参数
    AtmosphereModel     — 大气模型（无/恒定/指数衰减）
    GroundStation       — 单个地面站
    GroundStationNetwork — 地面站网络管理
    coordinate_utils    — 球面几何工具函数
"""

from .atmosphere import (
    AtmosphereModel as AtmosphereModel,
)
from .atmosphere import (
    ConstantHeightAtmosphere as ConstantHeightAtmosphere,
)
from .atmosphere import (
    ExponentialAtmosphere as ExponentialAtmosphere,
)
from .atmosphere import (
    NoAtmosphere as NoAtmosphere,
)
from .atmosphere import (
    create_atmosphere_for_body as create_atmosphere_for_body,
)
from .celestial_body import CelestialBody as CelestialBody
from .ground_station import (
    EXTENDED_GROUND_STATIONS as EXTENDED_GROUND_STATIONS,
)
from .ground_station import (
    PAPER_GROUND_STATIONS as PAPER_GROUND_STATIONS,
)
from .ground_station import (
    GroundStation as GroundStation,
)
from .ground_station import (
    GroundStationNetwork as GroundStationNetwork,
)
from .ground_station import (
    create_default_network as create_default_network,
)
from .ground_station import (
    create_extended_network as create_extended_network,
)

__all__ = [
    "EXTENDED_GROUND_STATIONS",
    "PAPER_GROUND_STATIONS",
    "AtmosphereModel",
    "CelestialBody",
    "ConstantHeightAtmosphere",
    "ExponentialAtmosphere",
    "GroundStation",
    "GroundStationNetwork",
    "NoAtmosphere",
    "create_atmosphere_for_body",
    "create_default_network",
    "create_extended_network",
]
