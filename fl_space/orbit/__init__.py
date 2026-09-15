"""
轨道力学层 — 轨道计算、卫星相位分布、可见性判断

双后端支持:
    - kepler  (默认): 轻量开普勒力学，无外部依赖
    - skyfield: 高精度 SGP4/JPL 星历，需 pip install skyfield

主要导出:
    KeplerOrbit          — 开普勒轨道计算器 (kepler 后端)
    OrbitalElements      — 轨道六要素
    MultiClusterConfig   — 多星簇星座配置
    ClusterSpec          — 单星簇规格
    SatelliteSpec        — 单星精细配置
    SatelliteRegistry    — 用户自定义卫星注册表
    ConstellationConfig  — 星座配置 (兼容旧接口)
    VisibilityEngine     — 可见性计算引擎
    MultiSatVisibility   — 多卫星批量可见性
    SkyfieldOrbitBackend — Skyfield 高精度后端 (可选)
"""

from .kepler_orbit import (
    KeplerOrbit as KeplerOrbit,
)
from .kepler_orbit import (
    OrbitalElements as OrbitalElements,
)
from .kepler_orbit import (
    create_circular_orbit as create_circular_orbit,
)
from .kepler_orbit import (
    create_polar_orbit as create_polar_orbit,
)
from .satellite_config import (
    ClusterSpec as ClusterSpec,
)
from .satellite_config import (
    MultiClusterConfig as MultiClusterConfig,
)
from .satellite_config import (
    SatelliteSpec as SatelliteSpec,
)
from .satellite_config import (
    orbits_from_legacy_config as orbits_from_legacy_config,
)
from .satellite_phases import (
    ConstellationConfig as ConstellationConfig,
)
from .satellite_phases import (
    generate_cluster_phases as generate_cluster_phases,
)
from .satellite_phases import (
    generate_orbits as generate_orbits,
)
from .satellite_phases import (
    generate_uniform_phases as generate_uniform_phases,
)
from .satellite_phases import (
    generate_walker_phases as generate_walker_phases,
)
from .satellite_registry import (
    SatelliteRegistry as SatelliteRegistry,
)
from .satellite_registry import (
    registry as registry,
)
from .visibility import (
    MultiSatVisibility as MultiSatVisibility,
)
from .visibility import (
    VisibilityEngine as VisibilityEngine,
)

# Skyfield 后端（可选依赖）
try:
    from .skyfield_backend import (
        SKYFIELD_AVAILABLE as SKYFIELD_AVAILABLE,
    )
    from .skyfield_backend import (
        SkyfieldOrbitBackend as SkyfieldOrbitBackend,
    )
    from .skyfield_backend import (
        SkyfieldProvider as SkyfieldProvider,
    )
    from .skyfield_backend import (
        get_precise_body_params as get_precise_body_params,
    )
    from .skyfield_backend import (
        list_supported_bodies as list_supported_bodies,
    )
except ImportError:
    SkyfieldOrbitBackend = None  # type: ignore
    SkyfieldProvider = None
    get_precise_body_params = None
    list_supported_bodies = None
    SKYFIELD_AVAILABLE = False

__all__ = [
    "SKYFIELD_AVAILABLE",
    "ClusterSpec",
    "ConstellationConfig",
    "KeplerOrbit",
    "MultiClusterConfig",
    "MultiSatVisibility",
    "OrbitalElements",
    "SatelliteRegistry",
    "SatelliteSpec",
    "SkyfieldOrbitBackend",
    "SkyfieldProvider",
    "VisibilityEngine",
    "create_circular_orbit",
    "create_polar_orbit",
    "generate_cluster_phases",
    "generate_orbits",
    "generate_uniform_phases",
    "generate_walker_phases",
    "get_precise_body_params",
    "list_supported_bodies",
    "orbits_from_legacy_config",
    "registry",
]
