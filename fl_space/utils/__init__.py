"""SpaceFL 工具模块 — 可视化、坐标转换和实验辅助。"""

from fl_space.utils.coordinate_utils import (
    DEFAULT_MIN_ELEVATION_DEG,
    EARTH_ANGULAR_VELOCITY_RAD_S,
    WGS84_A_KM,
    WGS84_E2,
    WGS84_F,
    eci_to_ecef,
    elevation_attenuation_factor,
    elevation_azimuth_deg,
    enu_from_ecef_delta,
    geodetic_to_ecef,
    gmst_from_time,
    occlude_by_table,
    simplified_elevation_far,
    spherical_visibility_coarse,
)
from fl_space.utils.precomputation_cache import (
    CachedWindow,
    HotColdWindowCache,
    StaticCachedParams,
    load_cache_from_json,
    precompute_all_gs_params,
    precompute_gs_static_params,
    save_cache_to_json,
)
from fl_space.utils.viz import (
    get_contact_statistics,
    plot_accuracy_comparison,
    plot_contact_heatmap,
    plot_ground_station_map,
    plot_time_breakdown,
    save_experiment_report,
)

__all__ = [
    "DEFAULT_MIN_ELEVATION_DEG",
    "EARTH_ANGULAR_VELOCITY_RAD_S",
    "WGS84_A_KM",
    "WGS84_E2",
    "WGS84_F",
    "CachedWindow",
    "HotColdWindowCache",
    "IncrementalWindowCache",
    "MatchScoreCache",
    "MultiplexCacheStore",
    "StaticCachedParams",
    "eci_to_ecef",
    "elevation_attenuation_factor",
    "elevation_azimuth_deg",
    "enu_from_ecef_delta",
    "geodetic_to_ecef",
    "get_contact_statistics",
    "gmst_from_time",
    "load_cache_from_json",
    "occlude_by_table",
    "plot_accuracy_comparison",
    "plot_contact_heatmap",
    "plot_ground_station_map",
    "plot_time_breakdown",
    "precompute_all_gs_params",
    "precompute_gs_static_params",
    "save_cache_to_json",
    "save_experiment_report",
    "simplified_elevation_far",
    "spherical_visibility_coarse",
]

# ── D4: IO/内存管理优化 ─────────────────────────────────────────
from fl_space.utils.io_optimizer import (
    ColumnarWindowSchema as ColumnarWindowSchema,
)
from fl_space.utils.io_optimizer import (
    ColumnarWindowStore as ColumnarWindowStore,
)
from fl_space.utils.io_optimizer import (
    PositionMemoryPool as PositionMemoryPool,
)
from fl_space.utils.io_optimizer import (
    TleIncrementalLoader as TleIncrementalLoader,
)
from fl_space.utils.io_optimizer import (
    TleRecord as TleRecord,
)

__all__ += [
    "ColumnarWindowSchema",
    "ColumnarWindowStore",
    "PositionMemoryPool",
    "TleIncrementalLoader",
    "TleRecord",
]
