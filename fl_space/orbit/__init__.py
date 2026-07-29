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
from .optimizer import (
    ErrorAdaptiveRefresher as ErrorAdaptiveRefresher,
)

# 论文维度二：轨道推演优化
from .optimizer import (
    ErrorCorrectionTracker as ErrorCorrectionTracker,
)
from .optimizer import (
    FarWindowDiscarder as FarWindowDiscarder,
)
from .optimizer import (
    OrbitCacheManager as OrbitCacheManager,
)
from .optimizer import (
    adaptive_sample_step as adaptive_sample_step,
)
from .optimizer import (
    compute_contacts_batch as compute_contacts_batch,
)
from .optimizer import (
    generate_adaptive_timeline as generate_adaptive_timeline,
)
from .optimizer import (
    propagate_sat_ecef_batch as propagate_sat_ecef_batch,
)
from .propagation_optimizer import (
    ChebyshevCoefficients as ChebyshevCoefficients,
)
from .propagation_optimizer import (
    GmstLookupTable as GmstLookupTable,
)
from .propagation_optimizer import (
    PolynomialPropagator as PolynomialPropagator,
)
from .propagation_optimizer import (
    build_gmst_table as build_gmst_table,
)
from .propagation_optimizer import (
    eci_to_ecef_lookup as eci_to_ecef_lookup,
)
from .propagation_optimizer import (
    evaluate_chebyshev as evaluate_chebyshev,
)
from .propagation_optimizer import (
    fit_chebyshev_position as fit_chebyshev_position,
)
from .propagation_optimizer import (
    solve_kepler_aitken as solve_kepler_aitken,
)
from .propagation_optimizer import (
    solve_kepler_pade as solve_kepler_pade,
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
    "ChebyshevCoefficients",
    "ClusterSpec",
    "ConstellationConfig",
    "ErrorAdaptiveRefresher",
    "ErrorCorrectionTracker",
    "FarWindowDiscarder",
    "GmstLookupTable",
    "KeplerOrbit",
    "MultiClusterConfig",
    "MultiSatVisibility",
    "OrbitCacheManager",
    "OrbitalElements",
    "PolynomialPropagator",
    "SatelliteRegistry",
    "SatelliteSpec",
    "SkyfieldOrbitBackend",
    "SkyfieldProvider",
    "VisibilityEngine",
    "adaptive_sample_step",
    "build_gmst_table",
    "compute_contacts_batch",
    "create_circular_orbit",
    "create_polar_orbit",
    "eci_to_ecef_lookup",
    "evaluate_chebyshev",
    "fit_chebyshev_position",
    "generate_adaptive_timeline",
    "generate_cluster_phases",
    "generate_orbits",
    "generate_uniform_phases",
    "generate_walker_phases",
    "get_precise_body_params",
    "list_supported_bodies",
    "orbits_from_legacy_config",
    "propagate_sat_ecef_batch",
    "registry",
    "solve_kepler_aitken",
    "solve_kepler_pade",
]

# ── D1: 半解析轨道传播 ──────────────────────────────────────────
# ── D10: 分层误差分配 + 闭环校验 ────────────────────────────────
from .error_budget import (
    AlertLevel as AlertLevel,
)
from .error_budget import (
    ClosedLoopValidator as ClosedLoopValidator,
)
from .error_budget import (
    FallbackToFullModelError as FallbackToFullModelError,
)
from .error_budget import (
    GlobalResidualMonitor as GlobalResidualMonitor,
)
from .error_budget import (
    LayeredErrorManager as LayeredErrorManager,
)
from .error_budget import (
    LayerError as LayerError,
)
from .error_budget import (
    PassObservation as PassObservation,
)
from .error_budget import (
    ValidationStats as ValidationStats,
)

# ── D2: 底层数值优化 ────────────────────────────────────────────
from .numerical_opt import (
    FloatPrecisionManager as FloatPrecisionManager,
)
from .numerical_opt import (
    LaguerreKeplerSolver as LaguerreKeplerSolver,
)
from .numerical_opt import (
    PrecisionConfig as PrecisionConfig,
)
from .numerical_opt import (
    PrecisionTier as PrecisionTier,
)
from .numerical_opt import (
    TrigLookupTable as TrigLookupTable,
)
from .numerical_opt import (
    get_default_lut as get_default_lut,
)

# ── D3: 时序预测补偿 ────────────────────────────────────────────
from .prediction_kalman import (
    CompensationLimitExceededError as CompensationLimitExceededError,
)
from .prediction_kalman import (
    CompensationResult as CompensationResult,
)
from .prediction_kalman import (
    ErrorTrendFitter as ErrorTrendFitter,
)
from .prediction_kalman import (
    HybridCompensator as HybridCompensator,
)
from .prediction_kalman import (
    KalmanConfig as KalmanConfig,
)
from .prediction_kalman import (
    KalmanTimingCompensator as KalmanTimingCompensator,
)
from .prediction_kalman import (
    TrendFitResult as TrendFitResult,
)
from .prediction_kalman import (
    estimate_refresh_interval_extension as estimate_refresh_interval_extension,
)
from .semi_analytic import (
    AdaptivePerturbationTruncator as AdaptivePerturbationTruncator,
)
from .semi_analytic import (
    BiasCompensationTable as BiasCompensationTable,
)
from .semi_analytic import (
    BiasState as BiasState,
)
from .semi_analytic import (
    BiasType as BiasType,
)
from .semi_analytic import (
    CalibrationPoint as CalibrationPoint,
)
from .semi_analytic import (
    ErrorBudget as ErrorBudget,
)
from .semi_analytic import (
    PeriodicCalibrationManager as PeriodicCalibrationManager,
)
from .semi_analytic import (
    PerturbationTier as PerturbationTier,
)
from .semi_analytic import (
    PolyFitResult as PolyFitResult,
)
from .semi_analytic import (
    RelativeMotionPropagator as RelativeMotionPropagator,
)
from .semi_analytic import (
    RelativeState as RelativeState,
)
from .semi_analytic import (
    ResidualExceededError as ResidualExceededError,
)
from .semi_analytic import (
    SlidingWindowPolyFitter as SlidingWindowPolyFitter,
)
from .semi_analytic import (
    TimeDomain as TimeDomain,
)
from .semi_analytic import (
    TimeDomainWindowController as TimeDomainWindowController,
)
from .semi_analytic import (
    TruncationConfig as TruncationConfig,
)
from .semi_analytic import (
    estimate_truncation_speedup as estimate_truncation_speedup,
)

__all__ += [
    "AdaptivePerturbationTruncator",
    "AlertLevel",
    "BiasCompensationTable",
    "BiasState",
    "BiasType",
    "CalibrationPoint",
    "ClosedLoopValidator",
    "CompensationLimitExceededError",
    "CompensationResult",
    "ErrorBudget",
    "ErrorTrendFitter",
    "FallbackToFullModelError",
    "FloatPrecisionManager",
    "GlobalResidualMonitor",
    "HybridCompensator",
    "KalmanConfig",
    "KalmanTimingCompensator",
    "LaguerreKeplerSolver",
    "LayerError",
    "LayeredErrorManager",
    "PassObservation",
    "PeriodicCalibrationManager",
    "PerturbationTier",
    "PolyFitResult",
    "PrecisionConfig",
    "PrecisionTier",
    "RelativeMotionPropagator",
    "RelativeState",
    "ResidualExceededError",
    "SlidingWindowPolyFitter",
    "TimeDomain",
    "TimeDomainWindowController",
    "TrendFitResult",
    "TrigLookupTable",
    "TruncationConfig",
    "ValidationStats",
    "estimate_refresh_interval_extension",
    "estimate_truncation_speedup",
    "get_default_lut",
]
