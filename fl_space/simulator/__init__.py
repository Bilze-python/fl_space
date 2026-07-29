"""
模拟器层 — 接触矩阵管理和轨道模拟主引擎

主要导出:
    OrbitSimulator  — 模块化轨道接触模拟器（主类）
    ContactMatrix   — 接触矩阵（兼容/完整两种模式）
    OrbitCacheCalculator — 轨道离线预计算缓存（优化模块）
    AdaptiveTimingScheduler — 自适应多层定时调度器（论文优化）
"""

from .business_scheduler import (
    GsLoadForecaster as GsLoadForecaster,
)
from .business_scheduler import (
    LoadForecast as LoadForecast,
)
from .business_scheduler import (
    PriorityDrivenScheduler as PriorityDrivenScheduler,
)
from .business_scheduler import (
    RelayChain as RelayChain,
)
from .business_scheduler import (
    StationRelayMerger as StationRelayMerger,
)
from .contact_matrix import ContactMatrix as ContactMatrix
from .heterogeneous_engine import (
    DistributedOrchestrator as DistributedOrchestrator,
)
from .heterogeneous_engine import (
    DomainConfig as DomainConfig,
)
from .heterogeneous_engine import (
    GpuBatchPropagator as GpuBatchPropagator,
)
from .heterogeneous_engine import (
    HeterogeneousScheduler as HeterogeneousScheduler,
)
from .orbit_cache import (
    OrbitCacheCalculator as OrbitCacheCalculator,
)
from .orbit_cache import (
    create_orbit_cache as create_orbit_cache,
)
from .orbit_simulator import (
    OrbitSimulator as OrbitSimulator,
)
from .orbit_simulator import (
    create_default_simulator as create_default_simulator,
)
from .orbit_simulator import (
    create_mars_simulator as create_mars_simulator,
)
from .pass_scheduler import (
    AllocationResult as AllocationResult,
)
from .pass_scheduler import (
    GroundStationAllocator as GroundStationAllocator,
)
from .pass_scheduler import (
    PassRecord as PassRecord,
)
from .pass_scheduler import (
    PassTimetable as PassTimetable,
)
from .pass_scheduler import (
    build_pass_schedule as build_pass_schedule,
)
from .precision_mode import (
    ECO_CONFIG as ECO_CONFIG,
)
from .precision_mode import (
    EMERGENCY_CONFIG as EMERGENCY_CONFIG,
)
from .precision_mode import (
    NORMAL_CONFIG as NORMAL_CONFIG,
)
from .precision_mode import (
    PrecisionConfig as PrecisionConfig,
)
from .precision_mode import (
    PrecisionMode as PrecisionMode,
)
from .precision_mode import (
    PrecisionModeSwitcher as PrecisionModeSwitcher,
)
from .preprocessor import (
    CleanedTLE as CleanedTLE,
)
from .preprocessor import (
    MaintenanceInterval as MaintenanceInterval,
)
from .preprocessor import (
    UnifiedTime as UnifiedTime,
)
from .preprocessor import (
    aggregate_maintenance as aggregate_maintenance,
)
from .preprocessor import (
    batch_clean_tles as batch_clean_tles,
)
from .preprocessor import (
    batch_unify_times as batch_unify_times,
)
from .preprocessor import (
    clean_tle as clean_tle,
)
from .preprocessor import (
    convert_to_utc_seconds as convert_to_utc_seconds,
)
from .preprocessor import (
    utc_seconds_to_timeslot as utc_seconds_to_timeslot,
)
from .sparse_computer import (
    CoarseScanResult as CoarseScanResult,
)
from .sparse_computer import (
    GeoGrid as GeoGrid,
)
from .sparse_computer import (
    GeoGridFilter as GeoGridFilter,
)
from .sparse_computer import (
    SatelliteOrbitGroups as SatelliteOrbitGroups,
)
from .sparse_computer import (
    coarse_scan_timeline as coarse_scan_timeline,
)
from .sparse_computer import (
    sparse_fine_scan_slots as sparse_fine_scan_slots,
)
from .timing_scheduler import (
    AdaptiveTimingScheduler as AdaptiveTimingScheduler,
)
from .timing_scheduler import (
    LoadLevel as LoadLevel,
)
from .timing_scheduler import (
    TimingLayer as TimingLayer,
)
from .timing_scheduler import (
    run_layered_schedule as run_layered_schedule,
)

__all__ = [
    "ECO_CONFIG",
    "EMERGENCY_CONFIG",
    "NORMAL_CONFIG",
    "AdaptiveTimingScheduler",
    "AllocationResult",
    "CleanedTLE",
    "CoarseScanResult",
    "ContactMatrix",
    "DistributedOrchestrator",
    "DomainConfig",
    "GeoGrid",
    "GeoGridFilter",
    "GpuBatchPropagator",
    "GroundStationAllocator",
    "GsLoadForecaster",
    "HeterogeneousScheduler",
    "LoadForecast",
    "LoadLevel",
    "MaintenanceInterval",
    "OrbitCacheCalculator",
    "OrbitSimulator",
    "PassRecord",
    "PassTimetable",
    "PrecisionConfig",
    "PrecisionMode",
    "PrecisionModeSwitcher",
    "PriorityDrivenScheduler",
    "RelayChain",
    "SatelliteOrbitGroups",
    "StationRelayMerger",
    "TimingLayer",
    "UnifiedTime",
    "aggregate_maintenance",
    "batch_clean_tles",
    "batch_unify_times",
    "build_pass_schedule",
    "clean_tle",
    "coarse_scan_timeline",
    "convert_to_utc_seconds",
    "create_default_simulator",
    "create_mars_simulator",
    "create_orbit_cache",
    "run_layered_schedule",
    "sparse_fine_scan_slots",
    "utc_seconds_to_timeslot",
]

# ── D5: 系统级定时调度 ──────────────────────────────────────────
# ── D9: 异常工况兜底 ────────────────────────────────────────────
from .fallback_engine import (
    DegradationManager as DegradationManager,
)
from .fallback_engine import (
    DegradationPolicy as DegradationPolicy,
)
from .fallback_engine import (
    EngineMode as EngineMode,
)
from .fallback_engine import (
    LightweightPropagator as LightweightPropagator,
)
from .fallback_engine import (
    create_fallback_schedule as create_fallback_schedule,
)

# ── D7: 混合离线查表 ────────────────────────────────────────────
from .hybrid_lookup import (
    EnvironmentCase as EnvironmentCase,
)
from .hybrid_lookup import (
    EnvironmentParamLibrary as EnvironmentParamLibrary,
)
from .hybrid_lookup import (
    HybridOrbitProvider as HybridOrbitProvider,
)
from .hybrid_lookup import (
    PassTemplate as PassTemplate,
)
from .hybrid_lookup import (
    PassTemplateManager as PassTemplateManager,
)
from .hybrid_lookup import (
    SatellitePassDB as SatellitePassDB,
)

# ── D8: 多地面站联合优化 ────────────────────────────────────────
from .multi_gs_optimizer import (
    BatchENUComputer as BatchENUComputer,
)
from .multi_gs_optimizer import (
    GSCluster as GSCluster,
)
from .multi_gs_optimizer import (
    ProximityGSClusterer as ProximityGSClusterer,
)
from .multi_gs_optimizer import (
    RelaySegment as RelaySegment,
)
from .multi_gs_optimizer import (
    RelayWindow as RelayWindow,
)
from .multi_gs_optimizer import (
    RelayWindowMerger as RelayWindowMerger,
)
from .multi_gs_optimizer import (
    estimate_batch_speedup as estimate_batch_speedup,
)
from .system_scheduler import (
    CpuAffinityBinder as CpuAffinityBinder,
)
from .system_scheduler import (
    HighPrecisionTimer as HighPrecisionTimer,
)
from .system_scheduler import (
    KernelClockTrigger as KernelClockTrigger,
)
from .system_scheduler import (
    LoadAwareDegrader as LoadAwareDegrader,
)
from .system_scheduler import (
    PriorityTaskScheduler as PriorityTaskScheduler,
)
from .system_scheduler import (
    ScheduledTask as ScheduledTask,
)
from .system_scheduler import (
    TaskPriority as TaskPriority,
)
from .system_scheduler import (
    TaskStats as TaskStats,
)

# ── D6: 可视窗口预筛选 ──────────────────────────────────────────
from .window_prefilter import (
    BoundingBox3D as BoundingBox3D,
)
from .window_prefilter import (
    CombinedPreFilter as CombinedPreFilter,
)
from .window_prefilter import (
    ElevationRatePruner as ElevationRatePruner,
)
from .window_prefilter import (
    ElevationState as ElevationState,
)
from .window_prefilter import (
    OrbitalBoundingBox as OrbitalBoundingBox,
)
from .window_prefilter import (
    PreFilterResult as PreFilterResult,
)

__all__ += [
    "BatchENUComputer",
    "BoundingBox3D",
    "CombinedPreFilter",
    "CpuAffinityBinder",
    "DegradationManager",
    "DegradationPolicy",
    "ElevationRatePruner",
    "ElevationState",
    "EngineMode",
    "EnvironmentCase",
    "EnvironmentParamLibrary",
    "GSCluster",
    "HighPrecisionTimer",
    "HybridOrbitProvider",
    "KernelClockTrigger",
    "LightweightPropagator",
    "LoadAwareDegrader",
    "OrbitalBoundingBox",
    "PassTemplate",
    "PassTemplateManager",
    "PreFilterResult",
    "PriorityTaskScheduler",
    "ProximityGSClusterer",
    "RelaySegment",
    "RelayWindow",
    "RelayWindowMerger",
    "SatellitePassDB",
    "ScheduledTask",
    "TaskPriority",
    "TaskStats",
    "create_fallback_schedule",
    "estimate_batch_speedup",
]
