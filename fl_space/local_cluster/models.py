"""Data contracts for isolated local satellite-cluster experiments.

The values in :func:`LocalClusterConfig.paper_reference` are traceable to
arXiv:2511.19468v2. They are research assumptions, not SpaceX or Google
engineering specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DATASET_PROFILES: dict[str, dict[str, float | int | str]] = {
    "mnist": {
        "bytes_per_sample": 784,
        "model_size_mb": 0.39,
        "flops_per_sample": 327_552.0,
        "description": "28x28 grayscale classification",
    },
    "fashion_mnist": {
        "bytes_per_sample": 784,
        "model_size_mb": 0.39,
        "flops_per_sample": 327_552.0,
        "description": "28x28 grayscale classification",
    },
    "cifar10": {
        "bytes_per_sample": 3_072,
        "model_size_mb": 1.20,
        "flops_per_sample": 2_000_000.0,
        "description": "32x32 RGB classification",
    },
    "femnist": {
        "bytes_per_sample": 784,
        "model_size_mb": 0.39,
        "flops_per_sample": 327_552.0,
        "description": "writer-level handwritten character classification",
    },
}


@dataclass
class OpticalLinkConfig:
    """Short-range optical ISL and spatial-multiplexing parameters.

    Distances use km, apertures use cm, optical power uses W, and data rates
    use Gbps. ``channels_per_link`` and ``rate_per_channel_gbps`` describe a
    nominal DWDM stream; actual delivered capacity is reduced by the link
    budget and pointing margin.
    """

    aperture_diameter_cm: float = 10.0
    link_distance_km: float = 5.0
    parallel_links: int = 1
    channels_per_link: int = 24
    rate_per_channel_gbps: float = 0.4
    wavelength_um: float = 1.55
    transmit_power_w: float = 5.0
    aperture_efficiency: float = 0.80
    other_loss_db: float = 3.0
    required_receive_power_dbm_per_channel: float = -20.0
    pointing_accuracy_urad: float = 1.0
    modulation: str = "PM-16QAM"
    source: str = "paper_reference:arXiv:2511.19468v2"

    def __post_init__(self) -> None:
        if self.aperture_diameter_cm <= 0 or self.link_distance_km <= 0:
            raise ValueError("口径和链路距离必须为正")
        if self.parallel_links < 1 or self.channels_per_link < 1:
            raise ValueError("并行链路数和DWDM通道数必须至少为1")
        if self.rate_per_channel_gbps <= 0 or self.wavelength_um <= 0:
            raise ValueError("通道速率和波长必须为正")
        if self.transmit_power_w <= 0 or not 0 < self.aperture_efficiency <= 1:
            raise ValueError("发射功率必须为正，口径效率必须在(0,1]内")
        if self.other_loss_db < 0 or self.pointing_accuracy_urad < 0:
            raise ValueError("损耗和指向误差不能为负")
        if not self.source.strip():
            raise ValueError("参数来源不能为空")

    @property
    def nominal_bandwidth_gbps(self) -> float:
        """Nominal aggregate one-way capacity before link-budget penalties."""
        return self.parallel_links * self.channels_per_link * self.rate_per_channel_gbps

    @property
    def nominal_duplex_bandwidth_gbps(self) -> float:
        """Nominal bidirectional sum, reported separately from one-way rate."""
        return self.nominal_bandwidth_gbps * 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "aperture_diameter_cm": self.aperture_diameter_cm,
            "link_distance_km": self.link_distance_km,
            "parallel_links": self.parallel_links,
            "channels_per_link": self.channels_per_link,
            "rate_per_channel_gbps": self.rate_per_channel_gbps,
            "wavelength_um": self.wavelength_um,
            "transmit_power_w": self.transmit_power_w,
            "aperture_efficiency": self.aperture_efficiency,
            "other_loss_db": self.other_loss_db,
            "required_receive_power_dbm_per_channel": self.required_receive_power_dbm_per_channel,
            "pointing_accuracy_urad": self.pointing_accuracy_urad,
            "modulation": self.modulation,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpticalLinkConfig:
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})


@dataclass
class LocalSatelliteVisualSpec:
    """Display-only satellite construction parameters.

    These values intentionally do not enter the mass, orbit, power, or link
    calculations. A future 2D/3D builder can replace the primitive while
    keeping the physical configuration stable.
    """

    primitive: str = "box"
    body_dimensions_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    solar_panel_span_m: float = 2.0
    telescope_scale_m_per_cm: float = 0.01

    def __post_init__(self) -> None:
        if not self.primitive.strip() or any(value <= 0 for value in self.body_dimensions_m):
            raise ValueError("视觉原语和卫星本体尺寸必须为正")
        if self.solar_panel_span_m <= 0 or self.telescope_scale_m_per_cm <= 0:
            raise ValueError("太阳能板跨度和望远镜缩放系数必须为正")

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "body_dimensions_m": list(self.body_dimensions_m),
            "solar_panel_span_m": self.solar_panel_span_m,
            "telescope_scale_m_per_cm": self.telescope_scale_m_per_cm,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LocalSatelliteVisualSpec:
        values = dict(data)
        if "body_dimensions_m" in values:
            values["body_dimensions_m"] = tuple(values["body_dimensions_m"])
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})


@dataclass
class LocalClusterConfig:
    """Independent local-cluster experiment configuration."""

    enabled: bool = False
    name: str = "paper-local-cluster"
    num_satellites: int = 81
    cluster_radius_km: float = 1.0
    altitude_km: float = 650.0
    topology: str = "ring"
    compute_gflops: float = 10.0
    memory_gb: float = 8.0
    link: OpticalLinkConfig = field(default_factory=OpticalLinkConfig)
    visual: LocalSatelliteVisualSpec = field(default_factory=LocalSatelliteVisualSpec)
    dataset: str = "mnist"
    samples_per_satellite: int = 1_000
    rounds: int = 10
    local_epochs: int = 1
    model_size_mb: float | None = None
    flops_per_sample: float | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_satellites < 2:
            raise ValueError("局部星簇至少需要2颗卫星")
        if self.cluster_radius_km <= 0 or self.altitude_km <= 0:
            raise ValueError("星簇半径和轨道高度必须为正")
        if self.topology not in {"ring", "grid", "mesh"}:
            raise ValueError("topology must be ring, grid, or mesh")
        if self.compute_gflops <= 0 or self.memory_gb <= 0:
            raise ValueError("算力和内存必须为正")
        if self.samples_per_satellite < 1 or self.rounds < 1 or self.local_epochs < 1:
            raise ValueError("样本数、轮次和本地epoch必须至少为1")
        if self.dataset not in DATASET_PROFILES and (
            self.model_size_mb is None or self.flops_per_sample is None
        ):
            raise ValueError("自定义数据集必须同时提供 model_size_mb 和 flops_per_sample")

    @classmethod
    def paper_reference(cls, **overrides: Any) -> LocalClusterConfig:
        """Return the paper-inspired 81-satellite baseline, still disabled by default."""
        values: dict[str, Any] = {"enabled": False}
        values.update(overrides)
        return cls(**values)

    def workload_profile(self) -> dict[str, float | int | str]:
        profile = DATASET_PROFILES.get(self.dataset, {})
        return {
            "dataset": self.dataset,
            "description": str(profile.get("description", "custom workload")),
            "bytes_per_sample": int(profile.get("bytes_per_sample", 0)),
            "model_size_mb": float(
                self.model_size_mb if self.model_size_mb is not None else profile["model_size_mb"]
            ),
            "flops_per_sample": float(
                self.flops_per_sample
                if self.flops_per_sample is not None
                else profile["flops_per_sample"]
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "name": self.name,
            "num_satellites": self.num_satellites,
            "cluster_radius_km": self.cluster_radius_km,
            "altitude_km": self.altitude_km,
            "topology": self.topology,
            "compute_gflops": self.compute_gflops,
            "memory_gb": self.memory_gb,
            "link": self.link.to_dict(),
            "visual": self.visual.to_dict(),
            "dataset": self.dataset,
            "samples_per_satellite": self.samples_per_satellite,
            "rounds": self.rounds,
            "local_epochs": self.local_epochs,
            "model_size_mb": self.model_size_mb,
            "flops_per_sample": self.flops_per_sample,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LocalClusterConfig:
        values = dict(data)
        values["link"] = OpticalLinkConfig.from_dict(values.get("link", {}))
        values["visual"] = LocalSatelliteVisualSpec.from_dict(values.get("visual", {}))
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})


@dataclass
class LocalClusterResult:
    """Serializable output of one independent local-cluster benchmark."""

    schema_version: str = "local_cluster_result_v1"
    name: str = "paper-local-cluster"
    dataset: str = "mnist"
    num_satellites: int = 0
    effective_bandwidth_gbps: float = 0.0
    nominal_bandwidth_gbps: float = 0.0
    received_power_mw: float = 0.0
    link_margin_db: float = 0.0
    pointing_loss_db: float = 0.0
    training_time_per_round_s: float = 0.0
    aggregation_time_per_round_s: float = 0.0
    per_round_latency_s: float = 0.0
    total_time_s: float = 0.0
    samples_per_second: float = 0.0
    cluster_score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_federated_profile(self) -> dict[str, Any]:
        """Return the explicit opt-in inputs understood by global FedLEO."""
        satellites = max(1, int(self.profile.get("num_satellites", self.num_satellites)))
        sats_per_plane = max(1, int(self.profile.get("sats_per_plane", satellites)))
        return {
            "schema_version": "local_cluster_profile_v1",
            "source": "local_cluster_benchmark",
            "num_satellites": satellites,
            "num_planes": max(1, satellites // sats_per_plane),
            "sats_per_plane": sats_per_plane,
            "bandwidth_mbps": float(self.effective_bandwidth_gbps * 1000.0),
            "timeslot_duration_sec": float(self.per_round_latency_s),
            "cluster_efficiency_score": float(self.cluster_score),
            "effective_bandwidth_gbps": float(self.effective_bandwidth_gbps),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "dataset": self.dataset,
            "num_satellites": self.num_satellites,
            "effective_bandwidth_gbps": self.effective_bandwidth_gbps,
            "nominal_bandwidth_gbps": self.nominal_bandwidth_gbps,
            "received_power_mw": self.received_power_mw,
            "link_margin_db": self.link_margin_db,
            "pointing_loss_db": self.pointing_loss_db,
            "training_time_per_round_s": self.training_time_per_round_s,
            "aggregation_time_per_round_s": self.aggregation_time_per_round_s,
            "per_round_latency_s": self.per_round_latency_s,
            "total_time_s": self.total_time_s,
            "samples_per_second": self.samples_per_second,
            "cluster_score": self.cluster_score,
            "score_components": dict(self.score_components),
            "profile": dict(self.profile),
            "federated_profile": self.to_federated_profile(),
            "assumptions": list(self.assumptions),
            "config": dict(self.config),
        }
