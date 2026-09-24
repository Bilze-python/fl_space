"""
SpaceFL 学习任务顶层模型。

学习任务是导师建议中的最高级单元：一个任务同时声明数据、模型、卫星轨道、
基础设施和调度策略。该层只做组合与序列化，不把经营模式或 UI 规则塞进物理层。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from fl_space.mission import ClassicalOrbitalElements, CommunicationBand, SatelliteDesign

DataDistribution = Literal["iid", "non_iid_label_skew", "non_iid_quantity_skew", "custom"]
LearningTaskKind = Literal["classification", "detection", "segmentation", "regression", "custom"]
LinkKind = Literal["space_to_ground", "inter_satellite", "crosslink"]
SchedulerPolicy = Literal["contact_aware", "energy_aware", "round_robin", "fedleo", "custom"]


@dataclass
class DatasetSpec:
    """学习任务数据声明。"""

    name: str = "mnist"
    data_type: str = "image"
    distribution: DataDistribution = "iid"
    sample_count: int = 60_000
    size_mb: float = 64.0
    shard_count: int = 1
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """验证数据配置。"""
        if not self.name.strip():
            raise ValueError("数据集名称不能为空")
        if self.sample_count < 0 or self.size_mb < 0:
            raise ValueError("数据规模不能为负")
        if self.shard_count < 1:
            raise ValueError("数据分片数量至少为 1")

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "distribution": self.distribution,
            "sample_count": self.sample_count,
            "size_mb": self.size_mb,
            "shard_count": self.shard_count,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DatasetSpec:
        """从字典创建数据声明。"""
        return cls(**data)


@dataclass
class ModelSpec:
    """学习模型和更新载荷声明。"""

    name: str = "cnn"
    task_kind: LearningTaskKind = "classification"
    framework: str = "pytorch"
    parameter_count: int = 100_000
    model_size_mb: float = 1.0
    update_size_mb: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """验证模型配置。"""
        if not self.name.strip():
            raise ValueError("模型名称不能为空")
        if self.parameter_count < 0:
            raise ValueError("模型参数量不能为负")
        if self.model_size_mb < 0 or self.update_size_mb < 0:
            raise ValueError("模型大小和更新大小不能为负")

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "task_kind": self.task_kind,
            "framework": self.framework,
            "parameter_count": self.parameter_count,
            "model_size_mb": self.model_size_mb,
            "update_size_mb": self.update_size_mb,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ModelSpec:
        """从字典创建模型声明。"""
        return cls(**data)


@dataclass
class SpaceNodeSpec:
    """参与学习任务的航天器节点。"""

    node_id: str
    satellite: SatelliteDesign = field(default_factory=SatelliteDesign)
    orbit: ClassicalOrbitalElements = field(
        default_factory=lambda: ClassicalOrbitalElements.from_circular_altitude(550.0)
    )
    role: str = "client"

    def __post_init__(self) -> None:
        """验证节点配置。"""
        if not self.node_id.strip():
            raise ValueError("节点 ID 不能为空")
        if not self.role.strip():
            raise ValueError("节点角色不能为空")

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "node_id": self.node_id,
            "satellite": self.satellite.to_dict(),
            "orbit": self.orbit.to_dict(),
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SpaceNodeSpec:
        """从字典创建航天器节点。"""
        values = dict(data)
        values["satellite"] = SatelliteDesign.from_dict(values["satellite"])
        values["orbit"] = ClassicalOrbitalElements.from_dict(values["orbit"])
        return cls(**values)


@dataclass
class GroundStationSpec:
    """学习任务基础设施中的地面站。"""

    name: str
    latitude_deg: float
    longitude_deg: float
    min_elevation_deg: float = 10.0
    role: str = "aggregator"

    def __post_init__(self) -> None:
        """验证地面站配置。"""
        if not self.name.strip():
            raise ValueError("地面站名称不能为空")
        if not -90 <= self.latitude_deg <= 90:
            raise ValueError("地面站纬度必须在 [-90, 90] 度")
        if not -180 <= self.longitude_deg <= 180:
            raise ValueError("地面站经度必须在 [-180, 180] 度")
        if not 0 <= self.min_elevation_deg <= 90:
            raise ValueError("最小仰角必须在 [0, 90] 度")

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "min_elevation_deg": self.min_elevation_deg,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GroundStationSpec:
        """从字典创建地面站。"""
        return cls(**data)


@dataclass
class CommunicationLinkSpec:
    """链路模拟声明。"""

    name: str = "s_band_downlink"
    link_kind: LinkKind = "space_to_ground"
    band: CommunicationBand = "S"
    frequency_hz: float = 2.2e9
    bandwidth_mhz: float = 5.0
    data_rate_mbps: float = 10.0
    max_range_km: float = 2_000.0
    source: str = "assumption"

    def __post_init__(self) -> None:
        """验证链路配置。"""
        if not self.name.strip():
            raise ValueError("链路名称不能为空")
        if self.frequency_hz <= 0:
            raise ValueError("链路频率必须为正")
        if self.bandwidth_mhz < 0 or self.data_rate_mbps <= 0 or self.max_range_km < 0:
            raise ValueError("链路带宽、速率和距离必须满足非负/正值约束")
        if not self.source.strip():
            raise ValueError("链路参数来源不能为空")

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "link_kind": self.link_kind,
            "band": self.band,
            "frequency_hz": self.frequency_hz,
            "bandwidth_mhz": self.bandwidth_mhz,
            "data_rate_mbps": self.data_rate_mbps,
            "max_range_km": self.max_range_km,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CommunicationLinkSpec:
        """从字典创建链路声明。"""
        return cls(**data)


@dataclass
class InfrastructureSpec:
    """地面站和通信链路集合。"""

    ground_stations: list[GroundStationSpec] = field(default_factory=list)
    links: list[CommunicationLinkSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        """验证基础设施配置。"""
        station_names = [station.name for station in self.ground_stations]
        if len(station_names) != len(set(station_names)):
            raise ValueError("地面站名称不能重复")
        link_names = [link.name for link in self.links]
        if len(link_names) != len(set(link_names)):
            raise ValueError("链路名称不能重复")

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "ground_stations": [station.to_dict() for station in self.ground_stations],
            "links": [link.to_dict() for link in self.links],
        }

    @classmethod
    def from_dict(cls, data: dict) -> InfrastructureSpec:
        """从字典创建基础设施声明。"""
        return cls(
            ground_stations=[
                GroundStationSpec.from_dict(item) for item in data.get("ground_stations", [])
            ],
            links=[CommunicationLinkSpec.from_dict(item) for item in data.get("links", [])],
        )


@dataclass
class SchedulerSpec:
    """学习任务调度策略。"""

    policy: SchedulerPolicy = "contact_aware"
    round_duration_s: float = 600.0
    participant_fraction: float = 1.0
    max_clients_per_round: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """验证调度配置。"""
        if self.round_duration_s <= 0:
            raise ValueError("训练轮时长必须为正")
        if not 0 < self.participant_fraction <= 1:
            raise ValueError("参与比例必须在 (0, 1] 内")
        if self.max_clients_per_round is not None and self.max_clients_per_round < 1:
            raise ValueError("每轮最大客户端数量至少为 1")

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "policy": self.policy,
            "round_duration_s": self.round_duration_s,
            "participant_fraction": self.participant_fraction,
            "max_clients_per_round": self.max_clients_per_round,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SchedulerSpec:
        """从字典创建调度声明。"""
        return cls(**data)


@dataclass
class LearningTaskScenario:
    """以学习任务为顶层单元的研究场景。"""

    name: str = "spacefl_learning_task"
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    nodes: list[SpaceNodeSpec] = field(default_factory=list)
    infrastructure: InfrastructureSpec = field(default_factory=InfrastructureSpec)
    scheduler: SchedulerSpec = field(default_factory=SchedulerSpec)
    assumptions: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """验证顶层学习任务。"""
        if not self.name.strip():
            raise ValueError("学习任务名称不能为空")
        if not self.nodes:
            raise ValueError("学习任务至少需要一个航天器节点")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("航天器节点 ID 不能重复")

    @property
    def satellite_count(self) -> int:
        """参与任务的航天器数量。"""
        return len(self.nodes)

    @property
    def per_round_update_volume_mb(self) -> float:
        """一轮训练中全量客户端上传模型更新的数据量估算。"""
        return (
            self.model.update_size_mb * self.satellite_count * self.scheduler.participant_fraction
        )

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "dataset": self.dataset.to_dict(),
            "model": self.model.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "infrastructure": self.infrastructure.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "assumptions": list(self.assumptions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> LearningTaskScenario:
        """从字典创建顶层学习任务。"""
        values = dict(data)
        values["dataset"] = DatasetSpec.from_dict(values["dataset"])
        values["model"] = ModelSpec.from_dict(values["model"])
        values["nodes"] = [SpaceNodeSpec.from_dict(item) for item in values["nodes"]]
        values["infrastructure"] = InfrastructureSpec.from_dict(values["infrastructure"])
        values["scheduler"] = SchedulerSpec.from_dict(values["scheduler"])
        return cls(**values)


__all__ = [
    "CommunicationLinkSpec",
    "DataDistribution",
    "DatasetSpec",
    "GroundStationSpec",
    "InfrastructureSpec",
    "LearningTaskKind",
    "LearningTaskScenario",
    "LinkKind",
    "ModelSpec",
    "SchedulerPolicy",
    "SchedulerSpec",
    "SpaceNodeSpec",
]
