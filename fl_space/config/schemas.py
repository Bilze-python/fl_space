"""
Pydantic v2 配置契约 — 可选的强类型配置校验层。

对齐师兄项目 autoFly_Stk 的 config/schemas.py，但与 SpaceFL 内部数据结构
兼容。Pydantic 是可选依赖，未安装时本模块导入会报 ImportError。

用法::

    try:
        from fl_space.config.schemas import SpaceFLScenario
        scenario = SpaceFLScenario.from_yaml("config.yaml")
    except ImportError:
        print("需要 pip install pydantic pyyaml")
"""

from __future__ import annotations

from typing import Optional

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    raise ImportError("Pydantic 未安装。请运行: pip install pydantic pyyaml")


# ── 轨道配置 ──────────────────────────────────────────────


class WalkerSpec(BaseModel):
    """Walker 星座规格。"""

    num_planes: int = Field(default=1, ge=1, description="轨道面数")
    sats_per_plane: int = Field(default=3, ge=1, description="每面卫星数")
    inclination_deg: float = Field(default=53.0, ge=0.0, le=180.0)
    altitude_km: float = Field(default=500.0, ge=100.0, le=2000.0)
    phasing: float = Field(default=1.0, description="相位因子 F")
    topology: str = Field(default="star", description="star | delta")

    @property
    def total_satellites(self) -> int:
        return self.num_planes * self.sats_per_plane


# ── 地面站 ────────────────────────────────────────────────


class GroundStation(BaseModel):
    """单个地面站定义。"""

    name: str
    lat_deg: float = Field(ge=-90.0, le=90.0)
    lon_deg: float = Field(ge=-180.0, le=180.0)
    min_elevation_deg: float = Field(default=10.0, ge=0.0, le=90.0)


# ── ISL 配置 ──────────────────────────────────────────────


class IntraClusterSpec(BaseModel):
    """星簇内 ISL 配置。"""

    enabled: bool = True
    atmosphere_buffer_km: float = Field(default=0.0, ge=0.0)
    step_seconds: float = Field(default=60.0, gt=0.0)
    cluster_mode: str = Field(default="plane")


# ── FL 实验配置 ───────────────────────────────────────────


class FLExperimentSpec(BaseModel):
    """FL 实验参数。"""

    algorithm: str = Field(default="fedavg")
    num_rounds: int = Field(default=300, ge=1)
    local_epochs: int = Field(default=2, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.01, gt=0.0)
    mu: float = Field(default=0.01, ge=0.0)
    early_stop_acc: Optional[float] = Field(default=0.90, ge=0.0, le=1.0)
    dataset: str = Field(default="mnist")
    device: str = Field(default="cpu")
    fraction: float = Field(default=1.0, gt=0.0, le=1.0)


# ── 输出配置 ──────────────────────────────────────────────


class OutputSpec(BaseModel):
    """输出产物开关。"""

    save_history_json: bool = True
    save_accuracy_plot: bool = True
    save_contact_heatmap: bool = True
    save_gs_positions: bool = True
    save_orbit_cross_section: bool = True
    save_sat_training_time: bool = True
    save_gs_sat_contacts: bool = True
    save_isl_report: bool = True


# ── 顶层场景 ──────────────────────────────────────────────


class SpaceFLScenario(BaseModel):
    """SpaceFL 实验场景完整配置（YAML 顶层）。"""

    name: str = Field(default="spacefl_experiment")
    constellation: WalkerSpec = Field(default_factory=WalkerSpec)
    ground_stations: list[GroundStation] = Field(default_factory=list)
    intra_cluster: IntraClusterSpec = Field(default_factory=IntraClusterSpec)
    fl_experiment: FLExperimentSpec = Field(default_factory=FLExperimentSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    sim_hours: float = Field(default=168.0, gt=0.0)
    timeslot_duration_min: float = Field(default=1.0, gt=0.0)
    seed: int = Field(default=42)

    @field_validator("ground_stations")
    @classmethod
    def _at_least_one_gs(cls, v: list[GroundStation]) -> list[GroundStation]:
        if len(v) < 1:
            raise ValueError("至少需要一个地面站")
        return v

    # ── 序列化 ────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> SpaceFLScenario:
        """从 YAML 文件加载场景配置。"""
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str) -> None:
        """导出为 YAML 文件。"""
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)

    def to_fl_config(self) -> dict:
        """转换为 SpaceFL FLConfig 兼容字典。"""
        return {
            "algorithm": self.fl_experiment.algorithm,
            "num_rounds": self.fl_experiment.num_rounds,
            "num_clients": self.constellation.total_satellites,
            "local_epochs": self.fl_experiment.local_epochs,
            "batch_size": self.fl_experiment.batch_size,
            "learning_rate": self.fl_experiment.learning_rate,
            "mu": self.fl_experiment.mu,
            "early_stop_acc": self.fl_experiment.early_stop_acc,
            "device": self.fl_experiment.device,
            "fraction": self.fl_experiment.fraction,
            "isl_enabled": self.intra_cluster.enabled,
            "isl_atmosphere_buffer_km": self.intra_cluster.atmosphere_buffer_km,
        }

    def to_simulator_kwargs(self) -> dict:
        """转换为 OrbitSimulator 构造参数。"""
        return {
            "num_satellites": self.constellation.total_satellites,
            "num_ground_stations": len(self.ground_stations),
            "orbit_altitude_km": self.constellation.altitude_km,
            "orbit_inclination_deg": self.constellation.inclination_deg,
            "distribution": self.constellation.topology,
            "timeslot_duration_min": self.timeslot_duration_min,
            "num_timeslots": int(self.sim_hours * 60 / self.timeslot_duration_min),
            "random_seed": self.seed,
        }
