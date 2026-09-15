"""
任务领域数据模型 — 火箭、载荷、卫星设计、发射窗口与发射场景。

支持:
    - 物理参数与视觉外观分离
    - 卫星分系统（电源、通信、姿态控制、推进）与质量分解
    - 用户自定义卫星设计
    - 火箭级间质量和推进剂参数、整流罩包络
    - 发射窗口、部署顺序和任务约束
    - 可序列化的任务场景配置

质量口径（全层一致）:

    卫星总质量 = 平台干质量 + 分系统结构质量 + 卫星推进剂 + 推进系统推进剂
    火箭起飞质量 = 各级湿质量 + 整流罩 + 有效载荷
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Literal

from .constraints import (
    MissionConstraint,
    constraint_from_dict,
    constraint_to_dict,
    default_constraints,
)
from .physics import STANDARD_GRAVITY_MPS2, staged_vehicle_delta_v_mps, tsiolkovsky_delta_v_mps
from .subsystems import (
    AttitudeControlSystem,
    CommunicationSystem,
    PowerSystem,
    PropulsionSystem,
)

VisualPrimitive = Literal["box", "cylinder", "sphere", "custom_mesh"]
PropellantKind = Literal["kerolox", "hydrolox", "methalox", "solid", "hypergolic", "other"]


def _parse_iso_utc(value: str, field_name: str) -> datetime:
    """解析 ISO 8601 时间字符串，并要求显式时区偏移。

    Python 3.9 的 ``datetime.fromisoformat`` 不接受 ``Z`` 结尾，请使用
    ``2026-01-01T00:00:00+00:00`` 形式。
    """
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} 必须是 ISO 8601 时间字符串: {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} 必须包含时区偏移，例如 2026-01-01T00:00:00+00:00")
    return parsed


@dataclass
class VisualAppearance:
    """卫星的可视化描述，不参与物理计算。"""

    primitive: VisualPrimitive = "box"
    dimensions_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    asset_uri: str | None = None
    color_hex: str = "#8ea3b7"
    scale: float = 1.0

    def __post_init__(self) -> None:
        """验证外观参数。"""
        if any(value <= 0 for value in self.dimensions_m):
            raise ValueError("卫星外观尺寸必须全部为正")
        if self.scale <= 0:
            raise ValueError("卫星外观缩放必须为正")
        if not (
            len(self.color_hex) == 7
            and self.color_hex.startswith("#")
            and all(char in "0123456789abcdefABCDEF" for char in self.color_hex[1:])
        ):
            raise ValueError("color_hex 必须是 #RRGGBB 格式")

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "primitive": self.primitive,
            "dimensions_m": list(self.dimensions_m),
            "asset_uri": self.asset_uri,
            "color_hex": self.color_hex,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VisualAppearance:
        """从字典创建外观描述。"""
        values = dict(data)
        if "dimensions_m" in values:
            values["dimensions_m"] = tuple(values["dimensions_m"])
        return cls(**values)


@dataclass
class OrbitalTarget:
    """目标轨道参数，角度使用度，距离使用 km。"""

    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    eccentricity: float = 0.0
    raan_deg: float = 0.0
    arg_perigee_deg: float = 0.0
    true_anomaly_deg: float = 0.0

    def __post_init__(self) -> None:
        """验证轨道参数。"""
        if self.altitude_km <= 0:
            raise ValueError("目标轨道高度必须为正")
        if not 0 <= self.eccentricity < 1:
            raise ValueError("当前基础模型只支持 0 <= eccentricity < 1")
        if not 0 <= self.inclination_deg <= 180:
            raise ValueError("轨道倾角必须在 [0, 180] 度")

    @property
    def is_circular(self) -> bool:
        """是否为圆轨道。"""
        return self.eccentricity < 1e-9

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "altitude_km": self.altitude_km,
            "inclination_deg": self.inclination_deg,
            "eccentricity": self.eccentricity,
            "raan_deg": self.raan_deg,
            "arg_perigee_deg": self.arg_perigee_deg,
            "true_anomaly_deg": self.true_anomaly_deg,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OrbitalTarget:
        """从字典创建目标轨道。"""
        return cls(**data)


@dataclass
class SatelliteDesign:
    """可部署卫星的物理与任务设计。

    ``visual`` 只描述渲染信息，用户可以替换为自定义模型文件而不影响
    质量、阻力和功率等研究参数。物理包络 ``body_dimensions_m`` 与
    ``VisualAppearance.dimensions_m`` 相互独立：前者用于容积和结构约束，
    后者只用于渲染。

    ``dry_mass_kg`` 是平台干质量，不含分系统结构质量和推进剂；分系统
    通过 ``power_system`` / ``communication_system`` /
    ``attitude_control_system`` / ``propulsion_system`` 单独声明。
    """

    name: str = "custom_satellite"
    dry_mass_kg: float = 100.0
    propellant_mass_kg: float = 0.0
    power_w: float = 250.0
    drag_coefficient: float = 2.2
    cross_section_area_m2: float = 1.0
    storage_gb: float = 64.0
    body_dimensions_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    stowed_dimensions_m: tuple[float, float, float] | None = None
    power_system: PowerSystem | None = None
    communication_system: CommunicationSystem | None = None
    attitude_control_system: AttitudeControlSystem | None = None
    propulsion_system: PropulsionSystem | None = None
    visual: VisualAppearance = field(default_factory=VisualAppearance)
    assumptions: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """验证卫星设计参数。"""
        if not self.name.strip():
            raise ValueError("卫星名称不能为空")
        if self.dry_mass_kg <= 0:
            raise ValueError("卫星干质量必须为正")
        if self.propellant_mass_kg < 0:
            raise ValueError("卫星推进剂质量不能为负")
        if self.power_w < 0:
            raise ValueError("卫星功率不能为负")
        if self.drag_coefficient <= 0:
            raise ValueError("阻力系数必须为正")
        if self.cross_section_area_m2 <= 0:
            raise ValueError("迎风面积必须为正")
        if self.storage_gb < 0:
            raise ValueError("存储容量不能为负")
        if any(value <= 0 for value in self.body_dimensions_m):
            raise ValueError("卫星物理包络尺寸必须全部为正")
        if self.stowed_dimensions_m is not None and any(
            value <= 0 for value in self.stowed_dimensions_m
        ):
            raise ValueError("卫星收拢尺寸必须全部为正")
        if (
            self.propulsion_system is not None
            and self.propulsion_system.propellant_mass_kg > 0
            and self.propellant_mass_kg > 0
        ):
            raise ValueError("推进剂不要同时在 propellant_mass_kg 和 propulsion_system 中声明")

    @property
    def launch_dimensions_m(self) -> tuple[float, float, float]:
        """发射状态包络尺寸 (m)，未单独给出收拢尺寸时取本体尺寸。"""
        return self.stowed_dimensions_m or self.body_dimensions_m

    @property
    def envelope_volume_m3(self) -> float:
        """发射状态包络体积 (m³)，按矩形包络一阶估算。"""
        length_m, width_m, height_m = self.launch_dimensions_m
        return length_m * width_m * height_m

    @property
    def subsystem_mass_kg(self) -> float:
        """已建模分系统的结构质量之和 (kg)，不含推进剂。"""
        components = (
            self.power_system,
            self.communication_system,
            self.attitude_control_system,
            self.propulsion_system,
        )
        return sum(component.mass_kg for component in components if component is not None)

    @property
    def total_propellant_mass_kg(self) -> float:
        """卫星自带推进剂总量 (kg)。"""
        system_propellant_kg = (
            self.propulsion_system.propellant_mass_kg if self.propulsion_system else 0.0
        )
        return self.propellant_mass_kg + system_propellant_kg

    @property
    def dry_total_mass_kg(self) -> float:
        """不含推进剂的卫星总质量 (kg)。"""
        return self.dry_mass_kg + self.subsystem_mass_kg

    @property
    def total_mass_kg(self) -> float:
        """卫星初始总质量 (kg)，含分系统结构质量和推进剂。"""
        return self.dry_total_mass_kg + self.total_propellant_mass_kg

    @property
    def available_power_w(self) -> float:
        """可用功率 (W)：建模电源分系统时取轨道平均功率，否则取 ``power_w``。"""
        if self.power_system is not None:
            return self.power_system.orbit_average_power_w
        return self.power_w

    @property
    def available_delta_v_mps(self) -> float:
        """卫星自带推进剂可提供的速度增量 (m/s)，无推进系统时为 0。"""
        if self.propulsion_system is None:
            return 0.0
        return self.propulsion_system.available_delta_v_mps(self.dry_total_mass_kg)

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "dry_mass_kg": self.dry_mass_kg,
            "propellant_mass_kg": self.propellant_mass_kg,
            "power_w": self.power_w,
            "drag_coefficient": self.drag_coefficient,
            "cross_section_area_m2": self.cross_section_area_m2,
            "storage_gb": self.storage_gb,
            "body_dimensions_m": list(self.body_dimensions_m),
            "stowed_dimensions_m": (
                list(self.stowed_dimensions_m) if self.stowed_dimensions_m else None
            ),
            "power_system": self.power_system.to_dict() if self.power_system else None,
            "communication_system": (
                self.communication_system.to_dict() if self.communication_system else None
            ),
            "attitude_control_system": (
                self.attitude_control_system.to_dict() if self.attitude_control_system else None
            ),
            "propulsion_system": (
                self.propulsion_system.to_dict() if self.propulsion_system else None
            ),
            "visual": self.visual.to_dict(),
            "assumptions": list(self.assumptions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SatelliteDesign:
        """从字典创建卫星设计。"""
        values = dict(data)
        visual = values.get("visual")
        if isinstance(visual, dict):
            values["visual"] = VisualAppearance.from_dict(visual)
        power_system = values.get("power_system")
        if isinstance(power_system, dict):
            values["power_system"] = PowerSystem.from_dict(power_system)
        communication_system = values.get("communication_system")
        if isinstance(communication_system, dict):
            values["communication_system"] = CommunicationSystem.from_dict(communication_system)
        attitude_control_system = values.get("attitude_control_system")
        if isinstance(attitude_control_system, dict):
            values["attitude_control_system"] = AttitudeControlSystem.from_dict(
                attitude_control_system
            )
        propulsion_system = values.get("propulsion_system")
        if isinstance(propulsion_system, dict):
            values["propulsion_system"] = PropulsionSystem.from_dict(propulsion_system)
        for key in ("body_dimensions_m", "stowed_dimensions_m"):
            if values.get(key):
                values[key] = tuple(values[key])
        return cls(**values)


@dataclass
class RocketStage:
    """火箭单级的推进和质量参数。

    ``thrust_kn`` 是该级有效总推力，``engine_count`` 记录并联发动机数量，
    单台推力由 ``thrust_per_engine_kn`` 给出。``specific_impulse_s`` 使用
    真空比冲口径，因为一级速度预算最关心真空段加速能力。

    ``burn_time_s`` 与推力、比冲、推进剂质量是四组独立输入；当它们不是
    同一套自洽数据时，``burn_time_deviation_ratio`` 会明显偏离 1，
    一阶模型不会自动改写参数，只提供诊断。
    """

    name: str = "stage"
    dry_mass_kg: float = 10000.0
    propellant_mass_kg: float = 100000.0
    specific_impulse_s: float = 300.0
    thrust_kn: float = 1000.0
    burn_time_s: float = 150.0
    engine_count: int = 1
    propellant_kind: PropellantKind = "kerolox"
    reusable: bool = False

    def __post_init__(self) -> None:
        """验证火箭级参数。"""
        if not self.name.strip():
            raise ValueError("火箭级名称不能为空")
        if self.dry_mass_kg <= 0 or self.propellant_mass_kg <= 0:
            raise ValueError("火箭级干质量和推进剂质量必须为正")
        if self.specific_impulse_s <= 0:
            raise ValueError("比冲必须为正")
        if self.thrust_kn <= 0 or self.burn_time_s <= 0:
            raise ValueError("推力和燃烧时间必须为正")
        if self.engine_count < 1:
            raise ValueError("发动机数量至少为 1")

    @property
    def wet_mass_kg(self) -> float:
        """火箭级湿质量 (kg)。"""
        return self.dry_mass_kg + self.propellant_mass_kg

    @property
    def structure_mass_fraction(self) -> float:
        """结构质量比 = 干质量 / 湿质量。"""
        return self.dry_mass_kg / self.wet_mass_kg

    @property
    def propellant_mass_fraction(self) -> float:
        """推进剂质量比 = 推进剂质量 / 湿质量。"""
        return self.propellant_mass_kg / self.wet_mass_kg

    @property
    def thrust_per_engine_kn(self) -> float:
        """单台发动机推力 (kN)。"""
        return self.thrust_kn / self.engine_count

    @property
    def mass_flow_kg_s(self) -> float:
        """由推力和比冲估算的推进剂流量 (kg/s)。"""
        return self.thrust_kn * 1000.0 / (self.specific_impulse_s * STANDARD_GRAVITY_MPS2)

    @property
    def nominal_burn_time_s(self) -> float:
        """按流量烧完全部推进剂所需时间 (s)。"""
        return self.propellant_mass_kg / self.mass_flow_kg_s

    @property
    def burn_time_deviation_ratio(self) -> float:
        """``burn_time_s`` 与额定燃烧时间之比，用于发现参数不自洽。"""
        return self.burn_time_s / self.nominal_burn_time_s

    def delta_v_mps(self, upper_stack_mass_kg: float) -> float:
        """计算该级对总速度增量的贡献 (m/s)。

        ``upper_stack_mass_kg`` 是该级之上所有质量和（上面级干质量 + 载荷）。
        """
        return tsiolkovsky_delta_v_mps(
            self.dry_mass_kg + self.propellant_mass_kg + upper_stack_mass_kg,
            self.dry_mass_kg + upper_stack_mass_kg,
            self.specific_impulse_s,
        )

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "dry_mass_kg": self.dry_mass_kg,
            "propellant_mass_kg": self.propellant_mass_kg,
            "specific_impulse_s": self.specific_impulse_s,
            "thrust_kn": self.thrust_kn,
            "burn_time_s": self.burn_time_s,
            "engine_count": self.engine_count,
            "propellant_kind": self.propellant_kind,
            "reusable": self.reusable,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RocketStage:
        """从字典创建火箭级。"""
        return cls(**data)


def _default_launch_stages() -> list[RocketStage]:
    """创建一组仅用于快速入门的研究用火箭级。"""
    return [
        RocketStage(
            name="default_booster",
            dry_mass_kg=20_000.0,
            propellant_mass_kg=160_000.0,
            specific_impulse_s=300.0,
            thrust_kn=2_000.0,
            burn_time_s=150.0,
        ),
        RocketStage(
            name="default_upper_stage",
            dry_mass_kg=3_000.0,
            propellant_mass_kg=30_000.0,
            specific_impulse_s=350.0,
            thrust_kn=800.0,
            burn_time_s=420.0,
        ),
    ]


@dataclass
class LaunchVehicle:
    """多级运载火箭的最小研究模型。

    ``stages`` 顺序为一级（助推/芯级）到末级，末级负责把有效载荷送入
    目标轨道。整流罩质量计入末级需要推动的质量，内径和高度只用于
    容积约束，不参与速度预算。
    """

    name: str = "research_launcher"
    stages: list[RocketStage] = field(default_factory=_default_launch_stages)
    payload_capacity_kg: float = 1000.0
    fairing_mass_kg: float = 500.0
    fairing_inner_diameter_m: float = 0.0
    fairing_inner_height_m: float = 0.0

    def __post_init__(self) -> None:
        """验证运载火箭配置。"""
        if not self.name.strip():
            raise ValueError("运载火箭名称不能为空")
        if not self.stages:
            raise ValueError("运载火箭至少需要一个级")
        if self.payload_capacity_kg <= 0:
            raise ValueError("有效载荷能力必须为正")
        if self.fairing_mass_kg < 0:
            raise ValueError("整流罩质量不能为负")
        if self.fairing_inner_diameter_m < 0 or self.fairing_inner_height_m < 0:
            raise ValueError("整流罩内径和高度不能为负")

    @property
    def stage_count(self) -> int:
        """火箭级数。"""
        return len(self.stages)

    @property
    def total_dry_mass_kg(self) -> float:
        """各级结构质量之和 (kg)。"""
        return sum(stage.dry_mass_kg for stage in self.stages)

    @property
    def total_propellant_mass_kg(self) -> float:
        """各级推进剂质量之和 (kg)。"""
        return sum(stage.propellant_mass_kg for stage in self.stages)

    @property
    def fairing_usable_volume_m3(self) -> float:
        """整流罩可用容积 (m³)，按圆柱包络估算；未给尺寸时返回 0。"""
        if self.fairing_inner_diameter_m <= 0 or self.fairing_inner_height_m <= 0:
            return 0.0
        radius_m = 0.5 * self.fairing_inner_diameter_m
        return math.pi * radius_m**2 * self.fairing_inner_height_m

    def liftoff_mass_kg(self, payload_mass_kg: float = 0.0) -> float:
        """起飞质量 (kg) = 各级湿质量 + 整流罩 + 有效载荷。"""
        if payload_mass_kg < 0:
            raise ValueError("载荷质量不能为负")
        return (
            sum(stage.wet_mass_kg for stage in self.stages) + self.fairing_mass_kg + payload_mass_kg
        )

    def ideal_delta_v_mps(self, payload_mass_kg: float) -> float:
        """按当前分级计算多级火箭理想速度增量 (m/s)。"""
        stage_specs = [
            (stage.dry_mass_kg, stage.propellant_mass_kg, stage.specific_impulse_s)
            for stage in self.stages
        ]
        return staged_vehicle_delta_v_mps(stage_specs, payload_mass_kg)

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "stages": [stage.to_dict() for stage in self.stages],
            "payload_capacity_kg": self.payload_capacity_kg,
            "fairing_mass_kg": self.fairing_mass_kg,
            "fairing_inner_diameter_m": self.fairing_inner_diameter_m,
            "fairing_inner_height_m": self.fairing_inner_height_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LaunchVehicle:
        """从字典创建运载火箭。"""
        values = dict(data)
        values["stages"] = [RocketStage.from_dict(stage) for stage in values["stages"]]
        return cls(**values)


@dataclass
class LaunchSite:
    """发射场位置、航区方位角范围和基本环境参数。"""

    name: str = "equatorial_test_site"
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    elevation_m: float = 0.0
    azimuth_range_deg: tuple[float, float] = (0.0, 360.0)

    def __post_init__(self) -> None:
        """验证发射场坐标。"""
        if not -90 <= self.latitude_deg <= 90:
            raise ValueError("发射场纬度必须在 [-90, 90] 度")
        if not -180 <= self.longitude_deg <= 180:
            raise ValueError("发射场经度必须在 [-180, 180] 度")
        if self.elevation_m < 0:
            raise ValueError("发射场海拔不能为负")
        low, high = (float(value) for value in self.azimuth_range_deg)
        if not 0.0 <= low <= high <= 360.0:
            raise ValueError("发射方位角允许范围必须满足 0 <= 下限 <= 上限 <= 360")
        self.azimuth_range_deg = (low, high)

    @property
    def minimum_inclination_deg(self) -> float:
        """从该场址可直接进入的最低轨道倾角 (度)，一阶几何估算。"""
        return abs(self.latitude_deg)

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "elevation_m": self.elevation_m,
            "azimuth_range_deg": list(self.azimuth_range_deg),
        }

    @classmethod
    def from_dict(cls, data: dict) -> LaunchSite:
        """从字典创建发射场。"""
        values = dict(data)
        if values.get("azimuth_range_deg") is not None:
            values["azimuth_range_deg"] = tuple(values["azimuth_range_deg"])
        return cls(**values)


@dataclass
class LaunchWindow:
    """发射窗口 — 用 ISO 8601 字符串声明，避免隐式时区假设。

    时间字符串必须带时区偏移，例如 ``2026-01-01T00:00:00+00:00``。
    """

    name: str = "primary_window"
    open_utc: str = "2026-01-01T00:00:00+00:00"
    close_utc: str = "2026-01-01T02:00:00+00:00"
    notes: str = ""

    def __post_init__(self) -> None:
        """验证窗口时间。"""
        if not self.name.strip():
            raise ValueError("发射窗口名称不能为空")
        open_time = _parse_iso_utc(self.open_utc, "open_utc")
        close_time = _parse_iso_utc(self.close_utc, "close_utc")
        if close_time <= open_time:
            raise ValueError("发射窗口关闭时刻必须晚于开启时刻")

    @property
    def duration_s(self) -> float:
        """窗口时长 (s)。"""
        open_time = _parse_iso_utc(self.open_utc, "open_utc")
        close_time = _parse_iso_utc(self.close_utc, "close_utc")
        return (close_time - open_time).total_seconds()

    def contains(self, epoch_utc: str) -> bool:
        """判断给定时刻是否落在窗口内（含边界）。"""
        epoch = _parse_iso_utc(epoch_utc, "epoch_utc")
        return (
            _parse_iso_utc(self.open_utc, "open_utc")
            <= epoch
            <= _parse_iso_utc(self.close_utc, "close_utc")
        )

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "open_utc": self.open_utc,
            "close_utc": self.close_utc,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LaunchWindow:
        """从字典创建发射窗口。"""
        return cls(**data)


@dataclass
class DeploymentStep:
    """单颗卫星的分离步骤。

    ``satellite_index`` 是 ``MissionScenario.payloads`` 的 1-based 序号；
    ``delay_s`` 相对上一步分离时刻，因此会累积成整个部署段时长。
    """

    satellite_index: int = 1
    delay_s: float = 60.0
    separation_delta_v_mps: float = 0.0
    satellite_name: str = ""
    target_slot: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        """验证分离步骤参数。"""
        if self.satellite_index < 1:
            raise ValueError("部署步骤的载荷序号必须从 1 开始")
        if self.delay_s < 0:
            raise ValueError("部署间隔不能为负")
        if self.separation_delta_v_mps < 0:
            raise ValueError("分离冲量不能为负")
        if self.target_slot < 0:
            raise ValueError("目标槽位编号不能为负")

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "satellite_index": self.satellite_index,
            "delay_s": self.delay_s,
            "separation_delta_v_mps": self.separation_delta_v_mps,
            "satellite_name": self.satellite_name,
            "target_slot": self.target_slot,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DeploymentStep:
        """从字典创建分离步骤。"""
        return cls(**data)


@dataclass
class DeploymentPlan:
    """部署顺序计划 — 载荷按 ``steps`` 顺序依次分离。"""

    name: str = "default_deployment"
    steps: list[DeploymentStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        """验证部署计划。"""
        if not self.name.strip():
            raise ValueError("部署计划名称不能为空")
        if not self.steps:
            raise ValueError("部署计划至少需要一个步骤")
        indices = [step.satellite_index for step in self.steps]
        if len(set(indices)) != len(indices):
            raise ValueError("部署计划中同一载荷不能出现两次")

    @property
    def total_delay_s(self) -> float:
        """部署段总时长 (s)。"""
        return sum(step.delay_s for step in self.steps)

    @property
    def total_separation_delta_v_mps(self) -> float:
        """分离冲量之和 (m/s)。"""
        return sum(step.separation_delta_v_mps for step in self.steps)

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> DeploymentPlan:
        """从字典创建部署计划。"""
        values = dict(data)
        values["steps"] = [DeploymentStep.from_dict(step) for step in values["steps"]]
        return cls(**values)


@dataclass
class MissionScenario:
    """一次发射、入轨并部署卫星的研究场景。

    ``constraints`` 为空时只执行内置的运载能力和速度预算检查；调用
    ``with_default_constraints()`` 或传入 ``default_constraints(scenario)``
    可以挂上推荐的完整约束集。发射窗口和部署计划都是可选的显式声明。
    """

    name: str = "deployment_demo"
    launch_vehicle: LaunchVehicle = field(default_factory=LaunchVehicle)
    payloads: list[SatelliteDesign] = field(default_factory=list)
    target_orbit: OrbitalTarget = field(default_factory=OrbitalTarget)
    launch_site: LaunchSite = field(default_factory=LaunchSite)
    launch_window: LaunchWindow | None = None
    launch_epoch_utc: str | None = None
    deployment_plan: DeploymentPlan | None = None
    constraints: list[MissionConstraint] = field(default_factory=list)
    gravity_loss_mps: float = 1500.0
    drag_loss_mps: float = 150.0
    steering_loss_mps: float = 100.0
    deployment_delay_s: float = 60.0
    assumptions: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """验证任务场景。"""
        if not self.name.strip():
            raise ValueError("任务名称不能为空")
        if not self.payloads:
            raise ValueError("任务至少需要一个载荷")
        if any(payload.total_mass_kg <= 0 for payload in self.payloads):
            raise ValueError("载荷总质量必须为正")
        if any(
            value < 0
            for value in (
                self.gravity_loss_mps,
                self.drag_loss_mps,
                self.steering_loss_mps,
                self.deployment_delay_s,
            )
        ):
            raise ValueError("任务损失和部署延迟不能为负")
        if self.launch_epoch_utc is not None:
            _parse_iso_utc(self.launch_epoch_utc, "launch_epoch_utc")
        if any(not isinstance(constraint, MissionConstraint) for constraint in self.constraints):
            raise ValueError("constraints 的元素必须是 MissionConstraint 实例")
        if self.deployment_plan is not None:
            payload_count = len(self.payloads)
            for step in self.deployment_plan.steps:
                if not 1 <= step.satellite_index <= payload_count:
                    raise ValueError(f"部署计划引用了不存在的载荷序号 {step.satellite_index}")
            if len(self.deployment_plan.steps) != payload_count:
                raise ValueError(f"部署计划必须覆盖全部 {payload_count} 个载荷")

    @property
    def payload_mass_kg(self) -> float:
        """载荷、整流罩和任务适配件之外的卫星总质量。"""
        return sum(payload.total_mass_kg for payload in self.payloads)

    @property
    def payload_volume_m3(self) -> float:
        """载荷收拢包络体积之和 (m³)。"""
        return sum(payload.envelope_volume_m3 for payload in self.payloads)

    @property
    def fairing_volume_margin_m3(self) -> float:
        """整流罩可用容积减载荷包络体积 (m³)，未给整流罩尺寸时为 0。"""
        fairing_volume_m3 = self.launch_vehicle.fairing_usable_volume_m3
        if fairing_volume_m3 <= 0:
            return 0.0
        return fairing_volume_m3 - self.payload_volume_m3

    def deployment_order(self) -> list[SatelliteDesign]:
        """按部署计划返回载荷顺序；未声明计划时按载荷列表顺序。"""
        if self.deployment_plan is None:
            return list(self.payloads)
        return [self.payloads[step.satellite_index - 1] for step in self.deployment_plan.steps]

    def deployment_intervals_s(self) -> list[float]:
        """每步分离间隔 (s)；未声明计划时每步都取 ``deployment_delay_s``。"""
        if self.deployment_plan is None:
            return [self.deployment_delay_s] * len(self.payloads)
        return [step.delay_s for step in self.deployment_plan.steps]

    def deployment_separation_delta_v_mps(self) -> list[float]:
        """每步分离冲量 (m/s)；未声明计划时全为 0。"""
        if self.deployment_plan is None:
            return [0.0] * len(self.payloads)
        return [step.separation_delta_v_mps for step in self.deployment_plan.steps]

    def with_default_constraints(self) -> MissionScenario:
        """挂上推荐默认约束集并返回自身，便于链式构造。"""
        self.constraints = default_constraints(self)
        return self

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "launch_vehicle": self.launch_vehicle.to_dict(),
            "payloads": [payload.to_dict() for payload in self.payloads],
            "target_orbit": self.target_orbit.to_dict(),
            "launch_site": self.launch_site.to_dict(),
            "launch_window": self.launch_window.to_dict() if self.launch_window else None,
            "launch_epoch_utc": self.launch_epoch_utc,
            "deployment_plan": (self.deployment_plan.to_dict() if self.deployment_plan else None),
            "constraints": [constraint_to_dict(constraint) for constraint in self.constraints],
            "gravity_loss_mps": self.gravity_loss_mps,
            "drag_loss_mps": self.drag_loss_mps,
            "steering_loss_mps": self.steering_loss_mps,
            "deployment_delay_s": self.deployment_delay_s,
            "assumptions": list(self.assumptions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MissionScenario:
        """从字典创建任务场景。"""
        values = dict(data)
        values["launch_vehicle"] = LaunchVehicle.from_dict(values["launch_vehicle"])
        values["payloads"] = [SatelliteDesign.from_dict(payload) for payload in values["payloads"]]
        values["target_orbit"] = OrbitalTarget.from_dict(values["target_orbit"])
        values["launch_site"] = LaunchSite.from_dict(values["launch_site"])
        launch_window = values.get("launch_window")
        if isinstance(launch_window, dict):
            values["launch_window"] = LaunchWindow.from_dict(launch_window)
        deployment_plan = values.get("deployment_plan")
        if isinstance(deployment_plan, dict):
            values["deployment_plan"] = DeploymentPlan.from_dict(deployment_plan)
        values["constraints"] = [
            constraint_from_dict(constraint) for constraint in values.get("constraints") or []
        ]
        return cls(**values)
