"""
卫星分系统模型 — 电源、通信、姿态控制和推进。

本模块只描述平台工程参数，不包含任何渲染信息，也不参与视觉外观推导。
所有单位遵循任务层 SI 约定：

    - 质量: kg
    - 功率: W
    - 能量: Wh
    - 推力: N
    - 比冲: s
    - 角度: deg
    - 数据量: MB / Mb/s

派生属性都是一阶工程估算，用于方案比较、约束检查和教学；不替代
热真空试验、详细链路计算、六自由度姿态动力学或推进剂预算软件。
每个分系统通过 ``mass_kg`` 暴露自身结构质量，推进剂由推进系统单独给出，
便于上层的质量汇总保持"干质量 + 推进剂"的可解释结构。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from .physics import STANDARD_GRAVITY_MPS2, tsiolkovsky_delta_v_mps

PropellantType = Literal[
    "none",
    "cold_gas",
    "hydrazine",
    "green_monoprop",
    "bipropellant",
    "electric",
]
CommunicationBand = Literal["UHF", "L", "S", "X", "Ka", "optical"]
CommunicationLinkKind = Literal["space_to_ground", "inter_satellite", "crosslink"]
AttitudeControlMode = Literal["none", "spin", "gravity_gradient", "three_axis"]


@dataclass
class PowerSystem:
    """电源分系统 — 发电、储能与负载。

    ``solar_array_power_w`` 是寿命末期太阳翼输出功率，``average_load_w``
    是平台与载荷的平均功耗。轨道平均功率按 (1 - 蚀期占比) 折算，
    蚀期能量需求由 ``eclipse_energy_wh`` 给出。
    """

    name: str = "solar_battery_bus"
    solar_array_power_w: float = 0.0
    battery_capacity_wh: float = 0.0
    bus_voltage_v: float = 28.0
    average_load_w: float = 0.0
    peak_load_w: float = 0.0
    eclipse_fraction: float = 0.35
    mass_kg: float = 0.0

    def __post_init__(self) -> None:
        """验证电源参数。"""
        if not self.name.strip():
            raise ValueError("电源分系统名称不能为空")
        if any(
            value < 0
            for value in (
                self.solar_array_power_w,
                self.battery_capacity_wh,
                self.average_load_w,
                self.peak_load_w,
                self.mass_kg,
            )
        ):
            raise ValueError("电源分系统功率、容量、负载和质量不能为负")
        if self.bus_voltage_v <= 0:
            raise ValueError("母线电压必须为正")
        if not 0.0 <= self.eclipse_fraction < 1.0:
            raise ValueError("蚀期占比必须在 [0, 1) 之间")
        if self.peak_load_w and self.peak_load_w < self.average_load_w:
            raise ValueError("峰值负载不能低于平均负载")

    @property
    def orbit_average_power_w(self) -> float:
        """轨道平均发电功率 (W)，按蚀期占比线性折算。"""
        return self.solar_array_power_w * (1.0 - self.eclipse_fraction)

    @property
    def power_margin_w(self) -> float:
        """轨道平均功率与平均负载之差 (W)，可为负。"""
        return self.orbit_average_power_w - self.average_load_w

    def eclipse_energy_wh(self, orbital_period_s: float) -> float:
        """单个轨道周期内蚀期负载能耗 (Wh)。"""
        if orbital_period_s <= 0:
            raise ValueError("轨道周期必须为正")
        return self.average_load_w * self.eclipse_fraction * orbital_period_s / 3600.0

    def battery_margin_ratio(self, orbital_period_s: float) -> float:
        """电池容量与蚀期能耗之比；无蚀期负载时返回 ``math.inf``。"""
        eclipse_energy_wh = self.eclipse_energy_wh(orbital_period_s)
        if eclipse_energy_wh <= 0:
            return math.inf
        return self.battery_capacity_wh / eclipse_energy_wh

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "solar_array_power_w": self.solar_array_power_w,
            "battery_capacity_wh": self.battery_capacity_wh,
            "bus_voltage_v": self.bus_voltage_v,
            "average_load_w": self.average_load_w,
            "peak_load_w": self.peak_load_w,
            "eclipse_fraction": self.eclipse_fraction,
            "mass_kg": self.mass_kg,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PowerSystem:
        """从字典创建电源分系统。"""
        return cls(**data)


@dataclass
class CommunicationSystem:
    """通信分系统 — 星地链路与可选星间链路。

    ``band``、``frequency_hz`` 和发射功率描述星地链路；``inter_satellite_link``
    为真时必须给出 ``isl_rate_mbps``。缺少官方工程参数时，调用方应把
    来源和假设写入 ``SatelliteDesign.assumptions``，不要把宣传峰值当作
    持续可用带宽。
    """

    name: str = "s_band_transceiver"
    link_kind: CommunicationLinkKind = "space_to_ground"
    band: CommunicationBand = "S"
    frequency_hz: float = 2.2e9
    channel_bandwidth_mhz: float = 0.0
    transmit_power_w: float = 5.0
    transmit_antenna_gain_dbi: float = 3.0
    receive_antenna_gain_dbi: float = 3.0
    ground_station_gain_dbi: float = 30.0
    downlink_rate_mbps: float = 10.0
    uplink_rate_mbps: float = 1.0
    inter_satellite_link: bool = False
    isl_rate_mbps: float = 0.0
    max_simultaneous_links: int = 1
    terminal_count: int = 1
    max_range_km: float = 0.0
    acquisition_time_s: float = 0.0
    pointing_accuracy_deg: float = 0.0
    weather_sensitive: bool = False
    mass_kg: float = 0.0
    power_w: float = 0.0
    source: str = "assumption"

    def __post_init__(self) -> None:
        """验证通信参数。"""
        if not self.name.strip():
            raise ValueError("通信分系统名称不能为空")
        if self.frequency_hz <= 0:
            raise ValueError("通信频率必须为正")
        if self.channel_bandwidth_mhz < 0:
            raise ValueError("信道带宽不能为负")
        if self.transmit_power_w <= 0:
            raise ValueError("发射功率必须为正")
        if self.downlink_rate_mbps <= 0 or self.uplink_rate_mbps <= 0:
            raise ValueError("上下行速率必须为正")
        if self.max_simultaneous_links < 1:
            raise ValueError("并发链路数至少为 1")
        if self.terminal_count < 1:
            raise ValueError("通信终端数量至少为 1")
        if self.inter_satellite_link and self.isl_rate_mbps <= 0:
            raise ValueError("启用星间链路时必须给出 isl_rate_mbps")
        if any(
            value < 0
            for value in (
                self.isl_rate_mbps,
                self.max_range_km,
                self.acquisition_time_s,
                self.pointing_accuracy_deg,
                self.mass_kg,
                self.power_w,
            )
        ):
            raise ValueError("链路速率、距离、捕获时间、指向精度、质量和功耗不能为负")

    @property
    def eirp_dbw(self) -> float:
        """等效全向辐射功率 EIRP (dBW)。"""
        return 10.0 * math.log10(self.transmit_power_w) + self.transmit_antenna_gain_dbi

    def downlink_time_s(self, data_mb: float) -> float:
        """下传给定数据量所需时间 (s)，不含建链和重传。"""
        if data_mb < 0:
            raise ValueError("数据量不能为负")
        return data_mb * 8.0 / self.downlink_rate_mbps

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "link_kind": self.link_kind,
            "band": self.band,
            "frequency_hz": self.frequency_hz,
            "channel_bandwidth_mhz": self.channel_bandwidth_mhz,
            "transmit_power_w": self.transmit_power_w,
            "transmit_antenna_gain_dbi": self.transmit_antenna_gain_dbi,
            "receive_antenna_gain_dbi": self.receive_antenna_gain_dbi,
            "ground_station_gain_dbi": self.ground_station_gain_dbi,
            "downlink_rate_mbps": self.downlink_rate_mbps,
            "uplink_rate_mbps": self.uplink_rate_mbps,
            "inter_satellite_link": self.inter_satellite_link,
            "isl_rate_mbps": self.isl_rate_mbps,
            "max_simultaneous_links": self.max_simultaneous_links,
            "terminal_count": self.terminal_count,
            "max_range_km": self.max_range_km,
            "acquisition_time_s": self.acquisition_time_s,
            "pointing_accuracy_deg": self.pointing_accuracy_deg,
            "weather_sensitive": self.weather_sensitive,
            "mass_kg": self.mass_kg,
            "power_w": self.power_w,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CommunicationSystem:
        """从字典创建通信分系统。"""
        return cls(**data)


@dataclass
class AttitudeControlSystem:
    """姿态控制分系统 — 指向精度、机动能力和角动量储备。"""

    name: str = "three_axis_reaction_wheels"
    mode: AttitudeControlMode = "three_axis"
    pointing_accuracy_deg: float = 0.1
    slew_rate_deg_s: float = 1.0
    momentum_capacity_nms: float = 5.0
    reaction_wheel_count: int = 3
    mass_kg: float = 0.0
    power_w: float = 0.0

    def __post_init__(self) -> None:
        """验证姿态控制参数。"""
        if not self.name.strip():
            raise ValueError("姿态控制分系统名称不能为空")
        if self.mode == "none":
            if self.mass_kg < 0 or self.power_w < 0:
                raise ValueError("质量和功耗不能为负")
            return
        if self.pointing_accuracy_deg <= 0:
            raise ValueError("指向精度必须为正")
        if self.slew_rate_deg_s <= 0:
            raise ValueError("机动角速度必须为正")
        if self.momentum_capacity_nms <= 0:
            raise ValueError("角动量容量必须为正")
        if self.reaction_wheel_count < 1:
            raise ValueError("动量轮数量至少为 1")
        if self.mode == "three_axis" and self.reaction_wheel_count < 3:
            raise ValueError("三轴稳定至少需要 3 个动量轮")
        if self.mass_kg < 0 or self.power_w < 0:
            raise ValueError("质量和功耗不能为负")

    @property
    def is_three_axis(self) -> bool:
        """是否为三轴稳定控制。"""
        return self.mode == "three_axis"

    def slew_time_s(self, angle_deg: float) -> float:
        """按恒定角速度估算机动时间 (s)，忽略加减速段。"""
        if angle_deg < 0:
            raise ValueError("机动角度不能为负")
        if self.mode == "none":
            raise ValueError("未配置姿态控制分系统，无法估算机动时间")
        return angle_deg / self.slew_rate_deg_s

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "mode": self.mode,
            "pointing_accuracy_deg": self.pointing_accuracy_deg,
            "slew_rate_deg_s": self.slew_rate_deg_s,
            "momentum_capacity_nms": self.momentum_capacity_nms,
            "reaction_wheel_count": self.reaction_wheel_count,
            "mass_kg": self.mass_kg,
            "power_w": self.power_w,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AttitudeControlSystem:
        """从字典创建姿态控制分系统。"""
        return cls(**data)


@dataclass
class PropulsionSystem:
    """推进分系统 — 推进剂、推力器和速度增量能力。

    ``dry_mass_kg`` 是推力器、贮箱和管路的结构质量，通过 ``mass_kg``
    暴露给上层的干质量汇总；``propellant_mass_kg`` 是可用推进剂，
    由上层单独计入总质量，避免干质量与推进剂混在一起。
    """

    name: str = "monoprop_hydrazine"
    propellant_type: PropellantType = "hydrazine"
    propellant_mass_kg: float = 0.0
    dry_mass_kg: float = 0.0
    thrust_n: float = 1.0
    specific_impulse_s: float = 220.0
    thruster_count: int = 1
    min_impulse_bit_ns: float = 0.0

    def __post_init__(self) -> None:
        """验证推进参数。"""
        if not self.name.strip():
            raise ValueError("推进分系统名称不能为空")
        if self.propellant_mass_kg < 0 or self.dry_mass_kg < 0:
            raise ValueError("推进剂和结构质量不能为负")
        if self.propellant_type == "none" and self.propellant_mass_kg > 0:
            raise ValueError("propellant_type 为 none 时不能携带推进剂")
        if self.thruster_count < 1:
            raise ValueError("推力器数量至少为 1")
        if self.propellant_type != "none":
            if self.thrust_n <= 0:
                raise ValueError("推力必须为正")
            if self.specific_impulse_s <= 0:
                raise ValueError("比冲必须为正")
        if self.min_impulse_bit_ns < 0:
            raise ValueError("最小冲量不能为负")

    @property
    def mass_kg(self) -> float:
        """推进系统结构质量 (kg)，不含推进剂。"""
        return self.dry_mass_kg

    @property
    def wet_mass_kg(self) -> float:
        """推进系统含推进剂质量 (kg)。"""
        return self.dry_mass_kg + self.propellant_mass_kg

    @property
    def total_thrust_n(self) -> float:
        """全部推力器同时工作的总推力 (N)。"""
        return self.thrust_n * self.thruster_count

    @property
    def propellant_mass_fraction(self) -> float:
        """推进剂占本分系统湿质量的比例。"""
        if self.wet_mass_kg <= 0:
            return 0.0
        return self.propellant_mass_kg / self.wet_mass_kg

    def mass_flow_kg_s(self, *, throttle: float = 1.0) -> float:
        """按节流比估算推进剂流量 (kg/s)。"""
        if not 0.0 < throttle <= 1.0:
            raise ValueError("节流比必须在 (0, 1]")
        if self.propellant_type == "none":
            raise ValueError("未配置推进系统，无法估算流量")
        return throttle * self.total_thrust_n / (self.specific_impulse_s * STANDARD_GRAVITY_MPS2)

    def burn_time_s(
        self, propellant_mass_kg: float | None = None, *, throttle: float = 1.0
    ) -> float:
        """燃烧给定推进剂所需时间 (s)。"""
        available = self.propellant_mass_kg if propellant_mass_kg is None else propellant_mass_kg
        if available < 0:
            raise ValueError("推进剂质量不能为负")
        return available / self.mass_flow_kg_s(throttle=throttle)

    def available_delta_v_mps(self, spacecraft_dry_mass_kg: float) -> float:
        """由齐奥尔科夫斯基公式估算可用速度增量 (m/s)。

        Parameters
        ----------
        spacecraft_dry_mass_kg : float
            不含推进剂的航天器干质量 (kg)，必须包含本推进系统的结构质量。
        """
        if spacecraft_dry_mass_kg <= 0:
            raise ValueError("航天器干质量必须为正")
        if self.propellant_mass_kg <= 0 or self.propellant_type == "none":
            return 0.0
        if self.specific_impulse_s <= 0:
            raise ValueError("比冲必须为正")
        return tsiolkovsky_delta_v_mps(
            spacecraft_dry_mass_kg + self.propellant_mass_kg,
            spacecraft_dry_mass_kg,
            self.specific_impulse_s,
        )

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "propellant_type": self.propellant_type,
            "propellant_mass_kg": self.propellant_mass_kg,
            "dry_mass_kg": self.dry_mass_kg,
            "thrust_n": self.thrust_n,
            "specific_impulse_s": self.specific_impulse_s,
            "thruster_count": self.thruster_count,
            "min_impulse_bit_ns": self.min_impulse_bit_ns,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PropulsionSystem:
        """从字典创建推进分系统。"""
        return cls(**data)


__all__ = [
    "AttitudeControlMode",
    "AttitudeControlSystem",
    "CommunicationBand",
    "CommunicationLinkKind",
    "CommunicationSystem",
    "PowerSystem",
    "PropellantType",
    "PropulsionSystem",
]
