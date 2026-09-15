"""
卫星设计注册表 — 为用户和插件提供可扩展的卫星蓝图入口。

内置设计的参数全部来自公开的量级常识，属于"研究默认值"：它们足以支撑
接口联调、约束检查和教学演示，但**不**代表任何真实型号的性能承诺。
需要真实数据的用户应当注册自己的设计，并在 ``SatelliteDesign.assumptions``
中写清来源。
"""

from __future__ import annotations

from collections.abc import Callable

from .models import SatelliteDesign, VisualAppearance
from .subsystems import (
    AttitudeControlSystem,
    CommunicationSystem,
    PowerSystem,
    PropulsionSystem,
)


class SatelliteDesignRegistry:
    """按名称管理可复用的卫星设计。"""

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._designs: dict[str, Callable[[], SatelliteDesign]] = {}
        self._metadata: dict[str, dict[str, str]] = {}

    def register(
        self,
        name: str,
        *,
        description: str = "",
        category: str = "custom",
    ) -> Callable[[Callable[[], SatelliteDesign]], Callable[[], SatelliteDesign]]:
        """返回装饰器，注册一个卫星设计工厂。"""
        if not name.strip():
            raise ValueError("注册名不能为空")

        def decorator(
            factory: Callable[[], SatelliteDesign],
        ) -> Callable[[], SatelliteDesign]:
            self._designs[name] = factory
            self._metadata[name] = {
                "description": description,
                "category": category,
            }
            return factory

        return decorator

    def register_design(
        self,
        name: str,
        design: SatelliteDesign,
        *,
        description: str = "",
        category: str = "custom",
    ) -> None:
        """注册一个固定卫星设计。"""
        if not name.strip():
            raise ValueError("注册名不能为空")
        self._designs[name] = lambda: SatelliteDesign.from_dict(design.to_dict())
        self._metadata[name] = {
            "description": description,
            "category": category,
        }

    def get(self, name: str) -> SatelliteDesign:
        """获取卫星设计副本。"""
        try:
            factory = self._designs[name]
        except KeyError as error:
            available = ", ".join(sorted(self._designs)) or "(empty)"
            raise KeyError(f"未知卫星设计 '{name}'，可用设计: {available}") from error
        return factory()

    def list_designs(self, category: str | None = None) -> list[dict[str, str]]:
        """列出注册设计的元数据。"""
        return [
            {
                "name": name,
                "description": self._metadata[name].get("description", ""),
                "category": self._metadata[name].get("category", "custom"),
            }
            for name in sorted(self._designs)
            if category is None or self._metadata[name].get("category") == category
        ]

    def unregister(self, name: str) -> None:
        """移除一个设计。"""
        self._designs.pop(name, None)
        self._metadata.pop(name, None)

    def __contains__(self, name: str) -> bool:
        """判断名称是否已注册。"""
        return name in self._designs


satellite_designs = SatelliteDesignRegistry()


@satellite_designs.register(
    "cubesat-3u",
    description="用于教学和接口测试的 3U 立方星，含电源、通信、姿态和冷气推进",
    category="builtin",
)
def create_cubesat_3u() -> SatelliteDesign:
    """创建一个轻量 3U 立方星示例。

    质量口径：``dry_mass_kg`` 只含平台结构，分系统结构质量由各分系统各自
    声明，因此 :attr:`SatelliteDesign.total_mass_kg` 会再加上它们和推进剂。
    """
    return SatelliteDesign(
        name="cubesat-3u",
        dry_mass_kg=2.2,
        power_w=20.0,
        drag_coefficient=2.2,
        cross_section_area_m2=0.034,
        storage_gb=8.0,
        body_dimensions_m=(0.1, 0.1, 0.34),
        power_system=PowerSystem(
            name="3u_solar_battery_bus",
            solar_array_power_w=20.0,
            battery_capacity_wh=40.0,
            bus_voltage_v=12.0,
            average_load_w=12.0,
            peak_load_w=25.0,
            eclipse_fraction=0.35,
            mass_kg=0.8,
        ),
        communication_system=CommunicationSystem(
            name="uhf_transceiver",
            band="UHF",
            frequency_hz=437.0e6,
            transmit_power_w=2.0,
            transmit_antenna_gain_dbi=0.0,
            receive_antenna_gain_dbi=0.0,
            ground_station_gain_dbi=24.0,
            downlink_rate_mbps=1.0,
            uplink_rate_mbps=0.1,
            mass_kg=0.35,
            power_w=6.0,
        ),
        attitude_control_system=AttitudeControlSystem(
            name="3u_three_axis_wheels",
            mode="three_axis",
            pointing_accuracy_deg=0.5,
            slew_rate_deg_s=2.0,
            momentum_capacity_nms=0.05,
            reaction_wheel_count=3,
            mass_kg=0.6,
            power_w=3.0,
        ),
        propulsion_system=PropulsionSystem(
            name="cold_gas_thrusters",
            propellant_type="cold_gas",
            propellant_mass_kg=0.3,
            dry_mass_kg=0.4,
            thrust_n=0.1,
            specific_impulse_s=60.0,
            thruster_count=4,
            min_impulse_bit_ns=0.002,
        ),
        visual=VisualAppearance(
            primitive="box",
            dimensions_m=(0.1, 0.1, 0.34),
            color_hex="#d7a84f",
        ),
        assumptions=[
            "研究默认值：参数取自立方星公开量级，不代表具体型号验收数据",
        ],
        metadata={"purpose": "teaching"},
    )


@satellite_designs.register(
    "earth-obs-smallsat",
    description="对地观测小卫星研究蓝图，含 X 波段下行、三轴稳定和肼推进",
    category="builtin",
)
def create_earth_obs_smallsat() -> SatelliteDesign:
    """创建一个中等规模对地观测卫星示例。

    该蓝图是"分系统齐全"的参考对象，用来验证质量分解、容积约束、
    功率余量和推进剂速度增量在一条链路上可以闭环。
    """
    return SatelliteDesign(
        name="earth-obs-smallsat",
        dry_mass_kg=59.0,
        drag_coefficient=2.2,
        cross_section_area_m2=1.2,
        storage_gb=256.0,
        body_dimensions_m=(1.2, 1.0, 1.4),
        stowed_dimensions_m=(1.2, 1.0, 0.9),
        power_system=PowerSystem(
            name="deployable_solar_battery_bus",
            solar_array_power_w=500.0,
            battery_capacity_wh=800.0,
            bus_voltage_v=28.0,
            average_load_w=220.0,
            peak_load_w=380.0,
            eclipse_fraction=0.36,
            mass_kg=22.0,
        ),
        communication_system=CommunicationSystem(
            name="x_band_transceiver",
            band="X",
            frequency_hz=8.2e9,
            transmit_power_w=20.0,
            transmit_antenna_gain_dbi=28.0,
            receive_antenna_gain_dbi=26.0,
            ground_station_gain_dbi=45.0,
            downlink_rate_mbps=150.0,
            uplink_rate_mbps=4.0,
            inter_satellite_link=False,
            max_simultaneous_links=1,
            mass_kg=9.0,
            power_w=45.0,
        ),
        attitude_control_system=AttitudeControlSystem(
            name="four_wheel_three_axis",
            mode="three_axis",
            pointing_accuracy_deg=0.05,
            slew_rate_deg_s=1.5,
            momentum_capacity_nms=12.0,
            reaction_wheel_count=4,
            mass_kg=11.0,
            power_w=18.0,
        ),
        propulsion_system=PropulsionSystem(
            name="hydrazine_monoprop",
            propellant_type="hydrazine",
            propellant_mass_kg=18.0,
            dry_mass_kg=9.0,
            thrust_n=4.0,
            specific_impulse_s=230.0,
            thruster_count=4,
            min_impulse_bit_ns=0.05,
        ),
        visual=VisualAppearance(
            primitive="box",
            dimensions_m=(1.2, 1.0, 1.4),
            color_hex="#4aa3df",
        ),
        assumptions=[
            "研究默认值：分系统参数按小卫星公开量级给定，未引用具体型号手册",
        ],
        metadata={"purpose": "earth_observation"},
    )


__all__ = [
    "SatelliteDesignRegistry",
    "satellite_designs",
]
