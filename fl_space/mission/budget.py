"""
发射速度预算 — 把任务场景转换成所需/可用速度增量和余量。

``LaunchBudget`` 是速度预算的唯一口径：``MissionSimulator``、
``DeltaVMarginConstraint`` 和研究报告脚本都消费同一个对象，避免出现
"约束里算一套、模拟器里算另一套"的隐患。全部为一阶估算，不替代
真实制导、飞行动力学和结构分析。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .physics import (
    insertion_delta_v_terms_mps,
    staged_vehicle_delta_v_mps,
)

if TYPE_CHECKING:
    from .models import MissionScenario


@dataclass(frozen=True)
class LaunchBudget:
    """一次发射的质量容量和速度预算分解 (一阶估算)。"""

    payload_mass_kg: float
    payload_capacity_kg: float
    orbital_velocity_mps: float
    rotation_credit_mps: float
    plane_change_mps: float
    losses_mps: float
    required_delta_v_mps: float
    available_delta_v_mps: float

    @property
    def delta_v_margin_mps(self) -> float:
        """理想速度增量余量 (m/s)。"""
        return self.available_delta_v_mps - self.required_delta_v_mps

    @property
    def delta_v_margin_ratio(self) -> float:
        """余量与所需速度增量之比；所需为零时返回 0。"""
        if self.required_delta_v_mps <= 0:
            return 0.0
        return self.delta_v_margin_mps / self.required_delta_v_mps

    @property
    def payload_margin_kg(self) -> float:
        """有效载荷质量余量 (kg)。"""
        return self.payload_capacity_kg - self.payload_mass_kg

    @property
    def mass_capacity_exceeded(self) -> bool:
        """载荷质量是否超出运载能力。"""
        return self.payload_mass_kg > self.payload_capacity_kg

    @property
    def delta_v_exceeded(self) -> bool:
        """可用速度增量是否低于所需速度增量。"""
        return self.available_delta_v_mps < self.required_delta_v_mps

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "payload_mass_kg": self.payload_mass_kg,
            "payload_capacity_kg": self.payload_capacity_kg,
            "payload_margin_kg": self.payload_margin_kg,
            "orbital_velocity_mps": self.orbital_velocity_mps,
            "rotation_credit_mps": self.rotation_credit_mps,
            "plane_change_mps": self.plane_change_mps,
            "losses_mps": self.losses_mps,
            "required_delta_v_mps": self.required_delta_v_mps,
            "available_delta_v_mps": self.available_delta_v_mps,
            "delta_v_margin_mps": self.delta_v_margin_mps,
            "delta_v_margin_ratio": self.delta_v_margin_ratio,
        }


def compute_launch_budget(scenario: MissionScenario) -> LaunchBudget:
    """计算场景的质量容量和速度预算 (一阶)。

    载荷质量包含整流罩，因为整流罩在入轨前同样是末级必须推动的质量。
    """
    payload_mass_kg = scenario.payload_mass_kg + scenario.launch_vehicle.fairing_mass_kg
    orbital_velocity_mps, rotation_credit_mps, plane_change_mps, losses_mps = (
        insertion_delta_v_terms_mps(
            scenario.target_orbit.altitude_km,
            scenario.target_orbit.inclination_deg,
            scenario.launch_site.latitude_deg,
            gravity_loss_mps=scenario.gravity_loss_mps,
            drag_loss_mps=scenario.drag_loss_mps,
            steering_loss_mps=scenario.steering_loss_mps,
        )
    )
    stage_specs = [
        (stage.dry_mass_kg, stage.propellant_mass_kg, stage.specific_impulse_s)
        for stage in scenario.launch_vehicle.stages
    ]
    return LaunchBudget(
        payload_mass_kg=payload_mass_kg,
        payload_capacity_kg=scenario.launch_vehicle.payload_capacity_kg,
        orbital_velocity_mps=orbital_velocity_mps,
        rotation_credit_mps=rotation_credit_mps,
        plane_change_mps=plane_change_mps,
        losses_mps=losses_mps,
        required_delta_v_mps=(
            orbital_velocity_mps + losses_mps + plane_change_mps - rotation_credit_mps
        ),
        available_delta_v_mps=staged_vehicle_delta_v_mps(stage_specs, payload_mass_kg),
    )


__all__ = [
    "LaunchBudget",
    "compute_launch_budget",
]
