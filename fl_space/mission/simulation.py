"""
发射任务模拟器 — 用离散事件连接火箭、入轨和卫星部署。

当前实现是研究/教学级一阶模型，重点是提供稳定的扩展接口：
后续可以在同一结果协议下接入六自由度、真实气象或高保真轨迹传播器。

模拟器本身不重新推导物理量，而是消费两个外部唯一的判定口径：

    - ``mission.budget.compute_launch_budget``: 质量容量与速度预算；
    - ``mission.constraints``: 点火前后必须满足的任务约束。

这样"约束里算一套、模拟器里算另一套"的结构性风险被消除。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from fl_space.environment import CelestialBody
from fl_space.orbit import KeplerOrbit, OrbitalElements

from .budget import LaunchBudget, compute_launch_budget
from .constraints import (
    ConstraintResult,
    ConstraintSeverity,
    MissionConstraint,
    builtin_gate_constraints,
)
from .models import MissionScenario, SatelliteDesign

FIDELITY_LABEL = "first_order"
"""本模块的保真度标签：一阶离散事件模型。"""


class MissionPhase(str, Enum):
    """任务阶段。"""

    COUNTDOWN = "countdown"
    ASCENT = "ascent"
    ORBITAL_INSERTION = "orbital_insertion"
    DEPLOYMENT = "deployment"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class MissionEvent:
    """任务事件记录。

    ``satellite_id`` 只在部署事件上有值；``severity`` 只在约束相关事件上有值。
    """

    phase: MissionPhase
    time_s: float
    message: str
    satellite_id: str = ""
    severity: ConstraintSeverity | None = None

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "phase": self.phase.value,
            "time_s": self.time_s,
            "message": self.message,
            "satellite_id": self.satellite_id,
            "severity": self.severity.value if self.severity is not None else None,
        }


@dataclass
class DeployedSatellite:
    """已部署卫星及其轨道入口。

    ``deployment_order_index`` 是分离顺序，``satellite_index`` 是它在
    ``MissionScenario.payloads`` 中的序号，二者在自定义部署顺序下可以不同。
    """

    satellite_id: str
    design: SatelliteDesign
    orbit: KeplerOrbit
    satellite_index: int = 1
    deployment_order_index: int = 1
    deployed_at_s: float = 0.0
    separation_delta_v_mps: float = 0.0
    target_slot: int = 0

    def to_dict(self) -> dict:
        """序列化卫星部署结果的元数据。"""
        return {
            "satellite_id": self.satellite_id,
            "satellite_index": self.satellite_index,
            "deployment_order_index": self.deployment_order_index,
            "deployed_at_s": self.deployed_at_s,
            "separation_delta_v_mps": self.separation_delta_v_mps,
            "target_slot": self.target_slot,
            "design": self.design.to_dict(),
            "orbit": self.orbit.elements.to_dict(),
        }


@dataclass
class MissionResult:
    """发射任务模拟结果。

    ``budget`` 是本次任务唯一的质量/速度预算口径，``constraint_results``
    记录所有被评估约束的实际值和判定；``fidelity`` 标明结果的保真度，
    避免一阶估算被误当成飞行认证数据。
    """

    scenario_name: str
    success: bool
    phase: MissionPhase
    fidelity: str
    budget: LaunchBudget
    payload_mass_kg: float
    elapsed_time_s: float
    events: list[MissionEvent] = field(default_factory=list)
    deployed_satellites: list[DeployedSatellite] = field(default_factory=list)
    constraint_results: list[ConstraintResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def required_delta_v_mps(self) -> float:
        """入轨所需速度增量 (m/s)。"""
        return self.budget.required_delta_v_mps

    @property
    def available_delta_v_mps(self) -> float:
        """运载火箭可提供的理想速度增量 (m/s)。"""
        return self.budget.available_delta_v_mps

    @property
    def delta_v_margin_mps(self) -> float:
        """理想速度增量余量 (m/s)。"""
        return self.budget.delta_v_margin_mps

    @property
    def blocking_constraints(self) -> list[ConstraintResult]:
        """全部未满足且严重程度为 ``ERROR`` 的约束。"""
        return [
            result
            for result in self.constraint_results
            if not result.satisfied and result.severity is ConstraintSeverity.ERROR
        ]

    @property
    def advisory_constraints(self) -> list[ConstraintResult]:
        """未满足但不阻塞任务的约束（``WARNING`` / ``INFO``）。"""
        return [
            result
            for result in self.constraint_results
            if not result.satisfied and result.severity is not ConstraintSeverity.ERROR
        ]

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "scenario_name": self.scenario_name,
            "success": self.success,
            "phase": self.phase.value,
            "fidelity": self.fidelity,
            "required_delta_v_mps": self.required_delta_v_mps,
            "available_delta_v_mps": self.available_delta_v_mps,
            "delta_v_margin_mps": self.delta_v_margin_mps,
            "payload_mass_kg": self.payload_mass_kg,
            "elapsed_time_s": self.elapsed_time_s,
            "budget": self.budget.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "deployed_satellites": [satellite.to_dict() for satellite in self.deployed_satellites],
            "constraint_results": [result.to_dict() for result in self.constraint_results],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _DeploymentStepSpec:
    """部署顺序的一步，由场景的 ``DeploymentPlan`` 或默认间隔归一化而来。"""

    satellite_index: int
    delay_s: float
    separation_delta_v_mps: float
    target_slot: int


def _deployment_sequence(scenario: MissionScenario) -> list[_DeploymentStepSpec]:
    """把场景归一化成统一的部署顺序序列。

    未声明 ``deployment_plan`` 时按载荷列表顺序部署，每步间隔取
    ``scenario.deployment_delay_s``，且不施加分离冲量。
    """
    if scenario.deployment_plan is not None:
        return [
            _DeploymentStepSpec(
                satellite_index=step.satellite_index,
                delay_s=step.delay_s,
                separation_delta_v_mps=step.separation_delta_v_mps,
                target_slot=step.target_slot,
            )
            for step in scenario.deployment_plan.steps
        ]
    return [
        _DeploymentStepSpec(
            satellite_index=index,
            delay_s=scenario.deployment_delay_s,
            separation_delta_v_mps=0.0,
            target_slot=0,
        )
        for index in range(1, len(scenario.payloads) + 1)
    ]


class MissionSimulator:
    """连接发射能力估算和轨道部署的轻量任务模拟器。"""

    def __init__(
        self,
        body: CelestialBody | None = None,
        *,
        countdown_s: float = 10.0,
        ascent_s: float = 480.0,
        insertion_s: float = 90.0,
    ) -> None:
        """初始化任务模拟器。"""
        self.body = body or CelestialBody.earth()
        self.countdown_s = countdown_s
        self.ascent_s = ascent_s
        self.insertion_s = insertion_s
        if any(value < 0 for value in (countdown_s, ascent_s, insertion_s)):
            raise ValueError("任务阶段时长不能为负")

    def simulate(self, scenario: MissionScenario) -> MissionResult:
        """执行一次发射、入轨和部署模拟。

        流程为：约束评估 → 倒计时 → 上升 → 入轨 → 按部署顺序分离 → 完成。
        内置门槛约束（运载能力、速度预算）始终执行；场景用同名约束显式
        声明时可以覆盖默认阈值，不会出现重复评估。
        """
        expected = self._evaluate_constraints(scenario)
        budget = compute_launch_budget(scenario)
        payload_mass_kg = budget.payload_mass_kg
        events = [MissionEvent(MissionPhase.COUNTDOWN, 0.0, "倒计时开始")]
        warnings = ["结果来自一阶速度预算，未模拟真实制导、姿态、结构和飞控约束"]
        warnings.extend(
            result.message
            for _, result in expected
            if not result.satisfied and result.severity is ConstraintSeverity.WARNING
        )

        preflight_failures = [
            (constraint, result)
            for constraint, result in expected
            if not result.satisfied
            and result.severity is ConstraintSeverity.ERROR
            and constraint.preflight_gate
        ]
        if preflight_failures:
            return self._failure(
                scenario,
                budget,
                payload_mass_kg,
                events,
                expected,
                warnings,
                preflight_failures,
                elapsed_time_s=0.0,
                phase=MissionPhase.COUNTDOWN,
            )

        elapsed = self.countdown_s + self.ascent_s
        events.append(
            MissionEvent(
                MissionPhase.ASCENT,
                self.countdown_s,
                "火箭离开发射台，执行一阶上升段估算",
            )
        )

        ascent_failures = [
            (constraint, result)
            for constraint, result in expected
            if not result.satisfied
            and result.severity is ConstraintSeverity.ERROR
            and not constraint.preflight_gate
        ]
        if ascent_failures:
            return self._failure(
                scenario,
                budget,
                payload_mass_kg,
                events,
                expected,
                warnings,
                ascent_failures,
                elapsed_time_s=elapsed,
                phase=MissionPhase.ASCENT,
            )

        events.append(
            MissionEvent(
                MissionPhase.ORBITAL_INSERTION,
                elapsed,
                f"达到目标轨道速度预算，执行入轨（余量 {budget.delta_v_margin_mps:.1f} m/s）",
            )
        )
        elapsed += self.insertion_s
        sequence = _deployment_sequence(scenario)
        events.append(
            MissionEvent(
                MissionPhase.DEPLOYMENT,
                elapsed,
                f"开始部署 {len(sequence)} 个卫星载荷",
            )
        )
        deployed: list[DeployedSatellite] = []
        for order_index, step in enumerate(sequence, start=1):
            elapsed += step.delay_s
            design = scenario.payloads[step.satellite_index - 1]
            satellite = self._deploy_satellite(
                scenario,
                design,
                step,
                order_index,
                deployed_at_s=elapsed,
            )
            deployed.append(satellite)
            events.append(
                MissionEvent(
                    MissionPhase.DEPLOYMENT,
                    elapsed,
                    f"第 {order_index}/{len(sequence)} 步分离 {design.name}"
                    f"（质量 {design.total_mass_kg:.1f} kg，"
                    f"分离冲量 {step.separation_delta_v_mps:.2f} m/s）",
                    satellite_id=satellite.satellite_id,
                )
            )
        events.append(MissionEvent(MissionPhase.COMPLETE, elapsed, "任务完成"))
        return MissionResult(
            scenario_name=scenario.name,
            success=True,
            phase=MissionPhase.COMPLETE,
            fidelity=FIDELITY_LABEL,
            budget=budget,
            payload_mass_kg=payload_mass_kg,
            elapsed_time_s=elapsed,
            events=events,
            deployed_satellites=deployed,
            constraint_results=[result for _, result in expected],
            warnings=warnings,
        )

    def _evaluate_constraints(
        self,
        scenario: MissionScenario,
    ) -> list[tuple[MissionConstraint, ConstraintResult]]:
        """评估内置门槛约束和场景声明的约束。

        内置门槛只补充"场景没有声明显式约束"的类型，因此使用者可以用
        自己的 ``PayloadMassConstraint`` 覆盖默认阈值和严重程度，而不会
        出现同一约束被评估两次。
        """
        declared = list(scenario.constraints)
        declared_kinds = {constraint.kind for constraint in declared}
        gates = [
            constraint
            for constraint in builtin_gate_constraints(scenario)
            if constraint.kind not in declared_kinds
        ]
        return [(constraint, constraint.evaluate(scenario)) for constraint in [*gates, *declared]]

    def _failure(
        self,
        scenario: MissionScenario,
        budget: LaunchBudget,
        payload_mass_kg: float,
        events: list[MissionEvent],
        expected: list[tuple[MissionConstraint, ConstraintResult]],
        warnings: list[str],
        failures: list[tuple[MissionConstraint, ConstraintResult]],
        *,
        elapsed_time_s: float,
        phase: MissionPhase,
    ) -> MissionResult:
        """构造失败结果，并把每条未满足约束写进事件流。"""
        for _constraint, result in expected:
            if not result.satisfied and result.severity is not ConstraintSeverity.INFO:
                events.append(
                    MissionEvent(
                        phase,
                        elapsed_time_s,
                        result.message,
                        severity=result.severity,
                    )
                )
        summary = "；".join(result.message for _, result in failures)
        events.append(MissionEvent(MissionPhase.FAILED, elapsed_time_s, summary))
        return MissionResult(
            scenario_name=scenario.name,
            success=False,
            phase=MissionPhase.FAILED,
            fidelity=FIDELITY_LABEL,
            budget=budget,
            payload_mass_kg=payload_mass_kg,
            elapsed_time_s=elapsed_time_s,
            events=events,
            deployed_satellites=[],
            constraint_results=[result for _, result in expected],
            warnings=[*warnings, *[result.message for _, result in failures]],
        )

    def _deploy_satellite(
        self,
        scenario: MissionScenario,
        design: SatelliteDesign,
        step: _DeploymentStepSpec,
        order_index: int,
        *,
        deployed_at_s: float,
    ) -> DeployedSatellite:
        """把卫星设计转换成现有轨道后端可消费的对象。"""
        target = scenario.target_orbit
        elements = OrbitalElements(
            semi_major_axis_km=self.body.radius_km + target.altitude_km,
            eccentricity=target.eccentricity,
            inclination_deg=target.inclination_deg,
            raan_deg=target.raan_deg,
            arg_perigee_deg=target.arg_perigee_deg,
            true_anomaly_deg=target.true_anomaly_deg,
        )
        return DeployedSatellite(
            satellite_id=f"{scenario.name}-sat-{step.satellite_index:03d}",
            design=design,
            orbit=KeplerOrbit(elements, self.body),
            satellite_index=step.satellite_index,
            deployment_order_index=order_index,
            deployed_at_s=deployed_at_s,
            separation_delta_v_mps=step.separation_delta_v_mps,
            target_slot=step.target_slot,
        )


__all__ = [
    "FIDELITY_LABEL",
    "DeployedSatellite",
    "MissionEvent",
    "MissionPhase",
    "MissionResult",
    "MissionSimulator",
]
