"""
任务约束 — 在"点火前"检查质量、容积、速度预算、方位角、窗口和部署顺序。

约束是一阶可行性判断，回答的是"这套配置值不值得发射"：

    - ``ERROR``: 违反后任务应当判定失败；
    - ``WARNING``: 需要人工确认，不阻塞模拟；
    - ``INFO``: 未建模信息或说明性记录。

约束只读取 ``MissionScenario`` 的公开字段并通过 ``mission.physics`` /
``mission.budget`` 计算，不写回场景、不修改物理公式。经营规则、任务模式和
用户插件可以注册自己的约束类型，而不要改写核心判定。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from .budget import compute_launch_budget
from .physics import launch_azimuth_deg

if TYPE_CHECKING:
    from .models import MissionScenario


class ConstraintSeverity(str, Enum):
    """约束严重程度。"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ConstraintResult:
    """单条约束的评估结果。"""

    kind: str
    severity: ConstraintSeverity
    satisfied: bool
    message: str
    actual: float | None = None
    limit: float | None = None

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "kind": self.kind,
            "severity": self.severity.value,
            "satisfied": self.satisfied,
            "message": self.message,
            "actual": self.actual,
            "limit": self.limit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConstraintResult:
        """从字典创建结果。"""
        values = dict(data)
        values["severity"] = ConstraintSeverity(values["severity"])
        return cls(**values)


class MissionConstraint(ABC):
    """任务约束基类。

    子类必须是 ``dataclass``，字段全部带默认值，并通过
    ``register_constraint`` 注册，才能参与 ``to_dict()`` / ``from_dict()``
    往返序列化。

    ``preflight_gate`` 说明违反该约束会在哪个阶段被判定：``True`` 表示
    必须在点火前满足（例如质量容量），``False`` 表示故障直到上升段结束
    才显现（例如速度增量不足）。``MissionSimulator`` 用它决定失败时刻。
    """

    kind: ClassVar[str] = "mission_constraint"
    default_severity: ClassVar[ConstraintSeverity] = ConstraintSeverity.ERROR
    preflight_gate: ClassVar[bool] = True
    severity: ConstraintSeverity

    @property
    def name(self) -> str:
        """约束名称。"""
        return self.kind

    @abstractmethod
    def evaluate(self, scenario: MissionScenario) -> ConstraintResult:
        """评估约束并返回结果。"""

    def parameters(self) -> dict:
        """返回参与序列化的构造参数（不含 severity）。"""
        if not is_dataclass(self):
            return {}
        return {
            item.name: getattr(self, item.name) for item in fields(self) if item.name != "severity"
        }

    def result(
        self,
        *,
        satisfied: bool,
        message: str,
        actual: float | None = None,
        limit: float | None = None,
        severity: ConstraintSeverity | None = None,
    ) -> ConstraintResult:
        """构造本约束的结果对象。"""
        return ConstraintResult(
            kind=self.kind,
            severity=severity or self.severity,
            satisfied=satisfied,
            message=message,
            actual=actual,
            limit=limit,
        )

    def __str__(self) -> str:
        """可读名称，便于日志和报告。"""
        return f"{self.kind}[{self.severity.value}]"


CONSTRAINT_TYPES: dict[str, type[MissionConstraint]] = {}


def register_constraint(
    constraint_class: type[MissionConstraint],
) -> type[MissionConstraint]:
    """注册约束类型，供序列化反查使用。"""
    kind = constraint_class.kind
    if not kind or kind == MissionConstraint.kind:
        raise ValueError("约束子类必须定义非默认的 kind")
    if kind in CONSTRAINT_TYPES:
        raise ValueError(f"约束类型 '{kind}' 已注册")
    CONSTRAINT_TYPES[kind] = constraint_class
    return constraint_class


def constraint_to_dict(constraint: MissionConstraint) -> dict:
    """把约束序列化为带类型标记的字典。"""
    return {
        "kind": constraint.kind,
        "severity": constraint.severity.value,
        "parameters": constraint.parameters(),
    }


def constraint_from_dict(data: dict) -> MissionConstraint:
    """从字典还原约束。"""
    kind = str(data["kind"])
    try:
        constraint_class = CONSTRAINT_TYPES[kind]
    except KeyError as error:
        available = ", ".join(sorted(CONSTRAINT_TYPES)) or "(empty)"
        raise KeyError(f"未知约束类型 '{kind}'，可用类型: {available}") from error
    parameters = dict(data.get("parameters") or {})
    parameters["severity"] = ConstraintSeverity(
        data.get("severity", constraint_class.default_severity.value)
    )
    return constraint_class(**parameters)


@register_constraint
@dataclass
class PayloadMassConstraint(MissionConstraint):
    """有效载荷质量约束 — 载荷加整流罩不超过运载能力。"""

    kind: ClassVar[str] = "payload_mass"
    limit_kg: float = 1_000.0
    include_fairing: bool = True
    severity: ConstraintSeverity = ConstraintSeverity.ERROR

    def __post_init__(self) -> None:
        """验证参数。"""
        if self.limit_kg <= 0:
            raise ValueError("载荷质量上限必须为正")

    def evaluate(self, scenario: MissionScenario) -> ConstraintResult:
        """比较载荷质量和运载能力。"""
        fairing_mass_kg = scenario.launch_vehicle.fairing_mass_kg if self.include_fairing else 0.0
        actual = scenario.payload_mass_kg + fairing_mass_kg
        satisfied = actual <= self.limit_kg
        note = "（含整流罩）" if self.include_fairing else ""
        message = (
            f"载荷质量 {actual:.1f} kg {'满足' if satisfied else '超出'}"
            f"上限 {self.limit_kg:.1f} kg{note}"
        )
        return self.result(
            satisfied=satisfied,
            message=message,
            actual=actual,
            limit=self.limit_kg,
        )


@register_constraint
@dataclass
class PayloadVolumeConstraint(MissionConstraint):
    """容积约束 — 载荷收拢包络不超过整流罩可用容积。

    ``packing_efficiency`` 是一阶装箱系数，用于把"等效装载容积"折算回
    整流罩容积口径；真实装载需要详细的三维干涉分析。
    """

    kind: ClassVar[str] = "payload_volume"
    limit_m3: float = 0.0
    packing_efficiency: float = 0.75
    severity: ConstraintSeverity = ConstraintSeverity.ERROR

    def __post_init__(self) -> None:
        """验证参数。"""
        if self.limit_m3 < 0:
            raise ValueError("容积上限不能为负")
        if not 0.0 < self.packing_efficiency <= 1.0:
            raise ValueError("装箱系数必须在 (0, 1]")

    def evaluate(self, scenario: MissionScenario) -> ConstraintResult:
        """比较等效装载容积和可用容积。"""
        limit_m3 = self.limit_m3 or scenario.launch_vehicle.fairing_usable_volume_m3
        if limit_m3 <= 0:
            return self.result(
                satisfied=True,
                message="未提供可用容积上限，跳过容积检查",
                severity=ConstraintSeverity.INFO,
            )
        actual = scenario.payload_volume_m3 / self.packing_efficiency
        satisfied = actual <= limit_m3
        message = (
            f"等效装载容积 {actual:.3f} m³ {'满足' if satisfied else '超出'}"
            f"上限 {limit_m3:.3f} m³（装箱系数 {self.packing_efficiency:.2f}）"
        )
        return self.result(
            satisfied=satisfied,
            message=message,
            actual=actual,
            limit=limit_m3,
        )


@register_constraint
@dataclass
class DeltaVMarginConstraint(MissionConstraint):
    """速度预算约束 — 可用速度增量相对所需速度增量的余量。

    速度增量不足只在上升段结束时才显现，因此 ``preflight_gate = False``。
    """

    kind: ClassVar[str] = "delta_v_margin"
    preflight_gate: ClassVar[bool] = False
    min_margin_mps: float = 0.0
    min_margin_ratio: float = 0.0
    severity: ConstraintSeverity = ConstraintSeverity.ERROR

    def __post_init__(self) -> None:
        """验证参数。"""
        if self.min_margin_mps < 0:
            raise ValueError("最小速度增量余量不能为负")
        if self.min_margin_ratio < 0:
            raise ValueError("最小余量比例不能为负")

    def evaluate(self, scenario: MissionScenario) -> ConstraintResult:
        """比较速度增量余量和要求。"""
        budget = compute_launch_budget(scenario)
        required_limit_mps = max(
            self.min_margin_mps,
            self.min_margin_ratio * budget.required_delta_v_mps,
        )
        margin_mps = budget.delta_v_margin_mps
        satisfied = margin_mps >= required_limit_mps
        message = (
            f"速度增量余量 {margin_mps:.1f} m/s {'满足' if satisfied else '低于'}"
            f"要求 {required_limit_mps:.1f} m/s"
            f"（所需 {budget.required_delta_v_mps:.1f} m/s，"
            f"可用 {budget.available_delta_v_mps:.1f} m/s）"
        )
        return self.result(
            satisfied=satisfied,
            message=message,
            actual=margin_mps,
            limit=required_limit_mps,
        )


@register_constraint
@dataclass
class LaunchAzimuthConstraint(MissionConstraint):
    """发射方位角约束 — 目标倾角对应的方位角必须落在航区允许范围内。

    ``azimuth_range_deg`` 为 ``None`` 时使用 ``LaunchSite.azimuth_range_deg``。
    方位角按自北顺时针为正、不跨越 0 度。
    """

    kind: ClassVar[str] = "launch_azimuth"
    azimuth_range_deg: tuple[float, float] | None = None
    severity: ConstraintSeverity = ConstraintSeverity.ERROR

    def __post_init__(self) -> None:
        """验证并归一化方位角范围。"""
        if self.azimuth_range_deg is None:
            return
        values = tuple(float(value) for value in self.azimuth_range_deg)
        if len(values) != 2:
            raise ValueError("发射方位角范围必须包含上下限两个端点")
        low, high = values
        if not 0.0 <= low <= high <= 360.0:
            raise ValueError("发射方位角范围必须满足 0 <= 下限 <= 上限 <= 360")
        self.azimuth_range_deg = (low, high)

    def evaluate(self, scenario: MissionScenario) -> ConstraintResult:
        """比较目标方位角与允许范围。"""
        low, high = self.azimuth_range_deg or scenario.launch_site.azimuth_range_deg
        try:
            azimuth_deg = launch_azimuth_deg(
                scenario.target_orbit.inclination_deg,
                scenario.launch_site.latitude_deg,
            )
        except ValueError as error:
            return self.result(satisfied=False, message=f"无法解算发射方位角: {error}")
        satisfied = low <= azimuth_deg <= high
        message = (
            f"发射方位角 {azimuth_deg:.2f} 度 {'落在' if satisfied else '不在'}"
            f"允许范围 [{low:.1f}, {high:.1f}] 度"
        )
        return self.result(
            satisfied=satisfied,
            message=message,
            actual=azimuth_deg,
            limit=high,
        )


@register_constraint
@dataclass
class LaunchWindowConstraint(MissionConstraint):
    """发射窗口约束 — 场景声明的发射时刻必须落在窗口内。"""

    kind: ClassVar[str] = "launch_window"
    require_bound_epoch: bool = False
    severity: ConstraintSeverity = ConstraintSeverity.ERROR

    def evaluate(self, scenario: MissionScenario) -> ConstraintResult:
        """检查发射时刻与窗口的关系。"""
        window = scenario.launch_window
        if window is None:
            return self.result(
                satisfied=True,
                message="场景未声明发射窗口，跳过窗口检查",
                severity=ConstraintSeverity.INFO,
            )
        if scenario.launch_epoch_utc is None:
            if self.require_bound_epoch:
                return self.result(
                    satisfied=False,
                    message=f"已声明发射窗口 {window.name}，但场景未绑定 launch_epoch_utc",
                )
            return self.result(
                satisfied=True,
                message=(
                    f"发射窗口 {window.name} 已定义，持续 {window.duration_s / 60.0:.1f} 分钟，"
                    "场景未绑定发射时刻"
                ),
                severity=ConstraintSeverity.INFO,
            )
        inside = window.contains(scenario.launch_epoch_utc)
        message = (
            f"发射时刻 {scenario.launch_epoch_utc} {'落在' if inside else '不在'}"
            f"窗口 {window.name} [{window.open_utc}, {window.close_utc}]"
        )
        return self.result(satisfied=inside, message=message)


@register_constraint
@dataclass
class DeploymentOrderConstraint(MissionConstraint):
    """部署顺序约束 — 载荷必须按声明的顺序和间隔分离。"""

    kind: ClassVar[str] = "deployment_order"
    require_plan: bool = True
    severity: ConstraintSeverity = ConstraintSeverity.WARNING

    def evaluate(self, scenario: MissionScenario) -> ConstraintResult:
        """检查部署顺序声明。"""
        plan = scenario.deployment_plan
        if plan is None:
            satisfied = not self.require_plan
            return self.result(
                satisfied=satisfied,
                message=(f"场景未声明部署顺序，按载荷列表顺序部署 {len(scenario.payloads)} 个卫星"),
            )
        order = " -> ".join(
            f"{step.satellite_index}:{scenario.payloads[step.satellite_index - 1].name}"
            for step in plan.steps
        )
        message = (
            f"部署顺序 {order}，共 {len(plan.steps)} 步，"
            f"总分离间隔 {plan.total_delay_s:.1f} s，"
            f"总分离冲量 {plan.total_separation_delta_v_mps:.1f} m/s"
        )
        return self.result(satisfied=True, message=message, actual=plan.total_delay_s)


@register_constraint
@dataclass
class PropellantReserveConstraint(MissionConstraint):
    """推进剂余量约束 — 每颗卫星的可用速度增量不低于预留任务需求。"""

    kind: ClassVar[str] = "propellant_reserve"
    reserve_delta_v_mps: float = 50.0
    severity: ConstraintSeverity = ConstraintSeverity.ERROR

    def __post_init__(self) -> None:
        """验证参数。"""
        if self.reserve_delta_v_mps < 0:
            raise ValueError("预留速度增量不能为负")

    def evaluate(self, scenario: MissionScenario) -> ConstraintResult:
        """取最差卫星的可用速度增量与预留需求比较。"""
        available = [(payload.name, payload.available_delta_v_mps) for payload in scenario.payloads]
        worst_name, worst_delta_v_mps = min(available, key=lambda item: item[1])
        satisfied = worst_delta_v_mps >= self.reserve_delta_v_mps
        message = (
            f"最低推进剂能力 {worst_delta_v_mps:.1f} m/s（{worst_name}）"
            f"{'满足' if satisfied else '低于'}预留 {self.reserve_delta_v_mps:.1f} m/s"
        )
        return self.result(
            satisfied=satisfied,
            message=message,
            actual=worst_delta_v_mps,
            limit=self.reserve_delta_v_mps,
        )


@register_constraint
@dataclass
class PowerBudgetConstraint(MissionConstraint):
    """功率预算约束 — 建模了电源分系统的卫星需要满足轨道平均功率余量。"""

    kind: ClassVar[str] = "power_budget"
    min_margin_w: float = 0.0
    severity: ConstraintSeverity = ConstraintSeverity.WARNING

    def evaluate(self, scenario: MissionScenario) -> ConstraintResult:
        """取最差卫星的轨道平均功率余量进行比较。"""
        modeled = [payload for payload in scenario.payloads if payload.power_system is not None]
        if not modeled:
            return self.result(
                satisfied=True,
                message="未建模电源分系统，跳过功率余量检查",
                severity=ConstraintSeverity.INFO,
            )
        worst = min(
            modeled,
            key=lambda payload: payload.power_system.power_margin_w,  # type: ignore[union-attr]
        )
        margin_w = worst.power_system.power_margin_w  # type: ignore[union-attr]
        satisfied = margin_w >= self.min_margin_w
        message = (
            f"最低轨道平均功率余量 {margin_w:.1f} W（{worst.name}）"
            f"{'满足' if satisfied else '低于'}要求 {self.min_margin_w:.1f} W"
        )
        return self.result(
            satisfied=satisfied,
            message=message,
            actual=margin_w,
            limit=self.min_margin_w,
        )


def builtin_gate_constraints(scenario: MissionScenario) -> list[MissionConstraint]:
    """返回模拟器始终执行的内置硬性门槛约束。

    只要运载能力或速度预算不满足，任务就不可能在物理上完成，因此这两条
    检查不依赖用户声明，由 ``MissionSimulator`` 无条件执行并写入结果。
    其余可选约束通过 ``scenario.constraints`` 或 ``default_constraints`` 声明。
    """
    return [
        PayloadMassConstraint(limit_kg=scenario.launch_vehicle.payload_capacity_kg),
        DeltaVMarginConstraint(min_margin_mps=0.0),
    ]


def default_constraints(scenario: MissionScenario) -> list[MissionConstraint]:
    """按场景生成一组推荐的默认约束。

    默认集合覆盖质量、容积、速度预算、方位角、发射窗口、部署顺序和功率预算。
    推进剂余量（``PropellantReserveConstraint``）需要显式的站内保持需求，
    因此不放进默认集合，由使用者按任务寿命单独声明。
    """
    return [
        PayloadMassConstraint(limit_kg=scenario.launch_vehicle.payload_capacity_kg),
        PayloadVolumeConstraint(),
        DeltaVMarginConstraint(min_margin_mps=50.0, min_margin_ratio=0.02),
        LaunchAzimuthConstraint(),
        LaunchWindowConstraint(),
        DeploymentOrderConstraint(),
        PowerBudgetConstraint(),
    ]


__all__ = [
    "CONSTRAINT_TYPES",
    "ConstraintResult",
    "ConstraintSeverity",
    "DeltaVMarginConstraint",
    "DeploymentOrderConstraint",
    "LaunchAzimuthConstraint",
    "LaunchWindowConstraint",
    "MissionConstraint",
    "PayloadMassConstraint",
    "PayloadVolumeConstraint",
    "PowerBudgetConstraint",
    "PropellantReserveConstraint",
    "builtin_gate_constraints",
    "constraint_from_dict",
    "constraint_to_dict",
    "default_constraints",
    "register_constraint",
]
