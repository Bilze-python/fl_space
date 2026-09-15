"""
发射任务与卫星设计框架测试。

覆盖四类要求：
    1. 序列化往返（含分系统、窗口、部署计划和约束）；
    2. 物理量级（轨道周期、方位角、霍曼转移、链路预算、分系统派生量）；
    3. 非法输入与边界；
    4. 最小集成（一阶预算 + 约束门槛 + 部署事件流）。
"""

import math

import pytest

from fl_space.mission import (
    AttitudeControlSystem,
    CommunicationSystem,
    ConstraintSeverity,
    DeltaVMarginConstraint,
    DeploymentPlan,
    DeploymentStep,
    LaunchSite,
    LaunchVehicle,
    LaunchWindow,
    LaunchWindowConstraint,
    MissionScenario,
    MissionSimulator,
    OrbitalTarget,
    PayloadMassConstraint,
    PayloadVolumeConstraint,
    PowerBudgetConstraint,
    PowerSystem,
    PropellantReserveConstraint,
    PropulsionSystem,
    RocketStage,
    SatelliteDesign,
    VisualAppearance,
    circular_orbit_period_s,
    circular_orbit_velocity_mps,
    compute_launch_budget,
    default_constraints,
    dynamic_pressure_pa,
    free_space_path_loss_db,
    hohmann_transfer_burns_mps,
    hohmann_transfer_delta_v_mps,
    launch_azimuth_deg,
    received_power_dbw,
    satellite_designs,
)

ISO_OPEN = "2026-01-01T00:00:00+00:00"
ISO_CLOSE = "2026-01-01T02:00:00+00:00"
ISO_INSIDE = "2026-01-01T00:30:00+00:00"
ISO_OUTSIDE = "2026-01-01T03:00:00+00:00"


def _payload() -> SatelliteDesign:
    """构造一个带完整分系统的测试载荷。"""
    return SatelliteDesign(
        name="test-sat",
        dry_mass_kg=30.0,
        power_w=180.0,
        cross_section_area_m2=0.8,
        body_dimensions_m=(1.0, 0.8, 0.6),
        power_system=PowerSystem(
            solar_array_power_w=200.0,
            battery_capacity_wh=300.0,
            average_load_w=120.0,
            peak_load_w=200.0,
            eclipse_fraction=0.35,
            mass_kg=12.0,
        ),
        communication_system=CommunicationSystem(
            band="S",
            frequency_hz=2.2e9,
            transmit_power_w=8.0,
            downlink_rate_mbps=20.0,
            mass_kg=4.0,
            power_w=15.0,
        ),
        attitude_control_system=AttitudeControlSystem(
            mode="three_axis",
            pointing_accuracy_deg=0.1,
            momentum_capacity_nms=4.0,
            reaction_wheel_count=3,
            mass_kg=6.0,
            power_w=9.0,
        ),
        propulsion_system=PropulsionSystem(
            propellant_type="hydrazine",
            propellant_mass_kg=6.0,
            dry_mass_kg=3.0,
            thrust_n=1.0,
            specific_impulse_s=220.0,
            thruster_count=2,
        ),
        visual=VisualAppearance(
            primitive="custom_mesh",
            dimensions_m=(1.0, 2.0, 0.5),
            asset_uri="assets/test.glb",
        ),
    )


def _vehicle(
    *,
    payload_capacity_kg: float = 1_000.0,
    upper_propellant_kg: float = 30_000.0,
) -> LaunchVehicle:
    """构造两级研究火箭。"""
    return LaunchVehicle(
        stages=[
            RocketStage(
                name="booster",
                dry_mass_kg=20_000.0,
                propellant_mass_kg=160_000.0,
                specific_impulse_s=300.0,
            ),
            RocketStage(
                name="upper",
                dry_mass_kg=3_000.0,
                propellant_mass_kg=upper_propellant_kg,
                specific_impulse_s=350.0,
            ),
        ],
        payload_capacity_kg=payload_capacity_kg,
        fairing_mass_kg=200.0,
        fairing_inner_diameter_m=3.0,
        fairing_inner_height_m=6.0,
    )


def _scenario() -> MissionScenario:
    """构造测试场景。"""
    return MissionScenario(
        name="test-mission",
        launch_vehicle=_vehicle(),
        payloads=[_payload()],
        target_orbit=OrbitalTarget(altitude_km=550.0, inclination_deg=53.0),
        launch_site=LaunchSite(latitude_deg=28.5, longitude_deg=-80.6),
    )


def test_physics_ranges_are_reasonable() -> None:
    """验证基础物理量级。"""
    assert 7_000 < circular_orbit_velocity_mps(550.0) < 8_000
    assert dynamic_pressure_pa(0.0, 100.0) > 0
    assert 5_600 < circular_orbit_period_s(550.0) < 5_900
    assert circular_orbit_period_s(550.0) < circular_orbit_period_s(1_200.0)


def test_launch_azimuth_matches_plane_geometry() -> None:
    """验证发射方位角几何关系及其不可行分支。"""
    azimuth_deg = launch_azimuth_deg(53.0, 28.5)
    assert 40.0 < azimuth_deg < 47.0
    assert math.isclose(
        math.sin(math.radians(azimuth_deg)),
        math.cos(math.radians(53.0)) / math.cos(math.radians(28.5)),
        rel_tol=1e-9,
    )
    with pytest.raises(ValueError):
        launch_azimuth_deg(20.0, 28.5)


def test_hohmann_and_link_budget_magnitudes() -> None:
    """验证霍曼转移和链路预算的量级与单调性。"""
    first_burn_mps, second_burn_mps = hohmann_transfer_burns_mps(550.0, 600.0)
    # 抬升轨道时第一次点火把圆轨道加速成转移椭圆，因此大于远地点圆化点火。
    assert 0.0 < second_burn_mps < first_burn_mps < 100.0
    assert hohmann_transfer_delta_v_mps(550.0, 550.0) == pytest.approx(0.0, abs=1e-9)
    # 目标轨道更高时总速度增量更大。
    assert hohmann_transfer_delta_v_mps(550.0, 1_000.0) > hohmann_transfer_delta_v_mps(550.0, 600.0)
    assert free_space_path_loss_db(1_000_000.0, 2.2e9) > free_space_path_loss_db(500_000.0, 2.2e9)
    assert received_power_dbw(10.0, 30.0, 30.0, 1_000_000.0, 2.2e9) < 0.0


def test_subsystem_derived_quantities() -> None:
    """验证分系统派生量的物理含义。"""
    design = _payload()
    assert design.total_mass_kg > design.dry_total_mass_kg > design.dry_mass_kg
    assert design.subsystem_mass_kg == 12.0 + 4.0 + 6.0 + 3.0
    assert design.total_propellant_mass_kg == 6.0
    assert design.available_power_w == pytest.approx(200.0 * 0.65)
    assert design.available_delta_v_mps > 0.0
    probe = PropulsionSystem(
        propellant_mass_kg=100.0,
        dry_mass_kg=20.0,
        thrust_n=1.0,
        specific_impulse_s=300.0,
    )
    assert probe.available_delta_v_mps(1_000.0) == pytest.approx(
        300.0 * 9.80665 * math.log(1_100.0 / 1_000.0)
    )
    assert probe.total_thrust_n == 1.0


def test_mission_deploys_satellite_into_existing_orbit_backend() -> None:
    """验证任务结果可以直接接到现有 Kepler 后端。"""
    result = MissionSimulator().simulate(_scenario())
    assert result.success
    assert result.phase.value == "complete"
    assert result.fidelity == "first_order"
    assert len(result.deployed_satellites) == 1
    assert result.deployed_satellites[0].orbit.altitude_km == 550.0
    assert result.deployed_satellites[0].design.visual.asset_uri == "assets/test.glb"
    assert {item.kind for item in result.constraint_results} >= {
        "payload_mass",
        "delta_v_margin",
    }


def test_design_round_trip_and_registry() -> None:
    """验证自定义外观序列化和设计注册。"""
    design = SatelliteDesign.from_dict(_scenario().payloads[0].to_dict())
    assert design.visual.dimensions_m == (1.0, 2.0, 0.5)
    assert design.body_dimensions_m == (1.0, 0.8, 0.6)
    assert design.power_system is not None
    assert design.power_system.solar_array_power_w == 200.0
    assert design.propulsion_system is not None
    assert design.propulsion_system.specific_impulse_s == 220.0
    assert "cubesat-3u" in satellite_designs
    assert satellite_designs.get("cubesat-3u").total_mass_kg > 0
    assert MissionScenario(payloads=[SatelliteDesign()]).launch_vehicle.stages
    smallsat = satellite_designs.get("earth-obs-smallsat")
    assert smallsat.available_delta_v_mps > 0.0
    assert smallsat.available_power_w > 0.0


def test_scenario_round_trip_with_window_plan_and_constraints() -> None:
    """验证带窗口、部署计划和约束的场景可以完整往返序列化。"""
    scenario = _scenario()
    scenario.launch_window = LaunchWindow(open_utc=ISO_OPEN, close_utc=ISO_CLOSE)
    scenario.launch_epoch_utc = ISO_INSIDE
    scenario.deployment_plan = DeploymentPlan(
        steps=[DeploymentStep(satellite_index=1, delay_s=90.0, separation_delta_v_mps=0.3)]
    )
    scenario.with_default_constraints()
    restored = MissionScenario.from_dict(scenario.to_dict())
    assert restored.launch_window is not None
    assert restored.launch_window.duration_s == pytest.approx(7200.0)
    assert restored.deployment_plan is not None
    assert restored.deployment_plan.total_separation_delta_v_mps == pytest.approx(0.3)
    assert restored.deployment_intervals_s() == [90.0]
    assert [item.kind for item in restored.constraints] == [
        item.kind for item in scenario.constraints
    ]
    assert restored.payloads[0].communication_system is not None


def test_launch_budget_is_single_source_of_truth() -> None:
    """验证预算对象与模拟器结果一致。"""
    scenario = _scenario()
    budget = compute_launch_budget(scenario)
    assert budget.payload_mass_kg == pytest.approx(
        scenario.payload_mass_kg + scenario.launch_vehicle.fairing_mass_kg
    )
    assert budget.required_delta_v_mps == pytest.approx(
        budget.orbital_velocity_mps
        + budget.losses_mps
        + budget.plane_change_mps
        - budget.rotation_credit_mps
    )
    assert not budget.mass_capacity_exceeded
    assert not budget.delta_v_exceeded
    result = MissionSimulator().simulate(scenario)
    assert result.required_delta_v_mps == pytest.approx(budget.required_delta_v_mps)
    assert result.available_delta_v_mps == pytest.approx(budget.available_delta_v_mps)


def test_mass_overflow_fails_before_liftoff() -> None:
    """验证质量超限在点火前就被拦截。"""
    scenario = MissionScenario(
        launch_vehicle=_vehicle(payload_capacity_kg=50.0),
        payloads=[_payload()],
        launch_site=LaunchSite(latitude_deg=28.5),
    )
    result = MissionSimulator().simulate(scenario)
    assert not result.success
    assert result.phase.value == "failed"
    assert result.elapsed_time_s == 0.0
    assert result.deployed_satellites == []
    assert [item.kind for item in result.blocking_constraints] == ["payload_mass"]


def test_delta_v_shortfall_fails_after_ascent() -> None:
    """验证速度增量不足在上升段结束时判定。"""
    scenario = MissionScenario(
        launch_vehicle=_vehicle(upper_propellant_kg=3_000.0),
        payloads=[_payload()],
        launch_site=LaunchSite(latitude_deg=28.5),
    )
    result = MissionSimulator().simulate(scenario)
    assert not result.success
    assert result.elapsed_time_s == pytest.approx(10.0 + 480.0)
    assert [item.kind for item in result.blocking_constraints] == ["delta_v_margin"]


def test_declared_constraints_are_enforced_without_duplication() -> None:
    """验证场景声明的约束被评估，并能覆盖内置门槛阈值。"""
    scenario = _scenario()
    scenario.launch_window = LaunchWindow(open_utc=ISO_OPEN, close_utc=ISO_CLOSE)
    scenario.launch_epoch_utc = ISO_OUTSIDE
    scenario.constraints = [
        PayloadMassConstraint(limit_kg=10.0),
        LaunchWindowConstraint(),
        PowerBudgetConstraint(),
    ]
    result = MissionSimulator().simulate(scenario)
    kinds = [item.kind for item in result.constraint_results]
    assert kinds.count("payload_mass") == 1
    assert "power_budget" in kinds
    assert "launch_window" in kinds
    assert not result.success
    blocking = {item.kind for item in result.blocking_constraints}
    assert blocking == {"payload_mass", "launch_window"}


def test_default_constraints_cover_expected_kinds() -> None:
    """验证默认约束集的覆盖面。"""
    kinds = {constraint.kind for constraint in default_constraints(_scenario())}
    assert {"payload_mass", "payload_volume", "delta_v_margin", "launch_azimuth"} <= kinds
    assert kinds.isdisjoint({"propellant_reserve"})


def test_deployment_plan_orders_and_times_separation() -> None:
    """验证部署顺序、间隔和分离冲量进入事件流。"""
    scenario = MissionScenario(
        name="two-payload",
        launch_vehicle=_vehicle(),
        payloads=[_payload(), satellite_designs.get("cubesat-3u")],
        target_orbit=OrbitalTarget(altitude_km=550.0, inclination_deg=53.0),
        launch_site=LaunchSite(latitude_deg=28.5),
        deployment_plan=DeploymentPlan(
            steps=[
                DeploymentStep(
                    satellite_index=2, delay_s=30.0, separation_delta_v_mps=0.4, target_slot=1
                ),
                DeploymentStep(
                    satellite_index=1, delay_s=120.0, separation_delta_v_mps=0.2, target_slot=2
                ),
            ]
        ),
    )
    result = MissionSimulator().simulate(scenario)
    assert result.success
    assert [item.orbit.altitude_km for item in result.deployed_satellites] == [550.0, 550.0]
    assert [item.design.name for item in result.deployed_satellites] == [
        "cubesat-3u",
        "test-sat",
    ]
    assert [item.satellite_index for item in result.deployed_satellites] == [2, 1]
    assert [item.deployment_order_index for item in result.deployed_satellites] == [1, 2]
    assert [item.separation_delta_v_mps for item in result.deployed_satellites] == [0.4, 0.2]
    assert result.elapsed_time_s == pytest.approx(10.0 + 480.0 + 90.0 + 30.0 + 120.0)
    assert sum(1 for event in result.events if event.phase.value == "deployment") == 3


def test_invalid_inputs_are_rejected() -> None:
    """验证非法输入与边界。"""
    with pytest.raises(ValueError):
        LaunchWindow(open_utc=ISO_CLOSE, close_utc=ISO_OPEN)
    with pytest.raises(ValueError):
        LaunchWindow(open_utc="2026-01-01T00:00:00", close_utc=ISO_CLOSE)
    with pytest.raises(ValueError):
        DeploymentPlan(
            steps=[
                DeploymentStep(satellite_index=1),
                DeploymentStep(satellite_index=1),
            ]
        )
    with pytest.raises(ValueError):
        DeploymentStep(satellite_index=0)
    with pytest.raises(ValueError):
        LaunchSite(azimuth_range_deg=(200.0, 100.0))
    with pytest.raises(ValueError):
        launch_azimuth_deg(200.0, 28.5)
    with pytest.raises(ValueError):
        PowerSystem(eclipse_fraction=1.0)
    with pytest.raises(ValueError):
        AttitudeControlSystem(mode="three_axis", reaction_wheel_count=2)
    with pytest.raises(ValueError):
        CommunicationSystem(inter_satellite_link=True, isl_rate_mbps=0.0)
    with pytest.raises(ValueError):
        PropulsionSystem(propellant_type="none", propellant_mass_kg=1.0)
    with pytest.raises(ValueError):
        SatelliteDesign(
            propellant_mass_kg=1.0,
            propulsion_system=PropulsionSystem(propellant_mass_kg=1.0),
        )
    with pytest.raises(ValueError):
        MissionScenario(
            payloads=[_payload()],
            deployment_plan=DeploymentPlan(steps=[DeploymentStep(satellite_index=3)]),
        )
    with pytest.raises(ValueError):
        DeltaVMarginConstraint(min_margin_ratio=-0.1)
    with pytest.raises(ValueError):
        PropellantReserveConstraint(reserve_delta_v_mps=-1.0)
    with pytest.raises(ValueError):
        PayloadVolumeConstraint(packing_efficiency=0.0)


def test_constraint_severity_and_reserve_semantics() -> None:
    """验证约束的严重程度语义和推进剂预留检查。"""
    scenario = _scenario()
    scenario.constraints = [PropellantReserveConstraint(reserve_delta_v_mps=50.0)]
    result = MissionSimulator().simulate(scenario)
    reserve = next(item for item in result.constraint_results if item.kind == "propellant_reserve")
    assert reserve.severity is ConstraintSeverity.ERROR
    assert reserve.satisfied
