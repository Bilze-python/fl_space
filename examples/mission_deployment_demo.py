"""
发射部署框架示例 — 从地面发射并把自定义卫星部署到 LEO。

示例演示了当前领域模型的完整能力：

    - 卫星：质量、物理包络、电源、通信、姿态控制和推进分系统；
    - 火箭：级数、结构质量、推进剂、发动机推力和比冲、整流罩包络；
    - 任务：载荷、目标轨道、发射窗口、部署顺序和任务约束。

运行:
    python examples/mission_deployment_demo.py
"""

import json

from fl_space.mission import (
    AttitudeControlSystem,
    CommunicationSystem,
    DeploymentPlan,
    DeploymentStep,
    LaunchSite,
    LaunchVehicle,
    LaunchWindow,
    MissionScenario,
    MissionSimulator,
    OrbitalTarget,
    PowerSystem,
    PropulsionSystem,
    RocketStage,
    SatelliteDesign,
    VisualAppearance,
    satellite_designs,
)

WINDOW_OPEN_UTC = "2026-03-01T02:00:00+00:00"
WINDOW_CLOSE_UTC = "2026-03-01T04:00:00+00:00"
LAUNCH_EPOCH_UTC = "2026-03-01T02:30:00+00:00"


def build_observer() -> SatelliteDesign:
    """构造主载荷：带完整分系统的对地观测卫星。"""
    return SatelliteDesign(
        name="research-observer",
        dry_mass_kg=80.0,
        power_w=180.0,
        drag_coefficient=2.2,
        cross_section_area_m2=0.8,
        storage_gb=128.0,
        body_dimensions_m=(1.2, 0.8, 0.6),
        stowed_dimensions_m=(1.2, 0.8, 0.6),
        power_system=PowerSystem(
            name="body_mounted_solar_battery_bus",
            solar_array_power_w=420.0,
            battery_capacity_wh=600.0,
            bus_voltage_v=28.0,
            average_load_w=180.0,
            peak_load_w=320.0,
            eclipse_fraction=0.35,
            mass_kg=18.0,
        ),
        communication_system=CommunicationSystem(
            name="x_band_downlink",
            band="X",
            frequency_hz=8.2e9,
            transmit_power_w=15.0,
            transmit_antenna_gain_dbi=26.0,
            receive_antenna_gain_dbi=24.0,
            ground_station_gain_dbi=44.0,
            downlink_rate_mbps=120.0,
            uplink_rate_mbps=4.0,
            mass_kg=7.5,
            power_w=38.0,
        ),
        attitude_control_system=AttitudeControlSystem(
            name="three_axis_reaction_wheels",
            mode="three_axis",
            pointing_accuracy_deg=0.05,
            slew_rate_deg_s=1.2,
            momentum_capacity_nms=10.0,
            reaction_wheel_count=4,
            mass_kg=9.0,
            power_w=16.0,
        ),
        propulsion_system=PropulsionSystem(
            name="hydrazine_monoprop",
            propellant_type="hydrazine",
            propellant_mass_kg=14.0,
            dry_mass_kg=7.0,
            thrust_n=3.0,
            specific_impulse_s=230.0,
            thruster_count=4,
            min_impulse_bit_ns=0.05,
        ),
        visual=VisualAppearance(
            primitive="custom_mesh",
            dimensions_m=(1.2, 0.8, 0.6),
            asset_uri="assets/research-observer.glb",
            color_hex="#4aa3df",
        ),
        assumptions=[
            "研究默认值：分系统参数按小卫星公开量级给定，不代表型号验收数据",
        ],
        metadata={"purpose": "open_research"},
    )


def build_secondary() -> SatelliteDesign:
    """构造次载荷：3U 立方星，使用注册表内置蓝图。"""
    design = satellite_designs.get("cubesat-3u")
    design.name = "cubesat-3u-secondary"
    return design


def build_launcher() -> LaunchVehicle:
    """构造两级研究运载火箭。"""
    return LaunchVehicle(
        name="two_stage_research_launcher",
        stages=[
            RocketStage(
                name="booster",
                dry_mass_kg=25_000.0,
                propellant_mass_kg=180_000.0,
                specific_impulse_s=300.0,
                thrust_kn=2_000.0,
                burn_time_s=150.0,
                engine_count=9,
                propellant_kind="kerolox",
            ),
            RocketStage(
                name="upper_stage",
                dry_mass_kg=4_000.0,
                propellant_mass_kg=35_000.0,
                specific_impulse_s=350.0,
                thrust_kn=800.0,
                burn_time_s=420.0,
                engine_count=1,
                propellant_kind="kerolox",
            ),
        ],
        payload_capacity_kg=2_000.0,
        fairing_mass_kg=500.0,
        fairing_inner_diameter_m=4.6,
        fairing_inner_height_m=8.0,
    )


def build_demo_scenario() -> MissionScenario:
    """构造一个可修改的两级火箭部署场景。"""
    return MissionScenario(
        name="leo_custom_satellite_demo",
        launch_vehicle=build_launcher(),
        payloads=[build_observer(), build_secondary()],
        target_orbit=OrbitalTarget(altitude_km=550.0, inclination_deg=53.0),
        launch_site=LaunchSite(
            name="demo_launch_site",
            latitude_deg=28.5,
            longitude_deg=-80.6,
            azimuth_range_deg=(35.0, 120.0),
        ),
        launch_window=LaunchWindow(
            name="primary_window",
            open_utc=WINDOW_OPEN_UTC,
            close_utc=WINDOW_CLOSE_UTC,
            notes="示例窗口，非真实航区约束",
        ),
        launch_epoch_utc=LAUNCH_EPOCH_UTC,
        deployment_plan=DeploymentPlan(
            name="primary_then_secondary",
            steps=[
                DeploymentStep(
                    satellite_index=1,
                    delay_s=90.0,
                    separation_delta_v_mps=0.25,
                    target_slot=1,
                    notes="先释放带推进的主载荷",
                ),
                DeploymentStep(
                    satellite_index=2,
                    delay_s=180.0,
                    separation_delta_v_mps=0.10,
                    target_slot=2,
                    notes="等待主载荷远离后再释放立方星",
                ),
            ],
        ),
        assumptions=[
            "一阶速度预算：重力/阻力/转向损失由场景显式给出",
            "未模拟真实制导、结构载荷、热流和航区安全",
        ],
    )


def summarize(scenario: MissionScenario, result) -> str:
    """生成人类可读的任务摘要。"""
    lines = [
        f"任务: {scenario.name} (保真度: {result.fidelity})",
        f"结果: {'成功' if result.success else '失败'} / {result.phase.value}",
        f"载荷质量: {result.payload_mass_kg:.1f} kg "
        f"(能力 {result.budget.payload_capacity_kg:.1f} kg)",
        f"所需速度增量: {result.required_delta_v_mps:.1f} m/s "
        f"(轨道 {result.budget.orbital_velocity_mps:.1f}, "
        f"自转收益 {result.budget.rotation_credit_mps:.1f}, "
        f"损失 {result.budget.losses_mps:.1f})",
        f"可用速度增量: {result.available_delta_v_mps:.1f} m/s "
        f"(余量 {result.delta_v_margin_mps:.1f} m/s)",
        f"任务时长: {result.elapsed_time_s:.0f} s",
        "约束判定:",
    ]
    lines.extend(
        f"  [{item.severity.value}] {item.kind}: "
        f"{'满足' if item.satisfied else '未满足'} — {item.message}"
        for item in result.constraint_results
    )
    lines.append("部署事件:")
    lines.extend(
        f"  t={event.time_s:.0f}s [{event.phase.value}] {event.message}" for event in result.events
    )
    return "\n".join(lines)


if __name__ == "__main__":
    demo_scenario = build_demo_scenario().with_default_constraints()
    demo_result = MissionSimulator().simulate(demo_scenario)
    print(summarize(demo_scenario, demo_result))
    print()
    print(json.dumps(demo_result.to_dict(), indent=2, ensure_ascii=False))
