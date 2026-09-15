"""
环境模拟演示脚本 — 展示模块化架构的各项功能

运行: python -m examples.demo_environment
"""

import os
import sys

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_space.environment import (
    CelestialBody,
    GroundStationNetwork,
    create_default_network,
    create_extended_network,
)
from fl_space.orbit import (
    ConstellationConfig,
    VisibilityEngine,
    create_circular_orbit,
    generate_cluster_phases,
    generate_walker_phases,
)
from fl_space.simulator import (
    OrbitSimulator,
    create_default_simulator,
    create_mars_simulator,
)


def demo_1_celestial_bodies():
    """演示1: 天体参数"""
    print("=" * 60)
    print("  演示1: 可配置天体参数")
    print("=" * 60)

    for body in [CelestialBody.earth(), CelestialBody.mars(), CelestialBody.moon()]:
        print(f"\n  {body.name}:")
        print(f"    半径: {body.radius_km} km")
        print(f"    GM: {body.GM:.1f} km³/s²")
        print(f"    自转周期: {body.rotation_period_hours} h")
        print(f"    大气高度: {body.atmosphere_height_km} km")
        print(f"    表面重力: {body.surface_gravity_ms2:.2f} m/s²")

        # 500km轨道的可见半角
        horizon = body.effective_horizon_angle_for_altitude(500)
        print(f"    500km轨道可见半角: {horizon*180/3.14159:.1f}°")

    # 自定义天体
    custom = CelestialBody(
        name="Kepler-22b",
        radius_km=16100,
        GM=4.8e6,
        rotation_period_hours=15.2,
        atmosphere_height_km=150,
    )
    print(f"\n  自定义天体 {custom.name}:")
    print(f"    半径: {custom.radius_km} km")

    input("\n  按 Enter 继续...")


def demo_2_orbit_mechanics():
    """演示2: 轨道力学"""
    print("\n" + "=" * 60)
    print("  演示2: 轨道力学计算")
    print("=" * 60)

    earth = CelestialBody.earth()
    orbit = create_circular_orbit(500, 90, 0, 0, earth)

    print("\n  轨道参数:")
    print(f"    高度: {orbit.altitude_km} km")
    print(f"    周期: {orbit.period_min:.1f} 分钟")
    print(f"    平均运动: {orbit.mean_motion_rad_per_min*180/3.14159:.4f} °/min")

    print("\n  前30分钟星下点轨迹:")
    for t in [0, 5, 10, 15, 20, 25, 30]:
        lat, lon = orbit.position_at_time_deg(t)
        print(f"    t={t:3d}min → ({lat:+.2f}°, {lon:+.2f}°)")

    input("\n  按 Enter 继续...")


def demo_3_constellation():
    """演示3: 星座分布"""
    print("\n" + "=" * 60)
    print("  演示3: 不同星座分布策略")
    print("=" * 60)

    earth = CelestialBody.earth()

    # Walker 5/2/1
    print("\n  [Walker 5/2/1 星座]")
    orbits_w = generate_walker_phases(5, num_planes=2, phasing_factor=1)
    for i, orb in enumerate(orbits_w):
        lat, lon = orb.position_at_time_deg(0)
        print(f"    SAT-{i}: (lat={lat:+.1f}°, lon={lon:+.1f}°), "
              f"raan={orb.elements.raan_deg:.1f}°, "
              f"ta0={orb.elements.true_anomaly_deg:.1f}°")

    # 星簇分布
    print("\n  [星簇分布 5卫星/2簇]")
    orbits_c = generate_cluster_phases(5, num_clusters=2)
    for i, orb in enumerate(orbits_c):
        lat, lon = orb.position_at_time_deg(0)
        cluster = i % 2
        print(f"    SAT-{i} (簇{cluster}): "
              f"(lat={lat:+.1f}°, lon={lon:+.1f}°), "
              f"raan={orb.elements.raan_deg:.1f}°")

    input("\n  按 Enter 继续...")


def demo_4_ground_stations():
    """演示4: 地面站管理"""
    print("\n" + "=" * 60)
    print("  演示4: 地面站网络管理")
    print("=" * 60)

    network = create_extended_network(7)
    print(f"\n  {network}")
    for gs in network:
        print(f"    {gs}")

    # 按名称查询
    beijing = network["Beijing"]
    print(f"\n  按名称查询 'Beijing': {beijing}")

    # 导出/导入
    network.save_json("d:/fl_space/examples/test_gs.json")
    loaded = GroundStationNetwork.load_json("d:/fl_space/examples/test_gs.json")
    print(f"  保存后重新加载: {loaded} (共 {loaded.count} 站)")

    # 清理
    os.remove("d:/fl_space/examples/test_gs.json")

    input("\n  按 Enter 继续...")


def demo_5_visibility():
    """演示5: 可见性计算"""
    print("\n" + "=" * 60)
    print("  演示5: 可见性计算引擎")
    print("=" * 60)

    earth = CelestialBody.earth()
    network = create_default_network(3)  # Sioux Falls, Sanya, Johannesburg

    orbit = create_circular_orbit(500, 90, 30, 60, earth)
    engine = VisibilityEngine(earth, orbit, network)

    print(f"\n  地平可见半角: {engine.horizon_angle_rad*180/3.14159:.1f}°")

    # 查看不同时刻的可见情况
    for t in [0, 15, 30, 60, 90, 120]:
        visible = engine.visible_stations_at_time(t)
        names = [network[gid].name for gid in visible]
        lat, lon = orbit.position_at_time_deg(t)
        print(f"    t={t:4d}min  ({lat:+7.2f}°, {lon:+7.2f}°) → "
              f"可见: {names if names else '(无)'}")

    input("\n  按 Enter 继续...")


def demo_6_orbit_simulator_basic():
    """演示6: 主模拟器 — 基础用法"""
    print("\n" + "=" * 60)
    print("  演示6: 主模拟器 — 基础用法")
    print("=" * 60)

    sim = create_default_simulator()
    print(sim.summary())

    # 查询
    print("\n  查询示例:")
    print(f"    SAT-0 @ TS60: 可见站 = {sim.get_all_contacts(0, 60)}")
    print(f"    SAT-1 @ TS60: 详细 = {sim.get_contact_detail(1, 60)}")
    print(f"    TS60 可通信卫星: {sim.get_satellites_in_contact(60)}")

    # 卫星位置
    lat, lon = sim.get_sat_position(0, 60)
    print(f"    SAT-0 @ TS60 位置: ({lat:+.2f}°, {lon:+.2f}°)")

    # 通信记录
    records = sim.get_communication_record(0)
    print(f"    SAT-0 通信记录: 共 {len(records)} 个接触窗口")
    if records:
        r = records[0]
        print(f"      首次接触: TS{r['timeslot']}, "
              f"基站={r['gs_names']}, "
              f"位置={r['sat_position_deg']}")

    # 导出
    sim.export("d:/fl_space/examples/test_contact.json")
    print("\n  接触矩阵已导出: examples/test_contact.json")

    input("\n  按 Enter 继续...")


def demo_7_orbit_simulator_mars():
    """演示7: 主模拟器 — 火星场景"""
    print("\n" + "=" * 60)
    print("  演示7: 主模拟器 — 火星场景")
    print("=" * 60)

    sim_mars = create_mars_simulator()
    print(sim_mars.summary())

    # 对比：同样的配置在地球和火星上的不同
    print("\n  对比分析:")
    print(f"    火星轨道周期: {sim_mars.orbit_period_min:.1f} min")
    print(f"    地球500km周期: ~{2*3.14159*(6371+500)**1.5/(398600.4418)**0.5/60:.1f} min")
    print(f"    火星接触率: {sim_mars.stats['contact_rate']*100:.1f}%")

    # 火星卫星通信记录
    records = sim_mars.get_communication_record(0)
    print(f"    MARS-SAT-0 通信记录: 共 {len(records)} 个接触窗口")

    input("\n  按 Enter 继续...")


def demo_8_configuration_loading():
    """演示8: 配置加载"""
    print("\n" + "=" * 60)
    print("  演示8: 从配置字典/JSON构建模拟器")
    print("=" * 60)

    from fl_space.config.loader import load_sim_config_from_dict

    # 用配置字典创建
    config = {
        "body": {"name": "earth"},
        "num_satellites": 5,
        "orbit_altitude_km": 400,
        "distribution": "walker",
        "ground_stations": [
            ("Beijing", 39.9, 116.4),
            ("Moscow", 55.75, 37.6),
            ("Cape Town", -33.9, 18.4),
        ],
        "num_timeslots": 720,  # 12小时
        "contact_mode": "full",
    }

    params = load_sim_config_from_dict(config)
    sim = OrbitSimulator(**params)
    print(sim.summary())

    print("\n  此配置完全通过字典定义，未使用硬编码默认值！")

    input("\n  按 Enter 继续...")


def demo_9_starlink_like():
    """演示9: Starlink-like 星座 + 地面站网络"""
    print("\n" + "=" * 60)
    print("  演示9: Starlink-like 星座 + 自定义地面站")
    print("=" * 60)

    from fl_space.config.defaults import CONSTELLATION_PRESETS

    cfg = CONSTELLATION_PRESETS["starlink_like"]
    const_config = ConstellationConfig(**{
        k: v for k, v in cfg.items()
        if k in ConstellationConfig.__dataclass_fields__
    })

    # 全球分布的15个地面站
    global_gs = [
        ("New York", 40.7, -74.0),
        ("London", 51.5, -0.1),
        ("Beijing", 39.9, 116.4),
        ("Tokyo", 35.7, 139.7),
        ("Sydney", -33.9, 151.2),
        ("Singapore", 1.3, 103.8),
        ("Moscow", 55.8, 37.6),
        ("Rio de Janeiro", -22.9, -43.2),
        ("Los Angeles", 34.1, -118.2),
        ("Nairobi", -1.3, 36.8),
        ("Frankfurt", 50.1, 8.7),
        ("Dubai", 25.2, 55.3),
        ("Seoul", 37.6, 127.0),
        ("Mexico City", 19.4, -99.1),
        ("Cairo", 30.0, 31.2),
    ]
    gs_network = GroundStationNetwork.from_tuples(global_gs)

    sim = OrbitSimulator(
        body=CelestialBody.earth(),
        constellation_config=const_config,
        ground_station_network=gs_network,
        timeslot_duration_min=1.0,
        num_timeslots=1440,
        verbose=False,  # 大星座不打印进度
    )
    print(sim.summary())

    # 某卫星的通信统计
    comm = sim.get_communication_record(0)
    print(f"\n  SAT-0 全天通信统计: {len(comm)} 个接触窗口")
    # 每个地面站的接触次数
    from collections import Counter
    gs_counter = Counter()
    for r in comm:
        for name in r['gs_names']:
            gs_counter[name] += 1
    print("  各基站联系次数:")
    for name, count in gs_counter.most_common(10):
        print(f"    {name}: {count} 次")


def main():
    """运行所有演示。"""
    demonstrations = [
        ("演示1: 天体参数", demo_1_celestial_bodies),
        ("演示2: 轨道力学", demo_2_orbit_mechanics),
        ("演示3: 星座分布", demo_3_constellation),
        ("演示4: 地面站管理", demo_4_ground_stations),
        ("演示5: 可见性计算", demo_5_visibility),
        ("演示6: 主模拟器-基础", demo_6_orbit_simulator_basic),
        ("演示7: 主模拟器-火星", demo_7_orbit_simulator_mars),
        ("演示8: 配置加载", demo_8_configuration_loading),
        ("演示9: Starlink-like", demo_9_starlink_like),
    ]

    print()
    print("╔" + "═" * 58 + "╗")
    print("║  SpaceFL 环境模拟 — 模块化框架演示" + " " * 21 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    for name, func in demonstrations:
        try:
            func()
        except KeyboardInterrupt:
            print("\n\n  用户中断。")
            break
        except Exception as e:
            print(f"\n  [错误] {name}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print("  所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
