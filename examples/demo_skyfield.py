"""
Skyfield 高精度后端演示 + 与环境模块集成测试

运行: python examples/demo_skyfield.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_space.environment import CelestialBody
from fl_space.orbit import SKYFIELD_AVAILABLE
from fl_space.simulator import OrbitSimulator


def demo_skyfield_basic():
    """演示: Skyfield后端基本用法"""
    print("=" * 60)
    print("  Skyfield 高精度后端 — 基本用法")
    print("=" * 60)

    sim = OrbitSimulator(
        backend="skyfield",
        num_satellites=3,
        num_ground_stations=3,
        orbit_altitude_km=500,
        num_timeslots=120,
        sim_start_date=(2024, 6, 1),
        verbose=False,
    )
    print(sim.summary())

    # 检查接触率
    rate = sim.stats['contact_rate']
    print(f"\n  Skyfield 接触率: {rate*100:.1f}%")

    # 查看某卫星通信记录
    comm = sim.get_communication_record(0)
    print(f"  SAT-0 通信窗口数: {len(comm)}")
    if comm:
        r = comm[0]
        print(f"    首窗口: TS={r['timeslot']}, GS={r['gs_names']}")


def demo_precise_body_params():
    """演示: JPL 星历精确参数 vs 默认参数"""
    print("\n" + "=" * 60)
    print("  JPL 星历精确参数对比")
    print("=" * 60)

    from fl_space.orbit.skyfield_backend import list_supported_bodies

    bodies = list_supported_bodies()
    print(f"  支持的天体: {bodies}\n")

    for name in ['earth', 'mars', 'moon', 'jupiter']:
        default = CelestialBody.from_name(name, precise=False)
        precise = CelestialBody.from_name(name, precise=True)

        if default:
            print(f"  {name.upper()}:")
            for attr in ['radius_km', 'GM', 'rotation_period_hours']:
                d_val = getattr(default, attr)
                p_val = getattr(precise, attr)
                diff = ""
                if d_val != p_val:
                    diff = f"  -> JPL: {p_val}"
                print(f"    {attr}: {d_val}{diff}")


def demo_kepler_vs_skyfield():
    """演示: Kepler vs Skyfield 后端对比"""
    print("\n" + "=" * 60)
    print("  Kepler vs Skyfield 后端对比")
    print("=" * 60)

    config = dict(
        num_satellites=3,
        num_ground_stations=3,
        orbit_altitude_km=500,
        num_timeslots=240,
        verbose=False,
    )

    t0 = __import__('time').time()
    sim_k = OrbitSimulator(backend="kepler", **config)
    t_k = __import__('time').time() - t0

    t0 = __import__('time').time()
    sim_s = OrbitSimulator(backend="skyfield", sim_start_date=(2024, 6, 1), **config)
    t_s = __import__('time').time() - t0

    print(f"\n  {'Metric':<25} {'Kepler':>12} {'Skyfield':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'接触率':<25} {sim_k.stats['contact_rate']:>11.1%} {sim_s.stats['contact_rate']:>11.1%}")
    print(f"  {'总接触次数':<25} {sim_k.stats['total_contacts']:>12} {sim_s.stats['total_contacts']:>12}")
    print(f"  {'生成耗时':<25} {t_k:>9.2f}s {t_s:>9.2f}s")
    print(f"  {'外部依赖':<25} {'无':>12} {'Skyfield+numpy':>12}")
    print(f"  {'轨道模型':<25} {'开普勒二体':>12} {'SGP4/JPL DE421':>12}")


def demo_planet_parameter_tuning():
    """演示: 自定义行星参数调优"""
    print("\n" + "=" * 60)
    print("  行星参数自定义 — 用户可调参展示")
    print("=" * 60)

    # 场景1: 地球默认 vs 小行星（半径1/3）
    custom_mini = CelestialBody(
        name="Mini-Earth",
        radius_km=2123.0,
        GM=14700.0,
        rotation_period_hours=12.0,
        atmosphere_height_km=30.0,
    )
    sim_mini = OrbitSimulator(
        body=custom_mini,
        num_satellites=3, num_ground_stations=2,
        orbit_altitude_km=300,
        num_timeslots=120,
        verbose=False,
    )
    print(f"\n  [自定义小行星] {custom_mini}")
    print(f"    地面站数量: {sim_mini.num_ground_stations}")
    print(f"    接触率: {sim_mini.stats['contact_rate']*100:.1f}%")
    print(f"    轨道周期: {sim_mini.orbit_period_min:.1f} min")


def demo_skyfield_visibility_windows():
    """演示: Skyfield 地面站可见窗口计算"""
    print("\n" + "=" * 60)
    print("  Skyfield 地面站可见窗口（rise/culminate/set）")
    print("=" * 60)

    from fl_space.orbit.skyfield_backend import SkyfieldOrbitBackend

    sf = SkyfieldOrbitBackend()
    sat = sf.create_satellite_from_kepler(
        altitude_km=500, inclination_deg=90.0,
        raan_deg=30.0, true_anomaly_deg=60.0,
        name="DEMO-SAT",
    )

    # 搜索北京上空可见窗口
    windows = sf.find_visibility_windows(
        sat, 39.9, 116.4,
        year=2024, month=6, day=1,
        duration_hours=12,
        min_elevation_deg=10.0,
    )

    print("\n  DEMO-SAT 对 Beijing [2024-06-01, 12h, min_elev=10°]:")
    print(f"  共 {len(windows)} 个可见窗口")
    for i, w in enumerate(windows[:5]):
        print(f"    [{i}] rise={w.get('rise_utc','?')} -> set={w.get('set_utc','?')}, "
              f"max_el={w.get('max_elevation_deg','?'):.1f}°")


if __name__ == "__main__":
    if not SKYFIELD_AVAILABLE:
        print("Skyfield 未安装。请运行: pip install skyfield")
        sys.exit(1)

    demos = [
        demo_skyfield_basic,
        demo_precise_body_params,
        demo_kepler_vs_skyfield,
        demo_planet_parameter_tuning,
        demo_skyfield_visibility_windows,
    ]

    print()
    print("=" * 60)
    print("  SpaceFL Skyfield 高精度后端集成演示")
    print("=" * 60)

    for demo in demos:
        try:
            demo()
            print()
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print("  演示完成")
    print("=" * 60)
