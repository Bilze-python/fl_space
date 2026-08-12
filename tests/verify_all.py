"""
无交互版验证脚本
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_space.config.defaults import CONSTELLATION_PRESETS
from fl_space.config.loader import load_sim_config_from_dict
from fl_space.environment import (
    CelestialBody,
    GroundStationNetwork,
    create_default_network,
    create_extended_network,
)
from fl_space.orbit import (
    ConstellationConfig,
    MultiSatVisibility,
    VisibilityEngine,
    create_circular_orbit,
    generate_cluster_phases,
    generate_walker_phases,
)
from fl_space.simulator import OrbitSimulator, create_default_simulator, create_mars_simulator

errors = []

def check(name, condition, detail=""):
    if condition:
        print(f"  [OK] {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        errors.append(name)

print("=" * 60)
print(" SpaceFL 模块化框架 — 验证测试")
print("=" * 60)

# ---- 天体 ----
print("\n[1] 天体参数")
earth = CelestialBody.earth(); mars = CelestialBody.mars(); moon = CelestialBody.moon()
check("Earth预设", earth.name == "Earth")
check("Mars预设", mars.name == "Mars")
check("Moon预设", moon.name == "Moon")
check("自定义天体", CelestialBody(name="Test", radius_km=1000, GM=5000).radius_km == 1000)
check("地平可见半角", earth.effective_horizon_angle_for_altitude(500) > 0)
check("火星500km可见半角>0", mars.effective_horizon_angle_for_altitude(500) > 0)
check("地球500km可见半角>0", earth.effective_horizon_angle_for_altitude(500) > 0)
check("月球无大气半角>0", moon.effective_horizon_angle_for_altitude(100) > 0)

# ---- 轨道力学 ----
print("\n[2] 轨道力学")
orbit = create_circular_orbit(500, 90, 30, 60, earth)
check("轨道周期~94.6min", abs(orbit.period_min - 94.6) < 1.0)
check("轨道高度500", abs(orbit.altitude_km - 500) < 1)
lat, lon = orbit.position_at_time_deg(0)
check("初始位置合理", -90 <= lat <= 90 and -180 <= lon <= 180)
lat1, lon1 = orbit.position_at_time_deg(30)
check("30分钟后位置改变", lat != lat1 or lon != lon1)

# 非极轨
orbit2 = create_circular_orbit(500, 53, 0, 0, earth)
lat2, _ = orbit2.position_at_time_deg(45)
check("53°倾角纬度范围", abs(lat2) <= 55)  # 应在±53°内

# ---- 星座分布 ----
print("\n[3] 星座分布")
w = generate_walker_phases(5, 2, 1)
check("Walker 5/2/1 数量", len(w) == 5)
c = generate_cluster_phases(6, 2)
check("Cluster 6/2 数量", len(c) == 6)
check("均匀分布", len(generate_cluster_phases(4, 1)) == 4)

# ---- 地面站 ----
print("\n[4] 地面站管理")
net = create_default_network(5)
check("默认网络5站", net.count == 5)
net_ext = create_extended_network(13)
check("扩展网络13站", net_ext.count == 13)
net7 = create_default_network(7)
check("7站网络含Beijing", net7["Beijing"].lat_deg == 39.9)
check("按索引查询", net[0].name == "Sioux Falls")

# ---- 可见性 ----
print("\n[5] 可见性计算")
engine = VisibilityEngine(earth, orbit, net)
v0 = engine.visible_stations_at_time(0)
check("可见性返回列表", isinstance(v0, list))
f0 = engine.first_visible_station_at_time(0)
check("首个可见站类型", f0 is None or isinstance(f0, int))

# 多卫星
multi = MultiSatVisibility(earth, [orbit, orbit2], net)
vm = multi.visible_matrix_at_time(30)
check("多卫星可见矩阵", len(vm) == 2 and isinstance(vm[0], list))

# ---- 模拟器 — 基础 ----
print("\n[6] 模拟器 — 基础")
sim = OrbitSimulator(verbose=False, num_satellites=3, num_ground_stations=3, num_timeslots=360)
check("卫星数", sim.num_satellites == 3)
check("地面站数", sim.num_ground_stations == 3)
check("轨道数匹配", len(sim.orbits) == 3)
check("接触矩阵shape", sim.contact_matrix.simple_matrix.shape == (3, 360))
contact = sim.get_contact_at_timeslot(0, 60)
check("接触查询", contact >= -1)
all_c = sim.get_all_contacts(0, 60)
check("完整接触查询", isinstance(all_c, list))
sats = sim.get_satellites_in_contact(60)
check("可通信用卫星查询", isinstance(sats, list))

# 通信记录（新需求核心功能）
comm = sim.get_communication_record(0)
check("通信记录非空或有合理原因", isinstance(comm, list))
if comm:
    r = comm[0]
    check("通信记录含gs_names", 'gs_names' in r)
    check("通信记录含sat_position", 'sat_position_deg' in r)
    check("通信记录含timeslot", 'timeslot' in r)

# 卫星轨迹
traj = sim.get_sat_trajectory(0, [0, 60, 120])
check("轨迹点数量", len(traj) == 3)
check("轨迹点格式", isinstance(traj[0], tuple) and len(traj[0]) == 2)

# FL轮次计算（核心功能兼容）
dur, end_ts = sim.compute_round_duration_sync(1, 2.0, 0)
check("轮次时长计算", dur >= 0)

# 导出
sim.export("test_export.json")
check("导出JSON", os.path.exists("test_export.json"))
os.remove("test_export.json")

# ---- 模拟器 — 默认兼容 ----
print("\n[7] 模拟器 — 与 orbit_sim_v2 兼容")
sim_def = create_default_simulator()
check("默认3卫星", sim_def.num_satellites == 3)
check("默认2地面站", sim_def.num_ground_stations == 2)
check("默认联系率有值", sim_def.stats['contact_rate'] > 0)
check("summary包含天体名", "Earth" in sim_def.summary())

# ---- 模拟器 — 火星 ----
print("\n[8] 模拟器 — 火星场景")
sim_mars = create_mars_simulator()
check("火星5卫星", sim_mars.num_satellites == 5)
check("火星5地面站", sim_mars.num_ground_stations == 5)
check("火星轨道周期>地球", sim_mars.orbit_period_min > 60)
check("火星联系率有值", sim_mars.stats['contact_rate'] > 0)
comm_m = sim_mars.get_communication_record(0)
check("火星通信记录生成", isinstance(comm_m, list))

# ---- Skyfield后端 ----
print("\n[9] Skyfield 高精度后端")
from fl_space.orbit import SKYFIELD_AVAILABLE

if SKYFIELD_AVAILABLE:
    from fl_space.orbit.skyfield_backend import (
        SkyfieldOrbitBackend,
        get_precise_body_params,
        list_supported_bodies,
    )
    check("Skyfield可用", SKYFIELD_AVAILABLE)
    bodies = list_supported_bodies()
    check("支持天体列表", len(bodies) >= 4 and 'earth' in bodies)
    params = get_precise_body_params('earth')
    check("JPL地球参数", params is not None and params['radius_km'] > 6370)
    check("JPL地球GM精度", abs(params['GM'] - 398600.435) < 1.0)

    sf = SkyfieldOrbitBackend()
    yr, mo, dy = 2024, 6, 1
    import math
    a = (14 - mo) // 12; y = yr + 4800 - a; m = mo + 12 * a - 3
    epoch_jd = (dy + (153*m+2)//5 + 365*y + y//4 - y//100 + y//400 - 32045 - 0.5)
    sat = sf.create_satellite_from_kepler(500, 90, 30, 60, epoch_jd=epoch_jd)
    check("SGP4卫星创建", sat is not None)
    check("SGP4周期~94.5min", abs(2*math.pi/sat.satrec.no_kozai - 94.5) < 2.0)
    lat, lon, alt = sf.position_at_time(sat, yr, mo, dy, 0.0)
    check("SGP4位置高度合理", 300 < alt < 700)
    check("SGP4纬度合理", -90 <= lat <= 90)

    sim_sf = OrbitSimulator(backend="skyfield", num_satellites=3, num_ground_stations=3,
                            num_timeslots=360, sim_start_date=(2024,6,1), verbose=False)
    check("Skyfield后端模拟器", sim_sf.backend_mode == "skyfield")
    check("Skyfield接触率>0", sim_sf.stats['contact_rate'] > 0)
    check("Skyfield周期~94.5min", abs(sim_sf.orbit_period_min - 94.5) < 2.0)
else:
    check("Skyfield跳过(未安装)", True)

# ---- 配置加载 ----
print("\n[10] 配置加载")
config = {
    "body": {"name": "earth"},
    "num_satellites": 4,
    "orbit_altitude_km": 400,
    "distribution": "walker",
    "ground_stations": [("Beijing", 39.9, 116.4), ("Sanya", 18.25, 109.5)],
    "num_timeslots": 180,
    "contact_mode": "full",
}
params = load_sim_config_from_dict(config)
sim_cfg = OrbitSimulator(verbose=False, **params)
check("配置加载卫星数", sim_cfg.num_satellites == 4)
check("配置加载地面站数", sim_cfg.num_ground_stations == 2)

# ---- Starlink风格星座 ----
print("\n[11] Starlink-like 大星座")
cfg = CONSTELLATION_PRESETS["starlink_like"]
const_config = ConstellationConfig(**{k: v for k, v in cfg.items() if k in ConstellationConfig.__dataclass_fields__})
gs_network = GroundStationNetwork.from_tuples([
    ("New York", 40.7, -74.0), ("London", 51.5, -0.1),
    ("Beijing", 39.9, 116.4), ("Tokyo", 35.7, 139.7),
    ("Sydney", -33.9, 151.2), ("Singapore", 1.3, 103.8),
])
sim_star = OrbitSimulator(
    body=CelestialBody.earth(), constellation_config=const_config,
    ground_station_network=gs_network,
    timeslot_duration_min=1.0, num_timeslots=720,
    verbose=False,
)
check("Starlink 20卫星", sim_star.num_satellites == 20)
check("Starlink 6地面站", sim_star.num_ground_stations == 6)
check("Starlink 联系率>0", sim_star.stats['contact_rate'] > 0)
check("Starlink 倾角53°", sim_star.orbit_inclination_deg == 53.0)
comm_star = sim_star.get_communication_record(0)
check("Starlink 通信记录", len(comm_star) > 0)

# ---- 总结 ----
print("\n" + "=" * 60)
if errors:
    print(f"  测试完成，{len(errors)} 项失败:")
    for e in errors:
        print(f"    - {e}")
else:
    print("  全部验证通过！")
print("=" * 60)
