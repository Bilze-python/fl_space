"""
快速验证脚本 — 测试所有模块的基本功能
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_space.environment import CelestialBody, create_default_network
from fl_space.orbit import create_circular_orbit, generate_cluster_phases, generate_walker_phases
from fl_space.simulator import OrbitSimulator, create_mars_simulator

print('=== 基础导入成功 ===')

# 天体
earth = CelestialBody.earth()
mars = CelestialBody.mars()
moon = CelestialBody.moon()
print(f'Earth: {earth}')
print(f'Mars: {mars}')
print(f'Moon: {moon}')

# 轨道计算
orbit = create_circular_orbit(500, 90, 0, 0, earth)
lat, lon = orbit.position_at_time_deg(60)
print(f'Position @60min: ({lat:.2f}, {lon:.2f})')
assert abs(orbit.period_min - 94.6) < 1.0, f"Period mismatch: {orbit.period_min}"

# Walker星座
orbs = generate_walker_phases(5, 2, 1)
print(f'Walker 5/2/1: {len(orbs)} orbits')
assert len(orbs) == 5

# 星簇
orbs_c = generate_cluster_phases(6, 2)
print(f'Cluster 6/2: {len(orbs_c)} orbits')
assert len(orbs_c) == 6

# 地面站
network = create_default_network(3)
print(f'GroundStations: {network}')
assert network.count == 3

# 模拟器快速测试（小规模）
sim = OrbitSimulator(verbose=False, num_satellites=2, num_ground_stations=2, num_timeslots=360)
print(f'Simulator: {sim}')
print(f'Contact rate: {sim.stats["contact_rate"]:.2%}')
comm = sim.get_communication_record(0)
print(f'SAT-0 comm records: {len(comm)}')
detail = sim.get_contact_detail(0, 60)
print(f'SAT-0 @ TS60 detail: {detail}')

# 火星模拟器
sim_mars = create_mars_simulator()
print(f'Mars Simulator: {sim_mars}')
print(f'Mars contact rate: {sim_mars.stats["contact_rate"]:.2%}')

# 接触矩阵导出
sim.export("test_contact.json")
print('Export OK')
os.remove("test_contact.json")

print('\n=== 全部测试通过 ===')
