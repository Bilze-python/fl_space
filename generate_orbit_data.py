"""生成轨道测试数据"""
import sys
sys.path.insert(0, '.')
from fl_space.orbit.orbit_sim import OrbitSimulator
import json

# 生成轨道数据
print("生成轨道数据...")
sim = OrbitSimulator(num_sats=10, num_gs=5, sim_hours=2, backend='kepler')
orbit_data = sim.build_orbit_data()

# 保存
with open('orbit_data.json', 'w', encoding='utf-8') as f:
    json.dump(orbit_data, f, ensure_ascii=False, indent=2)

print('orbit_data.json')
print(f'Satellites: {orbit_data["satellites"]}')
print(f'Timeslots: {len(orbit_data["timeslots"])}')
print('Ground Stations:', len(orbit_data["ground_stations"]))
