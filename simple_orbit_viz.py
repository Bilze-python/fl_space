"""
SpaceFL 轨道可视化工具 - 简化版
直接使用已有的轨道数据生成PNG图片
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端
import matplotlib.pyplot as plt
import json
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_orbit(data_path='orbit_viz_data.json', output='orbit_plot.png'):
    """生成轨道可视化图片"""

    # 读取数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#0a1929')
    ax.set_facecolor('#0d1b2a')

    # 设置坐标轴
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude', color='white', fontsize=12)
    ax.set_ylabel('Latitude', color='white', fontsize=12)
    ax.set_title('SpaceFL Satellite Orbit Visualization',
                 color='white', fontsize=16, pad=20)
    ax.tick_params(colors='white')
    ax.grid(True, color='#1b263b', alpha=0.3, linestyle='--')

    # 获取时隙0的数据
    timeslots = data.get('timeslots', [])
    if not timeslots:
        print("No timeslot data found")
        return

    slot = timeslots[0]
    positions = slot.get('positions', [])
    ground_stations = data.get('ground_stations', [])
    contacts = slot.get('contacts', [])

    print(f"Satellites: {len(positions)}")
    print(f"Ground Stations: {len(ground_stations)}")
    print(f"Contacts: {len(contacts)}")

    # 绘制地面站
    for i, gs in enumerate(ground_stations):
        ax.plot(gs['lon'], gs['lat'], 'rs', markersize=12,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax.text(gs['lon'], gs['lat']+5, f'GS{i}',
               color='red', fontsize=10, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # 绘制连接线
    for contact in contacts:
        sat = positions[contact['sat_id']]
        gs = ground_stations[contact['gs_id']]
        ax.plot([sat['lon'], gs['lon']], [sat['lat'], gs['lat']],
               'y-', alpha=0.5, linewidth=2, zorder=2)

    # 绘制卫星
    for i, sat in enumerate(positions):
        has_contact = any(c['sat_id'] == i for c in contacts)
        color = '#00ff88' if has_contact else '#3498db'

        ax.plot(sat['lon'], sat['lat'], 'o', color=color,
               markersize=10, markeredgecolor='white',
               markeredgewidth=1.5, zorder=4)
        ax.text(sat['lon'], sat['lat']-5, f'S{i}',
               color='white', fontsize=9, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # 添加图例
    ax.plot([], [], 'rs', markersize=10, label='Ground Station', markeredgecolor='white')
    ax.plot([], [], 'o', color='#00ff88', markersize=8, label='Active Satellite', markeredgecolor='white')
    ax.plot([], [], 'o', color='#3498db', markersize=8, label='Idle Satellite', markeredgecolor='white')
    ax.plot([], [], 'y-', linewidth=2, label='Link')
    ax.legend(loc='upper right', facecolor='#0a1929', edgecolor='white',
             labelcolor='white', framealpha=0.9)

    # 保存图片
    plt.tight_layout()
    plt.savefig(output, dpi=150, facecolor='#0a1929')
    print(f"\nVisualization saved: {output}")
    print("Open this file to view the orbit visualization!")

if __name__ == '__main__':
    visualize_orbit()
