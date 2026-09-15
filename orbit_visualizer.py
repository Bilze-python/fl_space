"""
SpaceFL 轨道可视化工具 - 独立版本
使用 Matplotlib 和 Cartopy 创建专业的卫星轨道可视化

依赖:
pip install matplotlib cartopy skyfield numpy
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

# 尝试导入cartopy（地球地图）
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not installed. Using simple projection.")


class OrbitVisualizer:
    """轨道可视化工具"""

    def __init__(self, orbit_data_path=None):
        """
        初始化可视化工具

        Args:
            orbit_data_path: 轨道数据JSON文件路径
        """
        self.orbit_data = None
        self.fig = None
        self.ax = None
        self.current_slot = 0

        if orbit_data_path:
            self.load_orbit_data(orbit_data_path)

    def load_orbit_data(self, path):
        """加载轨道数据"""
        with open(path, 'r', encoding='utf-8') as f:
            self.orbit_data = json.load(f)
        print(f"Loaded orbit data:")
        print(f"  Satellites: {self.orbit_data['satellites']}")
        print(f"  Ground Stations: {len(self.orbit_data['ground_stations'])}")
        print(f"  Timeslots: {len(self.orbit_data['timeslots'])}")
        print(f"  ISL: {'Enabled' if self.orbit_data.get('isl_enabled') else 'Disabled'}")

    def create_figure_with_cartopy(self):
        """使用Cartopy创建地球地图"""
        self.fig = plt.figure(figsize=(14, 8))
        self.ax = plt.axes(projection=ccrs.Orthographic(0, 30))

        # 添加地球特征
        self.ax.add_feature(cfeature.LAND, facecolor='#2d3436', edgecolor='#636e72')
        self.ax.add_feature(cfeature.OCEAN, facecolor='#0984e3')
        self.ax.add_feature(cfeature.COASTLINE, edgecolor='#74b9ff', linewidth=0.5)
        self.ax.add_feature(cfeature.BORDERS, edgecolor='#636e72', linewidth=0.3, alpha=0.5)

        # 添加网格
        self.ax.gridlines(color='#636e72', alpha=0.3, linestyle='--')

        self.ax.set_global()
        self.ax.set_title('SpaceFL 卫星轨道可视化', fontsize=16, pad=20,
                         fontproperties='SimHei', color='white')
        self.fig.patch.set_facecolor('#1e272e')
        self.ax.set_facecolor('#1e272e')

    def create_figure_simple(self):
        """创建简单的2D投影图"""
        self.fig, self.ax = plt.subplots(figsize=(14, 8), facecolor='#1e272e')
        self.ax.set_facecolor('#0a3d62')

        # 绘制经纬度网格
        for lon in range(-180, 181, 30):
            self.ax.axvline(lon, color='#34495e', alpha=0.3, linewidth=0.5)
        for lat in range(-90, 91, 30):
            self.ax.axhline(lat, color='#34495e', alpha=0.3, linewidth=0.5)

        # 设置坐标轴
        self.ax.set_xlim(-180, 180)
        self.ax.set_ylim(-90, 90)
        self.ax.set_xlabel('经度 (°)', color='white', fontproperties='SimHei')
        self.ax.set_ylabel('纬度 (°)', color='white', fontproperties='SimHei')
        self.ax.set_title('SpaceFL 卫星轨道可视化', fontsize=16,
                         fontproperties='SimHei', color='white')
        self.ax.tick_params(colors='white')

        # 绘制简单的大陆轮廓
        self._draw_simple_continents()

    def _draw_simple_continents(self):
        """绘制简化的大陆轮廓"""
        # 简化的大陆边界数据
        continents = [
            # 亚洲
            [(60, 5), (95, 5), (105, 20), (120, 25), (135, 35), (140, 45),
             (135, 50), (110, 55), (90, 55), (75, 45), (60, 35)],
            # 欧洲
            [(-10, 35), (20, 35), (30, 45), (30, 60), (10, 65), (-10, 60)],
            # 非洲
            [(-20, 35), (40, 35), (50, 0), (40, -35), (20, -35), (-20, 0)],
            # 北美
            [(-170, 60), (-130, 65), (-100, 50), (-80, 25), (-100, 15),
             (-110, 20), (-140, 40), (-170, 50)],
            # 南美
            [(-80, 10), (-60, 10), (-50, -10), (-60, -30), (-70, -55), (-80, -40)],
        ]

        for continent in continents:
            lons, lats = zip(*continent)
            self.ax.plot(lons, lats, color='#2d3436', linewidth=2)
            self.ax.fill(lons, lats, color='#2d3436', alpha=0.7)

    def plot_static(self, slot_idx=0):
        """绘制静态帧"""
        if not self.orbit_data:
            raise ValueError("请先加载轨道数据")

        # 创建图形
        if HAS_CARTOPY:
            self.create_figure_with_cartopy()
        else:
            self.create_figure_simple()

        # 绘制一个时隙
        self._draw_slot(slot_idx)

        plt.tight_layout()
        plt.show()

    def _draw_slot(self, slot_idx):
        """绘制指定时隙的卫星和连接"""
        slot = self.orbit_data['timeslots'][slot_idx]
        positions = slot['positions']
        contacts = slot.get('contacts', [])
        isl_links = slot.get('isl_links', [])

        # 绘制地面站
        for i, gs in enumerate(self.orbit_data['ground_stations']):
            if HAS_CARTOPY:
                self.ax.plot(gs['lon'], gs['lat'], 'rs', markersize=10,
                           transform=ccrs.PlateCarree(), zorder=5,
                           label='地面站' if i == 0 else '')
                self.ax.text(gs['lon'], gs['lat'] + 3, f'GS{i}',
                           transform=ccrs.PlateCarree(),
                           fontsize=8, ha='center', color='red',
                           fontproperties='SimHei')
            else:
                self.ax.plot(gs['lon'], gs['lat'], 'rs', markersize=10, zorder=5)
                self.ax.text(gs['lon'], gs['lat'] + 3, f'GS{i}',
                           fontsize=8, ha='center', color='red',
                           fontproperties='SimHei')

        # 绘制卫星-地面站连接
        for contact in contacts:
            sat = positions[contact['sat_id']]
            gs = self.orbit_data['ground_stations'][contact['gs_id']]

            if HAS_CARTOPY:
                self.ax.plot([sat['lon'], gs['lon']], [sat['lat'], gs['lat']],
                           'y-', alpha=0.4, linewidth=1.5,
                           transform=ccrs.PlateCarree(), zorder=2)
            else:
                self.ax.plot([sat['lon'], gs['lon']], [sat['lat'], gs['lat']],
                           'y-', alpha=0.4, linewidth=1.5, zorder=2)

        # 绘制ISL链路
        if self.orbit_data.get('isl_enabled') and isl_links:
            for link in isl_links:
                satA = positions[link['a_id']]
                satB = positions[link['b_id']]

                if HAS_CARTOPY:
                    self.ax.plot([satA['lon'], satB['lon']],
                               [satA['lat'], satB['lat']],
                               'c-', alpha=0.3, linewidth=1,
                               transform=ccrs.PlateCarree(), zorder=1)
                else:
                    self.ax.plot([satA['lon'], satB['lon']],
                               [satA['lat'], satB['lat']],
                               'c-', alpha=0.3, linewidth=1, zorder=1)

        # 绘制卫星
        for i, sat in enumerate(positions):
            has_contact = any(c['sat_id'] == i for c in contacts)
            color = '#00d2d3' if has_contact else '#4b7bec'

            if HAS_CARTOPY:
                self.ax.plot(sat['lon'], sat['lat'], 'o', color=color,
                           markersize=8, transform=ccrs.PlateCarree(), zorder=4)
                self.ax.text(sat['lon'], sat['lat'] - 5, f'S{i}',
                           transform=ccrs.PlateCarree(),
                           fontsize=7, ha='center', color='white',
                           fontproperties='SimHei')
            else:
                self.ax.plot(sat['lon'], sat['lat'], 'o', color=color,
                           markersize=8, zorder=4)
                self.ax.text(sat['lon'], sat['lat'] - 5, f'S{i}',
                           fontsize=7, ha='center', color='white',
                           fontproperties='SimHei')

        # 显示时间和统计信息
        time_str = slot.get('time', f'Slot {slot_idx}')
        info_text = f'时间: {time_str}\n卫星: {len(positions)}\n连接: {len(contacts)}'
        if isl_links:
            info_text += f'\nISL: {len(isl_links)}'

        self.ax.text(0.02, 0.98, info_text,
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                    color='white', fontsize=10, fontproperties='SimHei')

    def animate(self, save_path=None, interval=500):
        """创建动画"""
        if not self.orbit_data:
            raise ValueError("请先加载轨道数据")

        # 创建图形
        if HAS_CARTOPY:
            self.create_figure_with_cartopy()
        else:
            self.create_figure_simple()

        def update(frame):
            self.ax.clear()
            if HAS_CARTOPY:
                self.create_figure_with_cartopy()
            else:
                self.create_figure_simple()
            self._draw_slot(frame)
            return self.ax,

        anim = animation.FuncAnimation(
            self.fig, update,
            frames=len(self.orbit_data['timeslots']),
            interval=interval, blit=False, repeat=True
        )

        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=2)
            print("Animation saved successfully")
        else:
            plt.show()


def main():
    """主函数"""
    import sys

    print("=" * 60)
    print("SpaceFL 轨道可视化工具")
    print("=" * 60)

    # 检查依赖
    if not HAS_CARTOPY:
        print("\nWarning: cartopy not installed. Using simple projection")
        print("  Install: pip install cartopy")

    # 查找轨道数据文件
    orbit_files = list(Path('.').glob('orbit_*.json'))
    if not orbit_files:
        print("\n✗ 未找到轨道数据文件")
        print("  请先运行以下命令生成轨道数据:")
        print("  python -m fl_space.cli run simulate --output orbit_data.json")
        return

    # 使用最新的轨道数据
    orbit_file = sorted(orbit_files)[-1]
    print(f"\nUsing orbit data: {orbit_file}")

    # 创建可视化
    viz = OrbitVisualizer(orbit_file)

    # 显示选项
    print("\nOptions:")
    print("1. Show static plot (single timeslot)")
    print("2. Show animation (all timeslots)")
    print("3. Save animation as GIF")

    choice = input("\nSelect (1/2/3): ").strip()

    if choice == '1':
        slot = input(f"输入时隙编号 (0-{len(viz.orbit_data['timeslots'])-1}): ")
        viz.plot_static(int(slot))
    elif choice == '2':
        viz.animate()
    elif choice == '3':
        output = input("输入输出文件名 (如: orbit_anim.gif): ")
        viz.animate(save_path=output or 'orbit_animation.gif')
    else:
        print("无效选择")


if __name__ == '__main__':
    main()
