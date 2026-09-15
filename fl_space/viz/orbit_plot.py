"""
轨道可视化 — 2D 地图投影、卫星轨迹、地面站、接触矩阵热力图。

纯 matplotlib 实现，不需要 cartopy 等额外依赖。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

from fl_space.viz.i18n import setup_cjk_font, t, tf

if TYPE_CHECKING:
    from fl_space.environment import GroundStationNetwork
    from fl_space.orbit import KeplerOrbit

# 全局样式配置
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
})

# 地球颜色主题
EARTH_OCEAN = "#1a3a5c"
EARTH_LAND = "#2d5a27"
GRID_COLOR = "#ffffff33"
SATELLITE_COLORS = [
    "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff",
    "#ff922b", "#845ef7", "#20c997", "#f06595",
    "#339af0", "#fcc419", "#94d82d", "#ff8787",
]
GS_COLOR = "#ff6b6b"
ORBIT_COLOR = "#4d96ff55"
CONTACT_COLOR = "#51cf66"


def _wrap_lon(lon: float) -> float:
    """将经度包裹到 [-180, 180] 范围。"""
    return ((lon + 180) % 360) - 180


def _draw_earth_background(ax: plt.Axes):
    """绘制简化的地球背景（等距矩形投影）。"""
    # 海洋背景
    ax.fill_between([-180, 180], -90, 90, color=EARTH_OCEAN, zorder=0)

    # 简易大陆轮廓（关键区域）
    continents = [
        # 北美洲
        ([-170, -50], [15, 75]),
        # 南美洲
        ([-80, -35], [-55, 10]),
        # 欧洲
        ([-10, 40], [35, 70]),
        # 非洲
        ([-20, 50], [-35, 35]),
        # 亚洲
        ([40, 180], [5, 75]),
        # 澳大利亚
        ([110, 155], [-40, -10]),
        # 东亚（中国）
        ([70, 135], [15, 55]),
    ]
    for (lon_range, lat_range) in continents:
        ax.fill_between(lon_range, lat_range[0], lat_range[1],
                        color=EARTH_LAND, alpha=0.6, zorder=0)


def _compute_ground_track(
    orbit: KeplerOrbit,
    duration_hours: float = 24.0,
    n_points: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """计算卫星在地面的星下点轨迹。

    Returns
    -------
    lons, lats : np.ndarray
        经度和纬度数组。
    """
    times = np.linspace(0, duration_hours * 60, n_points)
    lats = np.zeros(n_points)
    lons = np.zeros(n_points)

    for i, t in enumerate(times):
        lat, lon = orbit.position_at_time(t)
        lats[i] = lat
        lons[i] = _wrap_lon(lon)

    # 修复跨越180度经线的不连续性
    for i in range(1, len(lons)):
        diff = lons[i] - lons[i - 1]
        if diff > 180:
            lons[i] -= 360
        elif diff < -180:
            lons[i] += 360

    return lons, lats


class OrbitVisualizer:
    """轨道可视化器。

    封装了常用的可视化方法，支持保存图片。
    """

    def __init__(self, figsize: tuple = (16, 8), lang: str = "en"):
        self.figsize = figsize
        self.lang = lang
        if lang == "zh":
            setup_cjk_font()

    def plot_map(
        self,
        orbits: list[KeplerOrbit],
        ground_stations: Optional[GroundStationNetwork] = None,
        duration_hours: float = 24.0,
        n_track_points: int = 300,
        title: str = "SpaceFL — 星座与地面站",
        sat_labels: Optional[list[str]] = None,
        show_ground_tracks: bool = True,
        snapshot_time_min: Optional[float] = None,
        save_path: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> plt.Figure:
        """2D 地图视图：卫星轨迹 + 地面站。

        Parameters
        ----------
        orbits : list of KeplerOrbit
            卫星轨道列表。
        ground_stations : GroundStationNetwork, optional
            地面站网络。
        duration_hours : float
            轨迹显示时长 (小时)。
        n_track_points : int
            轨迹采样点数。
        title : str
            图标题。
        sat_labels : list of str, optional
            卫星标签。
        show_ground_tracks : bool
            是否显示星下点轨迹。
        snapshot_time_min : float, optional
            快照时刻（标记卫星当前位置）。
        save_path : str, optional
            保存路径。

        Returns
        -------
        fig : matplotlib Figure
        """
        _lang = lang if lang is not None else self.lang
        fig, ax = plt.subplots(figsize=self.figsize)
        _draw_earth_background(ax)

        n_sats = len(orbits)

        # 轨迹
        if show_ground_tracks:
            for i, orbit in enumerate(orbits):
                lons, lats = _compute_ground_track(orbit, duration_hours, n_track_points)
                color = SATELLITE_COLORS[i % len(SATELLITE_COLORS)]
                ax.plot(lons, lats, color=color, alpha=0.3, linewidth=0.6, zorder=2)

        # 卫星当前位置（快照）
        if snapshot_time_min is not None:
            for i, orbit in enumerate(orbits):
                lat, lon = orbit.position_at_time(snapshot_time_min)
                lon = _wrap_lon(lon)
                color = SATELLITE_COLORS[i % len(SATELLITE_COLORS)]
                label = sat_labels[i] if sat_labels and i < len(sat_labels) else f"SAT-{i}"
                ax.scatter(lon, lat, c=color, s=40, edgecolors="white",
                          linewidths=0.8, zorder=5, label=label)

        # 地面站
        if ground_stations is not None:
            gs_lons = []
            gs_lats = []
            gs_names = []
            for i in range(len(ground_stations)):
                gs = ground_stations[i]
                if gs is not None:
                    gs_lons.append(gs.lon_deg)
                    gs_lats.append(gs.lat_deg)
                    gs_names.append(gs.name)

            ax.scatter(gs_lons, gs_lats, c=GS_COLOR, s=80, marker="^",
                      edgecolors="white", linewidths=1, zorder=6, label=t("Ground Station", _lang))

            # 标签
            for lon, lat, name in zip(gs_lons, gs_lats, gs_names):
                ax.annotate(name, (lon, lat), textcoords="offset points",
                           xytext=(5, 7), fontsize=7, color="white",
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="#00000088",
                                    edgecolor="none"),
                           zorder=7)

        # 图例
        if snapshot_time_min is not None:
            ax.legend(loc="upper left", fontsize=7, ncol=2,
                     framealpha=0.7, facecolor="#1a1a2e", labelcolor="white")

        # 样式
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.set_xlabel(t("Longitude (deg)", _lang))
        ax.set_ylabel(t("Latitude (deg)", _lang))
        ax.set_title(tf(title, _lang) if _lang == "zh" else title, fontweight="bold", color="white", pad=12)
        ax.set_facecolor(EARTH_OCEAN)
        fig.patch.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.15, color="white")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                       bbox_inches="tight")

        return fig

    def plot_contact_heatmap(
        self,
        contact_matrix: np.ndarray,
        num_satellites: int,
        num_stations: int,
        duration_hours: float,
        sat_labels: Optional[list[str]] = None,
        gs_labels: Optional[list[str]] = None,
        title: str = "SpaceFL — 接触矩阵热力图",
        save_path: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> plt.Figure:
        """接触矩阵热力图。

        Parameters
        ----------
        contact_matrix : np.ndarray
            形状为 (num_satellites, num_timeslots) 的接触矩阵 (simple模式)。
        num_satellites : int
        num_stations : int
        duration_hours : float
        sat_labels : list of str, optional
        gs_labels : list of str, optional
        title : str
        save_path : str, optional

        Returns
        -------
        fig : matplotlib Figure
        """
        _lang = lang if lang is not None else self.lang
        num_timeslots = contact_matrix.shape[1]
        fig, ax = plt.subplots(figsize=(14, max(4, num_satellites * 0.4)))

        # 二值化：有接触=1，无接触=0
        binary = (contact_matrix >= 0).astype(float)

        im = ax.imshow(binary, aspect="auto", cmap="YlOrRd",
                       interpolation="nearest", vmin=0, vmax=1)

        # 标签
        if sat_labels is None:
            sat_labels = [f"SAT-{i}" for i in range(num_satellites)]
        ax.set_yticks(range(num_satellites))
        ax.set_yticklabels(sat_labels[:num_satellites], fontsize=8)

        # 时间轴
        n_ticks = min(12, num_timeslots)
        tick_positions = np.linspace(0, num_timeslots - 1, n_ticks, dtype=int)
        tick_labels = [f"{p * duration_hours / num_timeslots:.1f}h" for p in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)

        ax.set_xlabel(t("Time", _lang))
        ax.set_ylabel(t("Satellite", _lang))
        ax.set_title(tf(title, _lang) if _lang == "zh" else title, fontweight="bold")

        # 接触率统计
        contact_rate = binary.mean() * 100
        ax.text(0.99, 1.02, tf("Contact Rate: {rate:.1f}%", _lang, rate=contact_rate),
                transform=ax.transAxes, ha="right", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_dashboard(
        self,
        orbits: list[KeplerOrbit],
        ground_stations: Optional[GroundStationNetwork] = None,
        contact_matrix: Optional[np.ndarray] = None,
        duration_hours: float = 24.0,
        snapshot_time_min: float = 0.0,
        sat_labels: Optional[list[str]] = None,
        gs_labels: Optional[list[str]] = None,
        cluster_map: Optional[dict[str, list[int]]] = None,
        title: str = "SpaceFL — 星座概览",
        save_path: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> plt.Figure:
        """综合仪表盘：地图 + 热力图 + 统计信息。

        Parameters
        ----------
        orbits, ground_stations, contact_matrix, duration_hours, snapshot_time_min,
        sat_labels, gs_labels, cluster_map, title, save_path
            参见上述各方法。

        Returns
        -------
        fig : matplotlib Figure
        """
        _lang = lang if lang is not None else self.lang
        fig = plt.figure(figsize=(16, 9), facecolor="#0d1117")
        gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 2],
                      hspace=0.35, wspace=0.35)

        # ---- 左上: 地图 ----
        ax_map = fig.add_subplot(gs[0, :2])
        _draw_earth_background(ax_map)

        n_sats = len(orbits)
        for i, orbit in enumerate(orbits):
            lons, lats = _compute_ground_track(orbit, duration_hours, 200)
            color = SATELLITE_COLORS[i % len(SATELLITE_COLORS)]
            ax_map.plot(lons, lats, color=color, alpha=0.25, linewidth=0.5, zorder=2)

        for i, orbit in enumerate(orbits):
            lat, lon = orbit.position_at_time(snapshot_time_min)
            lon = _wrap_lon(lon)
            color = SATELLITE_COLORS[i % len(SATELLITE_COLORS)]
            ax_map.scatter(lon, lat, c=color, s=30, edgecolors="white",
                          linewidths=0.5, zorder=5)

        if ground_stations is not None:
            for j in range(len(ground_stations)):
                station = ground_stations[j]
                if station is not None:
                    ax_map.scatter(station.lon_deg, station.lat_deg,
                                  c=GS_COLOR, s=60, marker="^",
                                  edgecolors="white", linewidths=0.8, zorder=6)
                    ax_map.annotate(station.name, (station.lon_deg, station.lat_deg),
                                   textcoords="offset points", xytext=(4, 5),
                                   fontsize=6, color="white",
                                   bbox=dict(boxstyle="round,pad=0.15",
                                            facecolor="#00000088", edgecolor="none"),
                                   zorder=7)

        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-90, 90)
        ax_map.set_xticks(np.arange(-180, 181, 60))
        ax_map.set_yticks(np.arange(-90, 91, 30))
        ax_map.set_facecolor(EARTH_OCEAN)
        ax_map.tick_params(colors="white")
        ax_map.grid(True, alpha=0.12, color="white")
        ax_map.set_title(t("Constellation Map", _lang), color="white", fontweight="bold")

        # ---- 右上: 统计面板 ----
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_stats.axis("off")
        ax_stats.set_facecolor("#161b22")

        stats_lines = [
            f"{t('Satellites:', _lang)} {n_sats}",
            f"{t('Ground Stations:', _lang)} {len(ground_stations) if ground_stations else 0}",
            f"{t('Altitude:', _lang)} {orbits[0].elements.semi_major_axis_km - orbits[0].body.radius_km:.0f} km" if orbits else "",
            f"{t('Duration:', _lang)} {duration_hours}h",
            f"{t('Snapshot:', _lang)} T={snapshot_time_min:.0f}min",
        ]
        if cluster_map:
            stats_lines.append("")
            stats_lines.append(f"{t('Clusters:', _lang)}")
            for name, indices in cluster_map.items():
                if name != "_custom":
                    stats_lines.append(f"  {name}: {len(indices)} {t('sats', _lang)}")

        if contact_matrix is not None:
            binary = contact_matrix >= 0
            rate = binary.mean() * 100
            stats_lines.append("")
            stats_lines.append(f"{t('Contact Rate:', _lang)} {rate:.1f}%")
            total_slots = contact_matrix.shape[1]
            avg_contacts = binary.sum(axis=0).mean()
            stats_lines.append(f"{t('Avg Contacts/slot:', _lang)} {avg_contacts:.1f}")

        y_pos = 0.95
        for line in stats_lines:
            ax_stats.text(0.05, y_pos, line, transform=ax_stats.transAxes,
                         fontsize=9, color="#c9d1d9", fontfamily="monospace",
                         verticalalignment="top")
            y_pos -= 0.04

        # ---- 底部: 接触热力图 ----
        if contact_matrix is not None:
            ax_heat = fig.add_subplot(gs[1, :])
            binary = (contact_matrix >= 0).astype(float)
            ax_heat.imshow(binary, aspect="auto", cmap="YlOrRd",
                          interpolation="nearest", vmin=0, vmax=1)

            if sat_labels is None:
                sat_labels = [f"SAT-{i}" for i in range(n_sats)]
            ax_heat.set_yticks(range(min(n_sats, len(sat_labels))))
            ax_heat.set_yticklabels(sat_labels[:n_sats], fontsize=7, color="white")
            ax_heat.tick_params(colors="white")

            n_slots = contact_matrix.shape[1]
            n_ticks = min(8, n_slots)
            tick_pos = np.linspace(0, n_slots - 1, n_ticks, dtype=int)
            tick_lab = [f"{p * duration_hours / n_slots:.1f}h" for p in tick_pos]
            ax_heat.set_xticks(tick_pos)
            ax_heat.set_xticklabels(tick_lab, fontsize=7, color="white", rotation=30)

            ax_heat.set_xlabel(t("Time", _lang), color="white")
            ax_heat.set_ylabel(t("Satellite", _lang), color="white")
            ax_heat.set_title(t("Contact Matrix", _lang), color="white", fontweight="bold")

        fig.suptitle(tf(title, _lang) if _lang == "zh" else title, color="white", fontsize=13, fontweight="bold", y=0.98)
        fig.patch.set_facecolor("#0d1117")

        if save_path:
            fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                       bbox_inches="tight")

        return fig


# ============================================================
# 便捷函数
# ============================================================


def plot_constellation_2d(
    orbits: list[KeplerOrbit],
    ground_stations: Optional[GroundStationNetwork] = None,
    duration_hours: float = 24.0,
    title: str = "SpaceFL Constellation",
    save_path: Optional[str] = None,
    lang: str = "en",
) -> plt.Figure:
    """快速绘制星座 2D 地图。"""
    viz = OrbitVisualizer(lang=lang)
    return viz.plot_map(
        orbits=orbits,
        ground_stations=ground_stations,
        duration_hours=duration_hours,
        snapshot_time_min=0.0,
        title=title,
        save_path=save_path,
        lang=lang,
    )


def plot_contact_heatmap(
    contact_matrix: np.ndarray,
    num_satellites: int,
    num_stations: int,
    duration_hours: float,
    title: str = "Contact Matrix",
    save_path: Optional[str] = None,
    lang: str = "en",
) -> plt.Figure:
    """快速绘制接触矩阵热力图。"""
    viz = OrbitVisualizer(lang=lang)
    return viz.plot_contact_heatmap(
        contact_matrix=contact_matrix,
        num_satellites=num_satellites,
        num_stations=num_stations,
        duration_hours=duration_hours,
        title=title,
        save_path=save_path,
        lang=lang,
    )


def plot_ground_track(
    orbit: KeplerOrbit,
    duration_hours: float = 24.0,
    title: str = "Ground Track",
    save_path: Optional[str] = None,
    lang: str = "en",
) -> plt.Figure:
    """快速绘制单星星下点轨迹。"""
    viz = OrbitVisualizer(lang=lang)
    return viz.plot_map(
        orbits=[orbit],
        duration_hours=duration_hours,
        snapshot_time_min=0.0,
        title=title,
        save_path=save_path,
        lang=lang,
    )


def quick_plot(
    simulator,
    title: str = "SpaceFL — Quick View",
    save_path: Optional[str] = None,
    lang: str = "en",
) -> plt.Figure:
    """一键可视化：传入模拟器，自动绘制仪表盘。

    Parameters
    ----------
    simulator : OrbitSimulator
        已运行的 OrbitSimulator 实例。
    title : str
    save_path : str, optional
    lang : str
        语言，'en' 或 'zh'。

    Returns
    -------
    fig : matplotlib Figure
    """
    viz = OrbitVisualizer(lang=lang)

    # 提取数据
    orbits = simulator.orbits
    ground_stations = simulator.ground_network
    contact_matrix = simulator.contact_matrix._simple
    duration_hours = (simulator.num_timeslots * simulator.timeslot_duration_min / 60.0)

    gs_labels = []
    if ground_stations is not None:
        for i in range(len(ground_stations)):
            gs = ground_stations[i]
            if gs:
                gs_labels.append(gs.name)

    return viz.plot_dashboard(
        orbits=orbits,
        ground_stations=ground_stations,
        contact_matrix=contact_matrix,
        duration_hours=duration_hours,
        snapshot_time_min=0.0,
        gs_labels=gs_labels,
        title=title,
        save_path=save_path,
        lang=lang,
    )
