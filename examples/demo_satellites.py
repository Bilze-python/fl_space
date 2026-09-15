"""
SpaceFL Step 1+2 综合演示 — 环境模拟 + 卫星设置 + 可视化

演示内容:
  1. 多星簇星座创建（极轨 + 中倾角 LEO 双壳层）
  2. 用户自定义卫星注册（装饰器注册）
  3. 从注册表获取配置
  4. 模拟运行 + 接触计算
  5. 可视化仪表盘
  6. 不同星座对比
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # 非交互后端

import matplotlib.pyplot as plt

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_space.environment import (
    CelestialBody,
    create_default_network,
)
from fl_space.orbit import (
    ClusterSpec,
    MultiClusterConfig,
    SatelliteSpec,
    registry,
)
from fl_space.simulator import OrbitSimulator
from fl_space.viz import OrbitVisualizer
from fl_space.viz.i18n import tf

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo1_multi_cluster(lang: str = "en"):
    """演示1: 多星簇星座"""
    print("=" * 60)
    print("Demo 1: 多星簇星座 — 极轨(4) + LEO壳层(6)")
    print("=" * 60)

    earth = CelestialBody.earth()

    # 创建双星簇
    config = MultiClusterConfig(
        clusters=[
            ClusterSpec(
                name="polar_cluster",
                num_satellites=4,
                altitude_km=500,
                inclination_deg=90,
                distribution="uniform",
                raan_offset_deg=0,
            ),
            ClusterSpec(
                name="leo_shell",
                num_satellites=6,
                altitude_km=550,
                inclination_deg=53,
                distribution="walker",
                num_planes=3,
                raan_offset_deg=20,
            ),
        ]
    )

    orbits, cluster_map = config.generate_orbits(earth)
    print(f"Total satellites: {config.total_satellites}")
    for name, indices in cluster_map.items():
        print(f"  {name}: {len(indices)} sats -> indices {indices}")

    # 地面站
    gss = create_default_network(7)

    # 模拟
    sim = OrbitSimulator(
        body=earth,
        orbits=orbits,
        ground_station_network=gss,

        backend="kepler",
        verbose=False,
    )
    print(f"Contact rate: {sim.stats['contact_rate']:.1%}")

    # 通信记录样例
    for sat_id in range(min(3, len(orbits))):
        rec = sim.get_communication_record(sat_id)
        contact_count = sum(1 for r in rec if r.get("contact_count", 0) > 0)
        print(f"  SAT-{sat_id}: {contact_count} contact slots")

    # 可视化
    viz = OrbitVisualizer(figsize=(16, 9), lang=lang)
    gs_labels = [gss[i].name for i in range(len(gss))]

    fig = viz.plot_dashboard(
        orbits=orbits,
        ground_stations=gss,
        contact_matrix=sim.contact_matrix._simple,

        snapshot_time_min=0.0,
        gs_labels=gs_labels,
        cluster_map=cluster_map,
        title=tf("Demo 1: Dual-Shell Constellation (Polar + LEO)", lang),
        save_path=os.path.join(OUTPUT_DIR, "demo1_multi_cluster.png"),
        lang=lang,
    )
    plt.close(fig)
    print("Saved: demo1_multi_cluster.png")

    return sim


def demo2_custom_registry(lang: str = "en"):
    """演示2: 用户自定义卫星注册"""
    print("\n" + "=" * 60)
    print("Demo 2: 用户自定义卫星注册表")
    print("=" * 60)

    # 注册自定义卫星类型
    @registry.register("my_research_sats", description="研究用混合轨道12星")
    def my_research_sats(body=None):
        if body is None:
            body = CelestialBody.earth()
        return MultiClusterConfig(
            clusters=[
                ClusterSpec("high_lat", num_satellites=3, altitude_km=600,
                           inclination_deg=85, distribution="uniform"),
                ClusterSpec("mid_lat", num_satellites=5, altitude_km=450,
                           inclination_deg=45, distribution="walker",
                           num_planes=2, raan_offset_deg=30),
                ClusterSpec("low_lat", num_satellites=4, altitude_km=550,
                           inclination_deg=20, distribution="uniform",
                           raan_offset_deg=60),
            ]
        )

    # 用字典注册
    registry.register_from_dict(
        "dict_geo_cluster",
        {
            "clusters": [
                {"name": "geo_belt", "num_satellites": 3, "altitude_km": 35786,
                 "inclination_deg": 0, "distribution": "uniform"},
            ]
        },
        description="3颗GEO同步轨道卫星",
    )

    # 查看注册表
    print(registry)

    # 使用注册的自定义类型
    earth = CelestialBody.earth()
    config = registry.get("my_research_sats", body=earth)
    orbits, cluster_map = config.generate_orbits(earth)
    print(f"\nCustom type 'my_research_sats': {config.total_satellites} satellites")

    gss = create_default_network(7)
    sim = OrbitSimulator(
        body=earth, orbits=orbits, ground_station_network=gss,
         backend="kepler", verbose=False,
    )
    print(f"Contact rate: {sim.stats['contact_rate']:.1%}")

    viz = OrbitVisualizer(lang=lang)
    fig = viz.plot_dashboard(
        orbits=orbits,
        ground_stations=gss,
        contact_matrix=sim.contact_matrix._simple,

        snapshot_time_min=0.0,
        cluster_map=cluster_map,
        title=tf("Demo 2: Custom Research Constellation (via Registry)", lang),
        save_path=os.path.join(OUTPUT_DIR, "demo2_custom_registry.png"),
        lang=lang,
    )
    plt.close(fig)
    print("Saved: demo2_custom_registry.png")

    return sim


def demo3_custom_satellites(lang: str = "en"):
    """演示3: 用户自定义单星加入星座"""
    print("\n" + "=" * 60)
    print("Demo 3: 自定义单星 + 星簇混合")
    print("=" * 60)

    earth = CelestialBody.earth()

    config = MultiClusterConfig(
        clusters=[
            ClusterSpec("walker_shell", num_satellites=5, altitude_km=500,
                       inclination_deg=53, distribution="walker",
                       num_planes=2),
        ],
        custom_satellites=[
            SatelliteSpec(
                name="Custom-SAT-A",
                altitude_km=800,
                inclination_deg=98,  # SSO 太阳同步轨道
                raan_deg=30,
                true_anomaly_deg=0,
                tx_power_dbm=40,
            ),
            SatelliteSpec(
                name="Custom-SAT-B",
                altitude_km=1200,
                inclination_deg=0,  # 赤道轨道
                raan_deg=0,
                true_anomaly_deg=180,
                tx_power_dbm=35,
            ),
        ],
    )

    orbits, cluster_map = config.generate_orbits(earth)
    print(f"Total: {config.total_satellites} sats (5 cluster + 2 custom)")
    for name, idx in cluster_map.items():
        print(f"  {name}: indices {idx}")

    gss = create_default_network(9)
    sim = OrbitSimulator(
        body=earth, orbits=orbits, ground_station_network=gss,
         backend="kepler", verbose=False,
    )
    print(f"Contact rate: {sim.stats['contact_rate']:.1%}")

    sat_labels = [f"SAT-{i}" for i in range(len(orbits))]
    sat_labels[5] = "Custom-A"
    sat_labels[6] = "Custom-B"

    viz = OrbitVisualizer(lang=lang)
    fig = viz.plot_dashboard(
        orbits=orbits,
        ground_stations=gss,
        contact_matrix=sim.contact_matrix._simple,

        snapshot_time_min=0.0,
        sat_labels=sat_labels,
        cluster_map=cluster_map,
        title=tf("Demo 3: Mixed — Walker Shell + Custom Satellites (SSO + Equatorial)", lang),
        save_path=os.path.join(OUTPUT_DIR, "demo3_custom_sats.png"),
        lang=lang,
    )
    plt.close(fig)
    print("Saved: demo3_custom_sats.png")

    return sim


def demo4_comparison(lang: str = "en"):
    """演示4: 不同星座配置对比"""
    print("\n" + "=" * 60)
    print("Demo 4: 星座配置对比")
    print("=" * 60)

    earth = CelestialBody.earth()
    gss = create_default_network(7)

    configs = [
        ("Polar Only (6 sats)", MultiClusterConfig.polar_only(6)),
        ("Starlink-like (20 sats)", MultiClusterConfig.starlink_like(20)),
        ("Mixed Orbit (10 sats)", MultiClusterConfig.mixed_orbit(4, 4, 2)),
        ("Dual Shell (10 sats)", MultiClusterConfig.demo_default()),
    ]

    results = []
    for name, config in configs:
        orbits, _cmap = config.generate_orbits(earth)
        sim = OrbitSimulator(
            body=earth, orbits=orbits, ground_station_network=gss,
             backend="kepler", verbose=False,
        )
        results.append((name, sim, _cmap, len(orbits)))
        print(f"  {name:30s} | Contact: {sim.stats['contact_rate']:6.1%} | {len(orbits)} sats")

    # 对比图：2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0d1117")
    viz = OrbitVisualizer()

    for ax, (name, sim, _cmap, _n_sats) in zip(axes.flat, results):
        from fl_space.viz.orbit_plot import (
            GS_COLOR,
            SATELLITE_COLORS,
            _compute_ground_track,
            _draw_earth_background,
            _wrap_lon,
        )

        _draw_earth_background(ax)
        orbits_list = sim.orbits

        for i, orbit in enumerate(orbits_list):
            lons, lats = _compute_ground_track(orbit, 24, 150)
            color = SATELLITE_COLORS[i % len(SATELLITE_COLORS)]
            ax.plot(lons, lats, color=color, alpha=0.25, linewidth=0.4)

        for i, orbit in enumerate(orbits_list):
            lat, lon = orbit.position_at_time(0)
            lon = _wrap_lon(lon)
            color = SATELLITE_COLORS[i % len(SATELLITE_COLORS)]
            ax.scatter(lon, lat, c=color, s=20, edgecolors="white", linewidths=0.4)

        for j in range(len(gss)):
            gs = gss[j]
            if gs:
                ax.scatter(gs.lon_deg, gs.lat_deg,
                          c=GS_COLOR, s=40, marker="^",
                          edgecolors="white", linewidths=0.5)

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_facecolor("#1a3a5c")
        ax.tick_params(colors="white", labelsize=7)
        ax.grid(True, alpha=0.1, color="white")
        ax.set_title(f"{name}\nContact: {sim.stats['contact_rate']:.1%}",
                    color="white", fontsize=10, fontweight="bold")

    fig.suptitle(tf("Demo 4: Constellation Comparison", lang),
                color="white", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "demo4_comparison.png"),
               dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("Saved: demo4_comparison.png")


def demo5_ground_track(lang: str = "en"):
    """演示5: 不同的接触模式表现"""
    print("\n" + "=" * 60)
    print("Demo 5: 单星星下点轨迹 + 地面站覆盖")
    print("=" * 60)

    earth = CelestialBody.earth()
    config = MultiClusterConfig(clusters=[
        ClusterSpec("polar", num_satellites=3, altitude_km=500,
                   inclination_deg=90, distribution="uniform"),
        ClusterSpec("mid", num_satellites=3, altitude_km=550,
                   inclination_deg=30, distribution="uniform",
                   raan_offset_deg=120),
    ])
    orbits, _cmap = config.generate_orbits(earth)
    gss = create_default_network(7)

    sim = OrbitSimulator(
        body=earth, orbits=orbits, ground_station_network=gss,
         backend="kepler", verbose=False,
    )

    # 绘制单星星下点轨迹（带接触高亮）
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#0d1117")

    from fl_space.viz.orbit_plot import (
        _compute_ground_track,
        _draw_earth_background,
        _wrap_lon,
    )

    for sat_id in range(3):
        ax = axes[sat_id]
        _draw_earth_background(ax)

        orbit = orbits[sat_id]
        lons, lats = _compute_ground_track(orbit, 24, 200)
        ax.plot(lons, lats, color="#4d96ff88", linewidth=0.6)

        # 高亮显示有接触的时刻
        contact_slots = [
            ts for ts in range(sim.contact_matrix.num_timeslots)
            if sim.contact_matrix._simple[sat_id, ts] >= 0
        ]

        if contact_slots:
            t_min = [ts * 24 * 60 / sim.contact_matrix.num_timeslots for ts in contact_slots]
            mark_lats = []
            mark_lons = []
            for t in t_min:
                lat, lon = orbit.position_at_time(t)
                mark_lats.append(lat)
                mark_lons.append(_wrap_lon(lon))
            ax.scatter(mark_lons, mark_lats, c="#51cf66", s=8, alpha=0.7,
                      zorder=5, label=f"{len(contact_slots)} contacts")

        # 地面站
        for j in range(len(gss)):
            gs = gss[j]
            if gs:
                ax.scatter(gs.lon_deg, gs.lat_deg,
                          c="#ff6b6b", s=50, marker="^",
                          edgecolors="white", linewidths=0.5)

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_facecolor("#1a3a5c")
        ax.tick_params(colors="white", labelsize=7)
        ax.grid(True, alpha=0.1, color="white")
        ax.set_title(f"SAT-{sat_id} ({orbits[sat_id].elements.inclination_deg:.0f}deg incl.)",
                    color="white", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="lower left", facecolor="#0d1117",
                 labelcolor="white", framealpha=0.5)

    fig.suptitle(tf("Demo 5: Ground Tracks with Contact Highlights", lang),
                color="white", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "demo5_ground_tracks.png"),
               dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("Saved: demo5_ground_tracks.png")


def main(lang: str = "en"):
    print(" SpaceFL — Step 1+2 综合可视化演示")
    print("=" * 60)

    # 运行所有演示
    demo1_multi_cluster(lang=lang)
    demo2_custom_registry(lang=lang)
    demo3_custom_satellites(lang=lang)
    demo4_comparison(lang=lang)
    demo5_ground_track(lang=lang)

    print("\n" + "=" * 60)
    print(f" All demos complete! Output dir: {OUTPUT_DIR}")
    print(" Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"   {f} ({size/1024:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "zh"], default="en",
                        help="语言 (默认: en)")
    args = parser.parse_args()
    main(lang=args.lang)
