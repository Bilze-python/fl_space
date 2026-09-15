"""
默认配置 — 太空联邦学习框架的预设参数

提供各种常用场景的配置预设，用户可直接使用或在此基础上修改。
"""

from typing import Any

# ============================================================
# 天体预设
# ============================================================

BODY_PRESETS: dict[str, dict[str, Any]] = {
    "earth": {
        "name": "Earth",
        "radius_km": 6371.0,
        "GM": 398600.4418,
        "rotation_period_hours": 24.0,
        "flattening": 1.0 / 298.257,
        "atmosphere_height_km": 100.0,
    },
    "mars": {
        "name": "Mars",
        "radius_km": 3389.5,
        "GM": 42828.3,
        "rotation_period_hours": 24.6597,
        "flattening": 1.0 / 169.8,
        "atmosphere_height_km": 80.0,
    },
    "moon": {
        "name": "Moon",
        "radius_km": 1737.4,
        "GM": 4902.8,
        "rotation_period_hours": 27.32 * 24,
        "flattening": 0.0,
        "atmosphere_height_km": 0.0,
    },
}


# ============================================================
# 星座预设
# ============================================================

CONSTELLATION_PRESETS = {
    "default": {
        "num_satellites": 3,
        "num_planes": 1,
        "inclination_deg": 90.0,
        "altitude_km": 500.0,
        "distribution": "uniform",
        "phasing_factor": 0,
    },
    "small_walker": {
        "num_satellites": 5,
        "num_planes": 1,
        "inclination_deg": 90.0,
        "altitude_km": 500.0,
        "distribution": "walker",
        "phasing_factor": 0,
    },
    "medium_walker": {
        "num_satellites": 10,
        "num_planes": 2,
        "inclination_deg": 90.0,
        "altitude_km": 500.0,
        "distribution": "walker",
        "phasing_factor": 1,
    },
    "large_walker": {
        "num_satellites": 20,
        "num_planes": 4,
        "inclination_deg": 90.0,
        "altitude_km": 500.0,
        "distribution": "walker",
        "phasing_factor": 1,
    },
    "starlink_like": {
        "num_satellites": 20,
        "num_planes": 4,
        "inclination_deg": 53.0,
        "altitude_km": 340.0,
        "distribution": "walker",
        "phasing_factor": 2,
    },
    "cluster_5x3": {
        "num_satellites": 15,
        "num_planes": 5,
        "inclination_deg": 90.0,
        "altitude_km": 500.0,
        "distribution": "cluster",
        "phasing_factor": 0,
    },
}


# ============================================================
# 地面站预设
# ============================================================

GS_PRESETS = {
    "paper_13": [
        ("Sioux Falls", 43.55, -96.72),
        ("Sanya", 18.25, 109.5),
        ("Johannesburg", -26.2, 28.03),
        ("Cordoba", -31.4, -64.18),
        ("Tromso", 69.65, 18.95),
        ("Kashi", 39.1, 77.2),
        ("Beijing", 39.9, 116.4),
        ("Neustrelitz", 53.1, 13.1),
        ("Parepare", -2.99, 119.8),
        ("Alice Springs", -25.1, 133.9),
        ("Fairbanks", 64.8, -147.7),
        ("Prince Albert", 53.2, -105.7),
        ("Shadnagar", 17.4, 78.5),
    ],
}


# ============================================================
# 实验配置预设
# ============================================================

EXPERIMENT_CONFIGS = [
    # (config_name, num_sats, num_gs, note)
    ("C1_S3_G2", 3, 2, "最小配置"),
    ("C2_S5_G3", 5, 3, "小型配置"),
    ("C3_S10_G5", 10, 5, "中型配置"),
    ("C4_S15_G7", 15, 7, "中大型配置"),
    ("C5_S20_G10", 20, 10, "大型配置"),
    ("C6_S30_G13", 30, 13, "大规模配置"),
]
