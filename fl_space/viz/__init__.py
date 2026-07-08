"""
可视化模块 — 地球卫星轨道可视化。

提供:
    - 2D 地图投影（卫星轨迹 + 地面站）
    - 接触矩阵热力图
    - 3D 轨道视图（可选）
    - 中英文双语支持
"""

from .orbit_plot import (
    OrbitVisualizer,
    plot_constellation_2d,
    plot_contact_heatmap,
    plot_ground_track,
    quick_plot,
)
from .i18n import (
    setup_cjk_font,
    t,
    tf,
)

__all__ = [
    "OrbitVisualizer",
    "plot_constellation_2d",
    "plot_contact_heatmap",
    "plot_ground_track",
    "quick_plot",
    "setup_cjk_font",
    "t",
    "tf",
]
