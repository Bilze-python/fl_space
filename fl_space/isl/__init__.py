"""
ISL (Inter-Satellite Link) 模块 — 可插拔星间链路计算。

提供：
    - WGS84 椭球遮挡判定 (los.py)
    - 簇内 ISL LOS 窗口计算 (intra_cluster.py)
    - 抽象基类 (base.py) — 支持自定义 ISL 计算器

设计原则：
    - 模块化可替换：用户可通过 --isl wgs84|disabled|path/to/custom.py:ClassName 切换
    - 纯数学计算：不依赖 Skyfield/SGP4，输入仅需 ECEF 坐标
    - 分层隔离：los.py 只做几何判定，intra_cluster.py 只做窗口合并

用法::

    from fl_space.isl import ISLCalculator, WGS84ISLCalculator, ISLConfig, ISLWindow

    # 默认 WGS84 计算器
    calc = WGS84ISLCalculator(atmosphere_buffer_km=80.0)

    # 自定义计算器（实现 ISLCalculator 接口）
    from fl_space.isl import ISLCalculator
    class MyISL(ISLCalculator):
        def compute(self, ecef_positions, cluster_map, sample_times, **kwargs):
            ...

参考：
    - 师兄项目 D:\\auot_fl_space\\src\\autofly\\core\\earth_los.py (WGS84 遮挡)
    - 师兄项目 D:\\auot_fl_space\\src\\autofly\\domain\\intra_cluster.py (ISL 窗口)
"""

from fl_space.isl.base import ISLCalculator, ISLConfig, ISLWindow, NoISLCalculator
from fl_space.isl.intra_cluster import (
    WGS84ISLCalculator,
    compute_intra_cluster_los_windows,
)
from fl_space.isl.los import WGS84_A_KM, WGS84_B_KM, is_los_clear_ecef

__all__ = [
    "WGS84_A_KM",
    "WGS84_B_KM",
    "ISLCalculator",
    "ISLConfig",
    "ISLWindow",
    "NoISLCalculator",
    "WGS84ISLCalculator",
    "compute_intra_cluster_los_windows",
    "is_los_clear_ecef",
]
