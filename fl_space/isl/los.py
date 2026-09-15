"""
WGS84 椭球视线遮挡判定 — ISL 物理核心。

给定两点 ECEF 位置矢量，判断连线是否被地球椭球遮挡。

算法（来自师兄项目 earth_los.py）：
    1. 按 (a+h, a+h, b+h) 归一化，椭球退化为单位球
    2. 求线段到原点的最近距离平方
    3. min_dist² > 1 → 链路畅通

关键约束：
    **禁止使用"距离 < 阈值"简化** — 500km LEO 同簇卫星距离可 < 2000km，
    但若分别位于地球两侧则完全遮挡。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# WGS84 椭球参数 (km)
WGS84_A_KM: float = 6378.137
WGS84_B_KM: float = 6356.7523142


def is_los_clear_ecef(
    r_a_km: Sequence[float],
    r_b_km: Sequence[float],
    atmosphere_buffer_km: float = 0.0,
) -> bool:
    """判断两点 ECEF 连线是否未被（带余量的）地球椭球遮挡。

    Parameters
    ----------
    r_a_km, r_b_km
        两点的 ECEF 位置矢量，单位 km。可为长度 3 的 list / tuple / np.ndarray。
    atmosphere_buffer_km
        大气余量 (km)。默认 0 = 纯 WGS84 椭球（与 STK 默认 LOS 对齐）。
        常用敏感性实验值 80。必须 >= 0。

    Returns
    -------
    bool
        True 表示链路畅通；False 表示被椭球遮挡。

    Raises
    ------
    ValueError
        输入维度不是 3、或大气余量为负。
    """
    if atmosphere_buffer_km < 0:
        raise ValueError("atmosphere_buffer_km 不能为负数")

    ra = np.asarray(r_a_km, dtype=float)
    rb = np.asarray(r_b_km, dtype=float)
    if ra.shape != (3,) or rb.shape != (3,):
        raise ValueError(f"r_a_km / r_b_km 必须是长度 3 的向量，得到 {ra.shape} / {rb.shape}")

    # 1) 归一化：按 (a+h, a+h, b+h) 缩放，椭球退化为单位球
    a_h = WGS84_A_KM + atmosphere_buffer_km
    b_h = WGS84_B_KM + atmosphere_buffer_km
    scale = np.array([a_h, a_h, b_h])
    ra_n = ra / scale
    rb_n = rb / scale

    # 2) 求线段到原点的最近距离平方
    d = rb_n - ra_n
    dd = float(d @ d)
    if dd == 0.0:
        # 两点重合：距离 = |ra_n|；若 < 1 则在椭球内 → 不可见
        return float(ra_n @ ra_n) > 1.0
    s_star = -float(ra_n @ d) / dd
    # 截断到 [0, 1]
    if s_star < 0.0:
        s_star = 0.0
    elif s_star > 1.0:
        s_star = 1.0
    closest = ra_n + s_star * d
    min_dist_sq = float(closest @ closest)

    return min_dist_sq > 1.0
