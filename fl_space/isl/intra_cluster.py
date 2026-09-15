"""
簇内 ISL LOS 窗口计算 — 适配 SpaceFL KeplerOrbit。

对每个 cluster 内按索引排序后取相邻对，沿仿真时间按 step 采样，
计算每对卫星间的 WGS84 椭球遮挡，合并连续可见采样点为 ISLWindow。

环形闭环：
    每簇 n 颗卫星枚举 [(0,1), (1,2), ..., (n-1, 0)]，
    n==1 跳过，n==2 仅枚举 (0,1)。

与师兄项目的差异：
    - 不依赖 Skyfield/SGP4 — 从 KeplerOrbit 直接计算 ECEF
    - 输入简化为 (sat_name → ECEF array) + cluster_map
    - 通过 ISLCalculator 接口暴露，可被替换
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from fl_space.isl.base import ISLCalculator, ISLConfig, ISLWindow
from fl_space.isl.los import is_los_clear_ecef

# ── 纯函数：合并布尔可见性序列为连续窗口 ────────────────────────


def _merge_visibility_to_windows(
    *,
    satellite_a: str,
    satellite_b: str,
    cluster_id: Optional[str],
    sample_times_utc: Sequence[datetime],
    visibility: Sequence[bool],
    ranges_km: Sequence[float],
) -> list[ISLWindow]:
    """把布尔可见性序列合并为连续可见窗口。"""
    n = len(sample_times_utc)
    if not (n == len(visibility) == len(ranges_km)):
        raise ValueError("可见性 / 距离序列长度不一致")

    windows: list[ISLWindow] = []
    start_idx: Optional[int] = None
    cur_min: Optional[float] = None
    cur_max: Optional[float] = None

    for i in range(n):
        if visibility[i]:
            if start_idx is None:
                start_idx = i
                cur_min = ranges_km[i]
                cur_max = ranges_km[i]
            else:
                if ranges_km[i] < cur_min:
                    cur_min = ranges_km[i]
                if ranges_km[i] > cur_max:
                    cur_max = ranges_km[i]
        else:
            if start_idx is not None:
                windows.append(
                    ISLWindow(
                        satellite_a=satellite_a,
                        satellite_b=satellite_b,
                        cluster_id=cluster_id,
                        start_utc=sample_times_utc[start_idx],
                        end_utc=sample_times_utc[i - 1],
                        min_range_km=float(cur_min),
                        max_range_km=float(cur_max),
                    )
                )
                start_idx = None
                cur_min = None
                cur_max = None

    if start_idx is not None:
        windows.append(
            ISLWindow(
                satellite_a=satellite_a,
                satellite_b=satellite_b,
                cluster_id=cluster_id,
                start_utc=sample_times_utc[start_idx],
                end_utc=sample_times_utc[n - 1],
                min_range_km=float(cur_min),
                max_range_km=float(cur_max),
            )
        )

    return windows


def _adjacent_pairs(n: int) -> list[tuple[int, int]]:
    """生成 [0..n) 的相邻对（含闭环 (n-1, 0)）。"""
    if n < 2:
        return []
    if n == 2:
        return [(0, 1)]
    return [(i, (i + 1) % n) for i in range(n)]


def _ensure_utc(dt: datetime) -> datetime:
    """确保 datetime 为 timezone-aware UTC。"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ── 顶层函数（兼容函数式调用）──────────────────────────────────


def compute_intra_cluster_los_windows(
    ecef_positions: dict[str, np.ndarray],
    cluster_assignments: dict[str, Optional[str]],
    sample_times: Sequence[datetime],
    *,
    atmosphere_buffer_km: float = 0.0,
) -> list[ISLWindow]:
    """计算每个 cluster 内相邻卫星对的 LOS 窗口。

    Parameters
    ----------
    ecef_positions : dict[str, np.ndarray]
        卫星名称 → ECEF 位置数组，shape (3, N_samples)，单位 km。
    cluster_assignments : dict[str, str or None]
        卫星名称 → 星簇 ID。None 表示不分组。
    sample_times : Sequence[datetime]
        采样时间点列表，长度 = N_samples。
    atmosphere_buffer_km : float
        椭球膨胀余量 (km)。默认 0。

    Returns
    -------
    list[ISLWindow]
        所有 ISL 窗口，按 start_utc 升序排列。
    """
    if len(sample_times) < 2:
        return []

    # 校验 ECEF 形状一致性
    n_samples = len(sample_times)
    for name, arr in ecef_positions.items():
        if arr.shape[0] != 3:
            raise ValueError(f"卫星 {name} ECEF 数组第一维应为 3，得到 {arr.shape[0]}")
        if arr.shape[1] != n_samples:
            raise ValueError(
                f"卫星 {name} ECEF 样本数 {arr.shape[1]} 与时间点数 {n_samples} 不一致"
            )

    # 按 cluster 分组
    by_cluster: dict[Optional[str], list[str]] = defaultdict(list)
    for name, cid in cluster_assignments.items():
        if name not in ecef_positions:
            continue
        by_cluster[cid].append(name)
    for names in by_cluster.values():
        names.sort()

    all_windows: list[ISLWindow] = []
    for cid, names in by_cluster.items():
        n_in_cluster = len(names)
        for i, j in _adjacent_pairs(n_in_cluster):
            name_a = names[i]
            name_b = names[j]
            ra = ecef_positions[name_a]  # (3, N)
            rb = ecef_positions[name_b]

            visibility: list[bool] = []
            ranges_km: list[float] = []
            for k in range(n_samples):
                r_a = ra[:, k]
                r_b = rb[:, k]
                visibility.append(
                    is_los_clear_ecef(r_a, r_b, atmosphere_buffer_km=atmosphere_buffer_km)
                )
                ranges_km.append(float(np.linalg.norm(r_a - r_b)))

            windows = _merge_visibility_to_windows(
                satellite_a=name_a,
                satellite_b=name_b,
                cluster_id=cid,
                sample_times_utc=sample_times,
                visibility=visibility,
                ranges_km=ranges_km,
            )
            all_windows.extend(windows)

    all_windows.sort(key=lambda w: w.start_utc)
    return all_windows


# ── 可插拔计算器实现 ───────────────────────────────────────────


class WGS84ISLCalculator(ISLCalculator):
    """基于 WGS84 椭球遮挡的 ISL 计算器（默认实现）。

    参数化大气余量，可从 ISLConfig 或直接传参。

    Parameters
    ----------
    atmosphere_buffer_km : float
        大气余量 (km)。0 = 纯 WGS84（STK 默认）。
    """

    def __init__(self, atmosphere_buffer_km: float = 0.0):
        self._buffer = atmosphere_buffer_km

    def compute(
        self,
        ecef_positions: dict[str, np.ndarray],
        cluster_assignments: dict[str, Optional[str]],
        sample_times: Sequence[datetime],
        **kwargs,
    ) -> list[ISLWindow]:
        buffer = kwargs.get("atmosphere_buffer_km", self._buffer)
        return compute_intra_cluster_los_windows(
            ecef_positions=ecef_positions,
            cluster_assignments=cluster_assignments,
            sample_times=sample_times,
            atmosphere_buffer_km=buffer,
        )

    @property
    def name(self) -> str:
        return f"WGS84ISL(buffer={self._buffer}km)"


# ── 工厂函数 ───────────────────────────────────────────────────


def create_isl_calculator(config: ISLConfig) -> ISLCalculator:
    """根据配置创建 ISL 计算器实例。

    支持：
        - 内置: "wgs84", "disabled"
        - 自定义: "path/to/module.py:ClassName"

    Parameters
    ----------
    config : ISLConfig
        ISL 配置。

    Returns
    -------
    ISLCalculator
        计算器实例。

    Raises
    ------
    ImportError
        自定义计算器模块无法导入。
    """
    if not config.enabled:
        from fl_space.isl.base import NoISLCalculator

        return NoISLCalculator()

    calc_spec = config.calculator

    if calc_spec == "wgs84":
        return WGS84ISLCalculator(atmosphere_buffer_km=config.atmosphere_buffer_km)
    elif calc_spec == "disabled":
        from fl_space.isl.base import NoISLCalculator

        return NoISLCalculator()
    elif ":" in calc_spec:
        # 自定义: "path/to/module.py:MyCalculator"
        import importlib.util
        import os

        module_path, class_name = calc_spec.rsplit(":", 1)
        if not os.path.isabs(module_path):
            raise ImportError(f"自定义 ISL 计算器路径必须是绝对路径: {module_path}")
        spec = importlib.util.spec_from_file_location("_custom_isl", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载自定义 ISL 模块: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, class_name, None)
        if cls is None:
            raise ImportError(f"模块 {module_path} 中找不到类 {class_name}")
        if not issubclass(cls, ISLCalculator):
            raise TypeError(f"{class_name} 必须继承 ISLCalculator")
        return cls()
    else:
        raise ValueError(
            f"未知 ISL 计算器: {calc_spec}。支持: wgs84, disabled, path/to/custom.py:ClassName"
        )
