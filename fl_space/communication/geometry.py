"""轨道适配：既有 ECEF km / 轨道分钟 → 通信层 ECEF m / 秒。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

from .models import GeometrySample, finite

PositionProvider = Callable[[float], Sequence[float]]
WGS84_A_M = 6378137.0
WGS84_B_M = 6356752.3142


def orbit_position_provider(orbit) -> PositionProvider:
    """适配 KeplerOrbit.ecef_at_time；time_s 相对同一轨道 epoch。"""

    def position_m(time_s: float) -> tuple[float, ...]:
        return tuple(float(component) * 1000 for component in orbit.ecef_at_time(time_s / 60))

    return position_m


def _position(position: Sequence[float]) -> tuple[float, ...]:
    if len(position) != 3 or not all(math.isfinite(value) for value in position):
        raise ValueError("ECEF 必须是 3 个有限的米制分量")
    return tuple(position)


def line_of_sight(tx_m: Sequence[float], rx_m: Sequence[float]) -> bool:
    """WGS84 线段遮挡，允许表面地面站；1e-12 仅用于浮点容差。"""
    tx = _position(tx_m)
    rx = _position(rx_m)
    scale = (WGS84_A_M, WGS84_A_M, WGS84_B_M)
    normalized = tuple(value / radius for value, radius in zip(tx, scale))
    delta = tuple((right - left) / radius for left, right, radius in zip(tx, rx, scale))
    length_sq = sum(value * value for value in delta)
    if length_sq == 0:
        raise ValueError("通信端点不能重合")
    fraction = max(
        0.0, min(1.0, -sum(left * right for left, right in zip(normalized, delta)) / length_sq)
    )
    closest_sq = sum((left + fraction * right) ** 2 for left, right in zip(normalized, delta))
    return closest_sq >= 1 - 1e-12


def sample_geometry(
    tx: PositionProvider,
    rx: PositionProvider,
    boundaries_s: Sequence[float],
    *,
    link_id: str = "link",
    visibility: Callable[[float], bool] | None = None,
) -> list[GeometrySample]:
    """按左端点采样，零阶保持到下一边界；不外推最后一个采样。

    visibility 可提供星地掩角/显式接触窗口；与 WGS84 遮挡取交集。
    此函数不定位窗口根，调用方须细化步长/插入已知窗口边界。
    """
    if len(boundaries_s) < 2:
        raise ValueError("至少需要两个时间边界")
    for value in boundaries_s:
        finite(value, "boundary_s", 0)
    if any(right <= left for left, right in zip(boundaries_s, boundaries_s[1:])):
        raise ValueError("时间边界必须严格递增")
    result = []
    for start, end in zip(boundaries_s, boundaries_s[1:]):
        tx_m, rx_m = _position(tx(start)), _position(rx(start))
        distance = math.dist(tx_m, rx_m)
        clear = line_of_sight(tx_m, rx_m)
        visible = clear and (visibility is None or bool(visibility(start)))
        result.append(GeometrySample(start, end, distance, visible, link_id))
    return result
