"""
坐标工具模块 — 球面几何与坐标变换

提供:
    - 角度转换 (度↔弧度)
    - Haversine大圆距离
    - 大圆角距计算
    - 地平可见半角
    - 经度归一化
    - 轨道平面坐标变换

所有函数均为纯函数，无副作用，便于测试和复用。
"""

import math

import numpy as np

# ---- 基础角度工具 ----

def deg_to_rad(deg: float) -> float:
    """度转弧度。"""
    return math.radians(deg)


def rad_to_deg(rad: float) -> float:
    """弧度转度。"""
    return math.degrees(rad)


def normalize_angle_rad(angle: float) -> float:
    """归一化角度到 [-π, π]。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def normalize_angle_deg(angle: float) -> float:
    """归一化角度到 [-180, 180]。"""
    return rad_to_deg(normalize_angle_rad(deg_to_rad(angle)))


def normalize_longitude_rad(lon: float) -> float:
    """归一化经度弧度到 [-π, π]。"""
    return normalize_angle_rad(lon)


def normalize_longitude_deg(lon: float) -> float:
    """归一化经度°到 [-180, 180]。"""
    return normalize_angle_deg(lon)


# ---- 球面距离与角距 ----

def haversine_distance_km(
    lat1_rad: float, lon1_rad: float,
    lat2_rad: float, lon2_rad: float,
    radius_km: float = 6371.0,
) -> float:
    """
    计算球面上两点的大圆距离 (Haversine公式)。

    Parameters
    ----------
    lat1_rad, lon1_rad : float
        点1的纬度和经度 (弧度)。
    lat2_rad, lon2_rad : float
        点2的纬度和经度 (弧度)。
    radius_km : float
        球体半径 (km)。

    Returns
    -------
    float
        大圆距离 (km)。
    """
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def angular_distance_rad(
    lat1_rad: float, lon1_rad: float,
    lat2_rad: float, lon2_rad: float,
) -> float:
    """
    计算两点之间的大圆角距 (弧度)。

    使用球面余弦定理，比 Haversine 更简洁，适合
    大距离计算（注意: 对极近距离可能有数值精度问题）。

    Parameters
    ----------
    lat1_rad, lon1_rad : float
        点1的纬度、经度 (弧度)。
    lat2_rad, lon2_rad : float
        点2的纬度、经度 (弧度)。

    Returns
    -------
    float
        角距 (弧度)，范围为 [0, π]。
    """
    cos_dist = (
        math.sin(lat1_rad) * math.sin(lat2_rad) +
        math.cos(lat1_rad) * math.cos(lat2_rad) *
        math.cos(lon1_rad - lon2_rad)
    )
    cos_dist = max(-1.0, min(1.0, cos_dist))  # 数值稳定性裁剪
    return math.acos(cos_dist)


# ---- 地平可见性 ----

def horizon_angle_rad(
    planet_radius_km: float,
    orbit_radius_km: float,
    atmosphere_height_km: float = 0.0,
) -> float:
    """
    计算从轨道看地平线的可见半角 (弧度)。

    这是卫星正下方点到地平线的角距。
    考虑大气折射后有效半径 = planet_radius + atmosphere_height。

    Parameters
    ----------
    planet_radius_km : float
        行星半径 (km)。
    orbit_radius_km : float
        轨道半径 (km) = planet_radius + altitude。
    atmosphere_height_km : float
        大气有效高度 (km)。

    Returns
    -------
    float
        可见半角 (弧度)。
    """
    effective_radius = planet_radius_km + atmosphere_height_km
    return math.acos(effective_radius / orbit_radius_km)


def is_visible(
    sat_lat_rad: float, sat_lon_rad: float,
    gs_lat_rad: float, gs_lon_rad: float,
    planet_radius_km: float,
    orbit_radius_km: float,
    atmosphere_height_km: float = 0.0,
    min_elevation_rad: float = 0.0,
) -> bool:
    """
    判断卫星能否看到地面站。

    Parameters
    ----------
    sat_lat_rad, sat_lon_rad : float
        卫星星下点位置 (弧度)。
    gs_lat_rad, gs_lon_rad : float
        地面站位置 (弧度)。
    planet_radius_km : float
        行星半径。
    orbit_radius_km : float
        轨道半径。
    atmosphere_height_km : float
        大气有效高度。
    min_elevation_rad : float
        地面站最小通信仰角。

    Returns
    -------
    bool
    """
    ang_dist = angular_distance_rad(sat_lat_rad, sat_lon_rad, gs_lat_rad, gs_lon_rad)
    h_angle = horizon_angle_rad(planet_radius_km, orbit_radius_km, atmosphere_height_km)
    return ang_dist < (h_angle - min_elevation_rad)


# ---- 轨道运动 ----

def earth_rotation_offset_deg(elapsed_min: float) -> float:
    """
    地球自转角度偏移 (°)。

    Parameters
    ----------
    elapsed_min : float
        经过的时间 (分钟)。

    Returns
    -------
    float
        积累的自转角度 (°)。
    """
    return elapsed_min * (360.0 / (24 * 60))


def earth_rotation_offset_rad(elapsed_min: float) -> float:
    """
    地球自转角度偏移 (弧度)。
    """
    return deg_to_rad(earth_rotation_offset_deg(elapsed_min))


def rotation_offset_rad(elapsed_min: float, rotation_period_hours: float = 24.0) -> float:
    """
    通用行星自转角度偏移 (弧度)。

    Parameters
    ----------
    elapsed_min : float
        经过的时间 (分钟)。
    rotation_period_hours : float
        行星自转周期 (小时)。
    """
    return math.radians(elapsed_min * (360.0 / (rotation_period_hours * 60)))


# ---- 轨道周期 ----

def orbital_period_min(semi_major_axis_km: float, GM: float) -> float:
    """
    开普勒第三定律 — 轨道周期 (分钟)。

    Parameters
    ----------
    semi_major_axis_km : float
        轨道半长轴 (km)。
    GM : float
        天体引力常数 (km³/s²)。

    Returns
    -------
    float
        轨道周期 (分钟)。
    """
    return (2 * math.pi * math.sqrt(semi_major_axis_km ** 3 / GM)) / 60.0


# ---- 批量运算 ----

def batch_angular_distances_rad(
    sat_lat_rad: float, sat_lon_rad: float,
    gs_coords_rad: np.ndarray,  # shape (N, 2): [[lat, lon], ...]
) -> np.ndarray:
    """
    批量计算卫星到多个地面站的角距（向量化版本，适合大规模计算）。

    Parameters
    ----------
    sat_lat_rad, sat_lon_rad : float
        卫星星下点 (弧度)。
    gs_coords_rad : np.ndarray, shape (N, 2)
        地面站坐标数组。

    Returns
    -------
    np.ndarray, shape (N,)
        到每个地面站的角距 (弧度)。
    """
    gs_lats = gs_coords_rad[:, 0]
    gs_lons = gs_coords_rad[:, 1]

    cos_dists = (
        np.sin(sat_lat_rad) * np.sin(gs_lats) +
        np.cos(sat_lat_rad) * np.cos(gs_lats) *
        np.cos(sat_lon_rad - gs_lons)
    )
    cos_dists = np.clip(cos_dists, -1.0, 1.0)
    return np.arccos(cos_dists)
