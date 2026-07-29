"""

WGS84 椭球坐标转换工具模块

============================



提供星地链路几何计算所需的全部坐标转换公式，基于 WGS84 椭球模型

（a=6378.137 km, f=1/298.257223563），包括：



1. 地面站大地坐标 → ECEF 地固直角坐标

2. 卫星 ECI 惯性坐标 → ECEF 地固坐标（格林威治恒星时旋转矩阵）

3. 站心 ENU（东北天）坐标系转换

4. 仰角 El / 方位角 Az 计算

5. 仰角衰减系数 k(El)（用于修正链路有效传输速率）



与 skyfield_backend.py 中的 `_compute_topocentric` 公式一致，

但提取为独立可复用的公共模块。

"""



from __future__ import annotations

import math

# ============================================================

# WGS84 椭球常数

# ============================================================

WGS84_A_KM = 6378.137          # 赤道半径 (km)

WGS84_F = 1.0 / 298.257223563  # 扁率

WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F  # 第一偏心率平方



# 地球自转角速率 (rad/s)

EARTH_ANGULAR_VELOCITY_RAD_S = 7.2921159e-5



# 默认最小通信仰角 (°)

DEFAULT_MIN_ELEVATION_DEG = 5.0





# ============================================================

# 1. 地面站大地坐标 → ECEF 地固直角坐标

# ============================================================

def geodetic_to_ecef(

    lat_deg: float,

    lon_deg: float,

    alt_km: float = 0.0,

) -> tuple[float, float, float]:

    """地面站大地坐标转 ECEF 地固直角坐标（WGS84 椭球）。



    公式：

        N = a / sqrt(1 - e?·sin?φ)

        X_g = (N + h)·cosφ·cosλ

        Y_g = (N + h)·cosφ·sinλ

        Z_g = [N·(1 - e?) + h]·sinφ



    Parameters

    ----------

    lat_deg : float

        大地纬度 (°)，北纬为正。

    lon_deg : float

        大地经度 (°)，东经为正。

    alt_km : float

        海拔高度 (km)。



    Returns

    -------

    (x, y, z) : tuple[float, float, float]

        ECEF 坐标 (km)。

    """

    lat = math.radians(lat_deg)

    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)

    cos_lat = math.cos(lat)



    # 卯酉圈曲率半径

    N = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)



    x = (N + alt_km) * cos_lat * math.cos(lon)

    y = (N + alt_km) * cos_lat * math.sin(lon)

    z = (N * (1.0 - WGS84_E2) + alt_km) * sin_lat

    return (x, y, z)





# ============================================================

# 2. 卫星 ECI 惯性坐标 → ECEF 地固坐标

# ============================================================

def eci_to_ecef(
    eci: tuple[float, float, float],
    gmst_rad: float,
    *,
    gmst_table: GmstLookupTable | None = None,  # noqa: F821
) -> tuple[float, float, float]:
    """卫星 ECI 惯性坐标 → ECEF 地固坐标（绕 Z 轴旋转格林威治恒星时）。

    两种模式:
        - 实时模式 (gmst_table=None): 调用 math.cos/math.sin
        - 查表模式 (gmst_table!=None): 从预计算表获取旋转值，避免实时三角函数

    .. math::

        R_z(θ_G) = [[ cosθ_G, -sinθ_G, 0 ],
                    [ sinθ_G,  cosθ_G, 0 ],
                    [ 0,        0,      1 ]]

        r_sat = R_z(θ_G) · r_ECI


    Parameters
    ----------
    eci : (x, y, z)
        卫星 ECI 坐标 (km)。
    gmst_rad : float
        格林威治恒星时 (rad)。
    gmst_table : GmstLookupTable, optional
        预计算恒星时-旋转矩阵映射表，提供时跳过实时 sin/cos。

    Returns
    -------
    (x, y, z) : tuple[float, float, float]
        ECEF 坐标 (km)。
    """
    if gmst_table is not None:
        c, s = gmst_table.lookup(gmst_rad)
    else:
        c = math.cos(gmst_rad)
        s = math.sin(gmst_rad)
    x, y, z = eci
    return (c * x + s * y, -s * x + c * y, z)





def gmst_from_time(seconds_since_epoch: float) -> float:

    """从历元秒数近似计算格林威治恒星时 (rad)。



    简化模型：θ_G(t) = θ_G0 + ω_E·t

    其中 ω_E = 7.2921159e-5 rad/s。



    Parameters

    ----------

    seconds_since_epoch : float

        从 J2000.0 起算的秒数（已含地球自转效应）。



    Returns

    -------

    float

        格林威治恒星时 (rad)，范围 [0, 2π)。

    """

    gmst = EARTH_ANGULAR_VELOCITY_RAD_S * seconds_since_epoch

    return gmst % (2.0 * math.pi)





# ============================================================

# 3. 站心 ENU（东北天）坐标变换矩阵

# ============================================================

def enu_from_ecef_delta(

    dx: float,

    dy: float,

    dz: float,

    lat_deg: float,

    lon_deg: float,

) -> tuple[float, float, float]:

    """将 ECEF 差分矢量转换至站心 ENU（东·北·天）坐标系。



    旋转矩阵 M（大地纬度 φ、经度 λ）：



        M = [[-sinλ,         cosλ,         0    ],

             [-sinφ·cosλ,   -sinφ·sinλ,   cosφ ],

             [ cosφ·cosλ,    cosφ·sinλ,   sinφ ]]



        [E, N, U]^T = M · [ΔX, ΔY, ΔZ]^T



    Parameters

    ----------

    dx, dy, dz : float

        卫星相对地面站的 ECEF 差分矢量 (km)：

        ΔX = X_sat - X_g,  ΔY = Y_sat - Y_g,  ΔZ = Z_sat - Z_g

    lat_deg : float

        地面站大地纬度 (°)。

    lon_deg : float

        地面站大地经度 (°)。



    Returns

    -------

    (east, north, up) : tuple[float, float, float]

        站心 ENU 分量 (km)。

    """

    lat = math.radians(lat_deg)

    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)

    cos_lat = math.cos(lat)

    sin_lon = math.sin(lon)

    cos_lon = math.cos(lon)



    east = -sin_lon * dx + cos_lon * dy

    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz

    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return (east, north, up)





# ============================================================

# 4. 仰角 El / 方位角 Az

# ============================================================

def elevation_azimuth_deg(

    sat_ecef: tuple[float, float, float],

    gs_lat_deg: float,

    gs_lon_deg: float,

    gs_alt_km: float = 0.0,

) -> tuple[float, float]:

    """计算卫星相对地面站的仰角 El 和方位角 Az (°)。



    完整公式（WGS84 椭球 + ENU 矩阵）：



        1. 地面站 ECEF：geodetic_to_ecef(φ, λ, h)

        2. 差分矢量：Δ = sat_ecef - gs_ecef

        3. ENU 变换：[E, N, U]^T = M · Δ

        4. 仰角：El = arcsin(U / sqrt(E? + N? + U?))

        5. 方位角：Az = arctan2(E, N)，修正至 [0°, 360°)



    Parameters

    ----------

    sat_ecef : (x, y, z)

        卫星 ECEF 坐标 (km)。

    gs_lat_deg, gs_lon_deg : float

        地面站大地纬度/经度 (°)。

    gs_alt_km : float

        地面站海拔 (km)。



    Returns

    -------

    (elevation_deg, azimuth_deg) : tuple[float, float]

        仰角 (°)，范围 [-90, 90]；方位角 (°)，范围 [0, 360)。

    """

    gx, gy, gz = geodetic_to_ecef(gs_lat_deg, gs_lon_deg, gs_alt_km)

    dx = sat_ecef[0] - gx

    dy = sat_ecef[1] - gy

    dz = sat_ecef[2] - gz



    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    if dist < 1e-9:

        return (90.0, 0.0)



    east, north, up = enu_from_ecef_delta(dx, dy, dz, gs_lat_deg, gs_lon_deg)



    # 仰角

    sin_el = up / dist

    sin_el = max(-1.0, min(1.0, sin_el))

    el_deg = math.degrees(math.asin(sin_el))



    # 方位角（正北顺时针，0→360）

    az_rad = math.atan2(east, north)

    az_deg = math.degrees(az_rad)

    if az_deg < 0.0:

        az_deg += 360.0



    return (el_deg, az_deg)





# ============================================================

# 5. 仰角衰减系数 k(El)

# ============================================================

def elevation_attenuation_factor(

    el_deg: float,

    min_elevation_deg: float = DEFAULT_MIN_ELEVATION_DEG,

) -> float:

    """计算仰角衰减系数 k(El) ∈ [0, 1]。



    仰角越低，大气路径越长、损耗越大，有效传输速率下降。

    采用线性模型：



        k(El) = clamp((El - El_min) / (90 - El_min), 0, 1)



    含义：

        - El ≤ El_min  → k = 0（不可用链路）

        - El = 45°     → k ≈ 0.47

        - El = 90°     → k = 1.0（天顶，无衰减）



    用于修正下行链路有效速率：



        Rate_eff = C · k(El_avg)

        Data_max = Rate_eff · ΔT



    Parameters

    ----------

    el_deg : float

        仰角 (°)。

    min_elevation_deg : float

        最低通信仰角 (°)，默认 5°。



    Returns

    -------

    float

        衰减系数，范围 [0.0, 1.0]。

    """

    if el_deg <= min_elevation_deg:

        return 0.0

    if el_deg >= 90.0:

        return 1.0

    return (el_deg - min_elevation_deg) / (90.0 - min_elevation_deg)



# ============================================================
# 6. Spherical coarse filter (Dimension 3)
# ============================================================

def spherical_visibility_coarse(
    sat_ecef: tuple[float, float, float],
    gs_lat_deg: float,
    gs_lon_deg: float,
    gs_alt_km: float,
    min_elevation_deg: float = DEFAULT_MIN_ELEVATION_DEG,
) -> bool:
    sx, sy, sz = sat_ecef
    rs_sq = sx * sx + sy * sy + sz * sz
    gx, gy, gz = geodetic_to_ecef(gs_lat_deg, gs_lon_deg, gs_alt_km)
    rg = math.sqrt(gx * gx + gy * gy + gz * gz)
    dx = sx - gx
    dy = sy - gy
    dz = sz - gz
    dr_sq = dx * dx + dy * dy + dz * dz
    sin_el_min = math.sin(math.radians(min_elevation_deg))
    denom = rs_sq - rg * rg - dr_sq
    if abs(denom) < 1e-9:
        return True
    threshold = (rg * math.sqrt(dr_sq)) / denom
    return sin_el_min <= threshold


def simplified_elevation_far(
    sat_ecef: tuple[float, float, float],
    gs_lat_deg: float,
    gs_lon_deg: float,
    gs_alt_km: float,
) -> float:
    sx, sy, sz = sat_ecef
    rs = math.sqrt(sx * sx + sy * sy + sz * sz)
    gx, gy, gz = geodetic_to_ecef(gs_lat_deg, gs_lon_deg, gs_alt_km)
    rg = math.sqrt(gx * gx + gy * gy + gz * gz)
    dot = sx * gx + sy * gy + sz * gz
    cos_theta = max(-1.0, min(1.0, dot / (rs * rg)))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return 90.0
    sin_theta = math.sin(theta)
    num = cos_theta - rg / rs
    el_rad = math.atan2(num, sin_theta)
    return math.degrees(el_rad)


def occlude_by_table(
    azimuth_deg: float,
    elevation_deg: float,
    occlusion_table: dict[int, float],
    bin_size_deg: int = 5,
) -> bool:
    if not occlusion_table:
        return False
    bin_key = (int(azimuth_deg) // bin_size_deg) * bin_size_deg
    min_clear_el = occlusion_table.get(bin_key)
    if min_clear_el is None:
        return False
    return elevation_deg < min_clear_el
