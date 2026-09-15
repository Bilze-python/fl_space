"""
基础太空物理 — 研究级低阶轨道、火箭和大气阻力估算。

本模块使用 SI 单位：
    - 距离: m
    - 质量: kg
    - 时间: s
    - 速度: m/s
    - 加速度: m/s²

这些公式用于快速实验和游戏/教学模拟，不替代高保真飞行动力学、
六自由度姿态、真实大气或任务认证软件。
"""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6_378_137.0
EARTH_MU_M3_S2 = 3.986004418e14
EARTH_ROTATION_RATE_RAD_S = 7.2921150e-5
STANDARD_GRAVITY_MPS2 = 9.80665
SEA_LEVEL_DENSITY_KG_M3 = 1.225
EARTH_SCALE_HEIGHT_M = 8_500.0
SPEED_OF_LIGHT_MPS = 299_792_458.0


def circular_orbit_velocity_mps(
    altitude_km: float,
    *,
    radius_m: float = EARTH_RADIUS_M,
    mu_m3_s2: float = EARTH_MU_M3_S2,
) -> float:
    """计算圆轨道速度。"""
    if altitude_km <= 0:
        raise ValueError("轨道高度必须为正")
    orbit_radius_m = radius_m + altitude_km * 1000.0
    return math.sqrt(mu_m3_s2 / orbit_radius_m)


def circular_orbit_period_s(
    altitude_km: float,
    *,
    radius_m: float = EARTH_RADIUS_M,
    mu_m3_s2: float = EARTH_MU_M3_S2,
) -> float:
    """计算圆轨道周期 T = 2π sqrt(r³ / μ)。

    该值用于卫星寿命、蚀期能量和通信调度的一阶估算。
    """
    if altitude_km <= 0:
        raise ValueError("轨道高度必须为正")
    orbit_radius_m = radius_m + altitude_km * 1000.0
    return 2.0 * math.pi * math.sqrt(orbit_radius_m**3 / mu_m3_s2)


def escape_velocity_mps(
    altitude_km: float,
    *,
    radius_m: float = EARTH_RADIUS_M,
    mu_m3_s2: float = EARTH_MU_M3_S2,
) -> float:
    """计算指定高度的逃逸速度。"""
    if altitude_km < 0:
        raise ValueError("高度不能为负")
    orbit_radius_m = radius_m + altitude_km * 1000.0
    return math.sqrt(2.0 * mu_m3_s2 / orbit_radius_m)


def launch_site_rotation_velocity_mps(
    latitude_deg: float,
    *,
    radius_m: float = EARTH_RADIUS_M,
    rotation_rate_rad_s: float = EARTH_ROTATION_RATE_RAD_S,
) -> float:
    """计算发射场因地球自转获得的切向速度。"""
    if not -90 <= latitude_deg <= 90:
        raise ValueError("纬度必须在 [-90, 90] 度")
    return rotation_rate_rad_s * radius_m * math.cos(math.radians(latitude_deg))


def plane_change_delta_v_mps(
    velocity_mps: float,
    inclination_change_deg: float,
) -> float:
    """计算同一轨道速度下的理想变轨速度。"""
    if velocity_mps < 0:
        raise ValueError("轨道速度不能为负")
    if not 0 <= inclination_change_deg <= 180:
        raise ValueError("倾角变化必须在 [0, 180] 度")
    return 2.0 * velocity_mps * math.sin(math.radians(inclination_change_deg) / 2.0)


def insertion_delta_v_terms_mps(
    altitude_km: float,
    inclination_deg: float,
    launch_latitude_deg: float,
    *,
    gravity_loss_mps: float = 1500.0,
    drag_loss_mps: float = 150.0,
    steering_loss_mps: float = 100.0,
) -> tuple[float, float, float, float]:
    """返回入轨速度预算的分解项。

    返回顺序为 ``(轨道速度, 自转收益, 平面变化代价, 总损失)``，单位均为 m/s。
    这是"入轨速度预算"的唯一实现入口，``estimated_insertion_delta_v_mps``
    只是把四项组合成总和，避免预算报告和约束检查出现两套公式。
    """
    if any(value < 0 for value in (gravity_loss_mps, drag_loss_mps, steering_loss_mps)):
        raise ValueError("速度损失不能为负")
    losses = gravity_loss_mps + drag_loss_mps + steering_loss_mps
    orbital_speed = circular_orbit_velocity_mps(altitude_km)
    rotation_credit = launch_site_rotation_velocity_mps(launch_latitude_deg)
    plane_change = 0.0
    if inclination_deg < abs(launch_latitude_deg):
        plane_change = plane_change_delta_v_mps(
            orbital_speed,
            abs(launch_latitude_deg) - inclination_deg,
        )
    return orbital_speed, rotation_credit, plane_change, losses


def estimated_insertion_delta_v_mps(
    altitude_km: float,
    inclination_deg: float,
    launch_latitude_deg: float,
    *,
    gravity_loss_mps: float = 1500.0,
    drag_loss_mps: float = 150.0,
    steering_loss_mps: float = 100.0,
) -> float:
    """估算从地面发射到目标圆轨道所需的速度预算。

    这是面向早期方案比较的低阶模型。它把轨道速度、地球自转收益、
    重力损失、大气阻力损失和转向损失分开，便于替换成更高保真模型。
    轨道倾角低于发射场纬度时，模型保守地加入平面变化代价。
    """
    orbital_speed, rotation_credit, plane_change, losses = insertion_delta_v_terms_mps(
        altitude_km,
        inclination_deg,
        launch_latitude_deg,
        gravity_loss_mps=gravity_loss_mps,
        drag_loss_mps=drag_loss_mps,
        steering_loss_mps=steering_loss_mps,
    )
    return orbital_speed + losses + plane_change - rotation_credit


def launch_azimuth_deg(
    inclination_deg: float,
    latitude_deg: float,
) -> float:
    """由目标倾角和发射场纬度估算发射方位角（自北顺时针为正，单位度）。

    一阶几何关系 ``sin(A) = cos(i) / cos(φ)``。当 ``|cos i| > cos φ``
    （即目标倾角低于发射场纬度）时，不加平面变化就无法直接从该场址进入
    目标轨道面，函数抛出 ``ValueError``；这正是一阶可行性判断需要的语义。

    由于 ``sin A`` 有两个解，函数按以下固定分支取值并写清假设:

    - 顺行轨道取东向分支，逆行轨道取西向分支；
    - 南半球发射场取北向分支。

    该结果不包含航区、射向安全、地面轨迹限制和真实发射窗口。
    """
    if not 0 <= inclination_deg <= 180:
        raise ValueError("轨道倾角必须在 [0, 180] 度")
    if not -90 <= latitude_deg <= 90:
        raise ValueError("纬度必须在 [-90, 90] 度")
    cos_latitude = math.cos(math.radians(latitude_deg))
    if cos_latitude <= 0:
        raise ValueError("极点发射场不在当前一阶模型范围内")
    ratio = math.cos(math.radians(inclination_deg)) / cos_latitude
    if not -1.0 <= ratio <= 1.0:
        raise ValueError(
            f"目标倾角 {inclination_deg:.1f} 度低于发射场纬度 {abs(latitude_deg):.1f} 度，"
            "需要额外的平面变化，无法给出直接发射方位角"
        )
    azimuth = math.degrees(math.asin(ratio))
    if inclination_deg > 90.0:
        azimuth = 180.0 - azimuth
    if latitude_deg < 0:
        azimuth = 180.0 - azimuth
    return (azimuth + 360.0) % 360.0


def hohmann_transfer_burns_mps(
    from_altitude_km: float,
    to_altitude_km: float,
    *,
    radius_m: float = EARTH_RADIUS_M,
    mu_m3_s2: float = EARTH_MU_M3_S2,
) -> tuple[float, float]:
    """计算两个圆轨道之间霍曼转移的两次点火速度增量 (m/s)。

    返回 ``(第一次点火, 第二次点火)``。抬高轨道时两次都为正，降低轨道时
    两次都为负。霍曼转移只在共面圆轨道之间最优，且忽略了点火弧段、
    有限推力和摄动。
    """
    if from_altitude_km <= 0 or to_altitude_km <= 0:
        raise ValueError("转移轨道高度必须为正")
    r1 = radius_m + from_altitude_km * 1000.0
    r2 = radius_m + to_altitude_km * 1000.0
    semi_major_m = 0.5 * (r1 + r2)
    v1 = math.sqrt(mu_m3_s2 / r1)
    v2 = math.sqrt(mu_m3_s2 / r2)
    perigee_speed = math.sqrt(mu_m3_s2 * (2.0 / r1 - 1.0 / semi_major_m))
    apogee_speed = math.sqrt(mu_m3_s2 * (2.0 / r2 - 1.0 / semi_major_m))
    return perigee_speed - v1, v2 - apogee_speed


def hohmann_transfer_delta_v_mps(
    from_altitude_km: float,
    to_altitude_km: float,
    *,
    radius_m: float = EARTH_RADIUS_M,
    mu_m3_s2: float = EARTH_MU_M3_S2,
) -> float:
    """计算霍曼转移两次点火速度增量之和 (m/s)。"""
    first_burn, second_burn = hohmann_transfer_burns_mps(
        from_altitude_km,
        to_altitude_km,
        radius_m=radius_m,
        mu_m3_s2=mu_m3_s2,
    )
    return first_burn + second_burn


def wavelength_m(frequency_hz: float) -> float:
    """由频率计算自由空间波长 (m)。"""
    if frequency_hz <= 0:
        raise ValueError("频率必须为正")
    return SPEED_OF_LIGHT_MPS / frequency_hz


def free_space_path_loss_db(
    distance_m: float,
    frequency_hz: float,
) -> float:
    """自由空间路径损耗 ``FSPL = 20 log10(4πd / λ)`` (dB)。

    只描述几何扩散损耗，不包含大气吸收、雨衰、指向误差和多径。
    """
    if distance_m <= 0:
        raise ValueError("链路距离必须为正")
    return 20.0 * math.log10(4.0 * math.pi * distance_m / wavelength_m(frequency_hz))


def parabolic_antenna_gain_dbi(
    diameter_m: float,
    frequency_hz: float,
    *,
    efficiency: float = 0.55,
) -> float:
    """抛物面天线增益 ``G = 10 log10(η (πD / λ)²)`` (dBi)。"""
    if diameter_m <= 0:
        raise ValueError("天线口径必须为正")
    if not 0 < efficiency <= 1.0:
        raise ValueError("天线效率必须在 (0, 1]")
    gain_linear = efficiency * (math.pi * diameter_m / wavelength_m(frequency_hz)) ** 2
    return 10.0 * math.log10(gain_linear)


def received_power_dbw(
    transmit_power_w: float,
    transmit_gain_dbi: float,
    receive_gain_dbi: float,
    distance_m: float,
    frequency_hz: float,
    *,
    additional_losses_db: float = 0.0,
) -> float:
    """估算接收功率 ``P_rx = P_tx G_tx G_rx / FSPL`` (dBW)。

    一阶链路预算，用于研究不同频段、口径和距离下的相对差异；
    不替代包含噪声温度、调制编码和多址干扰的完整链路计算。
    """
    if transmit_power_w <= 0:
        raise ValueError("发射功率必须为正")
    eirp_dbw = 10.0 * math.log10(transmit_power_w) + transmit_gain_dbi
    return (
        eirp_dbw
        + receive_gain_dbi
        - free_space_path_loss_db(distance_m, frequency_hz)
        - additional_losses_db
    )


def tsiolkovsky_delta_v_mps(
    initial_mass_kg: float,
    final_mass_kg: float,
    specific_impulse_s: float,
) -> float:
    """计算齐奥尔科夫斯基理想速度增量。"""
    if initial_mass_kg <= final_mass_kg or final_mass_kg <= 0:
        raise ValueError("初始质量必须大于末端质量，且末端质量必须为正")
    if specific_impulse_s <= 0:
        raise ValueError("比冲必须为正")
    return specific_impulse_s * STANDARD_GRAVITY_MPS2 * math.log(initial_mass_kg / final_mass_kg)


def staged_vehicle_delta_v_mps(
    stages: list[tuple[float, float, float]],
    payload_mass_kg: float,
) -> float:
    """计算多级火箭理想总速度增量。

    Parameters
    ----------
    stages : list[tuple[float, float, float]]
        每级的 ``(干质量 kg, 推进剂质量 kg, 比冲 s)``，顺序为一级到末级。
    payload_mass_kg : float
        末级需要推动的有效载荷质量。
    """
    if payload_mass_kg <= 0:
        raise ValueError("载荷质量必须为正")
    if not stages:
        raise ValueError("至少需要一个火箭级")

    upper_stack_mass = payload_mass_kg
    total_delta_v = 0.0
    for dry_mass_kg, propellant_mass_kg, specific_impulse_s in reversed(stages):
        if dry_mass_kg <= 0 or propellant_mass_kg <= 0:
            raise ValueError("火箭级质量必须为正")
        initial_mass = dry_mass_kg + propellant_mass_kg + upper_stack_mass
        final_mass = dry_mass_kg + upper_stack_mass
        total_delta_v += tsiolkovsky_delta_v_mps(
            initial_mass,
            final_mass,
            specific_impulse_s,
        )
        upper_stack_mass += dry_mass_kg
    return total_delta_v


def exponential_atmosphere_density_kg_m3(
    altitude_km: float,
    *,
    sea_level_density_kg_m3: float = SEA_LEVEL_DENSITY_KG_M3,
    scale_height_m: float = EARTH_SCALE_HEIGHT_M,
) -> float:
    """使用指数模型估算大气密度。"""
    if altitude_km < 0:
        raise ValueError("高度不能为负")
    if sea_level_density_kg_m3 <= 0 or scale_height_m <= 0:
        raise ValueError("大气模型参数必须为正")
    return sea_level_density_kg_m3 * math.exp(-(altitude_km * 1000.0) / scale_height_m)


def dynamic_pressure_pa(
    altitude_km: float,
    velocity_mps: float,
    *,
    density_model: str = "exponential",
) -> float:
    """估算动压 q = 1/2 rho v²。"""
    if velocity_mps < 0:
        raise ValueError("速度不能为负")
    if density_model != "exponential":
        raise ValueError(f"不支持的大气模型: {density_model}")
    density = exponential_atmosphere_density_kg_m3(altitude_km)
    return 0.5 * density * velocity_mps**2


def drag_acceleration_mps2(
    altitude_km: float,
    velocity_mps: float,
    drag_coefficient: float,
    cross_section_area_m2: float,
    mass_kg: float,
) -> float:
    """估算沿速度反方向的大气阻力加速度大小。"""
    if drag_coefficient <= 0 or cross_section_area_m2 <= 0 or mass_kg <= 0:
        raise ValueError("阻力系数、迎风面积和质量必须为正")
    return (
        dynamic_pressure_pa(altitude_km, velocity_mps)
        * drag_coefficient
        * cross_section_area_m2
        / mass_kg
    )
