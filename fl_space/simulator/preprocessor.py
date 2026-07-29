"""
输入数据预处理优化模块 (论文维度九)
====================================

实现:
1. TLE 批量降噪清洗 (过滤失效/误差过大两行轨道数据)
2. 地面站状态预聚合 (多天线/多站点检修时段合并为统一时间区间)
3. 时间系统统一预转换 (所有输入时刻统一转为 UTC 秒数)

减少定时启动耗时, 避免无意义推演。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# ============================================================
# 1. TLE 批量降噪清洗
# ============================================================

@dataclass
class CleanedTLE:
    """清洗后的 TLE 记录。"""

    sat_name: str
    line1: str
    line2: str
    sat_number: int = 0
    epoch_year: int = 0
    epoch_day: float = 0.0
    bstar: float = 0.0
    inclination_deg: float = 0.0
    eccentricity: float = 0.0
    mean_motion_rev_per_day: float = 0.0
    is_valid: bool = True
    reject_reason: str = ""


def parse_tle_epoch(line1: str) -> tuple[int, float]:
    """解析 TLE 第一行的历元 (年, 日)。"""
    try:
        epoch_str = line1[18:32].strip()
        year = int(epoch_str[:2])
        year = 2000 + year if year < 57 else 1900 + year
        day = float(epoch_str[2:])
        return (year, day)
    except (ValueError, IndexError):
        return (0, 0.0)


def clean_tle(
    name: str,
    line1: str,
    line2: str,
    max_bstar: float = 0.1,
    max_eccentricity: float = 0.3,
    min_altitude_km: float = 150.0,
    max_altitude_km: float = 45000.0,
    max_epoch_age_days: float = 30.0,
    current_year: float = 2026.0,
    current_day: float = 200.0,
) -> CleanedTLE:
    """对单条 TLE 进行有效性校验和清洗。

    Parameters
    ----------
    name : str
        卫星名称。
    line1, line2 : str
        两行轨道数据。
    max_bstar : float
        最大允许 B* (阻力系数), 超限视为数据异常。
    max_eccentricity : float
        最大偏心率, 超限视为深空/异常轨道。
    min_altitude_km, max_altitude_km : float
        有效轨道高度范围。
    max_epoch_age_days : float
        最大 TLE 历元年龄 (天), 超期失效。
    current_year, current_day : float
        当前日期 (年, 年积日)。

    Returns
    -------
    CleanedTLE
    """
    result = CleanedTLE(sat_name=name, line1=line1, line2=line2)

    # 校验行长度
    if len(line1) < 69 or len(line2) < 69:
        result.is_valid = False
        result.reject_reason = "line_too_short"
        return result

    # 校验行号
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        result.is_valid = False
        result.reject_reason = "bad_line_prefix"
        return result

    try:
        # 卫星编号
        result.sat_number = int(line1[2:7].strip())

        # 历元年龄
        epoch_year, epoch_day = parse_tle_epoch(line1)
        result.epoch_year = epoch_year
        result.epoch_day = epoch_day
        if epoch_year > 0:
            age_days = (current_year - epoch_year) * 365.25 + (current_day - epoch_day)
            if abs(age_days) > max_epoch_age_days:
                result.is_valid = False
                result.reject_reason = f"epoch_too_old ({age_days:.0f}d > {max_epoch_age_days}d)"
                return result

        # B* 阻力系数 (行1 chars 53-61)
        bstar_str = line1[53:61].strip()
        if bstar_str:
            float(bstar_str.replace(" ", ""))
            bstar_base = float(line1[53:59].replace(" ", ""))
            bstar_exp = int(line1[59:61])
            result.bstar = bstar_base * (10.0 ** bstar_exp)
            if abs(result.bstar) > max_bstar:
                result.is_valid = False
                result.reject_reason = f"bstar_too_large ({result.bstar:.2e})"
                return result

        # 轨道倾角 (行2 chars 8-16)
        result.inclination_deg = float(line2[8:16].strip())

        # 偏心率 (行2 chars 26-33) — "0.0012345" 隐含前导 "0."
        ecc_str = line2[26:33].strip()
        result.eccentricity = float("0." + ecc_str) if ecc_str else 0.0
        if result.eccentricity > max_eccentricity:
            result.is_valid = False
            result.reject_reason = f"eccentricity_too_high ({result.eccentricity:.4f})"
            return result

        # 平均运动 (rev/day), 行2 chars 52-63
        result.mean_motion_rev_per_day = float(line2[52:63].strip())

        # 推算轨道半长轴 -> 高度
        mu = 398600.4418  # km^3/s^2
        n_rad_s = result.mean_motion_rev_per_day * (2.0 * 3.141592653589793) / 86400.0
        if n_rad_s > 0:
            a_km = (mu / (n_rad_s ** 2)) ** (1.0 / 3.0)
            altitude_km = a_km - 6371.0
            if altitude_km < min_altitude_km:
                result.is_valid = False
                result.reject_reason = f"altitude_too_low ({altitude_km:.0f}km)"
                return result
            if altitude_km > max_altitude_km:
                result.is_valid = False
                result.reject_reason = f"altitude_too_high ({altitude_km:.0f}km)"
                return result

    except (ValueError, IndexError, ZeroDivisionError):
        result.is_valid = False
        result.reject_reason = "parse_error"
        return result

    return result


def batch_clean_tles(
    tles: list[tuple[str, str, str]],
    **kwargs: Any,
) -> tuple[list[CleanedTLE], list[CleanedTLE]]:
    """批量清洗 TLE 列表。

    Returns
    -------
    (valid_list, rejected_list)
    """
    valid: list[CleanedTLE] = []
    rejected: list[CleanedTLE] = []

    for name, l1, l2 in tles:
        cleaned = clean_tle(name, l1, l2, **kwargs)
        if cleaned.is_valid:
            valid.append(cleaned)
        else:
            rejected.append(cleaned)

    return (valid, rejected)


# ============================================================
# 2. 地面站状态预聚合
# ============================================================

@dataclass
class MaintenanceInterval:
    """检修时间区间。"""

    gs_id: int
    start_ts: int
    end_ts: int


@dataclass
class AggregatedState:
    """聚合后的地面站状态。"""

    gs_id: int
    unavailable_ranges: list[tuple[int, int]] = field(default_factory=list)
    reduced_concurrency_ranges: list[tuple[int, int, int]] = field(default_factory=list)


def aggregate_maintenance(
    intervals: list[MaintenanceInterval],
    merge_gap_slots: int = 3,
) -> list[AggregatedState]:
    """将多天线、多站点检修时段合并为统一时间区间。

    相邻检修区间间隔 <= merge_gap_slots 则合并。

    Parameters
    ----------
    intervals : list[MaintenanceInterval]
        原始检修时间区间列表。
    merge_gap_slots : int
        合并间隙阈值 (timeslots)。

    Returns
    -------
    list[AggregatedState]
        聚合后的站点状态。
    """
    by_gs: dict[int, list[tuple[int, int]]] = {}
    for mi in intervals:
        by_gs.setdefault(mi.gs_id, []).append((mi.start_ts, mi.end_ts))

    results: list[AggregatedState] = []
    for gs_id, ranges in by_gs.items():
        ranges.sort()
        merged: list[tuple[int, int]] = []
        for start, end in ranges:
            if merged and start <= merged[-1][1] + merge_gap_slots:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        results.append(AggregatedState(gs_id=gs_id, unavailable_ranges=merged))
    return results


# ============================================================
# 3. 时间系统统一预转换
# ============================================================

@dataclass
class UnifiedTime:
    """统一时间表示 (UTC 秒数)。"""

    utc_seconds: float          # 相对 J2000.0 的 UTC 秒数
    julian_date: float = 0.0    # 儒略日
    timeslot: int = 0           # 对应 timeslot 索引


def convert_to_utc_seconds(
    year: int,
    month: int = 1,
    day: int = 1,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
) -> float:
    """将年月日时分秒转为相对 J2000.0 (2000-01-01 12:00:00 UTC) 的秒数。

    定时演算内部统一使用 UTC 秒数, 不再做时区/历法换算。

    Returns
    -------
    float
        相对 J2000.0 的 UTC 秒数。
    """
    # 儒略日计算
    if month <= 2:
        year -= 1
        month += 12
    a_val = int(year / 100)
    b_val = 2 - a_val + int(a_val / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b_val - 1524.5
    jd += (hour + minute / 60.0 + second / 3600.0) / 24.0
    # J2000.0 = JD 2451545.0
    j2000_jd = 2451545.0
    return (jd - j2000_jd) * 86400.0


def utc_seconds_to_timeslot(
    utc_s: float,
    sim_start_utc_s: float,
    timeslot_duration_min: float,
    max_timeslots: int,
) -> int:
    """将 UTC 秒数映射到 timeslot 索引。

    Parameters
    ----------
    utc_s : float
        目标 UTC 秒数 (相对 J2000.0)。
    sim_start_utc_s : float
        模拟起始 UTC 秒数。
    timeslot_duration_min : float
        每 timeslot 分钟数。
    max_timeslots : int
        最大 timeslot 数。

    Returns
    -------
    int
        timeslot 索引, 钳制在 [0, max_timeslots-1]。
    """
    offset_s = utc_s - sim_start_utc_s
    ts = int(offset_s / (timeslot_duration_min * 60.0))
    return max(0, min(ts, max_timeslots - 1))


def batch_unify_times(
    events: list[dict[str, Any]],
) -> list[UnifiedTime]:
    """批量为事件列表统一时间转换。

    events 每项含: {"year", "month", "day", "hour", "minute", "second"}
    """
    results: list[UnifiedTime] = []
    for ev in events:
        utc_s = convert_to_utc_seconds(
            year=int(ev.get("year", 2026)),
            month=int(ev.get("month", 1)),
            day=int(ev.get("day", 1)),
            hour=int(ev.get("hour", 0)),
            minute=int(ev.get("minute", 0)),
            second=float(ev.get("second", 0.0)),
        )
        results.append(UnifiedTime(utc_seconds=utc_s))
    return results
